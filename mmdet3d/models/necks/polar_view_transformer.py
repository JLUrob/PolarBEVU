# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS


@NECKS.register_module()
class PolarLSSViewTransformer(BaseModule):
    """
        Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
    """
    def __init__(
        self,
        grid_config,
        input_img_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False        
        ):
        super(PolarLSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.accelerate = accelerate
        
        self.create_polargrid_infos(**grid_config)
        self.create_frustum(grid_config['depth'], input_img_size, downsample)
        self.depth_net = nn.Conv2d(in_channels, self.D + out_channels, kernel_size=1, padding=0) # self.D is created from self.create_frustum()
        
        self.initial_flag = False

    def create_polargrid_infos(self, rho, theta, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            rho (tuple(float)): Config of grid alone rho axis in format of
                (lower_bound, upper_bound, interval).
            theta (tuple(float)): Config of grid alone theta axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [rho, theta, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [rho, theta, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [rho, theta, z]])

    def create_frustum(self, depth_cfg, input_img_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_img_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_img_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, H_feat, W_feat) + depth_cfg[2] / 2  # !!! TODO 
        self.D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float).view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float).view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        self.frustum = torch.stack((x, y, d), -1)
        #  frustum[k,v_f,u_f]->(u, v, d),  下采样后特征图处(v_f,u_f,k)对应着原图像(u, v, d)
    
    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate transformer of image feature maps to polar grid 

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, h, w, 3)
        """
        B, N, _ = trans.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)  # 将像素坐标(u,v,d)变成齐次坐标(d*u,d*v,d)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        # point : B x 6 x D x h x w x 3
        
        rho = torch.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)
        theta = torch.atan2(points[..., 1], points[..., 0])
        points = torch.stack((rho, theta, points[..., 2]), -1)
    
        return points  #  points[b, n, k, u_f, v_f] -> [x, y, z] -> [rho, theta, z]
    
    
    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, rho_num, theta_num)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)  # B*N x D+C x H x W

        depth_digit = x[:, :self.D, ...]
        imgs_feat = x[:, self.D:self.D + self.out_channels, ...] # B*N x C x H x W
        depth = depth_digit.softmax(dim=1) # B*N x D x H x W
        return self.view_transform(input, depth, imgs_feat)
    
    def view_transform(self, input, depth, imgs_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, imgs_feat)
    
    def pre_compute(self, input):
        if self.initial_flag == False:
            to_polar_coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration(to_polar_coor)
            self.initial_flag = True
    
    def init_acceleration(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Image features coordinate to Polar coordinate
                (B, N_cams, D, H, W, 3).
        """

        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = self.voxel_pooling_prepare(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()
        
    def voxel_pooling_prepare(self, coor):
        """Data preparation for voxel pooling. 无改动

        Args:
            coor (torch.tensor): Image features coordinate to Polar coordinate. (B, N, D, H, W, 3).  coor[b, n, d, u, v] -> [x, y, z] -> [rho, theta, z]

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(coor))  # 栅格化 k, u, v -> x_g, y_g, 0
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1).expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
            
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(), ranks_feat.int().contiguous(), interval_starts.int().contiguous(), interval_lengths.int().contiguous()
    
    def view_transform_core(self, input, depth, imgs_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = imgs_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)  # B x N x H x W x out_C
            depth = depth.view(B, N, self.D, H, W) # B x N x D x H x W
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]), int(self.grid_size[0]), feat.shape[-1])  # (B, Z, Y, X, out_C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth, self.ranks_feat, self.ranks_bev,bev_feat_shape, self.interval_starts, self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])  # coor[b, n, d, u_f, v_f] -> [x, y, z] -> [rho, theta, z]
            bev_feat = self.voxel_pooling(coor, depth.view(B, N, self.D, H, W), imgs_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth

    def voxel_pooling(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = self.voxel_pooling_prepare(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]), int(self.grid_size[0]), feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat
    
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None
    
class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d, with_cp=False):
        super(ASPP, self).__init__()

        self.with_cp = with_cp
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        
        def _inner_forward(x):
        
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(
                x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            return self.relu(x)

        if self.with_cp and x.requires_grad:
            x = checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        
        def _inner_forward(x):
            x = self.fc1(x)
            return self.act(x)
        
        if self.with_cp and x.requires_grad:
            x = checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        
        x = self.drop1(x)
        if self.with_cp and x.requires_grad:
            x = checkpoint(self.fc2, x)
        else:
            x = self.fc2(x)
        
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()
        
    def forward(self, x, x_se):
        
        def _inner_forward(x, x_se):
            x_se = self.conv_reduce(x_se)
            x_se = self.act1(x_se)
            x_se = self.conv_expand(x_se)
            return x * self.gate(x_se)

        if self.with_cp and x.requires_grad:
            x = checkpoint(_inner_forward, x, x_se)
        else:
            x = _inner_forward(x, x_se)
        return x
        
        
class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False):
        super(DepthNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels, with_cp=self.with_cp)
        self.depth_se = SELayer(mid_channels, with_cp=self.with_cp)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels, with_cp=self.with_cp)
        self.context_se = SELayer(mid_channels, with_cp=self.with_cp)  # NOTE: add camera-aware
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels, with_cp=self.with_cp),
            BasicBlock(mid_channels, mid_channels, with_cp=self.with_cp),
            BasicBlock(mid_channels, mid_channels, with_cp=self.with_cp),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels, with_cp=self.with_cp))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        
        if self.with_cp and x.requires_grad:
            x = checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        
        if self.with_cp and context.requires_grad:
            context = checkpoint(self.context_conv, context)
        else:
            context = self.context_conv(context)
        
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        
        if self.with_cp and depth.requires_grad:
            depth = checkpoint(self.depth_conv, depth)
        else:
            depth = self.depth_conv(depth)
        
        return torch.cat([depth, context], dim=1)


class DepthNetV2(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=False):
        super(DepthNetV2, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

    def forward(self, x, mlp_input):
        x = self.reduce_conv(x)
        context = self.context_conv(x)
        depth = self.depth_conv(x)
        return torch.cat([depth, context], dim=1)



class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x


@NECKS.register_module()
class PolarLSSViewTransformerDepth(PolarLSSViewTransformer):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), depthnetv2=False, **kwargs):
        super(PolarLSSViewTransformerDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        if depthnetv2:
            self.depth_net  = DepthNetV2(self.in_channels, self.in_channels,
                                         self.out_channels, self.D, **depthnet_cfg)
        else:
            self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                    self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
                                dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]  

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)
