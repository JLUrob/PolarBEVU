# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import torch
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from mmdet3d.core import circle_nms, xywhr2xyxyr
from mmdet3d.core.post_processing import nms_bev
from mmdet3d.models import builder
from mmdet3d.models.utils import clip_sigmoid

from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from ..builder import HEADS, build_loss


@HEADS.register_module()
class RingSeparateHead(BaseModule):
    """RingSeparateHead for PolarCenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels.
            Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer.
            Default: 1.
        init_bias (float, optional): Initial bias. Default: -2.19.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=3,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None,
                 with_cp=False,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(RingSeparateHead, self).__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(RingConv(c_in, head_conv, final_kernel))
                conv_layers.append(build_norm_layer(norm_cfg, head_conv, postfix=0)[1])
                conv_layers.append(nn.ReLU(inplace=True))
                c_in = head_conv
                
            conv_layers.append(RingConv(head_conv, classes, final_kernel, bias=True))
            
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            if self.with_cp and x.requires_grad:
                ret_dict[head] = checkpoint(self.__getattr__(head), x)
            else:
                ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


class RingConv(nn.Module): ##
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=(0, 1), bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = F.pad(x, (0, 0, 1, 1), mode='circular')
        x = self.conv(x)
        return x



def span_bound(corners):  # 判断是否横跨二三象限
    t1, t2, t3, t4 = torch.atan2(corners[:, 1], corners[:, 0])
    if (t1 * t3 < 0 or t2 * t4 < 0) and (torch.abs(t1) + torch.abs(t2) + torch.abs(t3) + torch.abs(t4)) / 4 > np.pi / 2 :
        return True
    return False

def get_polar_bound(corners, pc_range=torch.tensor([-3.1488, 1., -5.]), voxel_size=torch.tensor([0.0492, 0.5, 8]), out_size_factor=1, n=6):
    """
    corners : 4 x 2, 笛卡尔坐标系下连续的角点, 可以是顺时针方向也可以是逆时针方向
    output:
        theta_bound: theta坐标的下限和上限, 已取整
        rho_bound: rho坐标的下限和上限, 已经取整
    """

    ## 可以优化, 因为rho需要等分点判断, 但是theta不需要
    theta_min = torch.tensor(100000.0).to(corners)
    theta_max = torch.tensor(-100000.0).to(corners)
    rho_min = torch.tensor(100000.0).to(corners)
    rho_max = torch.tensor(-100000.0).to(corners)

    span = span_bound(corners)

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        v = p2 - p1
        ps = p1 + v.expand((n, 2)) * torch.tensor(range(1, n + 1), dtype=torch.float32, device=corners.device).unsqueeze(-1) / n

        thetas = torch.atan2(ps[:, 1], ps[:, 0])
        rhos = torch.sqrt(ps[:, 0] ** 2 + ps[:, 1] ** 2)

        rho_min = torch.min(rho_min, rhos.min())
        rho_max = torch.max(rho_max, rhos.max())

        if span:
            if (thetas >= 0).sum() > 0:
                theta_min = torch.min(theta_min, thetas[thetas >= 0].min())
            if (thetas < 0).sum() > 0:
                theta_max = torch.max(theta_max, thetas[thetas < 0].max())

        else:
            theta_min = torch.min(theta_min, thetas.min())
            theta_max = torch.max(theta_max, thetas.max())

    theta_bound, rho_bound = torch.tensor([theta_min, theta_max]).to(corners), torch.tensor([rho_min, rho_max]).to(corners)

    theta_bound = (theta_bound - pc_range[0]) / voxel_size[0] / out_size_factor
    rho_bound = (rho_bound - pc_range[1]) / voxel_size[1] / out_size_factor

    # 下界舍去, 上界进位
    theta_bound = torch.tensor([torch.floor(theta_bound[0]), torch.ceil(theta_bound[1])], dtype=torch.int32, device=corners.device)
    rho_bound = torch.tensor([torch.floor(rho_bound[0]), torch.ceil(rho_bound[1])], dtype=torch.int32, device=corners.device)

    return theta_bound, rho_bound

def point_in_rect(p, corners):
    """
    判断点是否在corners围成的四边形里
    corners : 4 x 2, 笛卡尔坐标系下连续的角点, 可以是顺时针方向也可以是逆时针方向
    p: [x, y] or shape : n x 2
    """
    def cross(points1, points2):
        """
        points1 : n x 2 or [x, y]
        points2 : n x 2 or [x, y]
        """
        len1 = len(points1.size())
        len2 = len(points2.size())

        if len1 == len2 and len1 == 1:
            return points1[0] * points2[1] - points1[1] * points2[0]
        if len1 != len2:
            points1 = points1.expand_as(points2) if len1 == 1 else points1
            points2 = points2.expand_as(points1) if len2 == 1 else points2
        return points1[:, 0] * points2[:, 1] - points1[:, 1] * points2[:, 0]

    p1, p2, p3, p4 = corners
    return (cross(p2 - p1, p - p1) * cross(p3 - p4, p - p4) <= 0) & \
           (cross(p1 - p4, p - p4) * cross(p2 - p3, p - p3) <= 0)

def rot_angle(points, angle):
    """
    points : n x 2
    将points逆时针旋转angle弧度, 返回 n x 2
    不是基变换
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    R = torch.tensor([[cos, -sin], [sin, cos]], device=points.device)
    return torch.mm(R, points.T).T

def boxes_to_2dcorners(boxes):
    """Given boxes, output corners of 2d box.

    Args:
        boxes (torch.tensor): shape of n x 9

    Returns:
        torch.tensor: shape of n x 4 x 2
    """
    origin = torch.tensor([[-1, 1], [1, 1], [1, -1], [-1, -1]]).to(boxes) * 0.5
    corners = torch.stack([boxes[:, 3:5] * origin[i] for i in range(4)], axis=1) # n x 4 x 2

    return torch.stack([rot_angle(corners[i], boxes[i, 6]) + boxes[i, :2] for i in range(len(corners))], axis=0)


def draw_heatmap(heatmap, box, corners, theta_bound, rho_bound, pc_range=torch.tensor([-3.1488, 1., -5.]), voxel_size=torch.tensor([0.0492, 0.5, 8]), out_size_factor=1, feature_map_size=torch.tensor([128, 128]), heat_threshold=0.5):
    """
    heatmap     : 要赋值的特征图
    box         : x, y, z, x_size, y_size, z_size, yaw ...
    corners     : 4 x 2, 笛卡尔坐标系下连续的角点, 可以是顺时针方向也可以是逆时针方向
    theta_bound : theta坐标的下限和上限, 已取整
    rho_bound   : rho坐标的下限和上限, 已经取整
    feature_map_size : rho_size, theta_size
    """
    center = box[:2]
    wl = box[3:5]
    rot = box[6]

    theta_bound[1] = torch.min(theta_bound[1], feature_map_size[1] - 1)
    rho_bound[1] = torch.min(rho_bound[1], feature_map_size[0] - 1)

    center_theta = (torch.atan2(center[1], center[0]) - pc_range[0]) / voxel_size[0] / out_size_factor
    center_rho = (torch.sqrt(center[0] ** 2 + center[1] ** 2) - pc_range[1]) / voxel_size[1] / out_size_factor

    center_theta = torch.clamp(center_theta, 0, feature_map_size[1]).to(torch.int32)
    center_rho = torch.clamp(center_rho, 0, feature_map_size[0]).to(torch.int32)

    if theta_bound[1] < theta_bound[0]:
        """
        0 ~ theta_bound[1]
        theta_bound[0] ~ feature_map_size[0]-1
        """
        if center_theta <= theta_bound[1]:
            high_idx1 = draw_heatmap_sub(torch.tensor([0, theta_bound[1]]), rho_bound, heatmap, center, wl, rot, corners, torch.tensor([center_theta, center_rho]), pc_range, voxel_size, out_size_factor, feature_map_size, heat_threshold)
            high_idx2 = draw_heatmap_sub(torch.tensor([theta_bound[0], feature_map_size[1] - 1]), rho_bound, heatmap, center, wl, rot, corners, torch.tensor([center_theta + 128.0, center_rho]), pc_range, voxel_size, out_size_factor, feature_map_size, heat_threshold)
        else:
            high_idx1 = draw_heatmap_sub(torch.tensor([0, theta_bound[1]]), rho_bound, heatmap, center, wl, rot, corners, torch.tensor([center_theta - 128.0, center_rho]), pc_range, voxel_size, out_size_factor, feature_map_size, heat_threshold)
            high_idx2 = draw_heatmap_sub(torch.tensor([theta_bound[0], feature_map_size[1] - 1]), rho_bound, heatmap, center, wl, rot, corners, torch.tensor([center_theta, center_rho]), pc_range, voxel_size, out_size_factor, feature_map_size, heat_threshold)
            
        high_heat_idx = torch.cat([high_idx1, high_idx2], axis=0)

    else:
        high_heat_idx = draw_heatmap_sub(theta_bound, rho_bound, heatmap, center, wl, rot, corners, torch.tensor([center_theta, center_rho]), pc_range, voxel_size, out_size_factor, feature_map_size, heat_threshold)

    heatmap[center_theta][center_rho] = 1.0
    
    if ((high_heat_idx[:, 0] == center_theta) & (high_heat_idx[:, 1] == center_rho)).sum().item() == 0:
        high_heat_idx = torch.cat([high_heat_idx, torch.tensor([[center_theta, center_rho]], device=high_heat_idx.device)], axis=0)
    
    return high_heat_idx
    


def draw_heatmap_sub(theta_bound, rho_bound, heatmap, center, wl, rot, corners, pseudo_center, pc_range=torch.tensor([-3.1488, 1., -5.]), voxel_size=torch.tensor([0.0492, 0.5, 8]), out_size_factor=1, feature_map_size=torch.tensor([128, 128]), heat_threshold=0.5):

    # 生成grid中心
    thetas = torch.arange(int(theta_bound[0]), int(theta_bound[1]) + 1).to(heatmap) + 0.5 # m
    rhos = torch.arange(int(rho_bound[0]), int(rho_bound[1]) + 1).to(heatmap) + 0.5       # n

    thetas = thetas.expand((len(rhos), len(thetas))).T
    rhos = rhos.expand((len(thetas), len(rhos)))

    polar_idx = torch.stack((thetas, rhos), axis=-1).view(-1, 2)  # mxn x 2
    # 转化到grid坐标
    theta_coor = polar_idx[:, 0] * voxel_size[0] * out_size_factor + pc_range[0]
    rho_coor = polar_idx[:, 1] * voxel_size[1] * out_size_factor + pc_range[1]

    xs = rho_coor * torch.cos(theta_coor)
    ys = rho_coor * torch.sin(theta_coor)
    cart_coor = torch.stack((xs, ys), axis=-1)
    # 判断点是否在box内
    in_box = point_in_rect(cart_coor, corners).to(device=heatmap.device)

    inbox_polar_idx = polar_idx[in_box].long()
    cart_coor = cart_coor[in_box] - center
    rw_rl = torch.abs(rot_angle(cart_coor, -rot)) / wl * 2  # c x 2  # wl/2
    # 赋值
    heat = 1.0 - rw_rl.max(1)[0]
    heatmap[inbox_polar_idx[:, 0], inbox_polar_idx[:, 1]] = torch.max(heatmap[inbox_polar_idx[:, 0], inbox_polar_idx[:, 1]], heat)
    
    high_heat =  heat > heat_threshold
    high_polar_idx = inbox_polar_idx[high_heat]  # 供regress_box使用
    

    # 为中心不在box内的grid分配heat
    outbox_polar_idx = polar_idx[in_box == False]
    # 向pseudo_center偏移
    offset_vector = pseudo_center.to(outbox_polar_idx) - outbox_polar_idx
    corners_polar_idx = offset_vector / torch.abs(offset_vector) * 0.5 + outbox_polar_idx
    # 调整边界值
    corners_polar_idx[corners_polar_idx[:, 0] < 0.0, 0] = 127.0
    corners_polar_idx[corners_polar_idx[:, 0] >= 128.0, 0] = 0.0
    corners_polar_idx[:, 1] = torch.clamp(corners_polar_idx[:, 1], 0.0, feature_map_size[1]).to(torch.int32)
    # 赋值
    theta_coor = corners_polar_idx[:, 0] * voxel_size[0] * out_size_factor + pc_range[0]
    rho_coor = corners_polar_idx[:, 1] * voxel_size[1] * out_size_factor + pc_range[1]
    xs = rho_coor * torch.cos(theta_coor)
    ys = rho_coor * torch.sin(theta_coor)
    cart_coor = torch.stack((xs, ys), axis=-1) - center
    rw_rl = torch.abs(rot_angle(cart_coor, -rot)) / wl * 2  # c x 2  # wl/2

    in_edge = rw_rl.max(1)[0] <= 1.0
    rw_rl = rw_rl[in_edge]
    outbox_polar_idx = outbox_polar_idx[in_edge].long()
    heat = 1.0 - rw_rl.max(1)[0]
    heatmap[outbox_polar_idx[:, 0], outbox_polar_idx[:, 1]] = torch.max(heatmap[outbox_polar_idx[:, 0], outbox_polar_idx[:, 1]], heat)
    
    return high_polar_idx



def generate_regression_boxes(box, polar_idx=None, r=1, norm_bbox=True, pc_range=torch.tensor([-3.1488, 1., -5.]), voxel_size=torch.tensor([0.0492, 0.5, 8]), out_size_factor=1, feature_map_size=torch.tensor([128, 128]), subtract_center=False, velocity_resolve=False):
    """
    回归方案4 experiment.pdf
    Args:
        box (torch.tensor): x, y, z, x_size, y_size, z_size, yaw, v_x, v_y
        r (int, optional): 回归的半径. Defaults to 1.
        norm_bbox (bool, optional): box_dim log. Defaults to True.
        pc_range (torch.tensor, optional): Defaults to torch.tensor([-3.1488, 1., -5.]).
        voxel_size (torch.tensor, optional): Defaults to torch.tensor([0.0492, 0.5, 8]).
        out_size_factor (int, optional): Defaults to 1.
        feature_map_size (torch.tensor, optional): Defaults to torch.tensor([128, 128]).
    """

    center_cart = box[:2]
    center_polar = torch.tensor([torch.atan2(center_cart[1], center_cart[0]), torch.sqrt(center_cart[0] ** 2 + center_cart[1] ** 2)]).to(box)  # 极坐标系下的center坐标

    center = ((center_polar - pc_range[:2]) / voxel_size[:2] / out_size_factor).long()  # box中心的grid坐标

    if polar_idx == None:
        theta_range = torch.arange(center[0] - r, center[0] + r + 1).to(center)
        rho_range = torch.arange(center[1] - r, center[1] + r + 1).to(center)

        theta_range = theta_range[(theta_range >= 0) & (theta_range < feature_map_size[1])]
        rho_range = rho_range[(rho_range >= 0) & (rho_range < feature_map_size[0])]

        m = theta_range.size(0)
        n = rho_range.size(0)
        theta_range = theta_range.expand((n, m)).T
        rho_range = rho_range.expand((m, n))

        polar_idx = torch.stack((theta_range, rho_range), axis=-1).view(-1, 2)  # grid坐标

    grids_center = polar_idx + 0.5  # grid中心
    # ego坐标系下 极坐标值
    grids_center_theta = grids_center[:, 0] * voxel_size[0] * out_size_factor + pc_range[0]
    grids_center_rho = grids_center[:, 1] * voxel_size[1] * out_size_factor + pc_range[1]
    grids_center_polar = torch.stack((grids_center_theta, grids_center_rho), axis=-1) # n x 2

    # 生成 delta_theta, delta_rho
    delta_polar = center_polar - grids_center_polar

    # 生成 delta_yaw
    yaw = box[6]
    delta_yaw = yaw - (center_polar[0] if subtract_center else grids_center_theta)  #TODO 之前的方案 有点问题,减得是 center_polar[0]
    
    cos_delta_yaw = torch.cos(delta_yaw)
    sin_delta_yaw = torch.sin(delta_yaw)

    # vx, vy
    vx, vy = box[7:9]
    v_n2 = torch.sqrt(vx ** 2 + vy ** 2)
    alpha_v = torch.atan2(vy, vx)
    delta_alpha_v = alpha_v - (center_polar[0] if subtract_center else grids_center_theta)   #TODO 有点问题  之前的方案减的是center的theta值
    if velocity_resolve:
        v_sin = torch.sin(delta_alpha_v)
        v_cos = torch.cos(delta_alpha_v)
        # v = torch.tensor([v_sin, v_cos, v_n2])
        v = torch.stack([v_sin, v_cos, v_n2.expand_as(v_sin)], axis=-1)   
    else:
        new_vx = v_n2 * torch.cos(delta_alpha_v)
        new_vy = v_n2 * torch.sin(delta_alpha_v)
        # v = torch.tensor([new_vx, new_vy])
        v = torch.stack([new_vx, new_vy], axis=-1)   
        
        
    # w, h, l
    box_dim = box[3:6].log() if norm_bbox else box[3:6]

    box_size = 10
    if velocity_resolve:
        box_size += 1
        
    anno_boxes = torch.zeros((polar_idx.size(0), box_size)).to(box)
    anno_boxes[:, :2] = delta_polar
    anno_boxes[:, 2] = box[2]
    anno_boxes[:, 3:6] = box_dim
    anno_boxes[:, 6] = sin_delta_yaw
    anno_boxes[:, 7] = cos_delta_yaw
    anno_boxes[:, 8:] = v

    # 生成idx
    indices = polar_idx[:, 0] * feature_map_size[0] + polar_idx[:, 1]

    return anno_boxes, indices


@HEADS.register_module()
class PolarBEVHead(BaseModule):
    """PolarBEVHead for PolarBEVU.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None,
                 task_specific=True,
                 regress_radius=1,
                 max_objs=500,
                 point_cloud_range=None,
                 grid_size=[128, 128, 1],
                 voxel_size=None,
                 out_size_factor=1,
                 heat_threshold=0.5,
                 dynamic_regress_region=False,
                 subtract_center=False, 
                 velocity_resolve=False,
                 with_cp=False):


        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(PolarBEVHead, self).__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.regress_radius = regress_radius

        self.max_objs = max_objs
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.heat_threshold = heat_threshold
        self.dynamic_regress_region = dynamic_regress_region
        
        self.subtract_center = subtract_center
        self.velocity_resolve = velocity_resolve


        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = nn.Sequential(
            RingConv(in_channels, share_conv_channel),
            build_norm_layer(norm_cfg, share_conv_channel, postfix=0)[1],
            nn.ReLU(inplace=True)
        )

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

        self.with_velocity = 'vel' in common_heads.keys()
        self.task_specific = task_specific


    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        if self.with_cp and x.requires_grad:
            x = checkpoint(self.shared_conv, x)
        else:
            x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d)

        # Transpose heatmaps  cat x w x h => b x cat x w x h
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        device = gt_labels_3d.device
        
        gt_bboxes_3d = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1).to(device) # x,y,z,w,l,h,yaw,vx,vy 9个
        # max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg'] # 500
        max_objs = self.max_objs
        
        
        grid_size = torch.tensor(self.grid_size, device=device)
        # grid_size = torch.tensor(self.train_cfg['grid_size'], device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)
        # pc_range = torch.tensor(self.train_cfg['point_cloud_range'], device=device)
        voxel_size = torch.tensor(self.voxel_size, device=device)
        # voxel_size = torch.tensor(self.train_cfg['voxel_size'], device=device)

        feature_map_size = grid_size[:2] // self.out_size_factor

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag) for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)


        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = torch.zeros((len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]), dtype=torch.float32, device=device)  # !!

            if self.with_velocity:
                vel_box_size = 10
                if self.velocity_resolve:
                    vel_box_size += 1
                anno_box = gt_bboxes_3d.new_zeros((max_objs, vel_box_size), dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 8), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)
            if num_objs > 0:
                corners = boxes_to_2dcorners(task_boxes[idx])
        
            count = 0
            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                corner = corners[k]
                box = task_boxes[idx][k]

                center_theta = torch.atan2(box[1], box[0])
                center_rho = torch.sqrt(box[0] ** 2 + box[1] ** 2)

                grid_theta = ((center_theta - pc_range[0]) / voxel_size[0] / self.out_size_factor).long()
                grid_rho = ((center_rho - pc_range[1]) / voxel_size[1] / self.out_size_factor).long()

                if not (0 <= grid_rho < feature_map_size[0] and 0 <= grid_theta < feature_map_size[1]):
                    continue
                # deaw heatmap
                theta_bound, rho_bound = get_polar_bound(corner, pc_range, voxel_size, self.out_size_factor, n=6)
                high_idx = draw_heatmap(heatmap[cls_id], box, corner, theta_bound, rho_bound, pc_range, voxel_size, self.out_size_factor, feature_map_size, self.heat_threshold)


                assert (grid_theta * feature_map_size[0] + grid_rho < feature_map_size[0] * feature_map_size[1])
                if self.dynamic_regress_region == False:
                    high_idx = None
                # generate regression boxes
                regress_boxes, indices = generate_regression_boxes(box, high_idx, self.regress_radius, self.norm_bbox, pc_range, voxel_size, self.out_size_factor, feature_map_size, self.subtract_center, self.velocity_resolve)
                num_boxes = regress_boxes.size(0)
                # print(num_boxes)
                
                anno_box[count : count + num_boxes] = regress_boxes.to(device=device)
                ind[count : count + num_boxes] = indices.to(device=device)
                mask[count : count + num_boxes] = 1
                count += num_boxes

            heatmaps.append(heatmap.to(device=device))
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks  # heatmaps   : list, heatmaps[i]   : class x W x H

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # heatmaps   : list, heatmaps[i]   : B x class x W x H
        # anno_boxes : list, anno_boxes[i] : B x max_obj x 10
        # inds       : list, inds[i]       : B x max_obj
        # masks      : list, masks[i]      : B x max_obj
        heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        if not self.task_specific:
            loss_dict['loss'] = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item() ## heat=1的数量
            cls_avg_factor = torch.clamp(reduce_mean(heatmaps[task_id].new_tensor(num_pos)), min=1).item() # 等于num_pos
            # TODO heatmap 在这里计算损失
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'], heatmaps[task_id],   avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (
                    preds_dict[0]['reg'],
                    preds_dict[0]['height'],
                    preds_dict[0]['dim'],
                    preds_dict[0]['rot'],
                    preds_dict[0]['vel'],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous() # B x 10 x W x H -> B x W x H x 10
            pred = pred.view(pred.size(0), -1, pred.size(3))  # B x W*H x 10
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)), min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)

            if self.task_specific:
                name_list = ['xy', 'z', 'whl', 'yaw', 'vel']
                clip_index = [0, 2, 3, 6, 8, 10]
                if self.velocity_resolve:  # TODO
                    clip_index[-1] += 1
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[..., clip_index[reg_task_id] : clip_index[reg_task_id + 1]]

                    target_box_tmp = target_box[..., clip_index[reg_task_id] : clip_index[reg_task_id + 1]]

                    bbox_weights_tmp = bbox_weights[..., clip_index[reg_task_id] : clip_index[reg_task_id + 1]]
                    loss_bbox_tmp = self.loss_bbox(pred_tmp, target_box_tmp, bbox_weights_tmp, avg_factor=(num + 1e-4))

                    loss_dict[f'task{task_id}.loss_%s' % (name_list[reg_task_id])] = loss_bbox_tmp

                loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            else:
                loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=num)
                loss_dict['loss'] += loss_bbox
                loss_dict['loss'] += loss_heatmap

        return loss_dict


    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()
        
            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas,
                                             task_id))
        
        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            task_id):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):
            default_val = [1.0 for _ in range(len(self.task_heads))]
            factor = self.test_cfg.get('nms_rescale_factor',
                                       default_val)[task_id]
            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[cls_labels == cid, 3:6] = \
                        box_preds[cls_labels == cid, 3:6] * factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * factor

            # Apply NMS in birdeye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.
                if isinstance(self.test_cfg['nms_thr'], list):
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr']
                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            if isinstance(factor, list):
                for cid in range(len(factor)):
                    box_preds[top_labels == cid, 3:6] = \
                        box_preds[top_labels == cid, 3:6] / factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / factor

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:        
                    post_rho = torch.sqrt(final_box_preds[:, 0] ** 2 + final_box_preds[:, 1] ** 2)

                    mask = (post_rho >= post_center_range[1])
                    mask &= (post_rho <= post_center_range[4])
                    mask &= (final_box_preds[:, 2] >= post_center_range[2])
                    mask &= (final_box_preds[:, 2] <= post_center_range[5])

                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
