# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet3d.core import bbox3d2result
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector



@DETECTORS.register_module()
class PolarBEVU(Base3DDetector):
    r"""PolarBEVU paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, 
                img_backbone=None,
                img_neck=None,
                img_view_transformer=None, 
                img_bev_encoder_backbone=None, 
                img_bev_encoder_neck=None, 
                pts_bbox_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                **kwargs):
        super(PolarBEVU, self).__init__(**kwargs)
        
        if img_backbone:        
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
        if img_view_transformer:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
            

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def image_encoder(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.img_backbone(x)
        x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_feat(self, img, **kwargs):
        """Extract features from images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x) # !
    
        return ([x], depth)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img_inputs=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function. 

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats,  _ = self.extract_feat(img=img_inputs, **kwargs)
        losses = dict()
        out_losses = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d)
        losses.update(out_losses)
        return losses

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore=None, **kwargs):
        outs = self.pts_bbox_head(pts_feats) # list:tasks,  list:len=1, doct:regress+heatmap, 
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        out_losses = self.pts_bbox_head.loss(*loss_inputs)
        return out_losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))
            
        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            
            if gt_bboxes_3d is not None:
                return self.debug_test(img_metas[0], img_inputs[0], gt_bboxes_3d[0], gt_labels_3d[0], **kwargs)
            else:
                return self.simple_test(img_metas[0], img=img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False


    def simple_test(self,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(img=img, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        
        outs = self.pts_bbox_head(img_feats)
        bbox_lists = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_lists
        ]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _ = self.extract_feat(img=img_inputs, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs

    def debug_test(self, img_metas, img, gt_bboxes_3d, gt_labels_3d, rescale=False, **kwargs):
        # img_feats, _, _ = self.extract_feat(None, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.debug_simple_test_pts(gt_bboxes_3d, gt_labels_3d, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def debug_simple_test_pts(self, gt_bboxes_3d, gt_labels_3d, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_gt_bboxes(gt_bboxes_3d, gt_labels_3d, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    
@DETECTORS.register_module()
class PolarBEVU_4D(PolarBEVU):
    def __init__(self,
                pre_process=None,
                align_after_view_transfromation=False,
                num_adj=1,
                with_prev=True,
                **kwargs):
        super(PolarBEVU_4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        
    @force_fp32()  # TODO
    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape  # 1 x 80 x 128 x 128
        _, v, _ = trans[0].shape  # b x 6 x 3

        # generate grid
        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)  # 关键帧ego坐标系到邻帧ego坐标系
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]] # 1 x 1 x 1 x 3 x 3  二维变换

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        # tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)  # 格子坐标到邻ego
        
        ## transform and normalize
        # grid = tf.matmul(grid)  # 将格子坐标转化成了邻ego坐标 第三维固定为1

        ### 
        
        grid_polar_selfego = feat2bev.matmul(grid)
        
        x = grid_polar_selfego[:, :, :, 0:1, :] * torch.cos(grid_polar_selfego[:, :, :, 1:2, :])
        y = grid_polar_selfego[:, :, :, 0:1, :] * torch.sin(grid_polar_selfego[:, :, :, 1:2, :])
        
        grid_cart_selfego = torch.cat([x, y, grid_polar_selfego[:, :, :, 2:3, :]], axis=3)
        grid_cart_adjego = l02l1.matmul(grid_cart_selfego)
        
        theta = torch.atan2(grid_cart_adjego[:, :, :, 1:2, :], grid_cart_adjego[:, :, :, 0:1, :])
        rho = torch.sqrt(grid_cart_adjego[:, :, :, 0:1, :] ** 2 + grid_cart_adjego[:, :, :, 1:2, :] ** 2)
        
        grid_polar_adjego = torch.cat([rho, theta, grid_cart_adjego[:, :, :, 2:3, :]], axis=3)
        grid = torch.inverse(feat2bev).matmul(grid_polar_adjego) # to polar grid
        
        ###
        
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)  #之所以使用grid是邻ego坐标系下，
        return output
    
    
    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...],
                       intrins, post_rots, post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev =  self.shift_feature(feat_prev, [trans_curr, trans_prev], [rots_curr, rots_prev], bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth
    
    
    def prepare_bev_feat(self, x, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x = self.image_encoder(x)  # b x 6 x 512 x 16 x 44
        x, depth = self.img_view_transformer(  
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])  # bev_feat: b x 80 x 128 x 128  depth:6 x 59 x 16 x 44
        if self.pre_process:
            x = self.pre_process_net(x)[0]  # bev_feat: b x 80 x 128 x 128
        return x, depth
    
    
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda
    
    
    def extract_feat(self, img, pred_prev=False, sequential=False, **kwargs):
        if sequential: # false !!!!是推理的时候用的，可以节省时间
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda = self.prepare_inputs(img)
        
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]  # vt之后对齐的话，内外参选择第一帧的
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():  # 只有关键帧反向传播
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev: # false
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            trans_curr = trans[0].repeat(self.num_frame - 1, 1, 1)
            rots_curr = rots[0].repeat(self.num_frame - 1, 1, 1, 1)
            trans_prev = torch.cat(trans[1:], dim=0)
            rots_prev = torch.cat(rots[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]
    

@DETECTORS.register_module()
class PolarBEVU_4D_Depth(PolarBEVU_4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, depth = self.extract_feat(img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore)
        
        losses.update(losses_pts)
        return losses
