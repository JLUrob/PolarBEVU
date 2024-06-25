# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class CenterPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = torch.tensor(post_center_range)
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.  用inds的值去选择feats, 值的维度与inds相同
        """
        dim = feats.size(2)  
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)  # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].  W : theta_num, H : rho_num
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()
        # b x cat x 500, b x cat x 500
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # topk_inds 值范围 [0, W*H)

        topk_inds = topk_inds % (height * width)  # 数值不变,搞不懂加这一步干什么
        topk_ys = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()  # b x cat x 500  取值范围[0, W) 代表theta值
        topk_xs = (topk_inds % width).int().float() # b x cat x 500
        # topk_scores : 分类选择前K个, topk_score : 所有类别前K个  
        # topk_ind 取值范围 [0, c*k)   shape(b x K)
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) 
        # topk_ind 可以根据大小判断是哪一类的, topk_scores view之后的id : 0~k-1|k~2k-1|...
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K) # 前K个的id, 范围[0, cat*K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)   # 
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 2, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num) # 各个类别是混在一起的, 用clses区分, 一共max_num个

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)  # 用inds去索引reg
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1] # x + dx
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2] # y + dy
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = xs.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]  # 恢复到lidar坐标系
        ys = ys.view(
            batch, self.max_num,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1] 

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:    # TODO !!
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts

    
@BBOX_CODERS.register_module()
class PolarBEVUBBoxCoder(CenterPointBBoxCoder):
    def __init__(self, velocity_resolve=False, subtract_center=False, **kwargs):
        super(PolarBEVUBBoxCoder, self).__init__(**kwargs)
        self.velocity_resolve = velocity_resolve
        self.subtract_center = subtract_center
        
        
    def topk_scores(self, scores, K=100):
        """_summary_

        Args:
            scores (torch.Tensor): Heatmap
            K (int, optional): Defaults to 100.

        Returns:
            _type_: _description_
        """
        batch, cat, height, width = scores.size()
        K = min(K, height * cat)

        max_scores, max_rho_indices = torch.max(scores, dim=3)
        # b x c x height
        max_thetas_indices = torch.arange(0, height, device=scores.device).expand_as(max_rho_indices).contiguous()
        # b x K
        topk_score, topk_idx = torch.topk(max_scores.view(batch, -1), K) 
        # topk_idx 可以根据大小判断是哪一类的, 0~h-1|h~2h-1|... 
        topk_class = (topk_idx / torch.tensor(height, dtype=torch.float32)).int()
        
        # topk_indices = _gather_feat(topk_inds.view(batch, -1, 1), topk_idx).view(batch, K) # 前K个的id, 范围[0, cat*K)
        topk_rho = self._gather_feat(max_rho_indices.view(batch, -1, 1), topk_idx).view(batch, K)      # b x K 
        topk_theta = self._gather_feat(max_thetas_indices.view(batch, -1, 1), topk_idx).view(batch, K) # b x K
            
        topk_indices = topk_theta * width + topk_rho
        
        return topk_score, topk_indices, topk_class, topk_theta, topk_rho
        
    def decode(self,
            heat,
            rot_sine,
            rot_cosine,
            hei,
            dim,
            vel,
            reg=None,
            task_id=-1):
        """Decode bboxes for polargrid.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 2, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.  theta, rho
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        # scores, inds, clses, thetas, rhos = self.topk_scores(heat, K=self.max_num) # 各个类别是混在一起的, 用clses区分, 一共max_num个
        scores, inds, clses, thetas, rhos = self._topk(heat, K=self.max_num) # 各个类别是混在一起的, 用clses区分, 一共max_num个

        # grid坐标系统 还原到ego坐标系, 求出grid中心的极坐标
        grid_center_thetas = (thetas.view(batch, self.max_num, 1) + 0.5) * self.voxel_size[0] * self.out_size_factor + self.pc_range[0]
        grid_center_rhos = (rhos.view(batch, self.max_num, 1) + 0.5) * self.voxel_size[1] * self.out_size_factor + self.pc_range[1]

        # reg
        reg = self._transpose_and_gather_feat(reg, inds)  # 用inds去索引reg
        reg = reg.view(batch, self.max_num, 2)

        orin_thetas = grid_center_thetas.view(batch, self.max_num, 1) + reg[:, :, 0:1] # theta + dtheta
        orin_rhos = grid_center_rhos.view(batch, self.max_num, 1) + reg[:, :, 1:2]     # rho + drho
        
        xs = orin_rhos * torch.cos(orin_thetas)
        ys = orin_rhos * torch.sin(orin_thetas)
        

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)
        rot += orin_thetas if self.subtract_center else grid_center_thetas 

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)
        
        # velocity
        vel = self._transpose_and_gather_feat(vel, inds)

        if self.velocity_resolve:
            vel = vel.view(batch, self.max_num, 3)
            alpha_v = torch.atan2(vel[:, :, 0:1], vel[:, :, 1:2])
            alpha_v += orin_thetas if self.subtract_center else grid_center_thetas 
            vx = vel[:, :, 2:3] * torch.cos(alpha_v)
            vy = vel[:, :, 2:3] * torch.sin(alpha_v)
        
        else:
            vel = vel.view(batch, self.max_num, 2)
            v_n2 = torch.sqrt(vel[:, :, 0:1] ** 2 + vel[:, :, 1:2] ** 2)
            alpha_v = torch.atan2(vel[:, :, 1:2], vel[:, :, 0:1])
            alpha_v += orin_thetas if self.subtract_center else grid_center_thetas 
            vx = v_n2 * torch.cos(alpha_v)
            vy = v_n2 * torch.sin(alpha_v)
        
        final_box_preds = torch.cat([xs, ys, hei, dim, rot, vx, vy], dim=2)
        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:    # TODO !!
            self.post_center_range = self.post_center_range.to(device=heat.device)
            
            post_rho = torch.sqrt(final_box_preds[..., 1] ** 2 + final_box_preds[..., 0] ** 2)

            mask = (post_rho >= self.post_center_range[1])
            mask &= (post_rho <= self.post_center_range[4])
            mask &= (final_box_preds[..., 2] >= self.post_center_range[2])
            mask &= (final_box_preds[..., 2] <= self.post_center_range[5])
            
            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
    
