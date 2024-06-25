# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn
import torch.nn.functional as F
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmcv.cnn import build_norm_layer


@BACKBONES.register_module()
class CustomResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,  # [160, 320, 640]
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input  # 720
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class RingResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,  # [160, 320, 640]
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(RingResNet, self).__init__()
        
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
       
        if block_type == 'Basic':
            curr_numC = numC_input  # 720
            for i in range(len(num_layer)):
                layer = [
                    RingBasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride=stride[i], padding=(0, 1)),
                        norm_cfg=norm_cfg
                        )
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    RingBasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

class RingBasicBlock(nn.Module):
    '''(Conv => BN => ReLU) * 2'''
    def __init__(self, in_channel , out_channel, stride=1, downsample=None, norm_cfg=dict(type='BN')):
        super(RingBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=(0, 1), stride=stride, bias=False),
            build_norm_layer(norm_cfg, out_channel, postfix=0)[1],
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=(0, 1), bias=False),
            build_norm_layer(norm_cfg, out_channel, postfix=0)[1],
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        # add circular padding
        identity = x
        x = F.pad(x, (0, 0, 1, 1), mode='circular') # theta首尾填充
        out = self.conv1(x)
        out = self.relu(out)
        
        out = F.pad(out, (0, 0, 1, 1), mode='circular')
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        out = self.relu(out)

        return out
