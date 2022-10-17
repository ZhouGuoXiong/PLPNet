#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
from mmcv.ops import SAConv2d

from .PAC_darknet import PAC_CSPDarknet
from .LRAM import LRAM
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class SD_PFAN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = PAC_CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        # 开始处理特征提取过程

        # 底层到中层的 上采样
        # 512, 20, 20 -> 256, 40, 40
        self.upsample_L_to_M = nn.ConvTranspose2d(
            in_channels=int(in_channels[2] * width),
            out_channels=int(in_channels[1] * width),
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # 中层到高层的 上采样
        # 512, 40, 40 -> 128, 80, 80
        self.upsample_M_to_H = nn.ConvTranspose2d(
            in_channels=int(2 * in_channels[1] * width),
            out_channels=int(in_channels[1] // 2 * width),
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # 中层的过程特征 下采样
        # 512, 40, 40 -> 512, 20, 20
        self.downsample_M_to_L = SAConv2d(
            in_channels=int(2 * in_channels[1] * width),
            out_channels=int(in_channels[2] * width),
            kernel_size=3,
            stride=2,
            padding=1
        )

        # 高层输出特征层下采样
        # 256, 80, 80 -> 512, 40, 40
        self.downsample_H_to_M = SAConv2d(
            in_channels=int(2 * in_channels[0] * width),
            out_channels=int(2 * in_channels[1] * width),
            kernel_size=3,
            stride=2,
            padding=1
        )

        # 通道数 resize
        # 高层；256, 80, 80 -> 128, 80, 80
        self.resize_H = BaseConv(
            in_channels=int(2 * in_channels[0] * width),
            out_channels=int(in_channels[0] * width),
            ksize=1,
            stride=1,
            act=act
        )

        # 中层：1024, 40, 40 -> 256, 40, 40
        self.resize_M = BaseConv(
            in_channels=int(4 * in_channels[1] * width),
            out_channels=int(in_channels[1] * width),
            ksize=1,
            stride=1,
            act=act
        )

        # 底层：1024, 20, 20 -> 512, 20, 20
        self.resize_L = BaseConv(
            in_channels=int(2 * in_channels[2] * width),
            out_channels=int(in_channels[2] * width),
            ksize=1,
            stride=1,
            act=act
        )

        self.feat1_attention = LRAM(int(in_channels[0] * width), int(in_channels[0] * width))
        self.feat2_attention = LRAM(int(in_channels[1] * width), int(in_channels[1] * width))
        self.feat3_attention = LRAM(int(in_channels[2] * width), int(in_channels[2] * width))

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        # 应用注意力机制
        feat3 = self.feat3_attention(feat3)
        feat2 = self.feat2_attention(feat2)
        feat1 = self.feat1_attention(feat1)

        # 开始构建不同特征层，代码写法需要倒过来，但是可以从高层开始写
        # 其中，feat1~3 分别为 dark3~5，也即高层到底层，通道数分别为 128, 256, 512

        # 底层骨干特征上采样
        # 512, 20, 20 -> 256, 40, 40
        feat3_upsample = self.upsample_L_to_M(feat3)

        # 中层的过程特征
        # 256, 40, 40 + 256, 40, 40 -> 512, 40, 40,
        Fmap_M_prs = torch.cat([feat2, feat3_upsample], 1)

        # 中层的过程特征 上采样
        # 512, 40, 40 -> 128, 80, 80
        Fmap_M_prs_upsample = self.upsample_M_to_H(Fmap_M_prs)

        # 高层输出
        # 128, 80, 80 + 128, 80, 80 -> 256, 80, 80
        Fmap_H_out = torch.cat([feat1, Fmap_M_prs_upsample], 1)

        # 中层的过程特征 下采样
        # 512, 40, 40 -> 512, 20, 20
        Fmap_M_prs_downsample = self.downsample_M_to_L(Fmap_M_prs)

        # 高层输出特征层下采样
        # 256, 80, 80 -> 512, 40, 40
        Fmap_H_out_downsample = self.downsample_H_to_M(Fmap_H_out)

        # 中层输出
        # 512, 40, 40 + 512, 40, 40 -> 1024, 40, 40
        Fmap_M_out = torch.cat([Fmap_H_out_downsample, Fmap_M_prs] ,1)


        # 底层输出
        # 512, 20, 20 + 512, 20, 20 -> 1024, 20, 20
        Fmap_L_out = torch.cat([feat3, Fmap_M_prs_downsample], 1)


        # 输出降维通道数
        # 高层；256, 80, 80 -> 128, 80, 80
        Fmap_H_out = self.resize_H(Fmap_H_out)

        # 中层：1024, 40, 40 -> 256, 40, 40
        Fmap_M_out = self.resize_M(Fmap_M_out)

        # 底层：1024, 20, 20 -> 512, 20, 20
        Fmap_L_out = self.resize_L(Fmap_L_out)

        # 输出返回参数从 高层向底层
        return (Fmap_H_out, Fmap_M_out, Fmap_L_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = SD_PFAN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
