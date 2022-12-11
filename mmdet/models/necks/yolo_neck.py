# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..attentions import CBAM

from ..builder import NECKS


class DetectionBlock(BaseModule):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(DetectionBlock, self).__init__(init_cfg)
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YOLOV3Neck(BaseModule):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(YOLOV3Neck, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

        # add CBAM attention module
        self.cbam_1 = CBAM(int(in_channels[0]))
        self.cbam_2 = CBAM(int(in_channels[1]))
        self.cbam_3 = CBAM(int(in_channels[2]))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []

        # adjust cbam
        feats = list(feats)
        feats2 = feats[-1]
        feats[-1] = self.cbam_1(feats2)
        # adjust cbam

        out = self.detect1(feats[-1])

        outs.append(out)
        #
        # for i, x in enumerate(reversed(feats[:-1])):
        #     conv = getattr(self, f'conv{i+1}')
        #     tmp = conv(out)
        #     # Cat with low-lvl feats
        #     tmp = F.interpolate(tmp, scale_factor=2)
        #     tmp = torch.cat((tmp, x), 1)
        #
        #     detect = getattr(self, f'detect{i+2}')
        #     out = detect(tmp)
        #     outs.append(out)

        # assert CBAM module
        conv1 = getattr(self, f'conv{1}')
        tmp1 = conv1(out)

        tmp1 = F.interpolate(tmp1, scale_factor=2)
        feats1 = feats[1]
        feats[1] = self.cbam_2(feats1)
        tmp1 = torch.cat((tmp1, feats[1]), 1)
        detect2 = getattr(self, f'detect{2}')
        out1 = detect2(tmp1)
        outs.append(out1)
        #
        #
        conv2 = getattr(self, f'conv{2}')
        tmp2 = conv2(out1)
        tmp2 = F.interpolate(tmp2, scale_factor=2)
        feats0 = feats[0]
        feats[0] = self.cbam_3(feats0)
        tmp2 = torch.cat((tmp2, feats[0]), 1)
        detect3 = getattr(self, f'detect{3}')
        out2 = detect3(tmp2)
        outs.append(out2)

        return tuple(outs)