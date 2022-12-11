import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_norm_layer, constant_init,
                      normal_init)
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import BaseModule

class DyDCNv3(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset.contiguous(), mask)
        if self.with_norm:
            x = self.norm(x)
        return x

class MChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(MChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        # 裁剪后通道重新与原特征图各通道对应
        # self.max_pool1 = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # 通道排序裁剪，后面需要调整恢复原通道顺序
        # out_sort, idx1 = torch.sort(out, dim=1)
        # out_sort[:, :(out_sort.size(1) // 16), :, :] = 1e-03
        #
        # # 重排序
        # tmp, idx2 = torch.sort(idx1.squeeze())
        # out_sort = out_sort.index_select(1, idx2)

        # out = self.max_pool1(out_sort)
        # torch.save(out, './ca.pt')
        return self.sigmoid(out)

class DC_SpatialAttention(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                 kernel_size=7):
        super(DC_SpatialAttention, self).__init__()

        self.zero_init_offset = zero_init_offset
        self.offset_and_mask_dim = 3 * 7 * 7
        self.offset_dim = 2 * 7 * 7

        self.spatial_conv = DyDCNv3(in_channels, out_channels)
        self.spatial_conv_offset = nn.Conv2d(in_channels, self.offset_and_mask_dim, kernel_size, padding=3)
        self._init_weights()

        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def _init_weights(self):
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        offset_and_mask = self.spatial_conv_offset(x)
        offset = offset_and_mask[:, :self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
        x = self.spatial_conv(x, offset, mask)
        # y = self.relu(x)
        torch.save(x, './sa.pt')
        return self.sigmoid(x)


# CBAM注意力机制
class MCBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(MCBAM, self).__init__()
        self.channelattention = MChannelAttention(channel, ratio=ratio)
        self.spatialattention = DC_SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
