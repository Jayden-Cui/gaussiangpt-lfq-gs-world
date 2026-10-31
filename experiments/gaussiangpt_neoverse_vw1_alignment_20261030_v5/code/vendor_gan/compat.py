"""Only the FP32, single-process CALVIN discriminator dependency surface.

This is a scoped mathematical adapter, not an mmcv/mmengine replacement.
Unsupported modes fail instead of silently changing the discriminator.
All ops use ordinary PyTorch autograd, including the gradient penalty's
second derivative. No extensions are compiled or imported.
"""
from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        if init_cfg is not None:
            raise ValueError('Only the released CALVIN init_cfg=None is supported')
        super().__init__()


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, norm_cfg=None,
                 act_cfg=None, **kwargs):
        super().__init__()
        if norm_cfg is not None or act_cfg is not None or kwargs:
            raise ValueError('CALVIN only uses ConvModule without norm/activation')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        # MMCV ConvModule's preliminary MSRA initialization, subsequently
        # overwritten by the original EqualizedLRConvModule.normal_init.
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out',
                                nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


def normal_init(module, mean=0., std=1., bias=0.):
    nn.init.normal_(module.weight, mean, std)
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, bias)


def autocast(enabled=False, **kwargs):
    if enabled or kwargs:
        raise ValueError('The pinned CALVIN objective is FP32; AMP is not enabled')
    return nullcontext()


def upfirdn2d(x, kernel, up=1, down=1, padding=0, **kwargs):
    """The exact Blur call used here: no resampling, 2-D FIR convolution.

    Two-value padding follows MMCV [symmetric x, symmetric y] semantics.
    CALVIN uses (2,2) for the 3x3 branch and (1,1) for its 1x1 skip.
    """
    if up != 1 or down != 1 or kwargs or kernel.ndim != 2:
        raise ValueError('Only CALVIN Blur up=down=1 with a 2-D kernel is supported')
    if isinstance(padding, int):
        padding = (padding,) * 4
    elif len(padding) == 2:
        px, py = padding
        padding = (px, px, py, py)
    elif len(padding) != 4:
        raise ValueError('Invalid padding')
    px0, px1, py0, py1 = padding
    x = F.pad(x, (max(px0, 0), max(px1, 0), max(py0, 0), max(py1, 0)))
    x = x[:, :, max(-py0, 0):x.shape[2]-max(-py1, 0),
          max(-px0, 0):x.shape[3]-max(-px1, 0)]
    weight = kernel.to(device=x.device, dtype=x.dtype).flip((0, 1))[None, None]
    weight = weight.expand(x.shape[1], 1, *weight.shape[-2:])
    return F.conv2d(x, weight, groups=x.shape[1])


def fused_bias_leakyrelu(x, bias, negative_slope=.2, scale=math.sqrt(2)):
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
            raise ValueError('Bias must have one value per channel')
        x = x + bias.reshape(1, -1, *([1] * (x.ndim - 2)))
    return F.leaky_relu(x, negative_slope=negative_slope) * scale


class FusedBiasLeakyReLU(nn.Module):
    def __init__(self, num_channels, negative_slope=.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.negative_slope, self.scale = negative_slope, scale

    def forward(self, x):
        return fused_bias_leakyrelu(x, self.bias, self.negative_slope, self.scale)


class Registry:
    def register_module(self, *args, **kwargs):
        return lambda cls: cls

    def build(self, cfg):
        raise ValueError(f'Unexpected registry-built dependency: {cfg}')


def get_dist_info():
    raise ValueError('CALVIN single-process sync_std=False only')


def build_norm_layer(*args, **kwargs):
    raise ValueError('CALVIN discriminator has no ConvModule normalization')


def load_checkpoint_with_prefix(*args, **kwargs):
    raise ValueError('Pass an explicitly verified state_dict via load_state_dict; no implicit checkpoint loading')
