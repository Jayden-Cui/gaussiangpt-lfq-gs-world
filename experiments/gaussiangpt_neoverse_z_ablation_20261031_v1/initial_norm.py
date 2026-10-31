"""Decoder normalization whose shared statistics depend only on initial slots."""
from __future__ import annotations

import torch
from torch import nn


def _validate(x, active):
    if (x.ndim != 3 or x.shape[0] < 4 or x.shape[1] == 0 or x.shape[2] != 64
            or not x.is_floating_point() or active.shape != x.shape[:2]
            or active.dtype != torch.bool or active.device != x.device):
        raise ValueError('Expected [T>=4,N>0,64] and matching bool activity')
    if not bool(active[:4].any()):
        raise ValueError('Initial reference contains no active rows')


class InitialReferenceNorm64(nn.Module):
    def __init__(self, weight, bias, eps=1e-6):
        super().__init__()
        if tuple(weight.shape) != (64,) or tuple(bias.shape) != (64,) or eps != 1e-6:
            raise ValueError('Unexpected original normalization parameters')
        # Retain the SAME Parameters, including values and optimizer identity.
        self.weight, self.bias, self.eps = weight, bias, eps

    @classmethod
    def from_existing(cls, source):
        if not getattr(source, 'across_time', False):
            raise ValueError('Only full-window decoder norms may be replaced')
        result = cls(source.weight, source.bias, source.eps)
        result.training = source.training
        return result

    def forward(self, x, active):
        _validate(x, active)
        clean = torch.where(active[..., None], x, torch.zeros_like(x))
        g = clean.reshape(*x.shape[:2], 4, 16)
        ref = g[:4]
        mask = active[:4, :, None, None].to(x.dtype)
        count = mask.sum((0, 1, 3), keepdim=True) * 16
        mean = (ref * mask).sum((0, 1, 3), keepdim=True) / count
        var = ((ref-mean).square() * mask).sum((0, 1, 3), keepdim=True) / count
        y = ((g-mean) * torch.rsqrt(var+self.eps)).reshape_as(x)
        result = y*self.weight+self.bias
        return torch.where(active[..., None], result, torch.zeros_like(result))


class InitialConditionNorm64(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm = InitialReferenceNorm64.from_existing(source.norm)
        self.affine = source.affine
        self.training = source.training

    def forward(self, x, condition, active):
        _validate(x, active)
        if condition.shape != x.shape or condition.dtype != x.dtype or condition.device != x.device:
            raise ValueError('Condition must match decoder tensor')
        valid = active[:4, :, None]
        clean = torch.where(valid, condition[:4], torch.zeros_like(condition[:4]))
        pooled = clean.sum((0, 1)) / active[:4].sum()
        scale, shift = self.affine(pooled).chunk(2, -1)
        y = self.norm(x, active)*(1+scale)+shift
        return torch.where(active[..., None], y, torch.zeros_like(y))


def replace_decoder_norms(decoder):
    """Accept the actual anchored base or the pinned v9 spatial wrapper."""
    base = decoder.base if getattr(decoder, '_gs_spatial_action_wrapped', False) else decoder
    if getattr(base, 'mode', None) != 'decoder' or len(base.blocks) != 4:
        raise ValueError('Expected the four-block anchored decoder')
    replaced = []
    # Validate before mutating, so a duplicate call cannot partly replace a model.
    for block in base.blocks:
        for norm in (block.temporal.norm1, block.temporal.norm2, block.condition_norm.norm):
            if not getattr(norm, 'across_time', False):
                raise ValueError('Decoder already adapted or source structure changed')
    for index, block in enumerate(base.blocks):
        for name in ('norm1', 'norm2'):
            setattr(block.temporal, name, InitialReferenceNorm64.from_existing(getattr(block.temporal, name)))
            replaced.append(f'blocks.{index}.temporal.{name}')
        block.condition_norm = InitialConditionNorm64(block.condition_norm)
        replaced.append(f'blocks.{index}.condition_norm')
    return replaced
