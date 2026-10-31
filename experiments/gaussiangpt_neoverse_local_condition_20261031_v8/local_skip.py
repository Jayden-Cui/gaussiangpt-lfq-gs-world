"""Per-row initial-context shortcut for the original native-64 GS decoder.

Restore the original checkpoint before attaching; restore a new wrapped
checkpoint after attaching. Parameters become ``base.*`` and ``P.weight``.
The shortcut does not alter the original backbone's conditioning input.
"""
from __future__ import annotations

import torch
from torch import nn


class LocalSkipBackbone(nn.Module):
    """Add P(initial slot3) only to future slots4..12 after the base backbone.

The full output remains [13,N,64]. Slots0..3, including the actual h0 output,
are unchanged. The original decoder may itself mix full-window statistics;
this wrapper makes no new prefix-causality claim.
"""

    _gs_local_skip_wrapped = True

    def __init__(self, base: nn.Module, enabled: bool):
        super().__init__()
        if not isinstance(base, nn.Module) or getattr(base, '_gs_local_skip_wrapped', False):
            raise ValueError('Expected an original unwrapped decoder backbone')
        if type(enabled) is not bool:
            raise TypeError('enabled must be bool')
        self.base = base
        self.enabled = enabled
        reference = next((v for v in base.parameters() if v.is_floating_point()), None)
        if reference is None:
            reference = next((v for v in base.buffers() if v.is_floating_point()), None)
        # Explicit CPU construction consumes no CUDA RNG; fork_rng restores CPU
        # state even if initialization fails. Copying zero weights is not random.
        with torch.random.fork_rng(devices=[]):
            self.P = nn.Linear(64, 64, bias=False, device='cpu')
            nn.init.zeros_(self.P.weight)
        if reference is not None:
            self.P.to(device=reference.device, dtype=reference.dtype)
        self.P.requires_grad_(enabled)
        self.training = base.training
        self.P.train(base.training)

    def forward(self, x, coords, active, condition=None):
        if not self.enabled:
            return self.base(x, coords, active, condition=condition)
        anchor_source = x if condition is None else condition
        if (x.ndim != 3 or x.shape[0] != 13 or x.shape[-1] != 64
                or anchor_source.shape != x.shape or anchor_source.device != x.device
                or anchor_source.dtype != x.dtype
                or active.shape != x.shape[:2] or active.dtype != torch.bool
                or active.device != x.device):
            raise ValueError('Expected matching13-slot native64 input/condition and active mask')
        hidden = self.base(x, coords, active, condition=condition)
        if hidden.shape != x.shape:
            raise RuntimeError('Base backbone changed the original13-slot shape')
        local = self.P(anchor_source[3])
        future = hidden[4:] + local[None] * active[4:, :, None].to(hidden.dtype)
        return torch.cat((hidden[:4], future), dim=0)


def attach_local_skip(model: nn.Module, enabled: bool):
    """Attach once; return exactly the added trainable parameters, or []."""
    base = getattr(model, 'decoder_backbone', None)
    if not isinstance(base, nn.Module) or getattr(base, '_gs_local_skip_wrapped', False):
        raise ValueError('Model must have an original, not already wrapped decoder_backbone')
    wrapper = LocalSkipBackbone(base, enabled)
    model.decoder_backbone = wrapper
    return [wrapper.P.weight] if enabled else []


def local_skip_grad_norm(model: nn.Module):
    wrapper = getattr(model, 'decoder_backbone', None)
    if not isinstance(wrapper, LocalSkipBackbone):
        raise ValueError('Model has no LocalSkipBackbone')
    gradient = wrapper.P.weight.grad
    if gradient is None:
        return 0.
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('Nonfinite local-shortcut gradient')
    return float(gradient.detach().double().norm())


__all__ = ['LocalSkipBackbone', 'attach_local_skip', 'local_skip_grad_norm']
