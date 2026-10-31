"""Native-64 spatial action modulation; an unfrozen experimental candidate.

Restore the original checkpoint before attaching. Wrapped checkpoints restore
after attaching. The original backbone is registered under ``base.*``; every
other parameter belongs to the new branch and can use an appended optimizer
group. No LFQ, Q-Former, renderer, support coordinates, or output head is changed.
"""
from __future__ import annotations

import torch
from torch import nn


INITIALIZATION_SEED = 20261031


class Position3D(nn.Module):
    """Original .032-scale, six-frequency, 39-to-64 positional representation.

The formula matches v6/model_core.py. Construction is explicitly CPU-only;
the containing wrapper subsequently adopts the original backbone's dtype and
device. This module does not import the native GS or quantizer dependencies.
"""

    def __init__(self):
        super().__init__()
        self.coordinate_scale = .032
        self.register_buffer('frequencies', 2 ** torch.arange(6, dtype=torch.float32, device='cpu'))
        self.project = nn.Linear(39, 64, bias=False, device='cpu')

    def forward(self, coordinates):
        xyz = coordinates.float() * self.coordinate_scale
        phase = xyz[..., None] * self.frequencies * torch.pi
        positional = torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        return self.project(positional.to(dtype=self.project.weight.dtype))


def _require_unwrapped(base):
    if (not isinstance(base, nn.Module)
            or getattr(base, '_gs_local_skip_wrapped', False)
            or getattr(base, '_gs_spatial_action_wrapped', False)):
        raise ValueError('Expected an original, unwrapped decoder backbone')


class SpatialActionBackbone(nn.Module):
    """Add a position-dependent action term to future slots 4..12 only.

For each future slot h and initial row i:
    delta[h,i] = Wo(tanh(Wf(LN(initial[3,i])) + Position3D(coords[i]))
                    * Wz(LN(z[h,i])))

The existing caller broadcasts each z_h over its initial rows; this wrapper
does not pool or reorder them. Both LayerNorms act only on the final 64
channels. The *new branch* reads only slot 3, its own future code and initial
coordinates. The unchanged base may mix full-window statistics, so the whole
decoder does not acquire a strict prefix-causality guarantee.
"""

    _gs_spatial_action_wrapped = True

    def __init__(self, base: nn.Module, enabled: bool):
        super().__init__()
        _require_unwrapped(base)
        if type(enabled) is not bool:
            raise TypeError('enabled must be bool')
        self.base = base
        self.enabled = enabled
        reference = next((p for p in base.parameters() if p.is_floating_point()), None)
        if reference is None:
            reference = next((b for b in base.buffers() if b.is_floating_point()), None)
        # fork_rng saves CPU state; an empty device list avoids touching CUDA
        # RNG. Seed the CPU generator directly, rather than torch.manual_seed,
        # which also schedules seeding of CUDA generators.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
            self.initial_norm = nn.LayerNorm(64, device='cpu')
            self.action_norm = nn.LayerNorm(64, device='cpu')
            self.Wf = nn.Linear(64, 64, bias=False, device='cpu')
            self.position = Position3D()
            self.Wz = nn.Linear(64, 64, bias=False, device='cpu')
            self.Wo = nn.Linear(64, 64, bias=False, device='cpu')
            nn.init.zeros_(self.Wo.weight)
        self.training = base.training
        # Preserve any intentionally mixed train/eval modes inside the base.
        for name, module in self.named_children():
            if name != 'base':
                if reference is not None:
                    module.to(device=reference.device, dtype=reference.dtype)
                module.requires_grad_(enabled)
                module.train(base.training)

    def branch_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('base.')]

    @staticmethod
    def _validate(x, coords, active, source):
        if (x.ndim != 3 or x.shape[0] != 13 or x.shape[-1] != 64
                or not x.is_floating_point() or x.shape[1] == 0
                or source.shape != x.shape or source.device != x.device
                or source.dtype != x.dtype
                or active.shape != x.shape[:2] or active.dtype != torch.bool
                or active.device != x.device
                or coords.shape != (x.shape[1], 3) or coords.device != x.device
                or coords.dtype == torch.bool or coords.is_complex()):
            raise ValueError('Expected matching [13,N,64], initial coordinates [N,3], and bool active [13,N]')

    def forward(self, x, coords, active, condition=None):
        if not self.enabled:
            return self.base(x, coords, active, condition=condition)
        source = x if condition is None else condition
        self._validate(x, coords, active, source)
        hidden = self.base(x, coords, active, condition=condition)
        if hidden.shape != x.shape:
            raise RuntimeError('Base backbone changed the original [13,N,64] shape')
        # An inactive initial row supplies no anchor. All production anchored
        # rows are active, but masking remains local for padded/toy interfaces.
        valid = active[4:] & active[3][None]
        initial = torch.where(active[3, :, None], source[3], torch.zeros_like(source[3]))
        actions = torch.where(valid[..., None], source[4:], torch.zeros_like(source[4:]))
        spatial = torch.tanh(self.Wf(self.initial_norm(initial)) + self.position(coords))
        delta = self.Wo(spatial[None] * self.Wz(self.action_norm(actions)))
        delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
        return torch.cat((hidden[:4], hidden[4:] + delta), dim=0)


def attach_spatial_action(model: nn.Module, enabled: bool):
    """Attach once, returning exactly the new trainable parameters (or [])."""
    base = getattr(model, 'decoder_backbone', None)
    _require_unwrapped(base)
    wrapper = SpatialActionBackbone(base, enabled)
    model.decoder_backbone = wrapper
    return wrapper.branch_parameters() if enabled else []


def spatial_action_grad_norm(model: nn.Module):
    """Finite aggregate norm of new-branch gradients, excluding the base."""
    wrapper = getattr(model, 'decoder_backbone', None)
    if not isinstance(wrapper, SpatialActionBackbone):
        raise ValueError('Model has no SpatialActionBackbone')
    squares = []
    for parameter in wrapper.branch_parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError('Nonfinite spatial-action gradient')
            squares.append(parameter.grad.detach().double().square().sum())
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.


def spatial_action_gradients(model: nn.Module):
    """JSON-safe gradient evidence for each added parameter, excluding base.*.

An absent gradient has finite=None, norm=0. A nonfinite gradient has
finite=False, norm=None; it is never silently reported as a valid norm.
"""
    wrapper = getattr(model, 'decoder_backbone', None)
    if not isinstance(wrapper, SpatialActionBackbone):
        raise ValueError('Model has no SpatialActionBackbone')
    result = {}
    for name, parameter in wrapper.named_parameters():
        if name.startswith('base.'):
            continue
        gradient = parameter.grad
        finite = None if gradient is None else bool(torch.isfinite(gradient).all())
        result[name] = dict(requires_grad=parameter.requires_grad,
                            has_grad=gradient is not None, finite=finite,
                            norm=0. if gradient is None else
                            float(gradient.detach().double().norm()) if finite else None)
    return result


__all__ = ['SpatialActionBackbone', 'attach_spatial_action',
           'spatial_action_grad_norm', 'spatial_action_gradients']
