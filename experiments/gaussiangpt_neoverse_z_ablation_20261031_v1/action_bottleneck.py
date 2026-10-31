"""Paired 64->16->64 action paths; this module never owns LFQ parameters.

The caller owns one real, hash-verified LFQ. Both paths evaluate its original
per-horizon auxiliary; only the reconstruction representation differs.
"""
from __future__ import annotations

import math
import torch
from torch import nn

MODES = ('lfq16', 'continuous16')


def validate_quantizer(q):
    expected = dict(dim=64, codebook_dim=16, num_codebooks=1, codebook_size=65536,
                    codebook_scale=1., soft_clamp_input_value=10., spherical=False,
                    orthogonal_rotation=False, entropy_loss_weight=.1,
                    commitment_loss_weight=0., diversity_gamma=1.,
                    frac_per_sample_entropy=1., experimental_softplus_entropy_loss=True,
                    entropy_loss_offset=5., channel_first=False,
                    keep_num_codebooks_dim=False, has_projections=True,
                    force_quantization_f32=True)
    if any(getattr(q, key, None) != value for key, value in expected.items()):
        raise ValueError('Unexpected bound LFQ configuration')
    if not isinstance(q.activation, nn.Identity):
        raise ValueError('Expected identity straight-through activation')
    for name, shape in [('project_in', (16, 64)), ('project_out', (64, 16))]:
        layer = getattr(q, name)
        if not isinstance(layer, nn.Linear) or tuple(layer.weight.shape) != shape or layer.bias is not None:
            raise ValueError('Expected bias-free 64->16->64 LFQ projections')


def _check_vectors(z):
    if (z.ndim != 3 or z.shape[0] == 0 or z.shape[1:] != (9, 64)
            or not z.is_floating_point() or not bool(torch.isfinite(z).all())):
        raise ValueError('Expected finite posterior [B,9,64]')


def soft_codes(z, quantizer):
    """The exact bound LFQ pre-sign formula, in float32."""
    _check_vectors(z)
    with torch.autocast(device_type=z.device.type, enabled=False):
        return 10. * torch.tanh(quantizer.project_in(z.float()) / 10.)


class ActionBottleneck(nn.Module):
    """Mode is experiment metadata, not part of the common parameter state.

    s_ref is a once-bound buffer, shared by both clones. No online statistics,
    batch normalization, learnable scale or mode-specific parameter exists.
    """
    def __init__(self, mode='lfq16', *, s_ref=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown action mode')
        if s_ref is not None and (not math.isfinite(float(s_ref)) or float(s_ref) <= 1e-6):
            raise ValueError('s_ref must be finite and greater than 1e-6')
        self.mode = mode
        self.register_buffer('s_ref', torch.tensor(float('nan') if s_ref is None else s_ref, dtype=torch.float32))
        self.register_buffer('scale_bound', torch.tensor(s_ref is not None))

    @torch.no_grad()
    def calibrate(self, posterior_batches, quantizer):
        """Consume a fixed TRAIN plan at common initialization, exactly once.

        The runner must bind source/checkpoint/starts externally; tensor-only
        calibration deliberately cannot claim real-data or codec acceptance.
        """
        if bool(self.scale_bound):
            raise RuntimeError('Action scale is already bound')
        validate_quantizer(quantizer)
        sum_squares, count = 0., 0
        for z in posterior_batches:
            u = soft_codes(z, quantizer)
            sum_squares += float(u.double().square().sum())
            count += u.numel()
        value = math.sqrt(sum_squares / count) if count else float('nan')
        if not math.isfinite(value) or value <= 1e-6:
            raise ValueError('Empty or degenerate TRAIN calibration')
        self.s_ref.fill_(value)
        self.scale_bound.fill_(True)
        return dict(s_ref=float(self.s_ref), entries=count, sum_squares=sum_squares,
                    rule='global RMS of common-initialization soft LFQ codes; fixed')

    def forward(self, z, quantizer, *, inv_temperature=1.):
        validate_quantizer(quantizer)
        _check_vectors(z)
        if not bool(self.scale_bound) or not bool(torch.isfinite(self.s_ref)) or float(self.s_ref) <= 1e-6:
            raise RuntimeError('Bind a finite fixed action scale before encoding')
        if self.mode not in MODES or inv_temperature != 1.:
            raise ValueError('Unexpected mode or entropy temperature')
        if self.training != quantizer.training:
            raise ValueError('Bottleneck and LFQ train/eval modes differ')
        with torch.autocast(device_type=z.device.type, enabled=False):
            # Keep the real library auxiliary, independently for each h. Its
            # quantized reconstruction output is deliberately unused in BOTH arms.
            ids, auxiliaries = [], []
            for h in range(9):
                result = quantizer(z[:, h:h+1].float(), inv_temperature=inv_temperature)
                ids.append(result.indices)
                auxiliaries.append(result.entropy_aux_loss)
            u = soft_codes(z, quantizer)
            v = u / self.s_ref
            hard = torch.where(v > 0, torch.ones_like(v), -torch.ones_like(v))
            codes = v if self.mode == 'continuous16' else (
                v + (hard-v).detach() if self.training else hard)
            latents = quantizer.project_out(codes)
        diagnostic_ids = torch.cat(ids, dim=1)
        return dict(action_mode=self.mode, action_codes=codes, action_latents=latents,
                    diagnostic_ids=diagnostic_ids,
                    indices=diagnostic_ids if self.mode == 'lfq16' else None,
                    aux_loss=torch.stack(auxiliaries).mean())

    def codes_to_latents(self, codes, quantizer):
        if (codes.ndim != 3 or codes.shape[0] == 0 or codes.shape[1:] != (9, 16)
                or codes.dtype != torch.float32 or not bool(torch.isfinite(codes).all())):
            raise ValueError('Actual action codes must be finite float32 [B,9,16]')
        if self.mode == 'lfq16' and not bool(((codes == -1) | (codes == 1)).all()):
            raise ValueError('LFQ actual codes must be binary signs')
        return quantizer.project_out(codes)

    def packet(self, encoded):
        if encoded['action_mode'] != self.mode:
            raise ValueError('Cannot serialize another action mode')
        return dict(schema='paired_action_codes_v1', action_mode=self.mode,
                    codes=encoded['action_codes'].detach().cpu().float().clone(),
                    diagnostic_ids=encoded['diagnostic_ids'].detach().cpu().clone(),
                    indices=None if encoded['indices'] is None else encoded['indices'].detach().cpu().clone(),
                    s_ref=float(self.s_ref))

    def packet_to_latents(self, packet, quantizer):
        if (packet.get('schema') != 'paired_action_codes_v1' or packet.get('action_mode') != self.mode
                or packet.get('s_ref') != float(self.s_ref)):
            raise ValueError('Action packet mode/scale/schema does not match model')
        if self.mode == 'continuous16' and packet.get('indices') is not None:
            raise ValueError('Continuous actual actions cannot be represented by IDs')
        codes = packet['codes'].to(device=quantizer.project_out.weight.device)
        latents = self.codes_to_latents(codes, quantizer)
        if self.mode == 'lfq16':
            ids = packet.get('indices')
            if (ids is None or ids.dtype != torch.long or tuple(ids.shape) != tuple(codes.shape[:2])
                    or not bool(((ids >= 0) & (ids < 65536)).all())):
                raise ValueError('Missing LFQ replay IDs')
            expected = quantizer.indices_to_codes(ids.to(codes.device), project_out=False)
            if not torch.equal(expected, codes):
                raise ValueError('LFQ IDs and actual signs disagree')
        return latents
