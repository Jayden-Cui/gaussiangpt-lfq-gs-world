"""One matched native-render update for continuous or LFQ action bottlenecks.

The same teacher/base/delta objectives, model and staged VJP are used in both
arms. Continuous actions remain float codes and never require an ID.
"""
from __future__ import annotations

import math
import torch


def validate_encoded(encoded, mode, batch, quantizer):
    if mode not in ('continuous16', 'lfq16') or encoded.get('action_mode') != mode:
        raise ValueError('Action mode mismatch')
    codes, latent, ids = encoded.get('action_codes'), encoded.get('z_quantized'), encoded.get('indices')
    if (not isinstance(codes, torch.Tensor) or codes.dtype != torch.float32
            or tuple(codes.shape) != (batch,9,16) or not bool(torch.isfinite(codes).all())
            or not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (batch,9,64)
            or not bool(torch.isfinite(latent).all())):
        raise ValueError('Expected finite actual actions [B,9,16] and latents [B,9,64]')
    if mode == 'continuous16':
        if ids is not None:
            raise ValueError('Continuous actual actions must not be IDs')
    else:
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or tuple(ids.shape) != (batch,9)
                or not bool(((ids >= 0) & (ids < 65536)).all())):
            raise ValueError('LFQ actions need valid actual IDs')
        expected = quantizer.indices_to_codes(ids, project_out=False)
        if not torch.allclose(codes.detach(), expected, atol=1e-6, rtol=0):
            raise ValueError('LFQ IDs and actual signs differ')
    aux = encoded.get('aux_loss')
    if not isinstance(aux,torch.Tensor) or aux.ndim != 0 or not bool(torch.isfinite(aux)):
        raise ValueError('Expected finite shared auxiliary loss')


def grad_norm(module):
    values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def step_native(model, bridge, optimizer, data, starts, *, make_condition, temporal, teacher_loss,
                aux_multiplier=.1, grad_clip=1.):
    if len(starts) != 4 or len(set(starts)) != 4 or any(type(s) is not int or not 0 <= s < 168 for s in starts):
        raise ValueError('Full four distinct T10 windows required')
    if not math.isfinite(aux_multiplier) or aux_multiplier < 0 or grad_clip != 1.:
        raise ValueError('Invalid common loss/gradient settings')
    optimizer.zero_grad(set_to_none=True)
    windows = [data.window(s) for s in starts]
    conditions = [make_condition(window[0]) for window in windows]
    encoded = model.encode_windows(windows)
    mode = model.action_bottleneck.mode
    validate_encoded(encoded, mode, len(starts), model.quantizer)
    encoded['z_quantized'].retain_grad()
    predictions = model.decode_latents(encoded['z_quantized'], conditions, initial_contexts=encoded['initial_contexts'])
    if len(predictions) != 4 or any(len(row) != 10 for row in predictions):
        raise RuntimeError('Incomplete batch/window prediction')
    for b, row in enumerate(predictions):
        if any(not torch.equal(pred['coordinates'],windows[b][0]['coordinates']) for pred in row):
            raise RuntimeError('Initial latent support changed')
    terms, delta_terms, teacher_terms, counts = [], [], [], []

    def render(b,h,features):
        images = bridge.render_views(dict(predictions[b][h], features=features), data.records[starts[b]+h])
        counts.append(bridge.last_count)
        return images

    def base_loss(b,h,images):
        base, values = data.loss(images[0], starts[b], h)
        teacher, errors = teacher_loss(images, data.teacher_targets[starts[b]+h], (0.,.25),
                                       data.teacher_masks[starts[b]+h])
        terms.append(values)
        teacher_terms.append(errors.detach())
        return base+teacher

    def delta_loss(b,h,images,initial):
        start = starts[b]
        actual = data.records[start+h]['real']-data.records[start]['real']
        error = ((images[0]-initial[0])-actual).abs().mean(-1)
        full, roi = error.mean(), error[data.roi].mean()
        delta_terms.append(dict(delta_full_l1=float(full.detach()),delta_roi_l1=float(roi.detach())))
        return full+roi

    def observe():
        gradient = encoded['z_quantized'].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Reconstruction disconnected from action latents')
        query = model.queries.weight.grad
        return dict(reconstruction_latent_grad_per_sample_h=gradient.detach().norm(dim=2).cpu().tolist(),
                    reconstruction_query_grad_per_h=None if query is None else query.detach().norm(dim=1).cpu().tolist(),
                    reconstruction_project_in_grad=grad_norm(model.quantizer.project_in),
                    reconstruction_project_out_grad=grad_norm(model.quantizer.project_out))

    result = temporal.staged_temporal_backward(
        [[prediction['features'] for prediction in row] for row in predictions],
        render, base_loss, delta_loss, delta_weight=1.,
        auxiliary=encoded['aux_loss']*aux_multiplier, observe_rgb=observe)
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise RuntimeError('Nonfinite trainable gradient')
    norms = dict(residual_head=grad_norm(model.features), motion_encoder=grad_norm(model.encoder_backbone),
                 motion_decoder=grad_norm(model.decoder_backbone), native_trainable=grad_norm(bridge),
                 native_out=grad_norm(bridge.codec.ae.decoder.out), native_heads=grad_norm(bridge.codec.ae.feature_heads),
                 native_multiplier=grad_norm(bridge.codec.ae.gaussian_multiplier))
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    # Copy actual actions and optional IDs before the update; never dereference None.
    actual_codes = encoded['action_codes'].detach().cpu().clone()
    actual_ids = None if encoded['indices'] is None else encoded['indices'].detach().cpu().clone()
    optimizer.step()
    teacher_mean = torch.stack(teacher_terms).mean(0)
    average = lambda rows, key: sum(row[key] for row in rows)/len(rows)
    stats = dict(**{key:average(terms,key) for key in terms[0]},
                 **{key:average(delta_terms,key) for key in delta_terms[0]},
                 action_mode=mode, batch_size=4, starts=list(starts), teacher_weights=[0.,.25],
                 teacher_view0_l1=float(teacher_mean[0]), teacher_view1_l1=float(teacher_mean[1]),
                 auxiliary=result.auxiliary, **result.rgb_observation, gradient_norms=norms,
                 preclip_grad_norm=float(total_norm), gs_count_min=min(counts), gs_count_max=max(counts),
                 initial_rgb_grad_norm_per_sample=[float(g.norm()) for g in result.initial_rgb_gradients],
                 action_codes_shape=list(actual_codes.shape), indices_shape=None if actual_ids is None else list(actual_ids.shape))
    return stats, actual_codes, actual_ids
