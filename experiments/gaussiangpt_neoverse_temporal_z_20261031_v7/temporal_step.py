"""Exact first-order native rendering gradients for a temporal RGB objective.

The initial RGB image is a temporary leaf only while accumulating image-space
derivatives. Its complete derivative is then propagated through its original
render graph. No initial-image or initial-feature derivative is discarded.
Only PyTorch is imported until the real native step is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import torch


@dataclass
class TemporalBackwardResult:
    rgb_loss: float
    reconstruction_loss: float
    delta_loss: float
    feature_gradients: tuple[tuple[torch.Tensor, ...], ...]
    initial_rgb_gradients: tuple[torch.Tensor, ...]
    rgb_observation: Any
    auxiliary: float


def _scalar(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise RuntimeError(f'Nonfinite or nonscalar {name}')
    return value


def _gradient(leaf, name):
    if leaf.grad is None:
        raise RuntimeError(f'Disconnected {name}')
    if not bool(torch.isfinite(leaf.grad).all()):
        raise RuntimeError(f'Nonfinite {name} gradient')
    return leaf.grad.detach().clone()


def staged_temporal_backward(
    original_features: Sequence[Sequence[torch.Tensor]],
    render: Callable[[int, int, torch.Tensor], torch.Tensor],
    reconstruction_loss: Callable[[int, int, torch.Tensor], torch.Tensor],
    delta_loss: Callable[[int, int, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    delta_weight: float = 1.,
    auxiliary: torch.Tensor | None = None,
    observe_rgb: Callable[[], Any] | None = None,
) -> TemporalBackwardResult:
    """Base loss + lambda mean_B mean_future delta, then auxiliary once.

    Each reconstruction callback has the ORIGINAL within-window weighting
    (e.g. full/10 + ROI/10 + motion/9). Each delta callback returns unweighted
    full + ROI change error. This function divides every term by B and divides
    only delta by T-1. The callbacks may use shared native parameters, but may
    access upstream differentiable outputs only through their supplied leaves.

    At most the initial native render graph and one current future native graph
    remain live per window. Native forward order/count is B-major then T-major,
    and there is one upstream multi-output VJP. Ordinary first derivatives are
    preserved; higher-order derivatives or backward-hook count are not promised.
    Caller owns zero_grad, model modes, clipping and optimizer updates.
    """
    if not math.isfinite(delta_weight) or delta_weight < 0:
        raise ValueError('delta_weight must be finite and nonnegative')
    windows = tuple(tuple(window) for window in original_features)
    if not windows or len(windows[0]) < 2:
        raise ValueError('Expected a nonempty batch with initial plus future outputs')
    batch_size, length = len(windows), len(windows[0])
    if any(len(window) != length for window in windows):
        raise ValueError('Every window must have the same complete length')
    if any(not isinstance(f, torch.Tensor) or not f.requires_grad for window in windows for f in window):
        raise ValueError('Every original feature must have an upstream gradient graph')
    if auxiliary is not None:
        _scalar(auxiliary, 'auxiliary loss')
    all_gradients, initial_rgb_gradients = [], []
    total_reconstruction, total_delta = 0., 0.
    for b, features in enumerate(windows):
        initial_feature = features[0].detach().requires_grad_(True)
        initial_image = render(b, 0, initial_feature)
        if not initial_image.requires_grad:
            raise RuntimeError('Initial rendering has no derivative graph')
        initial_leaf = initial_image.detach().requires_grad_(True)
        initial_loss = _scalar(reconstruction_loss(b, 0, initial_leaf), 'initial RGB loss')
        (initial_loss / batch_size).backward()
        total_reconstruction += float(initial_loss.detach()) / batch_size
        del initial_loss
        future_gradients = []
        for h in range(1, length):
            feature = features[h].detach().requires_grad_(True)
            image = render(b, h, feature)
            base = _scalar(reconstruction_loss(b, h, image), 'future RGB loss')
            delta = _scalar(delta_loss(b, h, image, initial_leaf), 'temporal RGB loss')
            loss = (base + delta_weight * delta / (length - 1)) / batch_size
            _scalar(loss, 'combined RGB loss').backward()
            future_gradients.append(_gradient(feature, f'future feature[{b},{h}]'))
            total_reconstruction += float(base.detach()) / batch_size
            total_delta += float(delta.detach()) / ((length - 1) * batch_size)
            del feature, image, base, delta, loss
        # Includes initial reconstruction AND every future difference term.
        # Restoring this VJP makes the temporary leaf exact, not stop-gradient.
        image_gradient = _gradient(initial_leaf, f'initial RGB[{b}]')
        initial_image.backward(image_gradient)
        initial_rgb_gradients.append(image_gradient)
        all_gradients.append((_gradient(initial_feature, f'initial feature[{b},0]'),
                              *future_gradients))
        del initial_feature, initial_image, initial_leaf, image_gradient
    flat_features = tuple(feature for window in windows for feature in window)
    flat_gradients = tuple(gradient for window in all_gradients for gradient in window)
    torch.autograd.backward(flat_features, grad_tensors=flat_gradients, retain_graph=True)
    observation = None if observe_rgb is None else observe_rgb()
    auxiliary_value = 0. if auxiliary is None else float(auxiliary.detach())
    if auxiliary is not None and auxiliary.requires_grad:
        auxiliary.backward()
    return TemporalBackwardResult(
        total_reconstruction + delta_weight * total_delta, total_reconstruction,
        total_delta, tuple(all_gradients), tuple(initial_rgb_gradients),
        observation, auxiliary_value)


def step_native(model, bridge, optimizer, data, starts, cfg, delta_weight=1.0):
    """v6-compatible step with complete initial-image temporal gradients.

    delta_weight=0 dispatches directly to the separately verified staged base
    step, with no new forward/backward implementation on the RGB-only arms.
    Positive weight adds mean over B and h=1..9 of full+ROI RGB-difference L1.
    It does not add a second motion-weighted term or alter the original terms.
    """
    if not math.isfinite(delta_weight) or delta_weight < 0:
        raise ValueError('delta_weight must be finite and nonnegative')
    if delta_weight == 0:
        from staged_step import step_native as original_staged_step
        return original_staged_step(model, bridge, optimizer, data, starts, cfg)
    import native_experiment as native

    if not starts:
        raise ValueError('Expected a nonempty batch of complete windows')
    optimizer.zero_grad(set_to_none=True)
    windows = [data.window(start) for start in starts]
    conditions = [native.make_condition(window[0]) for window in windows]
    encoded = model.encode_windows(windows)
    encoded['z_quantized'].retain_grad()
    predictions = model.decode_latents(encoded['z_quantized'], conditions,
                                       initial_contexts=encoded['initial_contexts'])
    if len(predictions) != len(starts):
        raise RuntimeError('Missing batch predictions')
    for b, frames in enumerate(predictions):
        if len(frames) != 10:
            raise RuntimeError('Missing initial or future output')
        for prediction in frames:
            if not torch.equal(prediction['coordinates'], windows[b][0]['coordinates']):
                raise RuntimeError('Initial latent support was changed')
    terms, delta_terms, counts = [], [], []

    def render(b, h, feature):
        image = bridge.render(dict(predictions[b][h], features=feature), data.records[starts[b] + h])
        counts.append(bridge.last_count)
        return image

    def base_loss(b, h, image):
        loss, values = data.loss(image, starts[b], h)
        terms.append(values)
        return loss

    def difference_loss(b, h, image, initial_image):
        start = starts[b]
        actual_change = data.records[start + h]['real'] - data.records[start]['real']
        error = ((image - initial_image) - actual_change).abs().mean(-1)
        full, roi = error.mean(), error[data.roi].mean()
        delta_terms.append(dict(delta_full_l1=float(full.detach()), delta_roi_l1=float(roi.detach())))
        return full + roi

    def observe_rgb():
        gradient = encoded['z_quantized'].grad
        if gradient is None:
            raise RuntimeError('RGB reconstruction/difference losses are disconnected from codes')
        norms = gradient.detach().norm(dim=2)
        query_gradient = model.queries.weight.grad
        return dict(rgb_quantized_grad_per_sample_h=norms.cpu().tolist(),
                    rgb_quantized_grad_per_h=norms.mean(0).cpu().tolist(),
                    rgb_query_grad_per_h=None if query_gradient is None else
                    query_gradient.detach().norm(dim=1).cpu().tolist())

    result = staged_temporal_backward(
        [[prediction['features'] for prediction in frames] for frames in predictions],
        render, base_loss, difference_loss, delta_weight=delta_weight,
        auxiliary=encoded['aux_loss'] * cfg['train']['aux_multiplier'], observe_rgb=observe_rgb)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise RuntimeError('Nonfinite gradient')
    norms = dict(residual_head=native.grad_norm(model.features),
                 motion_encoder=native.grad_norm(model.encoder_backbone),
                 motion_decoder=native.grad_norm(model.decoder_backbone),
                 native_trainable=native.grad_norm(bridge),
                 native_out=native.grad_norm(bridge.codec.ae.decoder.out),
                 native_heads=native.grad_norm(bridge.codec.ae.feature_heads),
                 native_multiplier=native.grad_norm(bridge.codec.ae.gaussian_multiplier))
    torch.nn.utils.clip_grad_norm_(parameters, cfg['train']['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    stats = dict(**{key: float(native.np.mean([v[key] for v in terms])) for key in terms[0]},
                 **{key: float(native.np.mean([v[key] for v in delta_terms])) for key in delta_terms[0]},
                 delta_weight=float(delta_weight), auxiliary=result.auxiliary,
                 **result.rgb_observation, gradient_norms=norms,
                 initial_rgb_grad_norm_per_sample=[float(g.norm()) for g in result.initial_rgb_gradients],
                 initial_feature_grad_norm_per_sample=[float(g[0].norm()) for g in result.feature_gradients],
                 feature_grad_norm_per_sample_h=[[float(g.norm()) for g in row] for row in result.feature_gradients],
                 gs_count_min=min(counts), gs_count_max=max(counts),
                 batch_ids=encoded['indices'].detach().cpu().tolist(),
                 gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    if stats['gpu_peak_gib'] > cfg['resources']['peak_limit_gib']:
        raise RuntimeError('Recorded GPU allocation exceeded admitted limit')
    return stats
