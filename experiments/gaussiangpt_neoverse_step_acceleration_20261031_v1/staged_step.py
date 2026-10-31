"""Exact first-order gradient staging at the native GaussianGPT feature input.

Only PyTorch is imported here. The original native_experiment module is loaded
inside step_native, never while importing or testing staged_rgb_backward.
This is an unadopted candidate: real B4 numerical/resource parity is still due.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass
class StagedBackwardResult:
    rgb_loss: float
    feature_gradients: tuple[torch.Tensor, ...]
    rgb_observation: Any
    auxiliary: float


def staged_rgb_backward(
    original_features: Sequence[torch.Tensor],
    loss_for_leaf: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    batch_size: int,
    auxiliary: torch.Tensor | None = None,
    observe_rgb: Callable[[], Any] | None = None,
) -> StagedBackwardResult:
    """Accumulate native gradients, one upstream VJP, then auxiliary once.

    The callback must evaluate one original RGB loss with its supplied feature
    leaf replacing the corresponding upstream feature tensor. It must not read
    any other differentiable upstream/model value; native trainable parameters
    must be disjoint from the upstream model. Its temporal/spatial loss weights
    are unchanged. Only the original division by batch_size is applied here.

    Callers zero optimizer gradients before this function. Each local render
    graph is freed by backward without retain_graph. All original feature
    graphs survive until one multi-output VJP. observe_rgb runs immediately
    after that VJP and before auxiliary backward, and must copy/detach readings.
    No optimizer update, clipping, forward recomputation, autocast or RNG reset
    is performed here. This preserves first derivatives, not higher derivatives
    or the number/order of parameter backward-hook invocations.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    features = tuple(original_features)
    if not features or any(not isinstance(f, torch.Tensor) or not f.requires_grad for f in features):
        raise ValueError('Every original feature must belong to the upstream gradient graph')
    if auxiliary is not None and (auxiliary.ndim != 0 or not bool(torch.isfinite(auxiliary))):
        raise RuntimeError('Nonfinite or nonscalar auxiliary loss')
    gradients = []
    total_rgb = 0.0
    for index, feature in enumerate(features):
        leaf = feature.detach().requires_grad_(True)
        loss = loss_for_leaf(index, leaf)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite or nonscalar RGB loss at output {index}')
        (loss / batch_size).backward()
        if leaf.grad is None:
            raise RuntimeError(f'RGB loss is disconnected from feature output {index}')
        if not bool(torch.isfinite(leaf.grad).all()):
            raise RuntimeError(f'Nonfinite feature gradient at output {index}')
        gradients.append(leaf.grad.detach())
        total_rgb += float(loss.detach()) / batch_size
        del leaf, loss

    # Every original output participates in the same graph traversal. No sum
    # loss surrogate is re-rendered and no native parameter is visited twice.
    torch.autograd.backward(features, grad_tensors=tuple(gradients), retain_graph=True)
    observation = None if observe_rgb is None else observe_rgb()
    auxiliary_value = 0.0 if auxiliary is None else float(auxiliary.detach())
    if auxiliary is not None and auxiliary.requires_grad:
        auxiliary.backward()
    return StagedBackwardResult(total_rgb, tuple(gradients), observation, auxiliary_value)


def step_native(model, bridge, optimizer, data, starts, cfg):
    """v6 step-compatible candidate; imports original helpers only on invocation.

    Use with the original v6 code directory on sys.path. The caller retains
    ownership of model modes, native freeze scopes, optimizer and checkpoints.
    All original ten-frame losses, support checks and result keys are retained.
    """
    import native_experiment as native

    if not starts:
        raise ValueError('A native step requires a nonempty complete-window batch')
    optimizer.zero_grad(set_to_none=True)
    windows = [data.window(start) for start in starts]
    conditions = [native.make_condition(window[0]) for window in windows]
    encoded = model.encode_windows(windows)
    encoded['z_quantized'].retain_grad()
    predictions = model.decode_latents(encoded['z_quantized'], conditions,
                                       initial_contexts=encoded['initial_contexts'])
    if len(predictions) != len(starts):
        raise RuntimeError('Missing batch predictions')
    flat = []
    for b, start in enumerate(starts):
        if len(predictions[b]) != 10:
            raise RuntimeError('Missing initial/future output')
        for h, prediction in enumerate(predictions[b]):
            if not torch.equal(prediction['coordinates'], windows[b][0]['coordinates']):
                raise RuntimeError('Initial latent support was changed')
            flat.append((start, h, prediction))

    terms, counts = [], []

    def rgb_loss(index, leaf):
        start, horizon, original = flat[index]
        prediction = dict(original, features=leaf)
        image = bridge.render(prediction, data.records[start + horizon])
        loss, values = data.loss(image, start, horizon)
        terms.append(values)
        counts.append(bridge.last_count)
        return loss

    def rgb_observation():
        gradient = encoded['z_quantized'].grad
        if gradient is None:
            raise RuntimeError('RGB loss is disconnected from the quantized codes')
        norms = gradient.detach().norm(dim=2)
        query_gradient = model.queries.weight.grad
        return dict(rgb_quantized_grad_per_sample_h=norms.cpu().tolist(),
                    rgb_quantized_grad_per_h=norms.mean(0).cpu().tolist(),
                    rgb_query_grad_per_h=None if query_gradient is None else
                    query_gradient.detach().norm(dim=1).cpu().tolist())

    auxiliary = encoded['aux_loss'] * cfg['train']['aux_multiplier']
    backward = staged_rgb_backward([prediction['features'] for _, _, prediction in flat],
                                   rgb_loss, batch_size=len(starts), auxiliary=auxiliary,
                                   observe_rgb=rgb_observation)
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
    result = dict(**{key: float(native.np.mean([value[key] for value in terms])) for key in terms[0]},
                  auxiliary=backward.auxiliary, **backward.rgb_observation, gradient_norms=norms,
                  gs_count_min=min(counts), gs_count_max=max(counts),
                  batch_ids=encoded['indices'].detach().cpu().tolist(),
                  gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    if result['gpu_peak_gib'] > cfg['resources']['peak_limit_gib']:
        raise RuntimeError('Recorded GPU allocation exceeded admitted limit')
    return result
