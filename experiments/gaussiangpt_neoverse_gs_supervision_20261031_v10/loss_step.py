"""Two-view GS supervision on the unchanged v9 latent model.

Targets are used only by loss callbacks. The original temporal VJP also restores
the complete initial-image derivative; stacking views does not truncate it.
"""
from __future__ import annotations

import math
import torch


def checked_weights(weights):
    weights = tuple(float(x) for x in weights)
    if len(weights) != 2 or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError('Expected two finite nonnegative teacher weights')
    return weights


def teacher_loss(images, targets, weights, mask, length=10):
    weights = checked_weights(weights)
    if images.shape != targets.shape or images.ndim != 4 or images.shape[0] != 2:
        raise ValueError('Expected matching two-view [2,H,W,3] arrays')
    if targets.requires_grad or length != 10:
        raise ValueError('Teacher must be detached and window must contain all ten frames')
    if mask.dtype != torch.bool or mask.shape != images.shape[1:3] or not bool(mask.any()):
        raise ValueError('Expected nonempty fixed nearby alpha mask')
    if weights[0] != 0:
        raise ValueError('This experiment changes only masked nearby supervision')
    pixel_errors=(images-targets).abs().mean(-1)
    errors=torch.stack((pixel_errors[0].mean(),pixel_errors[1][mask].mean()))
    return sum(w*errors[v] for v,w in enumerate(weights))/length, errors


def render_views(bridge, prediction, record):
    points = bridge.codec.decode_state(prediction['coordinates'], prediction['features'])
    gs, sh = bridge.prediction_gs(points)
    images = []
    for camera in record['cameras']:
        image, _ = bridge.render_fn(gs, camera, record['intrinsics'], record['near'], record['far'], sh)
        if image.shape != (252,448,3) or not bool(torch.isfinite(image).all()):
            raise RuntimeError('Invalid complete native rendered image')
        images.append(image.clamp(0,1))
    if len(images) != 2:
        raise RuntimeError('Expected original two cameras')
    bridge.last_count = len(gs)
    return torch.stack(images)


def step_native(model, bridge, optimizer, data, starts, cfg, temporal, weights):
    """Zero teacher weights dispatch EXACTLY to the original native step."""
    weights = checked_weights(weights)
    if weights == (0.,0.):
        result = temporal.step_native(model,bridge,optimizer,data,starts,cfg,delta_weight=1.)
        result.update(teacher_weights=list(weights), teacher_view0_l1=None, teacher_view1_l1=None)
        return result
    import native_experiment as native
    optimizer.zero_grad(set_to_none=True)
    windows = [data.window(s) for s in starts]
    conditions = [native.make_condition(w[0]) for w in windows]
    encoded = model.encode_windows(windows)
    encoded['z_quantized'].retain_grad()
    predictions = model.decode_latents(encoded['z_quantized'],conditions,
                                       initial_contexts=encoded['initial_contexts'])
    if len(predictions) != len(starts) or any(len(row) != 10 for row in predictions):
        raise RuntimeError('Missing full batch/window outputs')
    for b,row in enumerate(predictions):
        if any(not torch.equal(p['coordinates'],windows[b][0]['coordinates']) for p in row):
            raise RuntimeError('Initial latent support changed')
    terms, delta_terms, teacher_terms, counts = [], [], [], []

    def render(b,h,feature):
        image = render_views(bridge,dict(predictions[b][h],features=feature),data.records[starts[b]+h])
        counts.append(bridge.last_count)
        return image

    def base_loss(b,h,images):
        base,values = data.loss(images[0],starts[b],h)
        teacher,errors = teacher_loss(images,data.teacher_targets[starts[b]+h],weights,
                                     data.teacher_masks[starts[b]+h])
        terms.append(values);teacher_terms.append(errors.detach())
        return base+teacher

    def delta_loss(b,h,images,initial):
        start=starts[b]
        actual=data.records[start+h]['real']-data.records[start]['real']
        error=((images[0]-initial[0])-actual).abs().mean(-1)
        full,roi=error.mean(),error[data.roi].mean()
        delta_terms.append(dict(delta_full_l1=float(full.detach()),delta_roi_l1=float(roi.detach())))
        return full+roi

    def observe():
        gradient=encoded['z_quantized'].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError('Combined reconstruction losses disconnected from quantized codes')
        norms=gradient.detach().norm(dim=2)
        query=model.queries.weight.grad
        return dict(rgb_quantized_grad_per_sample_h=norms.cpu().tolist(),
                    rgb_quantized_grad_per_h=norms.mean(0).cpu().tolist(),
                    rgb_query_grad_per_h=None if query is None else query.detach().norm(dim=1).cpu().tolist(),
                    gradient_scope='original RGB plus two-view teacher; auxiliary excluded')

    result=temporal.staged_temporal_backward(
        [[p['features'] for p in row] for row in predictions],render,base_loss,delta_loss,
        delta_weight=1.,auxiliary=encoded['aux_loss']*cfg['train']['aux_multiplier'],observe_rgb=observe)
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise RuntimeError('Nonfinite native/LDM gradient')
    norms=dict(residual_head=native.grad_norm(model.features),motion_encoder=native.grad_norm(model.encoder_backbone),
        motion_decoder=native.grad_norm(model.decoder_backbone),native_trainable=native.grad_norm(bridge),
        native_out=native.grad_norm(bridge.codec.ae.decoder.out),native_heads=native.grad_norm(bridge.codec.ae.feature_heads),
        native_multiplier=native.grad_norm(bridge.codec.ae.gaussian_multiplier))
    total_norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['train']['grad_clip'],error_if_nonfinite=True)
    optimizer.step()
    teacher_mean=torch.stack(teacher_terms).mean(0)
    return dict(**{key:float(native.np.mean([v[key] for v in terms])) for key in terms[0]},
        **{key:float(native.np.mean([v[key] for v in delta_terms])) for key in delta_terms[0]},
        teacher_weights=list(weights),teacher_view0_l1=float(teacher_mean[0]),teacher_view1_l1=float(teacher_mean[1]),
        delta_weight=1.,auxiliary=result.auxiliary,**result.rgb_observation,gradient_norms=norms,
        preclip_grad_norm=float(total_norm),
        initial_rgb_grad_norm_per_sample=[float(g.norm()) for g in result.initial_rgb_gradients],
        initial_feature_grad_norm_per_sample=[float(g[0].norm()) for g in result.feature_gradients],
        feature_grad_norm_per_sample_h=[[float(g.norm()) for g in row] for row in result.feature_gradients],
        gs_count_min=min(counts),gs_count_max=max(counts),batch_ids=encoded['indices'].detach().cpu().tolist(),
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
