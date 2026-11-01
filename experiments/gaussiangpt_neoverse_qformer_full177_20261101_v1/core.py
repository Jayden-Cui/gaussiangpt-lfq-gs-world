"""Endpoint capacity diagnostic; no Q-Former, quantizer, or motion decoder."""
from __future__ import annotations

import numpy as np
from PIL import Image
import torch


def future_features(initial, residual, std, mode):
    """Optimize residuals in feature units, with equal initial Jacobians.

    Bounded: F0 + a*tanh(R/a), a=3*std. This is exactly the original
    F0 + a*tanh(U) with U=R/a, but does not confound the comparison with
    an a-fold difference in initial feature-space learning rates.
    A zero-scale bounded channel remains fixed. Unbounded: F0 + R.
    """
    if mode == 'unbounded':
        return initial + residual
    if mode != 'bounded':
        raise ValueError(mode)
    amplitude = 3 * std
    safe = torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    return initial + amplitude * torch.tanh(residual / safe)


def image_losses(image, target, roi, motion_weight):
    error = (image - target).abs().mean(-1)
    full = error.mean()
    region = error[roi].mean()
    motion = (motion_weight * error).sum() / motion_weight.sum().clamp_min(1e-6)
    # Exactly 10 times the formal run's real-view h>0 reconstruction terms.
    return full + region + (10 / 9) * motion, dict(mae=full, roi_mae=region, motion_mae=motion)


@torch.no_grad()
def image_metrics(image, target, roi, motion_weight):
    _, values = image_losses(image, target, roi, motion_weight)
    result = {key: float(value) for key, value in values.items()}
    result['psnr'] = float(-10 * torch.log10((image-target).square().mean().clamp_min(1e-12)))
    result['object_box_mae'] = float((image[25:235, 140:320]-target[25:235, 140:320]).abs().mean())
    return result


@torch.no_grad()
def residual_statistics(initial, residual, std, mode):
    amplitude = 3 * std
    safe = amplitude.clamp_min(1e-12)
    delta = future_features(initial, residual, std, mode) - initial
    relative = delta.abs() / safe
    stats = dict(mode=mode, delta_rms=float(delta.square().mean().sqrt()),
                 delta_abs_max=float(delta.abs().max()),
                 delta_abs_max_by_channel=delta.abs().amax(0).tolist(),
                 delta_rms_by_channel=delta.square().mean(0).sqrt().tolist(),
                 above_original_bound_fraction=float((delta.abs() > amplitude + 1e-5).float().mean()),
                 relative_delta_p95=float(torch.quantile(relative.flatten(), .95)),
                 zero_std_channels=int((std == 0).sum()))
    if mode == 'bounded':
        value = torch.tanh(residual / safe)
        stats.update(tanh_abs_above_095_fraction=float((value.abs() > .95).float().mean()),
                     tanh_abs_above_099_fraction=float((value.abs() > .99).float().mean()),
                     local_derivative_mean=float((1-value.square()).mean()))
    return stats


def color_mask(image, spec):
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    hsv = np.asarray(Image.fromarray(array).convert('HSV'))
    h0, h1 = spec['hue_uint8']
    mask = (hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= spec['saturation_min']) & (hsv[..., 2] >= spec['value_min'])
    x0, y0, x1, y1 = spec['roi_xyxy']
    region = np.zeros(mask.shape, dtype=bool)
    region[y0:y1, x0:x1] = True
    return mask & region


def mask_geometry(mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 10:
        return dict(valid=False, pixels=len(xx), centroid_xy=None, axis_degrees=None, elongation=None)
    xy = np.stack([xx, yy], axis=1).astype(np.float64)
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    axis = vectors[:, -1]
    return dict(valid=True, pixels=len(xx), centroid_xy=xy.mean(0).tolist(),
                axis_degrees=float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
                elongation=float(values[-1] / max(values[0], 1e-9)))


def object_proxy(image, target, spec):
    predicted, actual = color_mask(image, spec), color_mask(target, spec)
    p, a = mask_geometry(predicted), mask_geometry(actual)
    result = dict(prediction=p, target=a, iou=float((predicted & actual).sum() / max((predicted | actual).sum(), 1)),
                  centroid_error_px=None, axis_error_degrees=None,
                  interpretation='Fixed color/ROI proxy, not a tracked object pose or automatic action-success label')
    if p['valid'] and a['valid']:
        result['centroid_error_px'] = float(np.linalg.norm(np.array(p['centroid_xy'])-a['centroid_xy']))
        angle = abs(p['axis_degrees']-a['axis_degrees'])
        result['axis_error_degrees'] = min(angle, 180-angle)
    return result


def coordinate_overlap(initial, target):
    """Spatial-cell overlap only: row indices are never physical correspondences."""
    a = {tuple(row) for row in initial.detach().cpu().tolist()}
    b = {tuple(row) for row in target.detach().cpu().tolist()}
    return dict(initial_nodes=len(a), target_nodes=len(b), shared_nodes=len(a & b),
                target_only_nodes=len(b-a), initial_only_nodes=len(a-b), union_nodes=len(a | b),
                coordinate_iou=len(a & b)/len(a | b), physical_point_correspondence=False)
