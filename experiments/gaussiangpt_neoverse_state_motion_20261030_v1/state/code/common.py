"""Shared, explicit GS schema and similarity transforms; no point deletion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
TEACHER = BASE / 'movies_sampling_best_fourclip_20261023_v1'
UPSTREAM = BASE / 'gaussiangpt_state_vae_20261028_v1'
VENDOR = UPSTREAM / 'vendor/GaussianGPT-00658a4ab3f48daed3bdf4ba50805bb2b7dc5e14'
CHECKPOINT = UPSTREAM / 'checkpoints/vqvae_photoshape_official.ckpt'
CLIPS = ['buildingblock_tower_17_176']
H, W, N = 252, 448, 252 * 448
C0 = 0.28209479177387814


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_quat(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def similarity(gs, camera, rotation, scale, translation):
    """mu'=s R mu+t, covariance'=s² R covariance R^T; C2W follows same map."""
    from scipy.spatial.transform import Rotation
    rotation = torch.as_tensor(rotation, dtype=torch.float32, device=gs.device)
    translation = torch.as_tensor(translation, dtype=torch.float32, device=gs.device)
    q = Rotation.from_matrix(rotation.cpu().numpy()).as_quat()[[3, 0, 1, 2]].copy()
    q = torch.as_tensor(q, dtype=torch.float32, device=gs.device)
    result = gs.clone()
    result[:, :3] = scale * (gs[:, :3] @ rotation.T) + translation
    result[:, 6:9] = scale * gs[:, 6:9]
    result[:, 9:13] = canonical_quat(quat_multiply(q, gs[:, 9:13]))
    c = camera.clone()
    c[:3, :3] = rotation @ camera[:3, :3]
    c[:3, 3] = scale * (rotation @ camera[:3, 3]) + translation
    return result, c


def render(gs, camera, intrinsics, near=.001, far=100., sh=None):
    from gsplat import rasterization
    k = torch.zeros((3, 3), dtype=torch.float32, device=gs.device)
    k[0, 0], k[1, 1], k[0, 2], k[1, 2] = intrinsics.float() * gs.new_tensor([W, H, W, H])
    k[2, 2] = 1
    color = gs[:, 3:6].clamp_min(0).contiguous() if sh is None else sh.contiguous()
    image, alpha, _ = rasterization(
        means=gs[:, :3].contiguous(), quats=gs[:, 9:13].contiguous(),
        scales=gs[:, 6:9].contiguous(), opacities=gs[:, 13].contiguous(),
        colors=color, viewmats=torch.linalg.inv(camera.float())[None], Ks=k[None],
        width=W, height=H, near_plane=float(near), far_plane=float(far),
        radius_clip=0., backgrounds=gs.new_zeros((1, 3)),
        sh_degree=None if sh is None else 1, render_mode='RGB',
        rasterize_mode='classic', distributed=False, packed=True)
    return image[0, :, :, :3].float(), alpha[0].float()


def psnr(a, b):
    return float(-10 * torch.log10((a.float()-b.float()).square().mean().clamp_min(1e-12)))


def metrics(image, target):
    # Fixed central workspace crop, deliberately not called a semantic hand mask.
    return {'psnr': psnr(image, target),
            'workspace_roi_psnr': psnr(image[38:227, 67:380], target[38:227, 67:380]),
            'mae': float((image.float()-target.float()).abs().mean())}


def region_metrics(image, target, boxes):
    x0, y0, x1, y1 = boxes['object']
    result = {'object_psnr': psnr(image[y0:y1, x0:x1], target[y0:y1, x0:x1]),
              'object_mae': float((image[y0:y1, x0:x1]-target[y0:y1, x0:x1]).abs().mean())}
    mask = torch.zeros(image.shape[:2], dtype=torch.bool, device=image.device)
    for x0, y0, x1, y1 in boxes['hand_activity']:
        mask[y0:y1, x0:x1] = True
    result['hand_activity_psnr'] = psnr(image[mask], target[mask])
    result['hand_activity_mae'] = float((image[mask]-target[mask]).abs().mean())
    return result


def external_dict(gs):
    # Imported lazily: preprocessing runs in the MoVieS environment.
    from conf.dataclasses import PCKeys as K
    return {K.COORDS: gs[:, :3].clone(), K.SH0: (gs[:, 3:6]-.5)/C0,
            K.SCALES: gs[:, 6:9].clamp_min(1e-12).log(),
            K.QUATS: canonical_quat(gs[:, 9:13]),
            K.OPACITIES: torch.logit(gs[:, 13:14].clamp(1e-6, 1-1e-6)),
            K.SH: gs.new_zeros((len(gs), 9)),
            K.BATCH: torch.zeros(len(gs), dtype=torch.int32, device=gs.device)}


def prediction_gs(points):
    from conf.dataclasses import PCKeys as K
    rgb = points[K.SH0] * C0 + .5
    gs = torch.cat([points[K.COORDS], rgb, points[K.SCALES].exp(),
                    canonical_quat(points[K.QUATS]), points[K.OPACITIES].sigmoid().reshape(-1, 1)], -1)
    sh = torch.cat([points[K.SH0].reshape(-1, 1, 3), points[K.SH].reshape(-1, 3, 3)], 1)
    return gs, sh


def voxelize(gs, grid, with_stats=True):
    """Deterministic representative and sufficient statistics using ALL source GS.

    Stable CPU sorting resolves alpha ties by original source/pixel order.
    Statistics are alpha-weighted mean/std of 14 internal attributes plus log count.
    Float64 reductions avoid GPU atomic reduction nondeterminism.
    """
    g = gs.detach().cpu().numpy().astype(np.float32)
    coord = np.rint(g[:, :3] / grid).astype(np.int32)
    order = np.lexsort((np.arange(len(g)), -g[:, 13], coord[:, 2], coord[:, 1], coord[:, 0]))
    ordered = coord[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1))]
    ids = order[starts]
    counts = np.diff(np.r_[starts, len(g)])
    result = {'representatives': torch.from_numpy(g[ids].copy()),
              'coordinates': torch.from_numpy(coord[ids].copy()),
              'source_indices': torch.from_numpy(ids.copy()),
              'counts': torch.from_numpy(counts.copy()), 'n_input': len(g)}
    assert int(counts.sum()) == len(g)
    if with_stats:
        q = g[:, 9:13].copy()
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        q *= np.where(q[:, :1] < 0, -1, 1)
        alpha = np.clip(g[:, 13:14], 1e-6, 1-1e-6)
        attributes = np.concatenate([g[:, :3]/grid-coord,
                                    g[:, 3:6],
                                    100*np.exp(np.clip(np.log(np.maximum(g[:, 6:9], 1e-12)), -10, -2)),
                                    q, np.clip(np.log(alpha/(1-alpha)), -10, 10)/10], 1)
        weights = np.maximum(g[order, 13:14], 1e-6).astype(np.float64)
        values = attributes[order].astype(np.float64)
        denominator = np.add.reduceat(weights, starts)
        mean = np.add.reduceat(values * weights, starts) / denominator
        second = np.add.reduceat(values * values * weights, starts) / denominator
        stats = np.concatenate([mean, np.sqrt(np.maximum(second-mean*mean, 0)),
                                np.log1p(counts[:, None])], 1).astype(np.float32)
        assert stats.shape[1] == 29 and np.isfinite(stats).all()
        result['set_statistics'] = torch.from_numpy(stats)
    return result
