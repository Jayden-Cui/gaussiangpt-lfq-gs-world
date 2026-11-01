"""Audit sensitivity of a color-proxy center error to detached mask pixels.

This is an auxiliary image-space analysis; it does not replace a tracked
object mask, modify the pre-registered metric, or decide experiment success.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCE = EXPERIMENTS / 'gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1/result/step_00360'
VISUAL = EXPERIMENTS / 'gaussiangpt_neoverse_two_future_visual_20261102_v2/result'
CORE = EXPERIMENTS / 'gaussiangpt_neoverse_qformer_support_20261101_v1/core.py'
CONFIG = EXPERIMENTS / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def component_stats(mask):
    """Describe all pixels and their largest 8-connected component."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Expected a 2D boolean mask')
    pixels = int(mask.sum())
    if not pixels:
        return dict(pixels=0, largest_pixels=0, largest_fraction=None,
                    all_bbox=None, largest_bbox=None, all_diagonal=None,
                    largest_diagonal=None, diagonal_inflation=None), mask.copy()
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = labels == counts.argmax()

    def box(m):
        yy, xx = np.nonzero(m)
        return [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]

    def diagonal(b):
        return float(np.hypot(b[2] - b[0] + 1, b[3] - b[1] + 1))

    all_bbox, largest_bbox = box(mask), box(largest)
    all_diag, largest_diag = diagonal(all_bbox), diagonal(largest_bbox)
    return dict(pixels=pixels, largest_pixels=int(largest.sum()),
                largest_fraction=float(largest.sum() / pixels),
                all_bbox=all_bbox, largest_bbox=largest_bbox,
                all_diagonal=all_diag, largest_diagonal=largest_diag,
                diagonal_inflation=all_diag / largest_diag), largest


def centroid(mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 10:
        return None
    return np.array([xx.mean(), yy.mean()], dtype=np.float64)


def measure(target_mask, prediction_mask):
    """Keep old center fixed for the denominator-only sensitivity comparison."""
    target, target_largest = component_stats(target_mask)
    prediction, prediction_largest = component_stats(prediction_mask)
    a, b = centroid(target_mask), centroid(prediction_mask)
    la, lb = centroid(target_largest), centroid(prediction_largest)
    distance = float(np.linalg.norm(a - b)) if a is not None and b is not None else None
    largest_distance = float(np.linalg.norm(la - lb)) if la is not None and lb is not None else None
    def divided(numerator, denominator):
        return numerator / denominator if numerator is not None and denominator else None
    return dict(target=target, prediction=prediction, original_center_distance_px=distance,
                original_center_over_all_bbox=divided(distance, target['all_diagonal']),
                same_center_over_largest_target_bbox=divided(distance, target['largest_diagonal']),
                largest_components_center_over_target_bbox=divided(largest_distance, target['largest_diagonal']),
                target_is_fragmented=target['largest_fraction'] is not None and target['largest_fraction'] < .95,
                prediction_is_fragmented=prediction['largest_fraction'] is not None and prediction['largest_fraction'] < .95,
                interpretation='Largest connected component is a sensitivity reference, not ground-truth object segmentation')


def main(output):
    manifest = load_json(VISUAL / 'MANIFEST.json')
    recorded = load_json(VISUAL / 'OBJECT_PROXY.json')
    evaluation = SOURCE / 'EVALUATION.json'
    if (manifest['source_evaluation_sha256'] != sha(evaluation)
            or recorded['source_evaluation_sha256'] != sha(evaluation)
            or recorded['visual_manifest_sha256'] != sha(VISUAL / 'MANIFEST.json')
            or recorded['proxy_code_sha256'] != sha(CORE)
            or recorded['proxy_config_sha256'] != sha(CONFIG)):
        raise RuntimeError('Frozen source, proxy or visual manifest differs')
    expected = {(r['start'], r['horizon']): r['source_sha256'] for r in manifest['frames']}
    if set(expected) != {(s, h) for s in (35, 123) for h in range(1, 10)}:
        raise RuntimeError('Expected all 18 visualized source frames')
    spec = importlib.util.spec_from_file_location('_pinned_existing_object_proxy', CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    configs = load_json(CONFIG)['object_proxies']
    results = []
    for start in (35, 123):
        cfg = configs[str(start)]
        for h in range(1, 10):
            path = SOURCE / f's{start:03d}' / f'frame_{h:02d}.png'
            if sha(path) != expected[(start, h)]:
                raise RuntimeError('Source PNG hash mismatch')
            image = np.asarray(Image.open(path).convert('RGB'))
            if image.shape != (252, 2240, 3):
                raise RuntimeError('Unexpected paired-frame layout')
            panels = [image[:, i * 448:(i + 1) * 448] for i in range(5)]
            for branch, target, prediction in [('motion', panels[0], panels[2]),
                                               ('static', panels[3], panels[4])]:
                result = measure(core.color_mask(target, cfg), core.color_mask(prediction, cfg))
                original = recorded['summary'][str(start)][branch]['frames'][h - 1]
                if original['horizon'] != h:
                    raise RuntimeError('Recorded horizon order differs')
                x, y = result['original_center_over_all_bbox'], original['center_error_over_bbox_diagonal']
                if (x is None) != (y is None) or (x is not None and not np.isclose(x, y, rtol=1e-10, atol=1e-10)):
                    raise RuntimeError('Could not reproduce pre-registered center proxy')
                results.append(dict(start=start, horizon=h, branch=branch, frame_sha256=sha(path), **result))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(status='complete', source_evaluation_sha256=sha(evaluation),
                   visual_manifest_sha256=sha(VISUAL / 'MANIFEST.json'),
                   original_proxy_sha256=sha(VISUAL / 'OBJECT_PROXY.json'),
                   original_core_sha256=sha(CORE), original_config_sha256=sha(CONFIG),
                   analysis_code_sha256=sha(__file__), source_rows=len(results), rows=results,
                   scope='Two training windows, two branches, 18 lossless PNGs; auxiliary proxy sensitivity only',
                   warning='Do not substitute connected components for tracked object labels or the pre-registered gate')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': payload['status'], 'rows': len(results), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
