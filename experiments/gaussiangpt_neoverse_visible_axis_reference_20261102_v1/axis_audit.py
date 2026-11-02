"""Source-image axis landmarks for reviewing two trained motion windows.

The line endpoints are visually estimated on the source images. This is a
diagnostic against a color mask, not a physical pose or segmentation metric.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
BASELINE = EXPERIMENTS / 'gaussiangpt_neoverse_eight_window_object_audit_20261102_v1/result_baseline/OBJECT_AUDIT.json'
ORIGINAL = EXPERIMENTS / 'gaussiangpt_neoverse_qformer_full177_20261101_v1'
TWO_FREE = EXPERIMENTS / 'gaussiangpt_neoverse_two_future_visual_20261102_v2/result'
TWO_Q = EXPERIMENTS / 'gaussiangpt_neoverse_two_future_qformer_20261102_v1/result/step_01194'
REFERENCE = ROOT / 'source_axis_reference.json'
RADII = (2, 4, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rgb_sha(image):
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (252, 448, 3):
        raise ValueError('Expected one 448 by 252 RGB panel')
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def samples(point_a, point_b, height, width):
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([*a, *b]).all():
        raise ValueError('Axis endpoints must be finite xy pairs')
    if np.array_equal(a, b):
        raise ValueError('Axis endpoints must differ')
    xy = np.rint(np.linspace(a, b, 100)).astype(np.int64)
    x, y = xy[:, 0], xy[:, 1]
    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError('Axis outside image')
    return x, y


def coverage(mask, point_a, point_b):
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Expected a two-dimensional boolean color mask')
    x, y = samples(point_a, point_b, *mask.shape)
    distance = distance_transform_edt(~mask) if mask.any() else np.full(mask.shape, np.inf)
    return {str(r): float(np.mean(distance[y, x] <= r)) for r in RADII}


def validate_reference(reference, baseline):
    if (reference.get('schema') != 'visible_source_axis_reference_v1'
            or reference.get('image_size_wh') != [448, 252]
            or len(reference.get('rows', [])) != 18):
        raise RuntimeError('Unexpected source axis reference')
    if sha(baseline) != reference['baseline_object_audit_sha256']:
        raise RuntimeError('Baseline object audit changed')
    original = read(baseline)
    baseline_rows = {(r['start'], r['horizon'], r['object']): r for r in original['rows']}
    found = set()
    for row in reference['rows']:
        key = (row['start'], row['horizon'], row['object'])
        if key in found or key not in baseline_rows:
            raise RuntimeError('Missing or duplicate landmark reference')
        found.add(key)
        if row['source_target_rgb_sha256'] != baseline_rows[key]['source_target_rgb_sha256']:
            raise RuntimeError('Landmark target pixels do not match baseline')
        samples(row['free_end_xy'], row['visible_interior_xy'], 252, 448)
    if found != {(s, h, obj) for s, obj in ((35, 'yellow'), (123, 'green')) for h in range(1, 10)}:
        raise RuntimeError('Expected 18 source-axis landmarks')


def load_core(reference):
    path, config = ORIGINAL / 'core.py', ORIGINAL / 'protocol.json'
    if sha(path) != reference['color_core_sha256'] or sha(config) != reference['color_config_sha256']:
        raise RuntimeError('Color mask implementation or configuration changed')
    spec = importlib.util.spec_from_file_location('_source_axis_color_core', path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core, read(config)['object_proxies']


def three_panels(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    if image.shape != (252, 1344, 3):
        raise RuntimeError('Expected three 448 by 252 panels: ' + str(path))
    return [image[:, 448 * i:448 * (i + 1)] for i in range(3)]


def four_panels(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    if image.shape != (276, 1792, 3):
        raise RuntimeError('Expected four 448 by 252 panels with a 24-pixel header: ' + str(path))
    return [image[24:, 448 * i:448 * (i + 1)] for i in range(4)]


def assert_source(image, row):
    actual = rgb_sha(image)
    if actual != row['source_target_rgb_sha256']:
        raise RuntimeError('Source target differs from bound landmark: ' + str((row['start'], row['horizon'])))
    return actual


def two_window_images(row):
    folder = 's%03d' % row['start']
    frame = 'frame_%02d.png' % row['horizon']
    free_path, q_path = TWO_FREE / folder / frame, TWO_Q / folder / frame
    free, q = three_panels(free_path), three_panels(q_path)
    assert_source(free[0], row)
    assert_source(q[0], row)
    return {'source': free[0], 'two_window_free_motion': free[1],
            'two_window_q_motion': q[1], 'two_window_q_static': q[2]}, {
                'free_image_sha256': sha(free_path), 'q_image_sha256': sha(q_path)}


def stage_b_data(root, expected_sha):
    path = Path(root) / 'EVALUATION.json'
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise RuntimeError('Full evaluation bytes do not match the supplied SHA-256')
    data = read(path)
    src = data['source']
    if (data.get('status') != 'complete' or data.get('mode') != 'free_z'
            or data.get('window_predictions') != 1512
            or src.get('schema') != 'stage_b_shared_decoder_evaluation_v1'
            or src.get('quantized') is not False
            or src.get('codec_sha256') != 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
            or src.get('cache_manifest_sha256') != 'cb5c2b30f1b4fdac4db642046a46773cc0b96a4dabdab4917e0d3c79ff6eaa4a'):
        raise RuntimeError('Expected the completed original-codec stage-B evaluation')
    future = [row for row in data['rows'] if row['horizon'] > 0]
    indexed = {(row['start'], row['horizon']): row for row in future}
    if len(future) != 1512 or set(indexed) != {(s, h) for s in range(168) for h in range(1, 10)}:
        raise RuntimeError('Stage-B evaluation coverage is incomplete')
    return data, indexed


def stage_b_images(root, indexed, row):
    path = Path(root) / 'snapshots' / ('s%03d' % row['start']) / ('frame_%02d.png' % row['horizon'])
    image = four_panels(path)
    assert_source(image[0], row)
    original = indexed[(row['start'], row['horizon'])]
    for name, panel in (('correct', image[1]), ('actual_target', image[3])):
        observed = float(np.abs(panel.astype(np.float64) - image[0].astype(np.float64)).mean() / 255)
        if abs(observed - original['metrics'][name]['metrics']['mae']) > 1 / 255 + 1e-6:
            raise RuntimeError('Saved stage-B panel disagrees with evaluation RGB MAE')
    return {'source': image[0], 'stage_b_free_motion': image[1], 'actual_state_codec': image[3]}, {
        'snapshot_sha256': sha(path)}


def marked(image, row, box, scale):
    canvas = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(canvas)
    a, b = tuple(row['free_end_xy']), tuple(row['visible_interior_xy'])
    draw.line((*a, *b), fill=(255, 40, 255), width=1)
    for p, color in ((a, (0, 255, 255)), (b, (255, 255, 0))):
        draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color, outline=(0, 0, 0))
    crop = canvas.crop(box)
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


def main(output, evaluation_root=None, expected_evaluation_sha=None):
    if bool(evaluation_root) != bool(expected_evaluation_sha):
        raise ValueError('Stage-B evaluation path and SHA-256 must be supplied together')
    reference = read(REFERENCE)
    validate_reference(reference, BASELINE)
    core, specs = load_core(reference)
    data, indexed = stage_b_data(evaluation_root, expected_evaluation_sha) if evaluation_root else (None, None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, views = [], {}
    for row in reference['rows']:
        images, binding = (stage_b_images(evaluation_root, indexed, row) if data is not None
                           else two_window_images(row))
        masks = {name: core.color_mask(image, specs[str(row['start'])]) for name, image in images.items()}
        values = {name: coverage(mask, row['free_end_xy'], row['visible_interior_xy'])
                  for name, mask in masks.items()}
        rows.append(dict(start=row['start'], horizon=row['horizon'], object=row['object'],
                         source_target_rgb_sha256=row['source_target_rgb_sha256'],
                         free_end_xy=row['free_end_xy'], visible_interior_xy=row['visible_interior_xy'],
                         input_binding=binding, axis_color_coverage_by_radius_px=values))
        views[(row['start'], row['horizon'])] = (row, images)
    if len(rows) != 18:
        raise RuntimeError('Incomplete axis analysis')
    summaries = {}
    for start in (35, 123):
        items = [row for row in rows if row['start'] == start]
        arms = items[0]['axis_color_coverage_by_radius_px']
        summaries[str(start)] = {name: {str(r): float(np.mean([
            row['axis_color_coverage_by_radius_px'][name][str(r)] for row in items]))
            for r in RADII} for name in arms}
        box = (210, 25, 325, 175) if start == 35 else (150, 0, 335, 95)
        w, h = (box[2] - box[0]) * 2, (box[3] - box[1]) * 2
        sheet = Image.new('RGB', ((2 * w + 24) * 3, (h + 30) * 3), 'white')
        draw = ImageDraw.Draw(sheet)
        pred_name = 'stage_b_free_motion' if data is not None else 'two_window_q_motion'
        for horizon in range(1, 10):
            row, images = views[(start, horizon)]
            x, y = ((horizon - 1) % 3) * (2 * w + 24), ((horizon - 1) // 3) * (h + 30)
            draw.text((x, y + 3), 'h%d  source | %s' % (horizon, pred_name), fill='black')
            sheet.paste(marked(images['source'], row, box, 2), (x, y + 25))
            sheet.paste(marked(images[pred_name], row, box, 2), (x + w + 8, y + 25))
        sheet.save(output / ('s%03d_axis_review.png' % start), optimize=True)
    result = dict(status='complete', schema='visible_source_axis_diagnostic_v1',
                  reference_sha256=sha(REFERENCE), analysis_code_sha256=sha(__file__),
                  source_baseline_audit_sha256=sha(BASELINE),
                  evaluation_sha256=expected_evaluation_sha, evaluation_source=data['source'] if data else None,
                  thresholds_px=list(RADII), sampled_points_per_axis=100, summaries=summaries, rows=rows,
                  interpretation='Fraction of source-axis points near a fixed HSV/ROI mask; diagnostic only',
                  limitations=reference['limitations'], manual_motion_review_required=True,
                  no_acceptance_decision=True)
    with (output / 'AXIS_DIAGNOSTIC.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', rows=len(rows), summaries=summaries), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--evaluation-root', help='Completed stage-B evaluation directory')
    parser.add_argument('--expected-evaluation-sha256')
    args = parser.parse_args()
    main(args.output, args.evaluation_root, args.expected_evaluation_sha256)
