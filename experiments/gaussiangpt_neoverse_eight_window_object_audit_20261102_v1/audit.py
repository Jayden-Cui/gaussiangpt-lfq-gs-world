"""Auxiliary eight-window color-proxy audit for completed 168-window evaluations.

The two old HSV/ROI definitions were calibrated on windows 35 and 123. Applying
them elsewhere is exploratory and cannot provide tracked-object ground truth.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
CORE = EXP / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/core.py'
CONFIG = EXP / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json'
SENSITIVITY = EXP / 'gaussiangpt_neoverse_object_proxy_sensitivity_20261102_v1/sensitivity.py'
STAGE_A_CONFIG = EXP / 'gaussiangpt_neoverse_action_stage_a_20261102_v1/stage_a.json'
STARTS = (0, 35, 56, 84, 112, 123, 140, 167)
OBJECTS = {'yellow': '35', 'green': '123'}
CACHE_SHA = 'cb5c2b30f1b4fdac4db642046a46773cc0b96a4dabdab4917e0d3c79ff6eaa4a'
CODEC_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and
                                          math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10))


def panels(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    if image.shape != (276, 1792, 3):
        raise RuntimeError('Expected four 448x252 panels with a 24px header: ' + str(path))
    return [image[24:, i * 448:(i + 1) * 448] for i in range(4)]


def visual_overlay(image, masks):
    rgba = Image.fromarray(image).convert('RGBA')
    for name, color in [('yellow', (255, 0, 255)), ('green', (0, 170, 255))]:
        layer = np.zeros((252, 448, 4), dtype=np.uint8)
        layer[masks[name], :3] = color
        layer[masks[name], 3] = 120
        rgba = Image.alpha_composite(rgba, Image.fromarray(layer))
    return rgba.convert('RGB')


def summarize(rows, arm):
    visible = [r for r in rows if r['target']['valid']]
    present = [r for r in visible if r[arm]['prediction']['valid']]
    angles = [r[arm]['axis_error_degrees'] for r in visible if r[arm]['axis_error_degrees'] is not None]
    missing = longest = 0
    for r in rows:
        missing = missing + 1 if r['target']['valid'] and not r[arm]['prediction']['valid'] else 0
        longest = max(longest, missing)
    return dict(target_mask_detected_frames=len(visible), prediction_mask_detected_frames=len(present),
                mean_color_iou=statistics.mean(r[arm]['iou'] for r in visible) if visible else None,
                median_axis_error_degrees=statistics.median(angles) if angles else None,
                original_center_within_10pct_frames=sum(r[arm]['center_over_original_bbox'] is not None and
                    r[arm]['center_over_original_bbox'] <= .1 for r in visible),
                sensitivity_center_within_10pct_frames=sum(r[arm]['same_center_over_largest_bbox'] is not None and
                    r[arm]['same_center_over_largest_bbox'] <= .1 for r in visible),
                longest_missing_mask_run=longest,
                target_fragmented_frames=sum(r['target_fragmented'] for r in visible),
                warning='Detection uses HSV/ROI, not manually verified visibility or physical object pose')


def displacement(rows, arm):
    first, last = rows[0], rows[-1]
    def delta(key):
        a = first[key]['centroid_xy'] if key == 'target' else first[key]['prediction']['centroid_xy']
        b = last[key]['centroid_xy'] if key == 'target' else last[key]['prediction']['centroid_xy']
        return [b[0] - a[0], b[1] - a[1]] if a is not None and b is not None else None
    return dict(target_h1_to_h9_px=delta('target'), prediction_h1_to_h9_px=delta(arm),
                warning='Image-mask centroid displacement; not a tracked 3D object trajectory')


def continuous_evidence(data):
    source = data['source']
    if source.get('quantized') is False:
        return 'evaluation source explicitly records quantized=false'
    # The older stage-A result omits that field; its bound config records it.
    if (source.get('schema') == 'full177_fixed_decoder_free_code_stage_a_v1'
            and data.get('mode') == 'free_z'
            and source.get('stage_a_config_sha256') == sha(STAGE_A_CONFIG)
            and load(STAGE_A_CONFIG).get('quantized') is False):
        return 'older stage-A source binds a config with quantized=false'
    raise RuntimeError('No bound evidence that this evaluation uses continuous codes')


def main(evaluation_root, expected_evaluation_sha, output, reference_root=None,
         reference_audit=None, expected_reference_audit_sha=None):
    evaluation_root = Path(evaluation_root)
    if reference_root is not None and Path(reference_root).resolve() == evaluation_root.resolve():
        raise ValueError('Reference root must be a distinct evaluation, not the input itself')
    if (reference_audit is None) != (expected_reference_audit_sha is None):
        raise ValueError('Reference audit and its expected hash must be supplied together')
    evaluation_path = evaluation_root / 'EVALUATION.json'
    if len(expected_evaluation_sha) != 64 or sha(evaluation_path) != expected_evaluation_sha:
        raise RuntimeError('Completed evaluation hash differs')
    data = load(evaluation_path)
    if (data.get('status') != 'complete' or data.get('window_predictions') != 1512
            or data['source']['cache_manifest_sha256'] != CACHE_SHA
            or data['source']['codec_sha256'] != CODEC_SHA):
        raise RuntimeError('Expected completed original-codec continuous full-168 evaluation')
    continuous_binding = continuous_evidence(data)
    future = [r for r in data['rows'] if r['horizon'] > 0]
    indexed = {(r['start'], r['horizon']): r for r in future}
    if len(future) != 1512 or set(indexed) != {(s, h) for s in range(168) for h in range(1, 10)}:
        raise RuntimeError('Full evaluation window/horizon coverage differs')
    reference_rows = None
    if reference_audit is not None:
        if len(expected_reference_audit_sha) != 64 or sha(reference_audit) != expected_reference_audit_sha:
            raise RuntimeError('Reference object audit hash differs')
        reference = load(reference_audit)
        if (reference.get('status') != 'complete' or reference.get('audited_object_rows') != 144
                or reference.get('evaluation_sha256') == expected_evaluation_sha
                or reference.get('analysis_code_sha256') != sha(__file__)
                or reference.get('config_sha256') != sha(CONFIG)
                or reference.get('core_sha256') != sha(CORE)
                or reference.get('sensitivity_sha256') != sha(SENSITIVITY)):
            raise RuntimeError('Reference audit is not a distinct comparable evaluation')
        reference_rows = {(r['start'], r['horizon'], r['object']): r for r in reference['rows']}
        if len(reference_rows) != 144:
            raise RuntimeError('Reference audit object rows are incomplete or duplicated')
    specs = load(CONFIG)['object_proxies']
    core, sensitivity = module('_eight_window_core', CORE), module('_eight_window_sensitivity', SENSITIVITY)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previews = {name: Image.new('RGB', (448 * 3, 274 * len(STARTS)), 'white')
                for name in ('source', 'correct')}
    rows, summaries, anchor_checks, max_mae_disagreement = [], {}, 0, 0.
    for index, start in enumerate(STARTS):
        by_object = {name: [] for name in OBJECTS}
        for h in range(1, 10):
            path = evaluation_root / 'snapshots' / f's{start:03d}' / f'frame_{h:02d}.png'
            images = panels(path)
            source, correct, _, codec_reference = images
            target_sha = hashlib.sha256(source.tobytes(order='C')).hexdigest()
            if reference_root is not None:
                ref = Path(reference_root) / 'snapshots' / f's{start:03d}' / f'frame_{h:02d}.png'
                if not np.array_equal(source, panels(ref)[0]):
                    raise RuntimeError('Source target panel differs from reference: ' + str(path))
            recorded = indexed[(start, h)]
            for arm, image in [('correct', correct), ('actual_target', codec_reference)]:
                pixel_mae = float(np.abs(image.astype(np.float64) - source.astype(np.float64)).mean() / 255)
                old_mae = recorded['metrics'][arm]['metrics']['mae']
                error = abs(pixel_mae - old_mae)
                max_mae_disagreement = max(max_mae_disagreement, error)
                if error > 1 / 255 + 1e-6:
                    raise RuntimeError('Saved panel disagrees with evaluation RGB MAE')
            image_masks = {}
            for object_name, spec_key in OBJECTS.items():
                if reference_rows is not None:
                    ref = reference_rows.get((start, h, object_name))
                    if ref is None or ref.get('source_target_rgb_sha256') != target_sha:
                        raise RuntimeError('Source target differs from bound reference audit')
                cfg = specs[spec_key]
                target_mask = core.color_mask(source, cfg)
                image_masks[object_name] = target_mask
                arms = {}
                for arm, image in [('correct', correct), ('codec_reference', codec_reference)]:
                    proxy = core.object_proxy(image, source, cfg)
                    impact = sensitivity.measure(target_mask, core.color_mask(image, cfg))
                    arms[arm] = dict(prediction=proxy['prediction'], iou=proxy['iou'],
                        centroid_error_px=proxy['centroid_error_px'], axis_error_degrees=proxy['axis_error_degrees'],
                        center_over_original_bbox=impact['original_center_over_all_bbox'],
                        same_center_over_largest_bbox=impact['same_center_over_largest_target_bbox'],
                        prediction_fragmented=impact['prediction_is_fragmented'])
                target_stats, _ = sensitivity.component_stats(target_mask)
                target = core.mask_geometry(target_mask)
                if str(start) == spec_key:
                    original = recorded['metrics']['correct']['object_proxy']
                    for key in ('iou', 'centroid_error_px', 'axis_error_degrees'):
                        if not close(original[key], arms['correct'][key]):
                            raise RuntimeError('Saved anchor proxy differs for ' + str((start, h, key)))
                    anchor_checks += 1
                row = dict(start=start, horizon=h, frame=start + h, object=object_name,
                    source_snapshot_sha256=sha(path), source_target_rgb_sha256=target_sha, target=target,
                    target_largest_fraction=target_stats['largest_fraction'],
                    target_bbox_inflation=target_stats['diagonal_inflation'],
                    target_fragmented=target_stats['largest_fraction'] is not None and
                                      target_stats['largest_fraction'] < .95,
                    **arms)
                rows.append(row)
                by_object[object_name].append(row)
            if h in (1, 5, 9):
                x, y = (h // 4) * 448, index * 274
                for name, image in [('source', source), ('correct', correct)]:
                    masks = image_masks if name == 'source' else {
                        n: core.color_mask(image, specs[k]) for n, k in OBJECTS.items()}
                    sheet = previews[name]
                    ImageDraw.Draw(sheet).text((x + 5, y + 3), f's{start:03d} h{h}', fill='black')
                    sheet.paste(visual_overlay(image, masks), (x, y + 22))
        summaries[str(start)] = {name: {**{arm: summarize(group, arm) for arm in ('correct', 'codec_reference')},
                                       'correct_displacement': displacement(group, 'correct'),
                                       'codec_displacement': displacement(group, 'codec_reference')}
                                 for name, group in by_object.items()}
    if len(rows) != 144 or anchor_checks != 18:
        raise RuntimeError('Unexpected row or original-anchor-check count')
    for name, image in previews.items():
        image.save(output / f'{name}_mask_samples.png', optimize=True)
    result = dict(status='complete', evaluation_sha256=sha(evaluation_path), mode=data['mode'],
        evaluation_source=data['source'], config_sha256=sha(CONFIG), core_sha256=sha(CORE),
        sensitivity_sha256=sha(SENSITIVITY), analysis_code_sha256=sha(__file__),
        reference_targets_checked=(reference_root is not None or reference_audit is not None),
        reference_audit_sha256=expected_reference_audit_sha, continuous_code_evidence=continuous_binding,
        initial_only_panel_used=False, full_evaluation_pairs=1512,
        audited_window_pairs=72, audited_object_rows=144, anchor_proxy_rows_reproduced=anchor_checks,
        maximum_png_vs_float_mae_disagreement=max_mae_disagreement, summaries=summaries, rows=rows,
        scope='Eight fixed training windows; HSV/ROI proxies extrapolated beyond their original anchor windows',
        limitations=['No manual object segmentation or tracking ground truth',
                     'Mask visibility can reflect occlusion or false colors',
                     'Connected-component sensitivity is not a corrected physical-object metric',
                     'Do not infer full-168 or cross-video action success from these eight windows'],
        manual_review_required=True)
    with (output / 'OBJECT_AUDIT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'window_pairs': 72, 'object_rows': 144,
                      'anchor_proxy_rows_reproduced': anchor_checks,
                      'max_png_mae_difference': max_mae_disagreement}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-root', required=True)
    parser.add_argument('--expected-evaluation-sha256', required=True)
    parser.add_argument('--reference-root')
    parser.add_argument('--reference-audit')
    parser.add_argument('--expected-reference-audit-sha256')
    parser.add_argument('--output', required=True, help='New, non-existing output directory')
    args = parser.parse_args()
    main(args.evaluation_root, args.expected_evaluation_sha256, args.output, args.reference_root,
         args.reference_audit, args.expected_reference_audit_sha256)
