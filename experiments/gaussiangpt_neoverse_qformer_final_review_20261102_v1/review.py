"""Review the completed two-window Q output against its bound free-code teacher."""
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
Q = EXP / 'gaussiangpt_neoverse_two_future_qformer_20261102_v1'
Q_RESULT = Q / 'result/step_01194'
PAIRED = EXP / 'gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1/result/step_00360'
VISUAL = EXP / 'gaussiangpt_neoverse_two_future_visual_20261102_v2/result'
CORE = EXP / 'gaussiangpt_neoverse_qformer_support_20261101_v1/core.py'
CONFIG = EXP / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json'
SENSITIVITY = EXP / 'gaussiangpt_neoverse_object_proxy_sensitivity_20261102_v1/sensitivity.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def close(actual, expected):
    return ((actual is None and expected is None) or
            (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)))


def summarize(rows, owner):
    visible = [r for r in rows if r[owner]['target_visible']]
    valid = [r for r in visible if r[owner]['prediction_visible']]
    original = sum(r[owner]['center_over_original_bbox'] is not None and
                   r[owner]['center_over_original_bbox'] <= .1 for r in visible)
    alternate = sum(r[owner]['same_center_over_largest_bbox'] is not None and
                    r[owner]['same_center_over_largest_bbox'] <= .1 for r in visible)
    angles = [r[owner]['axis_error_degrees'] for r in visible if r[owner]['axis_error_degrees'] is not None]
    run = longest = 0
    for r in rows:
        run = run + 1 if r[owner]['target_visible'] and not r[owner]['prediction_visible'] else 0
        longest = max(longest, run)
    return dict(target_visible_frames=len(visible), prediction_visible_frames=len(valid),
                original_center_within_10pct_frames=original,
                same_center_largest_bbox_within_10pct_frames=alternate,
                median_axis_error_degrees=statistics.median(angles) if angles else None,
                mean_color_iou=statistics.mean(r[owner]['iou'] for r in visible) if visible else None,
                maximum_consecutive_missing_frames=longest,
                target_fragmented_frames=sum(r[owner]['target_fragmented'] for r in visible))


def main(output):
    done, export = load(Q / 'DONE.json'), load(Q / 'result/FINAL_EXPORT.json')
    summary, evaluation = load(Q / 'result/SUMMARY.json'), load(Q_RESULT / 'EVALUATION.json')
    causal, replay = load(Q / 'result/CAUSALITY.json'), load(Q_RESULT / 'REPLAY.json')
    previous, manifest = load(VISUAL / 'OBJECT_PROXY.json'), load(VISUAL / 'MANIFEST.json')
    if (done['status'] != 'complete' or export['status'] != 'complete' or summary['updates'] != 1194
            or done['evaluation_sha256'] != sha(Q_RESULT / 'EVALUATION.json')
            or done['summary_sha256'] != sha(Q / 'result/SUMMARY.json')
            or done['replay_sha256'] != sha(Q_RESULT / 'REPLAY.json')
            or summary['causality_sha256'] != sha(Q / 'result/CAUSALITY.json')
            or causal['status'] != 'passed' or replay['status'] != 'passed'
            or replay['packets'] != 36 or replay['views'] != 72
            or evaluation['source']['posterior_checkpoint_sha256'] != summary['checkpoints']['1194']
            or evaluation['source']['teacher_checkpoint_sha256'] != '461c8391d09a27ba5f2b4f4f52878c8a9d8b287072e2fa6a52a9ac4877043744'
            or evaluation['source']['quantized'] is not False
            or evaluation['teacher_reproduction_max_abs'] > 5e-5):
        raise RuntimeError('Final Q checkpoint, teacher, causality or replay binding differs')
    expected = {x['path']: x['sha256'] for x in export['files']}
    if len(expected) != 26:
        raise RuntimeError('Final export is incomplete or contains duplicate paths')
    for path, digest in expected.items():
        if sha(Q / path) != digest:
            raise RuntimeError('Final Q output hash differs: ' + path)
    if (previous['source_evaluation_sha256'] != sha(PAIRED / 'EVALUATION.json')
            or previous['visual_manifest_sha256'] != sha(VISUAL / 'MANIFEST.json')
            or previous['proxy_code_sha256'] != sha(CORE)
            or previous['proxy_config_sha256'] != sha(CONFIG)):
        raise RuntimeError('Free-code teacher or object proxy binding differs')
    paired_hashes = {(r['start'], r['horizon']): r['source_sha256'] for r in manifest['frames']}
    if set(paired_hashes) != {(s, h) for s in (35, 123) for h in range(1, 10)}:
        raise RuntimeError('Expected 18 teacher frames')
    configs = load(CONFIG)['object_proxies']
    core, sensitivity = module('_object_proxy_final_review', CORE), module('_component_final_review', SENSITIVITY)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, summaries = [], {}
    for start in (35, 123):
        sheets = {b: Image.new('RGB', (896 * 3, 276 * 3), 'white') for b in ('motion', 'static')}
        window_rows = {'motion': [], 'static': []}
        for h in range(1, 10):
            q_path = Q_RESULT / f's{start:03d}' / f'frame_{h:02d}.png'
            t_path = PAIRED / f's{start:03d}' / f'frame_{h:02d}.png'
            if sha(t_path) != paired_hashes[(start, h)]:
                raise RuntimeError('Teacher frame hash differs')
            q_image, t_image = np.asarray(Image.open(q_path).convert('RGB')), np.asarray(Image.open(t_path).convert('RGB'))
            if q_image.shape != (252, 1344, 3) or t_image.shape != (252, 2240, 3):
                raise RuntimeError('Unexpected three- or five-panel PNG layout')
            q = [q_image[:, i * 448:(i + 1) * 448] for i in range(3)]
            t = [t_image[:, i * 448:(i + 1) * 448] for i in range(5)]
            if not np.array_equal(q[0], t[0]):
                raise RuntimeError('Q and teacher source frames differ')
            for branch, target, prediction, teacher in [('motion', q[0], q[1], t[2]), ('static', t[3], q[2], t[4])]:
                cfg = configs[str(start)]
                target_mask = core.color_mask(target, cfg)
                values = {}
                for name, image in [('q', prediction), ('teacher', teacher)]:
                    proxy = core.object_proxy(image, target, cfg)
                    impact = sensitivity.measure(target_mask, core.color_mask(image, cfg))
                    values[name] = dict(target_visible=proxy['target']['valid'], prediction_visible=proxy['prediction']['valid'],
                        iou=proxy['iou'], axis_error_degrees=proxy['axis_error_degrees'],
                        center_over_original_bbox=impact['original_center_over_all_bbox'],
                        same_center_over_largest_bbox=impact['same_center_over_largest_target_bbox'],
                        target_bbox_inflation=impact['target']['diagonal_inflation'],
                        target_fragmented=impact['target_is_fragmented'],
                        prediction_fragmented=impact['prediction_is_fragmented'])
                old = previous['summary'][str(start)][branch]['frames'][h - 1]
                if (old['horizon'] != h or old['target_visible'] != values['teacher']['target_visible']
                        or old['prediction_visible'] != values['teacher']['prediction_visible']
                        or not close(old['iou'], values['teacher']['iou'])
                        or not close(old['center_error_over_bbox_diagonal'], values['teacher']['center_over_original_bbox'])
                        or not close(old['axis_error_degrees'], values['teacher']['axis_error_degrees'])):
                    raise RuntimeError('Original free-code teacher proxy was not reproduced')
                row = dict(start=start, horizon=h, frame=start + h, branch=branch,
                           q_frame_sha256=sha(q_path), teacher_frame_sha256=sha(t_path), **values)
                results.append(row)
                window_rows[branch].append(row)
                x, y = ((h - 1) % 3) * 896, ((h - 1) // 3) * 276
                sheet = sheets[branch]
                ImageDraw.Draw(sheet).text((x + 6, y + 5), f'h{h}  TARGET                                      Q OUTPUT', fill='black')
                sheet.paste(Image.fromarray(target), (x, y + 24))
                sheet.paste(Image.fromarray(prediction), (x + 448, y + 24))
        summaries[str(start)] = {branch: {owner: summarize(rows, owner) for owner in ('q', 'teacher')}
                                 for branch, rows in window_rows.items()}
        for branch, sheet in sheets.items():
            sheet.save(output / f's{start:03d}_{branch}_contact.png', optimize=True)
    if len(results) != 36:
        raise RuntimeError('Missing visual proxy rows')
    payload = dict(status='complete', scope='Two known training windows; auxiliary image proxies only',
        q_checkpoint_sha256=summary['checkpoints']['1194'], q_evaluation_sha256=sha(Q_RESULT / 'EVALUATION.json'),
        q_export_sha256=sha(Q / 'result/FINAL_EXPORT.json'), q_replay_sha256=sha(Q_RESULT / 'REPLAY.json'),
        q_causality_sha256=sha(Q / 'result/CAUSALITY.json'), teacher_frame_manifest_sha256=sha(VISUAL / 'MANIFEST.json'),
        teacher_object_proxy_sha256=sha(VISUAL / 'OBJECT_PROXY.json'), core_sha256=sha(CORE),
        proxy_config_sha256=sha(CONFIG), sensitivity_sha256=sha(SENSITIVITY), review_code_sha256=sha(__file__),
        teacher_proxy_reproduced=True, source_frames_identical=True, summaries=summaries, rows=results,
        manual_visual_review_required=True, full_168_teacher_accepted=False,
        note='Color masks and connected components are not physical object labels; static target uses the same S0 and camera.')
    with (output / 'OBJECT_PROXY.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'rows': len(results), 'summaries': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New, non-existing output directory')
    args = parser.parse_args()
    main(args.output)
