"""Compare each C posterior with its exact free-code teacher by source motion group."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ('lowest', 'lower_middle', 'upper_middle', 'highest')
CONTROL_NAMES = ('correct', 'teacher_free', 'initial_only', 'swapped_window_codes', 'actual_target')
METRICS = ('mae', 'motion_mae')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_self():
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise ValueError('CPU witness is absent or differs')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != expected:
            raise ValueError('Bound report source changed: ' + name)


def validate_groups(groups, protocol, actual_sha):
    if (protocol.get('schema') != 'fixed_teacher_posterior_motion_group_report_v1'
            or actual_sha != protocol['groups_sha256'] or groups.get('status') != 'complete'
            or groups.get('schema') != 'target_frame_motion_groups_v1'
            or groups.get('cache_manifest_sha256') != protocol['cache_manifest_sha256']
            or groups.get('frame_cache_sha256') != protocol['frame_cache_sha256']
            or groups.get('model_predictions_used') is not False or groups.get('group_size') != 42):
        raise ValueError('Frozen groups or source binding differs')
    scores = {int(k): v for k, v in groups['scores_by_start'].items()}
    if (len(scores) != 168 or set(scores) != set(range(168))
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in scores.values())):
        raise ValueError('Invalid source-only motion scores')
    ordered = sorted(scores, key=lambda s: (scores[s], s))
    expected = {name: ordered[42*i:42*(i+1)] for i, name in enumerate(GROUPS)}
    if groups.get('groups') != expected:
        raise ValueError('Frozen motion group membership differs from source scores')
    return scores, expected


def future_rows(evaluation):
    if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
            or evaluation.get('unique_future_frames') != 176
            or evaluation.get('all_source_frames_preserved') is not True
            or evaluation.get('decoder_future_geometry_input') is not False):
        raise ValueError('Incomplete evaluation or future geometry in decoder')
    rows = [row for row in evaluation['rows'] if row.get('horizon', 0) > 0]
    pairs = [(row['start'], row['horizon']) for row in rows]
    if (len(rows) != 1512 or set(pairs) != {(s, h) for s in range(168) for h in range(1, 10)}
            or any(row['frame'] != row['start'] + row['horizon'] for row in rows)):
        raise ValueError('Window/horizon coverage or frame mapping differs')
    return {(row['start'], row['horizon']): row for row in rows}


def metric(row, control, name):
    value = row['metrics'][control]['metrics'][name]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid nonnegative image error')
    return float(value)


def summarize(rows, starts, control):
    chosen = [rows[(s, h)] for s in sorted(starts) for h in range(1, 10)]
    out = {}
    for name in METRICS:
        by_frame, by_h = defaultdict(list), defaultdict(list)
        for row in chosen:
            value = metric(row, control, name)
            by_frame[row['frame']].append(value)
            by_h[row['horizon']].append(value)
        if len(chosen) != 378 or any(len(by_h[h]) != 42 for h in range(1, 10)):
            raise ValueError('Unexpected group or horizon size')
        out[name] = {'window_prediction_equal': sum(metric(r, control, name) for r in chosen)/len(chosen),
            'physical_frame_equal_within_group': sum(sum(by_frame[f])/len(by_frame[f])
                                                     for f in sorted(by_frame))/len(by_frame),
            'by_horizon_window_equal': {str(h): sum(by_h[h])/len(by_h[h]) for h in range(1, 10)},
            'unique_physical_frames': len(by_frame), 'window_predictions': len(chosen)}
    return out


def compare(groups, protocol, group_sha, student, teacher, teacher_sha):
    scores, group_starts = validate_groups(groups, protocol, group_sha)
    s, t = student['source'], teacher['source']
    if (s.get('schema') != 'fixed_teacher_posterior_evaluation_v1' or s.get('variant') not in ('C0', 'C1', 'C2')
            or s.get('quantized') is not False or s.get('teacher_evaluation_sha256') != teacher_sha
            or s.get('cache_manifest_sha256') != protocol['cache_manifest_sha256']
            or any(s.get(k) != t.get(k) or not isinstance(s.get(k), str) for k in
                   ('decoder_digest', 'codec_sha256', 'cache_manifest_sha256'))
            or s.get('teacher_codes_sha256') != t.get('merged_codes_sha256', t.get('codes_sha256'))):
        raise ValueError('Student and teacher decoder, codec, cache, evaluation or codes differ')
    student_rows, teacher_rows = future_rows(student), future_rows(teacher)
    for pair, row in student_rows.items():
        if row['metrics']['teacher_free'] != teacher_rows[pair]['metrics']['correct']:
            raise ValueError('A copied teacher metric differs from its bound source: ' + str(pair))
        for control in CONTROL_NAMES:
            for name in METRICS:
                metric(row, control, name)
    by_group = {}
    for name in GROUPS:
        starts = group_starts[name]
        controls = {c: summarize(student_rows, starts, c) for c in CONTROL_NAMES}
        a, b = controls['correct']['motion_mae'], controls['teacher_free']['motion_mae']
        ratio = lambda x, y: x/y if y > 0 else None
        by_group[name] = {'window_starts': starts, 'source_score_min': min(scores[s] for s in starts),
            'source_score_max': max(scores[s] for s in starts), 'controls': controls,
            'student_over_teacher_motion_mae': {
                'physical_frame_equal': ratio(a['physical_frame_equal_within_group'], b['physical_frame_equal_within_group']),
                'by_horizon': {str(h): ratio(a['by_horizon_window_equal'][str(h)],
                                             b['by_horizon_window_equal'][str(h)]) for h in range(1, 10)}}}
    return {'status': 'complete', 'schema': protocol['schema'], 'groups_sha256': group_sha,
        'teacher_evaluation_sha256': teacher_sha, 'student_source': s, 'groups': by_group,
        'teacher_row_metrics_match': 1512, 'model_predictions_used_for_grouping': False,
        'aggregation': protocol['aggregation'], 'scope': protocol['limits'],
        'action_success': 'not determined; per-object and visual review still required'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('groups', 'student', 'teacher', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    verify_self()
    protocol = read(ROOT / 'protocol.json')
    result = compare(read(args.groups), protocol, sha(args.groups), read(args.student),
                     read(args.teacher), sha(args.teacher))
    result['student_evaluation_sha256'] = sha(args.student)
    result['report_code_sha256'] = sha(ROOT / 'CODE.sha256')
    with Path(args.output).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print('PASS: fixed-teacher C motion groups; dynamic visual review still required')


if __name__ == '__main__':
    main()
