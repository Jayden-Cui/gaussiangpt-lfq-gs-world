"""Freeze ground-truth motion strata and summarize fully covered evaluation rows.

No model, predicted image, training update or GPU is used to define strata.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = ('lowest', 'lower_middle', 'upper_middle', 'highest')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def partition(scores):
    if len(scores) != 168 or set(scores) != set(range(168)):
        raise ValueError('Expected one source-only score for each of 168 starts')
    if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in scores.values()):
        raise ValueError('Invalid motion intensity')
    ordered = sorted(scores, key=lambda start: (scores[start], start))
    return {name: ordered[42*i:42*(i+1)] for i, name in enumerate(NAMES)}


def freeze(manifest_path, output):
    # The data manifest is the exact one used for stages A and B.
    protocol = read(ROOT / 'protocol.json')
    manifest_path = Path(manifest_path)
    if sha(manifest_path) != protocol['cache_manifest_sha256']:
        raise ValueError('Unexpected source cache manifest')
    manifest = read(manifest_path)
    if manifest['status'] != 'complete' or manifest['starts'] != list(range(168)):
        raise ValueError('Incomplete source cache')
    entry = manifest['frames']
    frame_file = (manifest_path.parent / entry['file']).resolve()
    if sha(frame_file) != entry['sha256'] or entry['sha256'] != protocol['frame_cache_sha256']:
        raise ValueError('Unexpected source frame bytes')
    import torch
    torch.set_num_threads(2)
    data = torch.load(frame_file, map_location='cpu', weights_only=True)
    samples = data['samples']
    if len(samples) != 177:
        raise ValueError('Expected all 177 source frames')
    images = []
    for sample in samples:
        x = sample['record']['real']
        if tuple(x.shape) != (252, 448, 3) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):
            raise ValueError('Unexpected source frame')
        if float(x.min()) < 0 or float(x.max()) > 1:
            raise ValueError('Source RGB outside [0,1]')
        images.append(x.float())
    scores = {}
    with torch.no_grad():
        for start in range(168):
            values = [float(((images[start+h]-images[start]).abs().mean(-1)/0.1).clamp(0, 1).mean())
                      for h in range(1, 10)]
            scores[start] = sum(values) / len(values)
    groups = partition(scores)
    write_new(output, {'status': 'complete', 'schema': protocol['schema'],
        'protocol_sha256': sha(ROOT/'protocol.json'), 'code_sha256': sha(ROOT/'motion_groups.py'),
        'cache_manifest_sha256': sha(manifest_path), 'frame_cache_sha256': sha(frame_file),
        'scores_by_start': {str(k): scores[k] for k in sorted(scores)}, 'groups': groups,
        'group_size': 42, 'model_predictions_used': False, 'future_frames_are_evaluation_only': True,
        'scope': protocol['limits']})


def summarize(rows, starts, control):
    chosen = [r for r in rows if r['start'] in starts]
    if len(chosen) != 42 * 9:
        raise ValueError('Expected all 9 future predictions for all 42 windows')
    result = {}
    for metric in ('mae', 'motion_mae'):
        values = []
        by_frame = defaultdict(list)
        by_h = defaultdict(list)
        for row in chosen:
            value = row['metrics'][control]['metrics'][metric]
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Missing or nonfinite metric')
            values.append(value)
            by_frame[row['frame']].append(value)
            by_h[row['horizon']].append(value)
        result[metric] = {
            'window_prediction_equal': sum(values)/len(values),
            'physical_frame_equal_within_group': sum(sum(v)/len(v) for v in by_frame.values())/len(by_frame),
            'h9_window_equal': sum(by_h[9])/len(by_h[9]),
            'unique_physical_frames': len(by_frame),
            'window_predictions': len(values)}
    return result


def evaluate(group_path, evaluation_path, output):
    protocol = read(ROOT/'protocol.json')
    groups = read(group_path)
    if (groups.get('status') != 'complete' or groups['protocol_sha256'] != sha(ROOT/'protocol.json')
            or groups['code_sha256'] != sha(ROOT/'motion_groups.py')
            or groups['cache_manifest_sha256'] != protocol['cache_manifest_sha256']
            or groups['frame_cache_sha256'] != protocol['frame_cache_sha256']):
        raise ValueError('Frozen grouping does not match protocol/code/source')
    scores = {int(k): v for k,v in groups['scores_by_start'].items()}
    if groups['groups'] != partition(scores):
        raise ValueError('Group membership differs from source scores')
    evaluation = read(evaluation_path)
    if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
            or evaluation['source'].get('cache_manifest_sha256') != protocol['cache_manifest_sha256']):
        raise ValueError('Evaluation is incomplete or comes from another source')
    rows = [r for r in evaluation['rows'] if r['horizon'] > 0]
    pairs = [(r['start'], r['horizon']) for r in rows]
    if (len(pairs) != 1512 or set(pairs) != {(s,h) for s in range(168) for h in range(1,10)}
            or any(r['frame'] != r['start'] + r['horizon'] for r in rows)):
        raise ValueError('Evaluation coverage, pair uniqueness or frame mapping differs')
    common = set.intersection(*(set(row['metrics']) for row in rows))
    required = {'correct', 'actual_target'}
    if not required.issubset(common):
        raise ValueError('Required controls missing')
    controls = sorted(common & {'correct', 'actual_target', 'baseline_qformer', 'historical_qformer',
                                'initial_only', 'swapped_window_codes'})
    grouped = {}
    for name in NAMES:
        starts = set(groups['groups'][name])
        grouped[name] = {control: summarize(rows, starts, control) for control in controls}
    write_new(output, {'status': 'complete', 'groups_sha256': sha(group_path),
        'evaluation_sha256': sha(evaluation_path), 'evaluation_source': evaluation['source'],
        'aggregation': 'Within each group: each window-prediction equally; also average within each physical frame then average frames',
        'h9_aggregation': 'Window-prediction equal across the 42 windows in the group',
        'controls': controls, 'groups': grouped, 'scope': protocol['limits'],
        'action_success': 'not determined; object and visual review required'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--manifest', required=True)
    f.add_argument('--output', required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--groups', required=True)
    e.add_argument('--evaluation', required=True)
    e.add_argument('--output', required=True)
    args = p.parse_args()
    if args.command == 'freeze':
        freeze(args.manifest, args.output)
    else:
        evaluate(args.groups, args.evaluation, args.output)
