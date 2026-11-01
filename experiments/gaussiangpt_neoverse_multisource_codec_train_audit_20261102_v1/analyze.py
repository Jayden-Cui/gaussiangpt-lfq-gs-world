"""Audit existing native codec results on the two TRAIN videos only."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
DATA = OUTPUTS / 'gaussiangpt_neoverse_multisource_action_data_20261102_v1'
CODEC = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_20261031_v1/evaluations/baseline'
REFERENCE = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_reference_20261102_v2/result/REFERENCE.json'
sys.path.insert(0, str(DATA))
from source_store import TRAIN, EVAL, PINS, CODEC_SHA


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def quantile(values, fraction):
    values = sorted(float(x) for x in values)
    if not values or not 0 <= fraction <= 1 or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid quantile inputs')
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    weight = position - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def summarize(rows, clip):
    if len(rows) != 177 or {row['frame'] for row in rows} != set(range(177)):
        raise RuntimeError('Expected every frame exactly once')
    if any(row['clip'] != clip or row['split'] != 'TRAIN' or row['checkpoint_sha256'] != CODEC_SHA for row in rows):
        raise RuntimeError('A non-training source or different codec entered the audit')
    codec = [float(row['codec_real']['mae']) for row in rows]
    teacher = [float(row['teacher_real']['mae']) for row in rows]
    psnr = [float(row['codec_real']['psnr']) for row in rows]
    if not all(math.isfinite(v) and v >= 0 for v in codec + teacher + psnr):
        raise RuntimeError('Nonfinite codec metric')
    segments = {}
    for lo, hi in ((0, 43), (44, 87), (88, 131), (132, 176)):
        values = [float(row['codec_real']['mae']) for row in rows if lo <= row['frame'] <= hi]
        if len(values) != hi - lo + 1:
            raise RuntimeError('Incomplete frame segment')
        segments[f'{lo:03d}_{hi:03d}'] = {'frames': len(values), 'codec_real_mae': statistics.fmean(values)}
    return {'frames': len(rows), 'codec_real_mae_mean': statistics.fmean(codec),
            'codec_real_mae_median': statistics.median(codec), 'codec_real_mae_p10': quantile(codec, .1),
            'codec_real_mae_p90': quantile(codec, .9), 'codec_real_mae_min': min(codec),
            'codec_real_mae_max': max(codec), 'codec_real_psnr_mean_db': statistics.fmean(psnr),
            'teacher_real_mae_mean': statistics.fmean(teacher),
            'frames_codec_mae_over_005': sum(v > .05 for v in codec), 'segments': segments}


def main():
    test = read(ROOT / 'CPU_TESTS.json')
    if test.get('status') != 'passed' or test.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Codec audit tests missing or changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Codec audit source changed: ' + name)
    evaluation_path, targets_path = CODEC / 'EVALUATION.json', CODEC / 'targets.jsonl'
    if sha(evaluation_path) != PINS['evaluations/baseline/EVALUATION.json']:
        raise RuntimeError('Historic codec evaluation changed')
    evaluation = read(evaluation_path)
    if (evaluation.get('status') != 'evaluated' or evaluation.get('checkpoint_sha256') != CODEC_SHA
            or sha(targets_path) != evaluation.get('targets_sha256')):
        raise RuntimeError('Historic codec target records changed')
    per_clip = {clip: [] for clip in TRAIN}
    with targets_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('clip') in per_clip:
                per_clip[row['clip']].append(row)
    results = {clip: summarize(rows, clip) for clip, rows in per_clip.items()}
    for clip in TRAIN:
        expected = evaluation['summary_by_clip'][clip]['codec_real']['mae']
        if abs(results[clip]['codec_real_mae_mean'] - expected) > 1e-10:
            raise RuntimeError('Per-frame mean differs from historic summary')
    reference = read(REFERENCE)
    if (reference.get('status') != 'complete' or reference.get('clip') != TRAIN[1]
            or reference.get('target_frame') != 35 or reference.get('codec_sha256') != CODEC_SHA):
        raise RuntimeError('Actual-state reference changed')
    row = next(x for x in per_clip[TRAIN[1]] if x['frame'] == 35)
    if abs(row['codec_real']['mae'] - reference['actual_target_codec']['metrics']['mae']) > 1e-5:
        raise RuntimeError('Frame-35 reference differs from the original codec record')
    baseline = results[TRAIN[0]]['codec_real_mae_mean']
    output = {'status': 'complete', 'train_sources': results,
              'source10_over_source17_codec_mae_ratio': results[TRAIN[1]]['codec_real_mae_mean'] / baseline,
              'codec_sha256': CODEC_SHA, 'historic_evaluation_sha256': sha(evaluation_path),
              'historic_targets_sha256': sha(targets_path), 'frame35_reference_sha256': sha(REFERENCE),
              'source_code_sha256': sha(ROOT / 'CODE.sha256'),
              'eval_source_records_selected': False, 'model_training_performed': False,
              'scope': 'Existing codec reconstruction metrics for two TRAIN videos, 177 correlated frames each; no action-model or unseen-test claim'}
    with (ROOT / 'TRAIN_CODEC_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: original codec metrics reproduced for both complete TRAIN videos')


if __name__ == '__main__':
    main()
