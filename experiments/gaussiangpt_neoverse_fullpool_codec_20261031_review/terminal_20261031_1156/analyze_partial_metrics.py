"""Read-only, stdlib-only matched analysis of an interrupted codec evaluation."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import statistics

GROUPS = ('codec_real', 'codec_teacher', 'pseudo_view_teacher')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr')
INITIAL_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
ENCODER_SHA = 'c2ddb2b1dd5a43ee4a86286ea09ef141f61661963fa1d5979edf3d4f5917737b'


def read_rows(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    keys = [(row['clip'], row['frame']) for row in rows]
    assert len(keys) == len(set(keys)), 'Duplicate clip/frame rows'
    return rows, dict(zip(keys, rows)), hashlib.sha256(raw).hexdigest()


def metrics(rows):
    return {group: {metric: statistics.mean(row[group][metric] for row in rows)
                    for metric in METRICS} for group in GROUPS}


def comparison(before, after):
    result = {}
    for group in GROUPS:
        result[group] = {}
        for metric in METRICS:
            b, a = before[group][metric], after[group][metric]
            v = {'baseline': b, 'decoder_only': a, 'after_minus_before': a - b}
            if metric.endswith('mae'):
                v['relative_error_reduction_percent'] = 100 * (b - a) / b
            result[group][metric] = v
    return result


def run(root):
    trained, trained_by_key, trained_sha = read_rows(root / 'evaluations/decoder_only/targets.jsonl')
    baseline, baseline_by_key, baseline_sha = read_rows(root / 'evaluations/baseline/targets.jsonl')
    train_done_bytes = (root / 'arms/decoder_only/TRAIN_DONE.json').read_bytes()
    done = json.loads(train_done_bytes)
    failure_bytes = (root / 'evaluations/decoder_only/FAILURE.json').read_bytes()
    failure = json.loads(failure_bytes)
    assert len(trained) == 2238 and len(baseline) == 9027
    assert done['status'] == 'complete' and done['steps'] == 4426
    assert done['initial_encoder_sha256'] == done['final_encoder_sha256'] == ENCODER_SHA
    assert failure['frames_complete'] == len(trained)
    assert all(row['checkpoint_sha256'] == done['checkpoint_sha256'] for row in trained)
    assert all(row['encoder_sha256'] == ENCODER_SHA and row['additional_codec_updates'] == 4426 for row in trained)
    assert all(row['checkpoint_sha256'] == INITIAL_SHA and row['encoder_sha256'] == ENCODER_SHA
               and row['additional_codec_updates'] == 0 for row in baseline)
    teacher_parity = {m: 0.0 for m in METRICS}
    for key, row in trained_by_key.items():
        base = baseline_by_key[key]
        assert row['split'] == base['split'] == 'TRAIN'
        assert row['source_group'] == base['source_group']
        assert row['pts'] == base['pts']
        assert row['latent_channels'] == base['latent_channels'] == 64
        assert row['input_gaussians'] == base['input_gaussians'] == 112896
        for group in GROUPS:
            for metric in METRICS:
                assert math.isfinite(row[group][metric]) and math.isfinite(base[group][metric])
        for metric in METRICS:
            teacher_parity[metric] = max(teacher_parity[metric], abs(row['teacher_real'][metric] - base['teacher_real'][metric]))
        for view in ('real_view', 'nearby_view'):
            assert math.isfinite(row['replay'][view]['mae']) and math.isfinite(row['replay'][view]['psnr'])
            assert row['replay'][view]['mae'] <= 1e-5 and row['replay'][view]['psnr'] >= 60
    clips = collections.defaultdict(list)
    for row in trained:
        clips[row['clip']].append(row)
    per_clip = []
    complete = []
    for clip, rows in sorted(clips.items()):
        frames = sorted(row['frame'] for row in rows)
        assert frames == list(range(len(frames)))
        full = frames == list(range(177))
        if full:
            complete.append(clip)
        before_rows = [baseline_by_key[(row['clip'], row['frame'])] for row in rows]
        before, after = metrics(before_rows), metrics(rows)
        per_clip.append({'clip': clip, 'source_group': rows[0]['source_group'], 'frames': len(rows),
                         'frame_range': [frames[0], frames[-1]], 'complete_177_frames': full,
                         'comparison': comparison(before, after)})
    assert len(complete) == 12 and len(clips) == 13
    complete_rows = [row for row in trained if row['clip'] in complete]
    assert len(complete_rows) == 2124
    before_equal = {g: {m: statistics.mean(item['comparison'][g][m]['baseline'] for item in per_clip
                                          if item['complete_177_frames']) for m in METRICS} for g in GROUPS}
    after_equal = {g: {m: statistics.mean(item['comparison'][g][m]['decoder_only'] for item in per_clip
                                         if item['complete_177_frames']) for m in METRICS} for g in GROUPS}
    all_before = metrics([baseline_by_key[(row['clip'], row['frame'])] for row in trained])
    all_after = metrics(trained)
    replay = {view: {'rows': len(trained), 'all_passed': True,
                      'max_mae': max(row['replay'][view]['mae'] for row in trained),
                      'min_psnr': min(row['replay'][view]['psnr'] for row in trained)}
              for view in ('real_view', 'nearby_view')}
    return {'analysis_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'read_only': True, 'gpu_executed': False,
            'source_root': str(root),
            'source_sha256': {'trained_targets': trained_sha, 'baseline_targets': baseline_sha,
                              'train_done': hashlib.sha256(train_done_bytes).hexdigest(),
                              'evaluation_failure': hashlib.sha256(failure_bytes).hexdigest()},
            'checks': {'unique_pairs': True, 'exact_matched_pts': True, 'matching_roles_groups': True,
                       'consistent_checkpoint_encoder_step': True,
                       'trained_checkpoint_sha256': done['checkpoint_sha256'],
                       'encoder_sha256': ENCODER_SHA, 'additional_codec_updates': 4426,
                       'baseline_total_frames': len(baseline), 'evaluated_rows': len(trained),
                       'complete_clips': len(complete), 'partial_clips': len(clips) - len(complete),
                       'source_groups': sorted({row['source_group'] for row in trained}),
                       'all_rows_train': True, 'tower14_evaluated': False,
                       'teacher_metric_max_abs_difference': teacher_parity, 'replay': replay},
            'all_2238_frames_descriptive': {'weighting': 'frame_equal', 'frames': len(trained),
                                           'comparison': comparison(all_before, all_after)},
            'complete_12_clips_equal': {'weighting': 'clip_equal_12_complete_clips', 'frames': len(complete_rows),
                                       'comparison': comparison(before_equal, after_equal)},
            'per_clip': per_clip,
            'failure': {k: failure[k] for k in ('status', 'error', 'context', 'frames_complete', 'valid_complete_evaluation')},
            'limitations': ['Evaluation stopped in fixed order at frame 2239; the selected prefix does not represent all 50 clips.',
                            'Twelve complete clips contain 2124 frames; one partial clip contains 114 frames. All are TRAIN; tower14 was not evaluated.',
                            'Overlapping clips are not independent samples; no population confidence interval is reported.',
                            'This trains only the state codec, not action QFormer/LFQ; better reconstruction does not establish learned action Z.',
                            'Decoder-only training completed; full-codec training was terminated by the queue and has no final evaluation.',
                            'Replay passes cover only the 2238 written rows; the next frame remains failed and thresholds are unchanged.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2, allow_nan=False))
