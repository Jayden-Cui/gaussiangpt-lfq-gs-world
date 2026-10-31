"""CPU-only historical baseline adapter. Never constructs/executes a codec.

49 prepared clips supply float32 reconstructions; tower17 and exposed tower14
retain the exactly bound old baseline metrics. This is not new GPU evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import pool_index as pool

RUN = 'gaussiangpt_neoverse_fullpool_codec_20261031_v1'
INITIAL_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
BINDING_SHA = '695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715'
OLD_EVALUATION_SHA = '68a5b3da1c122c0339f3171e53698b8c54a55dbbc051bf4987bfd0d04b58b88a'
OLD_TARGETS_SHA = '465d4a65070f7b5a00b379eb0f67105552f65f83c17e0063f4974a0e1e7ce16b'
OLD_MANIFEST_SHA = '4eaa25cc6d2759fc7a03ed96bb7a3041c0c3e2a60aeeba616b3e23da45ad963b'
ENCODER_SHA = 'c2ddb2b1dd5a43ee4a86286ea09ef141f61661963fa1d5979edf3d4f5917737b'
MODEL_DIGEST = '2dc414eaeb1fc03bfc3a77fe8d7cc607a23502ba48e6675bbc1816bcb304fb41'
PREPARED_SCHEMA = 'gsworld_continuous_state_v1'
HISTORICAL = {'buildingblock_tower_17_176', 'buildingblock_tower_14_0'}
EVAL_CLIP = 'buildingblock_tower_14_0'
CATEGORIES = ('codec_real', 'teacher_real', 'codec_teacher', 'pseudo_view_teacher')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr')
ROI = (67, 38, 380, 227)
# Absolute tolerances: CPU versus recorded GPU reduction is not bitwise equality.
TOLERANCE = {'mae': 2e-6, 'psnr': 2e-4, 'workspace_roi_psnr': 2e-4}


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Budget:
    def __init__(self, seconds):
        require(type(seconds) is int and 0 < seconds <= 86400, 'Invalid bounded CPU budget')
        self.seconds, self.start = seconds, time.monotonic()

    def check(self):
        require(not torch.cuda.is_initialized(), 'CPU baseline must never initialize CUDA')
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError('CPU baseline budget exceeded; no partial completion or GPU fallback')


def bound_bytes(path, digest, size, audit):
    """Hash the exact bytes passed to the CPU deserializer, with TOCTOU checks."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/symlink tensor file')
    before = pool.signature(path)
    require(type(size) is int and before[2] == size and pool.valid_sha(digest), 'Invalid tensor size/binding')
    data = path.read_bytes()
    require(len(data) == size and hashlib.sha256(data).hexdigest() == digest, 'Tensor bytes SHA mismatch')
    require(pool.signature(path) == before, 'Tensor changed while reading')
    prior = audit.seen.get(path)
    require(prior is None or prior == (before, digest), 'Tensor changed after admission')
    audit.seen[path] = (before, digest)
    return data


def load_tensor(path, record, audit):
    data = bound_bytes(path, record['sha256'], record['bytes'], audit)
    result = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    require(isinstance(result, dict), 'Expected tensor packet dictionary')
    return result


def finite_cpu(value, shape, label, dtype=None):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and tuple(value.shape) == tuple(shape), 'CPU tensor shape differs: ' + label)
    require(dtype is None or value.dtype == dtype, 'Tensor dtype differs: ' + label)
    require(bool(torch.isfinite(value).all()), 'Nonfinite tensor: ' + label)


def image_metrics(image, target):
    """Same four formulas/float32 reduction and ROI as old worker.image_metrics."""
    finite_cpu(image, (252, 448, 3), 'image', torch.float32)
    finite_cpu(target, (252, 448, 3), 'target', torch.float32)
    difference = image.float() - target.float()
    x0, y0, x1, y1 = ROI
    crop = difference[y0:y1, x0:x1]
    return {'mae': float(difference.abs().mean()),
            'psnr': float(-10 * difference.square().mean().clamp_min(1e-12).log10()),
            'workspace_roi_mae': float(crop.abs().mean()),
            'workspace_roi_psnr': float(-10 * crop.square().mean().clamp_min(1e-12).log10())}


def validate_metric_set(value):
    require(isinstance(value, dict) and set(value) == set(METRICS), 'Metric schema differs')
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in value.values()),
            'Nonfinite/invalid bounded-image metric')


def compute_metrics(images, cache):
    finite_cpu(images, (2, 252, 448, 3), 'saved reconstruction', torch.float32)
    require(bool(((images >= 0) & (images <= 1)).all()), 'Historical image was not clamped float32')
    finite_cpu(cache['real'], (252, 448, 3), 'real', torch.float32)
    finite_cpu(cache['targets'], (2, 252, 448, 3), 'teacher targets', torch.float32)
    teacher = cache['targets'].clamp(0, 1)
    return dict(codec_real=image_metrics(images[0], cache['real']),
                teacher_real=image_metrics(teacher[0], cache['real']),
                codec_teacher=image_metrics(images[0], teacher[0]),
                pseudo_view_teacher=image_metrics(images[1], teacher[1]))


def compare_recorded(actual, recorded):
    differences = {}
    for category in CATEGORIES:
        validate_metric_set(actual[category])
        require(set(recorded[category]) == set(TOLERANCE), 'Preparation metric schema changed')
        for metric, tolerance in TOLERANCE.items():
            old = recorded[category][metric]
            require(type(old) in (int, float) and math.isfinite(old), 'Invalid historical metric')
            delta = abs(actual[category][metric] - old)
            require(delta <= tolerance, f'CPU/historical metric mismatch: {category}.{metric}: {delta}')
            differences[category + '.' + metric] = delta
    return differences


def validate_state(state, clip, frame, normal, cache_sha, *, prepared):
    require(state.get('schema') == (PREPARED_SCHEMA if prepared else 'multisource_continuous_encoder_state_v1'),
            'Historical state schema differs')
    require(state.get('stride') == 4 and state.get('grid') == .008 and state.get('feature_dtype') == 'float32',
            'Historical state units differ')
    meta = state['metadata']
    for key, expected in dict(clip=clip, frame=frame, normalization=normal, source_cache_sha256=cache_sha).items():
        require(meta.get(key) == expected, 'State metadata mismatch: ' + key)
    require(abs(meta.get('pts', -1) - frame / 29.97) <= 1e-9, 'State PTS differs')
    checkpoint_key = 'codec_checkpoint_sha256' if prepared else 'checkpoint_sha256'
    require(meta.get(checkpoint_key) == INITIAL_SHA, 'State belongs to a different source checkpoint')
    coords, features = state['coordinates'], state['features']
    finite_cpu(features, (len(coords), 64), 'state features', torch.float32)
    finite_cpu(coords, (len(features), 3), 'state coordinates')
    require(coords.dtype in (torch.int32, torch.int64) and len(coords) > 0
            and len(torch.unique(coords, dim=0)) == len(coords), 'Invalid latent coordinates')
    if prepared:
        for key, expected in dict(clip=clip, frame=frame, normalization=normal, source_cache_sha256=cache_sha).items():
            require(state.get(key) == expected, 'Prepared state outer metadata differs: ' + key)
        require(abs(state.get('pts', -1) - frame / 29.97) <= 1e-9, 'Prepared outer PTS differs')
    else:
        require(meta.get('arm') == 'baseline' and meta.get('additional_codec_updates') == 0
                and meta.get('inherited_codec_steps') == 3000 and meta.get('decoder_target_key', 'MISSING') is None
                and meta.get('binding_sha256') == BINDING_SHA and meta.get('encoder_sha256') == ENCODER_SHA,
                'Historical baseline state is not the original target-free codec')


def check_coverage(rows):
    expected = {(c, f) for c in [*pool.EXPECTED, EVAL_CLIP] for f in range(177)}
    ids = [(r['clip'], r['frame']) for r in rows]
    require(all(type(f) is int for _, f in ids) and len(ids) == 9027 and len(set(ids)) == 9027
            and set(ids) == expected, 'Incomplete/duplicate/foreign 9027-frame baseline')
    for r in rows:
        require(r['split'] == ('EVAL-only' if r['clip'] == EVAL_CLIP else 'TRAIN'), 'Baseline split leakage')
        for category in CATEGORIES:
            validate_metric_set(r[category])


def summaries(rows):
    check_coverage(rows)
    def average(selected):
        return {c: {m: float(np.mean([r[c][m] for r in selected])) for m in METRICS} for c in CATEGORIES}
    return dict(summary_by_clip={c: average([r for r in rows if r['clip'] == c])
                                 for c in [*sorted(pool.EXPECTED), EVAL_CLIP]},
                summary_train_frames=average([r for r in rows if r['split'] == 'TRAIN']),
                summary_exposed_eval=average([r for r in rows if r['split'] == 'EVAL-only']))


def historical_rows(evaluation, rows, manifests):
    for name, expected in dict(status='evaluated', arm='baseline', frames=531, checkpoint_sha256=INITIAL_SHA,
            encoder_sha256=ENCODER_SHA, additional_codec_updates=0, decoder_eval=True,
            decoder_target_key=None, every_packet_saved_replay_passed=True, roi_xyxy=list(ROI),
            targets_sha256=OLD_TARGETS_SHA, state_manifest_sha256=OLD_MANIFEST_SHA).items():
        require(evaluation.get(name) == expected, 'Historical evaluation differs: ' + name)
    require(evaluation['source']['binding_sha256'] == BINDING_SHA, 'Historical data binding differs')
    expected = {(c, f) for c in (*HISTORICAL, 'buildingblock_tower_10_176') for f in range(177)}
    tables = []
    for values in (rows, manifests):
        require(len(values) == 531 and len({(r['clip'], r['frame']) for r in values}) == 531
                and {(r['clip'], r['frame']) for r in values} == expected, 'Old baseline coverage differs')
        tables.append({(r['clip'], r['frame']): r for r in values})
    for identity in expected:
        r, m = (t[identity] for t in tables)
        require(r.get('checkpoint_sha256') == m.get('checkpoint_sha256') == INITIAL_SHA
                and r.get('encoder_sha256') == m.get('encoder_sha256') == ENCODER_SHA
                and r.get('additional_codec_updates') == m.get('additional_codec_updates') == 0,
                'Old row is not frozen baseline')
        require(m.get('decoder_target_key', 'MISSING') is None and m.get('arm') == 'baseline', 'Old decode mode differs')
        require(r.get('replay') == m.get('saved_packet_replay'), 'Old replay evidence differs')
        require(set(r['replay']) == {'real_view', 'nearby_view'}, 'Old two-view replay absent')
        for metric in r['replay'].values():
            validate_metric_set(metric)
            require(metric['psnr'] >= 60 and metric['mae'] <= 1e-5, 'Old replay failed')
        for c in CATEGORIES:
            validate_metric_set(r[c])
    return tables


def read_lines(path, digest, audit):
    audit.check(path, digest)
    values = [pool.parse(line) for line in Path(path).read_bytes().splitlines()]
    # Protect the second metadata read before accepting its values.
    require(sha(path) == digest, 'JSONL changed while reading')
    return values


def adapt(root, max_seconds=None):
    """Production entry; synthetic tests never call this against real data."""
    import worker as w  # imports CPU definitions; do NOT call load_runtime/setup_gpu.
    root = Path(root).resolve(strict=True)
    require(root.name == RUN and root == Path(__file__).resolve().parent, 'Dedicated adapter root required')
    p, evidence = w.verify_root(root)
    seconds = p['baseline']['max_seconds'] if max_seconds is None else max_seconds
    require(type(seconds) is int and 0 < seconds <= p['baseline']['max_seconds'], 'CPU budget exceeds frozen protocol')
    budget = Budget(seconds); budget.check(); torch.set_num_threads(6)
    require(p['baseline']['mode'] == 'reuse_preparation_with_cpu_metrics'
            and p['baseline']['train_prepare_clips'] == 49
            and set(p['baseline']['legacy_baseline_clips']) == HISTORICAL
            and p['baseline']['inherited_state_replay_frames'] == 354
            and p['baseline']['all_frames_saved_replay'] is False
            and p['baseline']['independent_native_gs_replay'] is False, 'Baseline protocol differs')
    require(p['evaluation']['records_per_model'] == 9027 and not p['evaluation']['target_forcing']
            and p['source']['checkpoint_sha256'] == INITIAL_SHA, 'Evaluation contract changed')
    folder = root/'evaluations'/'baseline'
    require(not folder.exists(), 'Baseline output already exists; never overwrite/retry automatically')
    folder.mkdir(parents=True, exist_ok=False)
    audit = pool.Audit(); rows = []; context = {}; max_differences = {}; verified_cache = set(); verified_state = set()
    try:
        pool_root = Path(p['data']['pool_root']); legacy_root = Path(p['source']['legacy_root'])
        index = pool.load_pool_index(pool_root); budget.check()
        by_id = {(r.clip, r.frame): r for r in index}
        plans = w.make_epochs(index)
        evidence['epoch_plan_sha256'] = [pool.plan_summary(index, plan)['plan_sha256'] for plan in plans]
        complete, _ = audit.check(pool_root/'POOL_COMPLETE.json', pool.COMPLETE_SHA, json_data=True)
        records = pool.table(complete['files'])
        binding, _ = audit.check(legacy_root/'DATA_BINDING.json', BINDING_SHA, json_data=True)
        require(binding['initial_codec_sha256'] == INITIAL_SHA, 'Legacy binding checkpoint changed')
        legacy_records = pool.table(binding['files'])
        # Read actual source checkpoint/config/code bytes; metadata hashes alone do not authorize reuse.
        for record in legacy_records.values():
            budget.check(); audit.check(record['path'], record['sha256'], record['bytes'])
        sources = w.load_bound(legacy_root/'sources.py', pool.LEGACY_SHA, '_baseline_cpu_sources')
        codec_root = legacy_root.parent/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
        require(audit.check(codec_root/'codec_final.pt', INITIAL_SHA) == INITIAL_SHA, 'Source weight changed')
        old_root = legacy_root/'evaluations'/'baseline'
        old_eval, _ = audit.check(old_root/'EVALUATION.json', OLD_EVALUATION_SHA, json_data=True)
        old_rows, old_manifests = historical_rows(old_eval,
            read_lines(old_root/'targets.jsonl', OLD_TARGETS_SHA, audit),
            read_lines(old_root/'state_manifest.jsonl', OLD_MANIFEST_SHA, audit))
        prepared = {}
        for clip in sorted(set(pool.EXPECTED) - HISTORICAL):
            directory, _, _, quality_path = pool.layout(pool_root, clip)
            q = pool.bound_json(quality_path, records, audit)
            require(len(q['frames']) == 177 and [r['frame'] for r in q['frames']] == list(range(177)), 'Prepared coverage differs')
            summary = pool.bound_json(directory/'SUMMARY.json', records, audit)
            require(summary['status'] == 'complete' and summary['codec_frozen'] is True
                    and summary['training_performed'] is False and summary['frames'] == 177
                    and summary['codec_state_sha256_before_and_after'] == MODEL_DIGEST, 'Prepared codec differs')
            inputs = pool.bound_json(directory/'inputs_hashes.json', records, audit)
            require(inputs['schema'] == 'gs_transfer_inputs_v1' and inputs['teacher_hashes_computed_after_export'], 'Input receipt differs')
            static = pool.table(inputs['static_files'])
            for path, record in static.items():
                pool.record(path, records, record)
                if path in legacy_records:
                    require(record['sha256'] == legacy_records[path]['sha256']
                            and record['bytes'] == legacy_records[path]['bytes'], 'Prepared shared static source differs')
            for name in ('codec_final.pt', 'selection.json', 'protocol.json',
                         'code/codec.py', 'code/common.py', 'code/train.py'):
                path = pool.key(codec_root/name)
                require(path in static and path in legacy_records, 'Prepared receipt omitted model/config: '+name)
            prepared[clip] = (directory, q['frames'], pool.record(quality_path, records)['sha256'])
        require(len(prepared) == 49, 'Exactly 49 prepared reconstruction sources required')
        # Exposed tower14 is bound separately and may never enter TRAIN.
        eval_directory = legacy_root.parent/'gaussiangpt_neoverse_tower14_transfer_20261031_v1'
        eval_quality = pool.bound_json(eval_directory/'quality'/f'{EVAL_CLIP}.json', legacy_records, audit)['frames']
        eval_normal = pool.bound_json(eval_directory/'normalization.json', legacy_records, audit)
        eval_source = SimpleNamespace(clip=EVAL_CLIP, role='EVAL-only', normalization=eval_normal,
            frames={r['frame']: dict(path=str(eval_directory/'cache'/EVAL_CLIP/f"{r['frame']:03d}.pt"),
                sha256=r['cache_sha256'], bytes=r['cache_bytes'], pts_seconds=r['pts_seconds'],
                native_gs_sha256=r['native_gs_sha256']) for r in eval_quality})
        require(set(eval_source.frames) == set(range(177)), 'Exposed eval coverage differs')
        with (folder/'targets.jsonl').open('x', encoding='utf-8', buffering=1) as output, \
             (folder/'state_manifest.jsonl').open('x', encoding='utf-8', buffering=1) as manifest:
            for clip in [*sorted(pool.EXPECTED), EVAL_CLIP]:
                for frame in range(177):
                    budget.check(); context = dict(clip=clip, frame=frame)
                    ref = by_id.get((clip, frame))
                    if ref is not None:
                        cache_path = Path(ref.cache_path)
                        cache_record = pool.record(cache_path, records)
                        normal = ref.normalization
                        source = SimpleNamespace(clip=clip, role='TRAIN', normalization=normal,
                            frames={frame: ref.legacy_row()})
                        # This isolated validator's ALL_CLIPS covers the 50 admitted clips.
                        validator = pool.isolated(ref.legacy_source_path, pool.LEGACY_SHA, 'sources')
                    else:
                        source = eval_source; validator = sources; normal = eval_normal
                        cache_path = Path(source.frames[frame]['path'])
                        cache_record = pool.record(cache_path, legacy_records)
                    cache = load_tensor(cache_path, cache_record, audit)
                    validator.validate_frame_metadata(cache, source, frame)
                    for value in cache.values():
                        if isinstance(value, torch.Tensor):
                            require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Invalid CPU cache')
                    verified_cache.add((clip, frame))
                    replay = None
                    if clip in prepared:
                        directory, quality, quality_sha = prepared[clip]; q = quality[frame]
                        require(q['clip'] == clip and q['cache_sha256'] == cache_record['sha256']
                                and q['source_count'] == 112896 and q['input_voxels'] == len(cache['representatives']),
                                'Preparation row/cache identity differs')
                        state_path = directory/'states'/clip/f'{frame:03d}.pt'
                        state_record = pool.record(state_path, records)
                        require(q['state_sha256'] == state_record['sha256'] and q['state_bytes'] == state_record['bytes'],
                                'Preparation state binding differs')
                        state = load_tensor(state_path, state_record, audit)
                        validate_state(state, clip, frame, normal, cache_record['sha256'], prepared=True)
                        require(len(state['features']) == q['latent_positions'], 'Prepared latent count differs')
                        metrics = compute_metrics(state['reconstruction'], cache)
                        for name, delta in compare_recorded(metrics, q).items():
                            max_differences[name] = max(max_differences.get(name, 0.), delta)
                        latent, decoded = q['latent_positions'], q['decoded_gaussians']
                        origin = dict(kind='prepared_float32_reconstruction_cpu_metrics', quality_sha256=quality_sha,
                            serialized_coordinates_features_checked_historically=True,
                            saved_latent_state_replay=False, replay_performed_in_this_run=False)
                    else:
                        old = old_rows[clip, frame]; m = old_manifests[clip, frame]
                        require(m['source_cache_sha256'] == cache_record['sha256'] and m['normalization'] == normal,
                                'Old baseline cache/normalization changed')
                        relative = Path(m['state_path'])
                        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe old state path')
                        state_path = old_root/relative
                        state_record = dict(sha256=m['state_sha256'], bytes=m['state_bytes'])
                        state = load_tensor(state_path, state_record, audit)
                        validate_state(state, clip, frame, normal, cache_record['sha256'], prepared=False)
                        require(len(state['features']) == old['latent_positions']
                                and old['input_voxels'] == len(cache['representatives'])
                                and old['input_gaussians'] == 112896, 'Old baseline counts differ')
                        metrics = {c: old[c] for c in CATEGORIES}; replay = old['replay']
                        latent, decoded = old['latent_positions'], old['decoded_gaussians']
                        origin = dict(kind='exact_historical_multisource_baseline_metrics',
                            historical_targets_sha256=OLD_TARGETS_SHA, saved_latent_state_replay=True,
                            replay_performed_in_this_run=False)
                    require(type(decoded) is int and 0 < decoded <= 2400000, 'Invalid decoded count')
                    verified_state.add((clip, frame))
                    row = dict(**context, split='EVAL-only' if clip == EVAL_CLIP else 'TRAIN',
                        source_group=pool.EXPECTED.get(clip, 'buildingblock_tower_14'), pts=frame/29.97,
                        **metrics, latent_positions=latent, latent_channels=64, input_gaussians=112896,
                        input_voxels=len(cache['representatives']), decoded_gaussians=decoded, replay=replay,
                        checkpoint_sha256=INITIAL_SHA, encoder_sha256=ENCODER_SHA, additional_codec_updates=0,
                        baseline_reused=True, baseline_origin=origin)
                    rows.append(row); output.write(json.dumps(row, allow_nan=False)+'\n')
                    manifest.write(json.dumps(dict(**context, split=row['split'], source_group=row['source_group'],
                        pts=row['pts'], normalization=normal, source_cache_sha256=cache_record['sha256'],
                        state_path=str(state_path), state_path_is_external=True,
                        state_sha256=state_record['sha256'], state_bytes=state_record['bytes'], shape=[latent,64],
                        checkpoint_sha256=INITIAL_SHA, encoder_sha256=ENCODER_SHA, fresh_encode=False,
                        decoder_target_key=None, additional_codec_updates=0, inherited_codec_steps=3000,
                        saved_packet_replay=replay, baseline_origin=origin), allow_nan=False)+'\n')
                    del cache, state, metrics
                print('CPU_BASELINE', clip, len(rows), flush=True)
        require(len(verified_cache) == len(verified_state) == 9027, 'Not all actual cache/state bytes verified')
        aggregate = summaries(rows); audit.finish(); budget.check()
        _, current = w.verify_root(root)
        require(all(evidence[k] == v for k, v in current.items()), 'Run binding changed')
        # No new screenshots or native GS are generated; empty manifest is intentional.
        (folder/'screenshot_manifest.jsonl').open('x', encoding='utf-8').close()
        write(folder/'EVALUATION.json', dict(status='evaluated', arm='baseline', frames=9027,
            train_frames=8850, exposed_eval_only_frames=177, additional_codec_updates=0, inherited_codec_steps=3000,
            checkpoint_sha256=INITIAL_SHA, encoder_sha256=ENCODER_SHA, source=evidence, **aggregate,
            baseline_reused=True, baseline_adapter_sha256=sha(Path(__file__)),
            reconstruction_cpu_metric_frames=8673, exact_historical_metric_frames=354,
            every_packet_saved_replay_passed=False, all_frames_freshly_encoded=False,
            saved_latent_state_replay=False, historical_saved_latent_state_replay_frames=354,
            saved_latent_state_replay_unavailable_frames=8673, replay_performed_in_this_run=False,
            independent_native_gs_replay=False, decoder_eval=True, decoder_target_key=None,
            roi_xyxy=list(ROI), cpu_gpu_absolute_tolerance=TOLERANCE,
            maximum_cpu_vs_historical_metric_differences=max_differences, bitwise_equivalence_claimed=False,
            actual_cache_bytes_sha_verified_frames=9027, actual_state_bytes_sha_verified_frames=9027,
            screenshots_generated=False, targets_sha256=sha(folder/'targets.jsonl'),
            state_manifest_sha256=sha(folder/'state_manifest.jsonl'),
            screenshot_manifest_sha256=sha(folder/'screenshot_manifest.jsonl'), gpu_executed=False,
            training_executed=False, goal_complete=False, completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-budget.start, max_seconds=seconds,
            scope='Historical original codec baseline only; exposed EVAL-only tower14; no Z/AR/transfer claim'))
    except BaseException as error:
        write(folder/'FAILURE.json', dict(status='failed', error=repr(error), context=context,
            frames_complete=len(rows), valid_complete_evaluation=False, gpu_executed=False, goal_complete=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, help='Optional smaller cap; default is frozen protocol baseline.max_seconds')
    args = parser.parse_args()
    adapt(args.root, args.max_seconds)


if __name__ == '__main__':
    main()
