"""CPU-only admission of existing full sequences and the complete fixed TRAIN pool.

No teacher inference, codec execution, rendering, optimization or old-run writes.
Old cached states are not promoted as encodings of a newly trained codec.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import bindings
import prepare_clip as prep

FRAMES, POINTS, HEIGHT, WIDTH = 177, 112896, 252, 448
TOWER17_PINS = {
    '../source_binding.json': 'e255a60f3ea473b34708569914f1e72aaf00d03075f813f55bd5c1cf34a5ae22',
    'prepare_done.json': 'a3119d29e7476326ff547a54a3f26fdc881e3880d2c9372591c462a4dea57a58',
    'normalization.json': '8bb690a5f7c16d822a4b99908692b1741cce0c48bb1f6f5ab7bf3afaab630ee3',
    'mappings/buildingblock_tower_17_176.json': '3fc2db2a3cb8642dbae28cb247bdd030b0d5038502ec0c2d89e48d01bb0ac983',
    'prepare/buildingblock_tower_17_176.json': '7b63c8c50eb9a3e7f3e2f4b55576894f272c6c0f72de3502b6b7636c56dd702a',
}
TOWER10_SUMMARY = '85ddbe1c465364c2bff82926e0682df8a19cca964435c0c51b9793e31fcfce53'
require = bindings.require
read = bindings.read_json


def fixed(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, f'{label} mismatch')


def check_pts(times, item):
    require(isinstance(times, list) and len(times) == FRAMES, 'Incomplete PTS sequence')
    for i, (actual, measured) in enumerate(zip(times, item['probe']['pts_seconds'])):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and abs(actual - i / 29.97) <= 1e-9 and abs(actual - measured) <= 1e-6,
                f'PTS mismatch at frame {i}')


class Audit:
    """Hash actual files once per stage; retain evidence for final revalidation."""
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None):
        path = Path(path)
        # Inputs may be symlinks (e.g. a warm HuggingFace cache); the resolved
        # regular file is hashed. This function never writes to input paths.
        require(path.is_file(), f'Missing input: {path}')
        key = str(path.resolve(strict=True))
        if key not in self.records:
            self.records[key] = prep.fingerprint(path)
        current = self.records[key]
        if expected is not None:
            require(current['sha256'] == expected['sha256'], f'File hash changed: {path}')
            if 'bytes' in expected:
                require(current['bytes'] == expected['bytes'], f'File size changed: {path}')
        return current

    def recheck(self):
        for old in self.records.values():
            current = prep.fingerprint(old['path'])
            require(current == old, f'Input changed during admission: {old["path"]}')


def validate_mapping(mapping, item):
    fixed(mapping.get('source_frame_indices'), list(range(FRAMES)), 'source frame coverage')
    fixed(mapping.get('model_timestamps'), list(range(FRAMES)), 'model timestamp coverage')
    fixed(mapping.get('source_sha256'), item['video_sha256'], 'mapping video SHA')
    check_pts(mapping.get('source_pts_seconds'), item)
    if 'adjacent_pairs' in mapping:
        fixed(mapping['adjacent_pairs'], [[i, i + 1] for i in range(FRAMES - 1)], 'adjacent pairs')


def validate_native(arrays, frame):
    """Inspect every saved native GS, not only provenance counts."""
    import numpy as np
    expected = {'means': (POINTS, 3), 'scales': (POINTS, 3), 'rotations': (POINTS, 4),
                'harmonics': (POINTS, 1, 3), 'opacities': (POINTS,)}
    for key, shape in expected.items():
        require(key in arrays and arrays[key].shape == shape
                and np.issubdtype(arrays[key].dtype, np.floating)
                and np.isfinite(arrays[key]).all(), f'Native {key} shape/value mismatch at {frame}')
    require((arrays['scales'] > 0).all() and (arrays['opacities'] >= 0).all()
            and (arrays['opacities'] <= 1).all(), f'Invalid native scale/opacity at {frame}')
    meta = json.loads(str(arrays['metadata_json']))
    require(meta.get('timestamp') == frame and meta.get('forward_vel') is None
            and meta.get('backward_vel') is None, f'Native timestamp/motion mismatch at {frame}')


def validate_cache(payload, item, frame, normal, rgb, camera, intrinsics):
    import numpy as np
    import torch
    fixed(payload.get('clip'), item['clip_id'], 'cache clip')
    fixed(payload.get('frame'), frame, 'cache frame')
    fixed(payload.get('n_input'), POINTS, 'cache all-source count')
    fixed(payload.get('normalization'), normal, 'cache normalization')
    metrics = payload.get('metrics', {})
    require(metrics.get('frame') == frame and metrics.get('source_count') == POINTS
            and abs(metrics.get('pts_seconds', -1) - frame / 29.97) <= 1e-9, 'Cache metrics mismatch')
    n = len(payload['coordinates'])
    shapes = {'coordinates': (n, 3), 'representatives': (n, 14), 'source_indices': (n,),
              'counts': (n,), 'set_statistics': (n, 29), 'cameras': (2, 4, 4),
              'intrinsics': (4,), 'targets': (2, HEIGHT, WIDTH, 3), 'real': (HEIGHT, WIDTH, 3)}
    require(0 < n <= 80000, 'Cache input voxel guard changed')
    for key, shape in shapes.items():
        value = payload.get(key)
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                f'Cache {key} shape/value mismatch at {frame}')
    require(payload['coordinates'].dtype in (torch.int32, torch.int64)
            and payload['counts'].dtype == torch.int64 and payload['source_indices'].dtype == torch.int64,
            'Cache integer schema changed')
    require(int(payload['counts'].sum()) == POINTS and bool((payload['counts'] > 0).all()),
            'Cache dropped or duplicated source point count')
    source = payload['source_indices']
    require(bool(((source >= 0) & (source < POINTS)).all()) and len(source.unique()) == n,
            'Cache source indices invalid/duplicated')
    require(torch.equal((payload['real'] * 255).round().to(torch.uint8), torch.from_numpy(np.array(rgb)))
            and torch.allclose(payload['real'], torch.from_numpy(np.array(rgb)).float() / 255, atol=1e-7, rtol=0),
            'Cache RGB does not match its source frame')
    expected_camera = torch.from_numpy(np.array(camera)).float()
    expected_camera[:3, 3] *= normal['scale']
    require(torch.allclose(payload['cameras'][0], expected_camera, atol=1e-7, rtol=0), 'Cache primary camera changed')
    nearby = expected_camera.clone(); nearby[:3, 3] += expected_camera[:3, 0] * .02
    require(torch.allclose(payload['cameras'][1], nearby, atol=1e-7, rtol=0), 'Cache pseudo-view camera changed')
    expected_intr = torch.tensor([intrinsics[0, 0] / WIDTH, intrinsics[1, 1] / HEIGHT,
                                  intrinsics[0, 2] / WIDTH, intrinsics[1, 2] / HEIGHT], dtype=torch.float32)
    require(torch.allclose(payload['intrinsics'], expected_intr, atol=1e-7, rtol=0), 'Cache intrinsics changed')
    require(payload['near'] == .01 * normal['scale'] and payload['far'] == 1e10 * normal['scale'],
            'Cache normalized clipping planes changed')


def reuse_metadata(root, item, audit):
    """Authenticate historical cache hashes before loading any trusted Torch payload."""
    clip = item['clip_id']; paths = bindings._reuse_paths(clip)
    old = Path(paths['cache']).parent.parent
    if clip == bindings.AUTHORIZED_DEV:
        for relative, digest in TOWER17_PINS.items():
            audit.add(old / relative, {'sha256': digest})
        source = read(old / '../source_binding.json')
        teacher_records = {r['path']: r for r in source['files']}
        require(len(teacher_records) == len(source['files']), 'Duplicate original source binding')
        done = read(old / 'prepare_done.json')
        require(done.get('status') == 'complete' and done.get('states') == FRAMES
                and done.get('all_source_points_per_state') == POINTS and done.get('grid') == .008,
                'Original tower17 preparation incomplete')
        prepared = read(old / 'prepare' / f'{clip}.json'); rows = prepared['rows']
    else:
        audit.add(old / 'SUMMARY.json', {'sha256': TOWER10_SUMMARY})
        audit.add(old / 'input_binding.json', {'sha256': bindings.TEMPLATE_SHA256})
        # Same bound helper verifier checks all saved states, parity and hashes.
        checked = prep.verify_outputs(old, item, bindings.TEMPLATE_SHA256)
        for record in checked['output_files']:
            audit.add(record['path'], record)
        source = read(old / 'inputs_hashes.json')
        teacher_records = {r['path']: r for r in source['teacher_and_source_files']}
        require(len(teacher_records) == len(source['teacher_and_source_files']), 'Duplicate teacher hash')
        rows = read(old / 'quality' / f'{clip}.json')['frames']
    normal = read(old / 'normalization.json')
    require(normal.get('source_frame') == 0 and normal.get('fixed_for_all_frames') is True
            and normal.get('center') == [0., 0., 0.] and normal.get('scale') == .5,
            'Original first-frame-only normalization changed')
    validate_mapping(read(old / 'mappings' / f'{clip}.json'), item)
    require(len(rows) == FRAMES and [r.get('frame') for r in rows] == list(range(FRAMES)),
            'Original preparation row coverage changed')
    for row in rows:
        require(row.get('clip') == clip and row.get('source_count') == POINTS
                and abs(row.get('pts_seconds', -1) - row['frame'] / 29.97) <= 1e-9,
                'Original preparation row identity/count/PTS mismatch')
        for key in ['native_render_parity', 'similarity_parity']:
            metric = row.get(key, {})
            require(math.isfinite(metric.get('psnr', float('nan'))) and metric['psnr'] >= 60
                    and 0 <= metric.get('mae', -1) <= 1e-5, 'Original parity evidence failed')
    return paths, normal, rows, teacher_records


def admit_reuse_clip(root, item, audit):
    import numpy as np
    import torch
    clip = item['clip_id']; paths, normal, rows, expected_files = reuse_metadata(root, item, audit)
    teacher = Path(paths['teacher']); cache = Path(paths['cache'])
    require(set((teacher / 'gaussians').glob('*.npz')) ==
            {teacher / 'gaussians' / f'native_group_{i:03d}.npz' for i in range(FRAMES)},
            'Reuse native GS coverage differs from 000..176')
    require(set(cache.glob('*.pt')) == {cache / f'{i:03d}.pt' for i in range(FRAMES)},
            'Reuse cache coverage differs from 000..176')
    for name in ['provenance.json', 'cameras.npz', 'input_rgb.npy', 'rendered_rgb.npy']:
        path = teacher / name
        require(str(path) in expected_files, f'Teacher lacks historical hash: {path}')
        audit.add(path, expected_files[str(path)])
    _, expected, _ = bindings._declared_inputs(root, item, bindings.load_pool(root))
    helper = prep.bind_helper(prep.load_helper(root), item)
    times = helper.validate_provenance(read(teacher / 'provenance.json'), expected, teacher.resolve())
    check_pts(times, item)
    rgb = np.load(teacher / 'input_rgb.npy', mmap_mode='r', allow_pickle=False)
    rendered = np.load(teacher / 'rendered_rgb.npy', mmap_mode='r', allow_pickle=False)
    require(rgb.shape == rendered.shape == (FRAMES, HEIGHT, WIDTH, 3)
            and rgb.dtype == np.uint8, 'Reuse native RGB resolution/dtype changed')
    with np.load(teacher / 'cameras.npz', allow_pickle=False) as cameras:
        require(np.array_equal(cameras['timestamps'], np.arange(FRAMES))
                and cameras['camera_to_world'].shape == (FRAMES, 4, 4)
                and cameras['intrinsics'].shape == (FRAMES, 3, 3), 'Reuse camera sequence changed')
        for frame, row in enumerate(rows):
            path = teacher / 'gaussians' / f'native_group_{frame:03d}.npz'
            require(str(path) in expected_files, 'Native GS missing original source binding')
            record = audit.add(path, expected_files[str(path)])
            require(row['native_gs_sha256'] == record['sha256'], 'Preparation consumed different native GS')
            with np.load(path, allow_pickle=False) as arrays:
                validate_native(arrays, frame)
            cache_path = cache / f'{frame:03d}.pt'
            audit.add(cache_path, {'sha256': row['cache_sha256'], 'bytes': row['cache_bytes']})
            # Payload hash matched its immutable historical preparation before load.
            payload = torch.load(cache_path, map_location='cpu', weights_only=False)
            validate_cache(payload, item, frame, normal, rgb[frame],
                           cameras['camera_to_world'][frame], cameras['intrinsics'][frame])
            del payload
    return {'clip_id': clip, 'status': 'complete', 'source_group': item['source_group'],
            'original_role': item['original_role'], 'current_role': item['current_role'],
            'role_basis': item['role_basis'], 'source_video_sha256': item['video_sha256'],
            'paths': paths, 'actual_frames': FRAMES, 'actual_adjacent_pairs': 176,
            'actual_native_gs_files': FRAMES, 'actual_cache_files': FRAMES,
            'actual_source_points_per_frame': POINTS, 'normalization': normal,
            'origin': 'admitted_existing_full177_sequence', 'native_arrays_and_cache_tensors_checked': True,
            'rendering_performed': False, 'historical_render_parity_revalidated_by_hash': True,
            'state_encodings_reused_for_new_codec': False}


def base_receipt(root):
    return {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'code_manifest_sha256': bindings.sha256(root / 'CODE_MANIFEST.json'),
            'input_pool_sha256': bindings.sha256(root / 'INPUT_POOL.json'),
            'training_executed': False, 'training_performed': False, 'goal_complete': False,
            'official_independent_holdout': False, 'cross_clip_physical_frame_unique_count': None}


def static_inputs(root, pool, audit):
    audit.add(root / 'CODE_MANIFEST.json'); audit.add(root / 'INPUT_POOL.json')
    for relative, digest in bindings.verify_manifest(root).items():
        audit.add(root / relative, {'sha256': digest})
    # Union of all 50 current videos plus unchanged bound codec/teacher dependencies.
    row = pool['clips'][0]
    files, _, _ = bindings._declared_inputs(root, row, pool)
    for record in files:
        audit.add(record['path'], record)
    for item in pool['clips']:
        audit.add(item['video_path'], {'sha256': item['video_sha256'], 'bytes': item['video_bytes']})


def reuse(root):
    pool = bindings.load_pool(root); audit = Audit(); static_inputs(root, pool, audit)
    # Preflight both reference paths first; never continue an interrupted attempt.
    for clip in sorted(bindings.REUSE):
        bindings._assert_fresh(bindings.clip_root(root, clip))
    for clip in sorted(bindings.REUSE):
        bindings.build_reuse_reference(root, clip)
        audit.add(bindings.clip_root(root, clip) / 'reuse_reference.json')
    clips = [admit_reuse_clip(root, row, audit) for row in pool['clips'] if row['clip_id'] in bindings.REUSE]
    audit.recheck(); bindings.verify_manifest(root)
    result = dict(base_receipt(root), schema='gs_fulltrain_reuse_admission_v1', clips=clips,
                  actual_clip_internal_frames_total=354, files=list(audit.records.values()),
                  scope='CPU integrity admission of two existing full177 TRAIN sequences; no new rendering')
    bindings._write_new(root / 'REUSE_ADMISSION.json', result)
    return result


def validate_reuse_receipt(receipt, root, pool):
    for key in ['status', 'code_manifest_sha256', 'input_pool_sha256', 'training_executed',
                'training_performed', 'goal_complete', 'official_independent_holdout',
                'cross_clip_physical_frame_unique_count']:
        fixed(receipt.get(key), base_receipt(root)[key], f'reuse receipt {key}')
    fixed(receipt.get('schema'), 'gs_fulltrain_reuse_admission_v1', 'reuse schema')
    clips = receipt.get('clips', [])
    require(len(clips) == 2 and {r.get('clip_id') for r in clips} == bindings.REUSE,
            'Reuse clip coverage/duplicates changed')
    for record in clips:
        item = next(x for x in pool['clips'] if x['clip_id'] == record['clip_id'])
        for key in ['source_group', 'original_role', 'current_role', 'role_basis']:
            fixed(record.get(key), item[key], f'reuse {key}')
        fixed(record.get('source_video_sha256'), item['video_sha256'], 'reuse video hash')
        fixed(record.get('paths'), bindings._reuse_paths(item['clip_id']), 'reuse paths')
        for key, value in {'status': 'complete', 'actual_frames': 177, 'actual_adjacent_pairs': 176,
                           'actual_native_gs_files': 177, 'actual_cache_files': 177,
                           'actual_source_points_per_frame': POINTS,
                           'native_arrays_and_cache_tensors_checked': True,
                           'state_encodings_reused_for_new_codec': False}.items():
            fixed(record.get(key), value, f'reuse {key}')
    paths = [r.get('path') for r in receipt.get('files', [])]
    require(paths and len(paths) == len(set(paths)), 'Reuse file hashes missing/duplicated')
    required = {str((root / 'CODE_MANIFEST.json').resolve()), str((root / 'INPUT_POOL.json').resolve())}
    for item in pool['clips']:
        required.add(str(Path(item['video_path']).resolve()))
        if item['clip_id'] not in bindings.REUSE:
            continue
        p = bindings._reuse_paths(item['clip_id']); teacher, cache = Path(p['teacher']), Path(p['cache'])
        required.update(str((teacher / n).resolve()) for n in
                        ['provenance.json', 'cameras.npz', 'input_rgb.npy', 'rendered_rgb.npy'])
        required.update(str((teacher / 'gaussians' / f'native_group_{i:03d}.npz').resolve()) for i in range(FRAMES))
        required.update(str((cache / f'{i:03d}.pt').resolve()) for i in range(FRAMES))
        required.add(str((bindings.clip_root(root, item['clip_id']) / 'reuse_reference.json').resolve()))
    require(required <= set(paths), 'Reuse receipt omitted mandatory source/teacher/cache hashes')


def admit_fresh(root, item, audit):
    clip = item['clip_id']; directory = bindings.clip_root(root, clip)
    require(not any((directory / n).exists() for n in ['FAILED.json', 'ADAPTER_FAILED.json']),
            f'Failed preparation cannot be admitted: {clip}')
    receipt_path = directory / 'PREPARATION_RECEIPT.json'
    receipt_record = audit.add(receipt_path); receipt = read(receipt_path)
    required = {'schema': 'gs_fulltrain_clip_preparation_receipt_v1', 'status': 'complete', 'clip': clip,
                'source_group': item['source_group'], 'original_role': item['original_role'],
                'current_role': 'TRAIN', 'data_origin': 'new_full177_export',
                'training_performed': False, 'goal_complete': False, 'helper_sha256': prep.HELPER_SHA256,
                'code_manifest_sha256': bindings.sha256(root / 'CODE_MANIFEST.json'),
                'input_pool_sha256': bindings.sha256(root / 'INPUT_POOL.json')}
    for key, value in required.items():
        fixed(receipt.get(key), value, f'{clip} receipt {key}')
    binding_path = directory / 'input_binding.json'
    binding = audit.add(binding_path, {'sha256': receipt['input_binding_sha256']})
    bindings.validate_binding(root, clip, read(binding_path))
    for name, key in [('SUMMARY.json', 'summary_sha256'), ('HELPER_SUMMARY.json', 'helper_summary_sha256')]:
        audit.add(directory / name, {'sha256': receipt[key]})
    # Verify all teacher/cache/state hashes, shape/count/PTS metadata and prior replay evidence.
    actual = prep.verify_outputs(directory, item, binding['sha256'])
    for key, value in actual.items():
        fixed(receipt.get(key), value, f'{clip} actual preparation {key}')
    for record in actual['output_files']:
        audit.add(record['path'], record)
    inputs = read(directory / 'inputs_hashes.json')
    for record in inputs['static_files'] + inputs['teacher_and_source_files']:
        audit.add(record['path'], record)
    return {'clip_id': clip, 'status': 'complete', 'source_group': item['source_group'],
            'original_role': item['original_role'], 'current_role': item['current_role'],
            'source_video_sha256': item['video_sha256'], 'origin': 'new_full177_export',
            'actual_frames': 177, 'actual_adjacent_pairs': 176, 'actual_source_points_per_frame': POINTS,
            'actual_cache_files': 177, 'actual_native_gs_files': 177, 'actual_state_files': 177,
            'paths': {'teacher': str(directory / 'teacher' / clip), 'cache': str(directory / 'cache' / clip)},
            'preparation_receipt': receipt_record,
            'state_encodings_reused_for_new_codec': False}


def final(root):
    pool = bindings.load_pool(root); audit = Audit(); static_inputs(root, pool, audit)
    reuse_path = root / 'REUSE_ADMISSION.json'; reuse_record = audit.add(reuse_path); accepted = read(reuse_path)
    validate_reuse_receipt(accepted, root, pool)
    for record in accepted['files']:
        audit.add(record['path'], record)
    by_clip = {row['clip_id']: row for row in accepted['clips']}
    for item in pool['clips']:
        if item['clip_id'] not in bindings.REUSE:
            by_clip[item['clip_id']] = admit_fresh(root, item, audit)
    require(set(by_clip) == set(bindings.EXPECTED_CLIPS) and len(by_clip) == 50, 'Final pool coverage differs')
    audit.recheck(); bindings.verify_manifest(root)
    result = dict(base_receipt(root), schema='gs_fulltrain_pool_complete_v1',
                  clips=[by_clip[row['clip_id']] for row in pool['clips']],
                  all_clips_complete=True, planned_clip_internal_frames_total=8850,
                  actual_clip_internal_frames_total=sum(v['actual_frames'] for v in by_clip.values()),
                  actual_adjacent_pairs_total=8800, fresh_clips=48, reused_clips=2, source_groups=19,
                  native_width=448, native_height=252, gs_per_frame=POINTS, frame_stride=1,
                  reuse_admission= reuse_record, files=list(audit.records.values()),
                  scope='All 50 predeclared TRAIN clips admitted; 8850 clip-local frames, global uniqueness unknown',
                  state_encodings_reused_for_new_codec=False)
    bindings._write_new(root / 'POOL_COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['reuse', 'final'], required=True)
    args = parser.parse_args(); root = args.root.resolve(strict=True)
    require(root.name == bindings.RUN and Path(__file__).resolve().parent == root,
            'Admission must run from its dedicated new pool root')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU admission requires CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(6)
    require(not torch.cuda.is_initialized(), 'CPU admission must not initialize CUDA')
    output = root / ('REUSE_ADMISSION.json' if args.mode == 'reuse' else 'POOL_COMPLETE.json')
    require(not output.exists() and not output.is_symlink(), 'Refusing overwrite or automatic retry')
    result = reuse(root) if args.mode == 'reuse' else final(root)
    print(json.dumps({'status': result['status'], 'mode': args.mode, 'clips': len(result['clips']),
                      'output': str(output), 'training_executed': False, 'goal_complete': False}))


if __name__ == '__main__':
    main()
