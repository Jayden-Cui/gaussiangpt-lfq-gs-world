"""Verified full-frame inputs for the two-TRAIN, one-EVAL codec comparison.

This module never reads historical latent states. Only teacher/voxel caches are
reused; every evaluated state must be encoded by the checkpoint under test.
Import is CPU-only and performs no filesystem writes or old-module imports.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
RUN = 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
BASE = Path('/data/CJH/AGS-V6Direct/outputs')
CODEC_ROOT = BASE / 'gaussiangpt_neoverse_state_motion_20261030_v1/state'
TOWER10_ROOT = BASE / 'gaussiangpt_neoverse_tower10_replay_20261031_v1'
TOWER14_ROOT = BASE / 'gaussiangpt_neoverse_tower14_transfer_20261031_v1'
V4_ROOT = BASE / 'gaussiangpt_neoverse_vw1_h9_lfq_condentropy_20261030_v4'
UPSTREAM = BASE / 'gaussiangpt_state_vae_20261028_v1'
VENDOR = UPSTREAM / 'vendor/GaussianGPT-00658a4ab3f48daed3bdf4ba50805bb2b7dc5e14'
TRAIN_CLIPS = ('buildingblock_tower_17_176', 'buildingblock_tower_10_176')
EVAL_CLIP = 'buildingblock_tower_14_0'
ALL_CLIPS = (*TRAIN_CLIPS, EVAL_CLIP)
FRAMES, N, H, W, GRID = 177, 112896, 252, 448, 0.008
ROLES = {TRAIN_CLIPS[0]: 'TRAIN', TRAIN_CLIPS[1]: 'TRAIN', EVAL_CLIP: 'EVAL-only'}
HISTORICAL_ROLES = {TRAIN_CLIPS[0]: 'DEV; user-authorized TRAIN for this run',
                    TRAIN_CLIPS[1]: 'TRAIN', EVAL_CLIP: 'TEST; previously exposed'}
CODEC_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
TOWER10_MANIFEST_SHA = 'f9d8db6b8f6c77636dc78cc11c7cbdd5da91d710e5a0675620fb4dd6facd0418'
TOWER10_BINDING_SHA = 'af208495ed3df26487a59913cdd306f45038ee2f2bb47bee552f5228702c6b79'
TOWER14_MANIFEST_SHA = '8c9381593b2a7bce009e38d73c07f9de1b0ab1a411cfa7feb9ce6729d268ef57'
TOWER14_BINDING_SHA = 'a422de97bdfeb7b9dcae47a00478f43ddb63185b9b352b85e221cb914c7d1e8d'
TOWER14_SUMMARY_SHA = '70626a0fc9ba6ed811a52352b3637dfbb8a1bdc5c2534b32b40a6dbd262b4e88'
TOWER14_QUALITY_SHA = '4255b687a81f5b45e29e8607dca7c93827a9878c274fb9bb42778ae15d8d18d6'
HELPER_SHA = '3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85'
V4_MANIFEST_SHA = '5014977c343158e23e082dde67b0033202f64b5b2effb0f12966578666c801a7'
VIDEO_SHA = {TRAIN_CLIPS[1]: 'a8d76842e1d7de31cb8bedfe437264a28b3203c9f3bfbaebf2a6f0864a051dab',
             EVAL_CLIP: '456d178c177d6790196f2e27f3bf5e1f5eebbb9fdaecc716794be90fbbab57f8',
             TRAIN_CLIPS[0]: '966d0fca220959e95608b383e808ecda897b4140c54fdeab93fb1a791cd24a90'}
# Additional immutable metadata pins are populated only from read-back evidence.
STATIC_METADATA = {
    CODEC_ROOT / 'prepare/buildingblock_tower_17_176.json': '7b63c8c50eb9a3e7f3e2f4b55576894f272c6c0f72de3502b6b7636c56dd702a',
    CODEC_ROOT / 'mappings/buildingblock_tower_17_176.json': '3fc2db2a3cb8642dbae28cb247bdd030b0d5038502ec0c2d89e48d01bb0ac983',
    CODEC_ROOT / 'normalization.json': '8bb690a5f7c16d822a4b99908692b1741cce0c48bb1f6f5ab7bf3afaab630ee3',
    CODEC_ROOT / 'prepare_done.json': 'a3119d29e7476326ff547a54a3f26fdc881e3880d2c9372591c462a4dea57a58',
    CODEC_ROOT / 'train_done.json': 'e86f7541360455a1930bc0d5a217e0038453d51fc2355f9e8fe789e48a5da0d2',
    TOWER14_ROOT / 'mappings/buildingblock_tower_14_0.json': '3e6bda385aabffdf88e3fc58114fef09e3c7e29cf8acf57006981442ba6120f8',
    TOWER14_ROOT / 'normalization.json': 'f44b1e88518aace5d183f544de2814fd958cd6ddfd5e0ad359807f9b26ec2e3c',
}
VGG_PATH = CODEC_ROOT / 'torch_cache/checkpoints/vgg19-dcbb9e9d.pth'
VGG_SHA = 'dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def fingerprint(path, expected=None, size=None, allow_empty=False):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f'Expected file: {path}')
    before = path.stat()
    require(before.st_size > 0 or allow_empty, f'Empty source file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'Source changed during hashing: {path}')
    actual = digest.hexdigest()
    require(expected is None or (valid_sha(expected) and actual == expected), f'SHA256 mismatch: {path}')
    require(size is None or after.st_size == size, f'Size mismatch: {path}')
    return {'path': str(path), 'bytes': after.st_size, 'sha256': actual}


def bind(records, path, expected=None, size=None, allow_empty=False):
    row = fingerprint(path, expected, size, allow_empty)
    old = records.get(Path(row['path']))
    require(old is None or old == row, f'Conflicting binding: {path}')
    records[Path(row['path'])] = row
    return row


def balanced_plan(seed, steps=300):
    """Two equal-weight samples per update; first 177 updates cover all 354."""
    require(type(seed) is int and type(steps) is int and steps >= FRAMES,
            'Sampler requires an integer seed and at least 177 complete updates')
    rng = random.Random(seed)
    result = []
    while len(result) < steps:
        orders = [list(range(FRAMES)), list(range(FRAMES))]
        for order in orders:
            rng.shuffle(order)
        result.extend(tuple((clip, orders[i][t]) for i, clip in enumerate(TRAIN_CLIPS))
                      for t in range(min(FRAMES, steps - len(result))))
    return result


def validate_protocol(cfg):
    """Prevent a changed protocol from silently changing the data/model contract."""
    roots = dict(zip(ALL_CLIPS, (p.as_posix() for p in (CODEC_ROOT, TOWER10_ROOT, TOWER14_ROOT))))
    fixed = {
        'source': {'codec_root': CODEC_ROOT.as_posix(), 'checkpoint_path': (CODEC_ROOT / 'codec_final.pt').as_posix(),
                   'checkpoint_sha256': CODEC_SHA, 'original_completed_steps': 3000,
                   'original_protocol_sha256': '24add801f064c181658b43da98e1b9a42779c2b876105231ded2ef044b1205da',
                   'train_done_sha256': STATIC_METADATA[CODEC_ROOT / 'train_done.json'],
                   'vgg_weights': {'path': VGG_PATH.as_posix(), 'bytes': 574673361, 'sha256': VGG_SHA}},
        'data': {'train_clips': list(TRAIN_CLIPS), 'eval_only_clips': [EVAL_CLIP],
                 'evaluation_clips': list(ALL_CLIPS), 'roots': roots, 'frames_per_clip': FRAMES,
                 'teacher_points_per_frame': N, 'height': H, 'width': W, 'frame_stride': 1,
                 'fps': 29.97, 'source_group_disjoint': True, 'official_independent_test_split': False,
                 'all_raw_cache_rows_retained': True},
        'dependency': {'root': TOWER10_ROOT.as_posix(), 'pid': 437557, 'start_ticks': '38609998',
                       'manifest_sha256': TOWER10_MANIFEST_SHA, 'input_binding_sha256': TOWER10_BINDING_SHA,
                       'require_done_complete': True},
        'model': {'latent_width': 64, 'grid': GRID, 'stride': 4, 'gaussians_per_voxel': 4,
                  'state_quantization': False, 'max_input_voxels': 80000,
                  'max_decoder_candidates': 600000, 'max_decoded_gaussians': 2400000},
    }
    require(cfg.get('schema') == 'gs_multisource_codec_v1' and cfg.get('run') == RUN, 'Unexpected protocol schema/run')
    for section, fields in fixed.items():
        for key, value in fields.items():
            actual = cfg.get(section, {}).get(key)
            require(type(actual) is type(value) and actual == value, f'Fixed protocol changed: {section}.{key}')


def verify_tower10(root, records):
    """No inferred completion: exact preparation manifest + DONE + summary hash."""
    root = Path(root)
    for path in (root / 'FAILED.json', root / 'queue/FAILED.json'):
        require(not path.exists(), f'Tower10 preparation has a failure receipt: {path}')
    done_path = root / 'queue/DONE.json'
    require(done_path.is_file(), f'Tower10 preparation has no DONE receipt: {done_path}')
    bind(records, root / 'CODE_MANIFEST.json', TOWER10_MANIFEST_SHA)
    bind(records, root / 'input_binding.json', TOWER10_BINDING_SHA)
    bind(records, done_path)
    done = read(done_path)
    admission = done.get('admission', {})
    require(done.get('status') == 'complete' and done.get('preparation_contract_passed') is True
            and done.get('training_performed') is False and done.get('source_role') == 'TRAIN',
            'Tower10 DONE does not confirm complete frozen preparation')
    require(admission.get('code_manifest_sha256') == TOWER10_MANIFEST_SHA
            and admission.get('input_binding_sha256') == TOWER10_BINDING_SHA,
            'Tower10 DONE belongs to another preparation')
    require(valid_sha(done.get('summary_sha256')), 'Tower10 DONE lacks an exact summary SHA256')
    bind(records, root / 'SUMMARY.json', done['summary_sha256'])
    summary = read(root / 'SUMMARY.json')
    validate_summary(summary, TRAIN_CLIPS[1], TOWER10_BINDING_SHA)
    require(summary.get('source_role') == 'TRAIN'
            and summary.get('source_group') == 'buildingblock_tower_10'
            and summary.get('official_independent_test_split') is False,
            'Tower10 source role differs from original TRAIN')
    require(done.get('codec_state_sha256_before_and_after') == summary.get('codec_state_sha256_before_and_after')
            and valid_sha(summary.get('codec_state_sha256_before_and_after')),
            'Tower10 DONE/summary frozen codec digest differs')
    outcomes = done.get('outcomes', [])
    require([row.get('name') for row in outcomes] == ['teacher_export', 'frozen_codec_preparation']
            and all(row.get('exit_code') == 0 and row.get('timed_out') is False for row in outcomes),
            'Tower10 child processes did not both complete successfully')
    return summary


def validate_summary(summary, clip, binding_sha):
    fixed = dict(status='complete', preparation_contract_passed=True, clip=clip, frames=FRAMES,
                 adjacent_pairs=FRAMES - 1, source_points_per_frame=N, all_input_points_aggregated=True,
                 codec_frozen=True, training_performed=False, input_binding_sha256=binding_sha,
                 state_quantization='Identity; continuous float32 Nx64')
    for key, value in fixed.items():
        require(type(summary.get(key)) is type(value) and summary[key] == value,
                f'{clip} summary mismatch: {key}')


def validate_normalization(normal):
    require(normal.get('center') == [0.0, 0.0, 0.0] and normal.get('source_frame') == 0
            and normal.get('fixed_for_all_frames') is True
            and normal.get('calibration_only_alpha_threshold') == 0.01,
            'Normalization must use the whole clip with frame-zero, scale-only calibration')
    scale = normal.get('scale')
    require(isinstance(scale, (float, int)) and math.isfinite(scale) and scale > 0
            and math.frexp(scale)[0] == 0.5, 'Normalization scale must be a positive power of two')
    require(normal.get('uses_future_frames', False) is False, 'Future-frame normalization is forbidden')


def validate_index(clip, rows, mapping, normal):
    require(clip in ALL_CLIPS, 'Unrecognized clip')
    validate_normalization(normal)
    require(isinstance(rows, list) and len(rows) == FRAMES, 'Exactly 177 full-frame cache records required')
    require(mapping.get('source_frame_indices') == list(range(FRAMES))
            and mapping.get('model_timestamps') == list(range(FRAMES)), 'Missing/reordered source frames')
    require(mapping.get('clip', clip) == clip, 'Mapping belongs to another clip')
    times = mapping.get('source_pts_seconds')
    require(isinstance(times, list) and len(times) == FRAMES
            and all(isinstance(t, (float, int)) and math.isfinite(t) for t in times)
            and all(b > a for a, b in zip(times, times[1:])), 'Invalid original source PTS')
    require(all(abs(t - i / 29.97) <= 1e-9 for i, t in enumerate(times)),
            'PTS must preserve every original consecutive frame at 29.97 FPS')
    require(valid_sha(mapping.get('source_sha256')), 'Missing source video identity')
    if clip in VIDEO_SHA:
        require(mapping['source_sha256'] == VIDEO_SHA[clip], 'Mapping names another source video')
    for frame, row in enumerate(rows):
        require(row.get('clip') == clip and row.get('frame') == frame
                and row.get('source_count') == N, 'Source/frame/point coverage differs')
        require(abs(float(row.get('pts_seconds', float('nan'))) - times[frame]) <= 1e-9,
                'Quality row PTS differs from clip mapping')
        require(valid_sha(row.get('cache_sha256')) and valid_sha(row.get('native_gs_sha256'))
                and type(row.get('cache_bytes')) is int and row['cache_bytes'] > 0,
                'Empty or unhashed cache record')
    return times


def build_source(clip, directory, rows, mapping, normal, records):
    times = validate_index(clip, rows, mapping, normal)
    expected = {directory / 'cache' / clip / f'{t:03d}.pt' for t in range(FRAMES)}
    require(set((directory / 'cache' / clip).glob('*.pt')) == expected,
            f'{clip} cache coverage must be exactly 000..176')
    frames = []
    for t, row in enumerate(rows):
        file = bind(records, directory / 'cache' / clip / f'{t:03d}.pt', row['cache_sha256'], row['cache_bytes'])
        frames.append(dict(file, frame=t, pts_seconds=times[t], native_gs_sha256=row['native_gs_sha256']))
    return SimpleNamespace(clip=clip, role=ROLES[clip], historical_role=HISTORICAL_ROLES[clip],
                           root=directory, frames=tuple(frames), normalization=normal,
                           mapping=mapping, source_video_sha256=mapping['source_sha256'])


def _static_inputs(records):
    """Only executed old codec/vendor sources and loaded weights are rebound."""
    bind(records, TOWER14_ROOT / 'input_binding.json', TOWER14_BINDING_SHA)
    binding = read(TOWER14_ROOT / 'input_binding.json')
    require(binding.get('schema') == 'gs_frozen_codec_transfer_binding_v1', 'Unexpected static input schema')
    for row in binding['files']:
        path = Path(row['path'])
        if path.is_relative_to(CODEC_ROOT) or path.is_relative_to(VENDOR) or path == UPSTREAM / 'checkpoints/vqvae_photoshape_official.ckpt':
            bind(records, path, row['sha256'], row['bytes'], allow_empty=path.suffix == '.py')
    for name in ('codec_final.pt', 'protocol.json', 'selection.json', 'code/common.py',
                 'code/codec.py', 'code/train.py', 'code/perceptual.py'):
        require((CODEC_ROOT / name).resolve() in records, f'Missing static codec input: {name}')
    require(records[(CODEC_ROOT / 'codec_final.pt').resolve()]['sha256'] == CODEC_SHA,
            'Initial codec checkpoint differs')
    for path, digest in STATIC_METADATA.items():
        bind(records, path, digest)
    bind(records, VGG_PATH, VGG_SHA, 574673361)
    train_done = read(CODEC_ROOT / 'train_done.json')
    require(train_done.get('status') == 'complete' and train_done.get('steps') == 3000
            and train_done.get('checkpoint_sha256') == CODEC_SHA,
            'Initial checkpoint has no exact completed 3000-step source receipt')
    helper_path = TOWER14_ROOT / 'prepare_frozen_codec.py'
    bind(records, helper_path, HELPER_SHA)
    return helper_path


def load_verified(root, *, import_modules=True):
    """Fully hash actual inputs; write immutable DATA_BINDING only after success.

    A missing tower10 terminal fails before any cache or old model is loaded.
    import_modules=False is only a data audit; it does not imply CUDA readiness.
    """
    root = Path(root).resolve(strict=True)
    require(root.name == RUN, 'Require the dedicated new multisource run directory')
    require(not any(root.is_relative_to(p.resolve()) for p in (CODEC_ROOT, TOWER10_ROOT, TOWER14_ROOT, UPSTREAM)),
            'New data binding must not be written inside any old source run')
    records = {}
    bind(records, root / 'protocol.json')
    protocol = read(root / 'protocol.json')
    validate_protocol(protocol)
    bind(records, root / 'CODE_MANIFEST.json')
    code_manifest = read(root / 'CODE_MANIFEST.json')
    require(isinstance(code_manifest, dict) and code_manifest, 'Missing current run code manifest')
    for relative, digest in code_manifest.items():
        path = (root / relative).resolve(strict=True)
        require(not Path(relative).is_absolute() and path.is_relative_to(root.parent), 'Unbounded current code path')
        bind(records, path, digest)
    tower10_summary = verify_tower10(TOWER10_ROOT, records)
    helper_path = _static_inputs(records)
    bind(records, TOWER14_ROOT / 'CODE_MANIFEST.json', TOWER14_MANIFEST_SHA)
    bind(records, TOWER14_ROOT / 'SUMMARY.json', TOWER14_SUMMARY_SHA)
    summary14 = read(TOWER14_ROOT / 'SUMMARY.json')
    validate_summary(summary14, EVAL_CLIP, TOWER14_BINDING_SHA)
    bind(records, V4_ROOT / 'cache_manifest.json', V4_MANIFEST_SHA)
    old_manifest = {row['path']: row for row in read(V4_ROOT / 'cache_manifest.json')}
    sources = {}
    for clip, directory in zip(ALL_CLIPS, (CODEC_ROOT, TOWER10_ROOT, TOWER14_ROOT)):
        index_path = directory / ('prepare' if clip == TRAIN_CLIPS[0] else 'quality') / f'{clip}.json'
        bind(records, index_path, TOWER14_QUALITY_SHA if clip == EVAL_CLIP else STATIC_METADATA.get(index_path))
        index = read(index_path)
        rows = index['rows' if clip == TRAIN_CLIPS[0] else 'frames']
        mapping_path, normal_path = directory / 'mappings' / f'{clip}.json', directory / 'normalization.json'
        bind(records, mapping_path, STATIC_METADATA.get(mapping_path))
        bind(records, normal_path, STATIC_METADATA.get(normal_path))
        mapping, normal = read(mapping_path), read(normal_path)
        if clip == TRAIN_CLIPS[0]:
            require(index.get('clip') == clip and index.get('normalization') == normal, 'Old prepare normalization differs')
            for row in rows:
                path = str(directory / 'cache' / clip / f'{row["frame"]:03d}.pt')
                authoritative = old_manifest.get(path, {})
                require(authoritative.get('sha256') == row.get('cache_sha256')
                        and authoritative.get('bytes') == row.get('cache_bytes'),
                        'Old prepare cache hashes differ from pinned v4 authoritative manifest')
            done_path = directory / 'prepare_done.json'
            bind(records, done_path, STATIC_METADATA.get(done_path))
            done = read(done_path)
            require(done.get('status') == 'complete' and done.get('states') == FRAMES
                    and done.get('all_source_points_per_state') == N and done.get('grid') == GRID,
                    'Original tower17 full preparation is incomplete')
        else:
            summary = tower10_summary if clip == TRAIN_CLIPS[1] else summary14
            require(normal == summary['normalization'], 'Clip-local normalization differs from completed summary')
        sources[clip] = build_source(clip, directory, rows, mapping, normal, records)
    binding = {'schema': 'gs_multisource_codec_data_binding_v1', 'frames_per_clip': FRAMES,
               'source_points_per_frame': N, 'height': H, 'width': W, 'train_clips': list(TRAIN_CLIPS),
               'eval_only_clips': [EVAL_CLIP], 'historical_roles': HISTORICAL_ROLES,
               'official_independent_holdout': False, 'historical_latents_reused': False,
               'initial_codec_sha256': CODEC_SHA, 'state_quantization': 'Identity; continuous float32 Nx64',
               'files': sorted(records.values(), key=lambda row: row['path'])}
    path = root / 'DATA_BINDING.json'
    if path.exists():
        require(read(path) == binding, 'DATA_BINDING is frozen; source changes or resume/rebinding are forbidden')
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(binding, indent=2, allow_nan=False) + '\n')
    bundle = SimpleNamespace(root=root, sources=sources, records=records, binding=binding,
                             binding_sha256=fingerprint(path)['sha256'], codec_root=CODEC_ROOT,
                             checkpoint=CODEC_ROOT / 'codec_final.pt', protocol=protocol,
                             common=None, codec=None, train=None, perceptual=None)
    if import_modules:
        source = helper_path.read_bytes()
        require(hashlib.sha256(source).hexdigest() == HELPER_SHA, 'Bound helper changed before import')
        spec = importlib.util.spec_from_file_location('_multisource_bound_preparer', helper_path)
        helper = importlib.util.module_from_spec(spec)
        exec(compile(source, str(helper_path), 'exec'), helper.__dict__)
        # Reuse the verified successful import finder, never copy vendor code.
        budget = helper.Budget(float('inf'))
        # Its normal guard starts after common is imported. Install the same
        # finder temporarily before that first import to cover common as well.
        startup_guard = helper.BoundSourceGuard([CODEC_ROOT / 'code', VENDOR], records, budget)
        sys.meta_path.insert(0, startup_guard)
        try:
            bundle.common, bundle.codec, _vendor = helper.load_bound_codec(CODEC_ROOT, records, budget)
        finally:
            sys.meta_path.remove(startup_guard)
        for name in ('train', 'perceptual'):
            require(name not in sys.modules or Path(sys.modules[name].__file__).resolve() == CODEC_ROOT / 'code' / f'{name}.py',
                    f'Module name collision: {name}')
            module = importlib.import_module(name)
            require(Path(module.__file__).resolve() == CODEC_ROOT / 'code' / f'{name}.py', f'Unexpected module: {name}')
            setattr(bundle, name, module)
        require((bundle.common.H, bundle.common.W, bundle.common.N) == (H, W, N), 'Renderer dimensions changed')
    return bundle


def verify_sources(bundle):
    """Hash all actually used sources again at phase/end boundaries."""
    for path, row in bundle.records.items():
        fingerprint(path, row['sha256'], row['bytes'], allow_empty=path.suffix == '.py')
    fingerprint(bundle.root / 'DATA_BINDING.json', bundle.binding_sha256)
    require(read(bundle.root / 'DATA_BINDING.json') == bundle.binding, 'Frozen data binding changed')
    require(not (TOWER10_ROOT / 'queue/FAILED.json').exists() and not (TOWER10_ROOT / 'FAILED.json').exists(),
            'Tower10 acquired a failure receipt after binding')
    return {'status': 'passed', 'files': len(bundle.records), 'binding_sha256': bundle.binding_sha256}


def validate_frame_metadata(payload, source, frame):
    require(type(frame) is int and 0 <= frame < FRAMES, 'Frame index must be 000..176')
    require(source.clip in ALL_CLIPS and source.role == ROLES[source.clip], 'Source role changed')
    require(payload.get('clip') == source.clip and payload.get('frame') == frame, 'Cache belongs to another clip/frame')
    require(payload.get('normalization') == source.normalization, 'Cache normalization belongs to another clip')
    require(payload.get('n_input') == N, 'Full 112896-point teacher aggregation is required')
    metrics = payload.get('metrics', {})
    require(metrics.get('clip') == source.clip and metrics.get('frame') == frame
            and metrics.get('source_count') == N
            and metrics.get('native_gs_sha256') == source.frames[frame]['native_gs_sha256'],
            'Cache source/point/provenance metadata differs')
    pts = source.frames[frame]['pts_seconds']
    for value in (metrics.get('pts_seconds'), payload.get('pts', pts)):
        require(isinstance(value, (float, int)) and math.isfinite(value) and abs(value - pts) <= 1e-9,
                'Cache PTS belongs to another source/frame')
    count = len(payload['representatives'])
    require(0 < count <= 80000, 'Input voxel guard exceeded; truncation forbidden')
    shapes = {'representatives': (count, 14), 'coordinates': (count, 3), 'set_statistics': (count, 29),
              'counts': (count,), 'source_indices': (count,), 'cameras': (2, 4, 4), 'intrinsics': (4,),
              'targets': (2, H, W, 3), 'real': (H, W, 3)}
    for key, shape in shapes.items():
        require(tuple(payload[key].shape) == shape, f'Cache shape mismatch: {key}; no frame/point/image reduction')
    require(int(payload['counts'].sum()) == N and int(payload['counts'].min()) > 0,
            'Every teacher point must be aggregated exactly once')
    scale = source.normalization['scale']
    require(payload.get('near') == 0.01 * scale and payload.get('far') == 1e10 * scale,
            'Camera clipping planes do not match clip-local normalization')
    require(not any(key in payload for key in ('features', 'codes', 'reconstruction')),
            'Historical latent/reconstruction cache cannot be reused under a new encoder')


def load_frame(source, frame, device='cuda'):
    """Load exact bytes that were hashed, then validate full cache before transfer."""
    import torch
    require(type(frame) is int and 0 <= frame < FRAMES, 'Frame index must be 000..176')
    row = source.frames[frame]
    content = Path(row['path']).read_bytes()
    require(len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256'],
            f'Cache changed after binding: {row["path"]}')
    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    validate_frame_metadata(payload, source, frame)
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all()), f'Nonfinite cache tensor: {key}')
    for key in ('counts', 'coordinates', 'source_indices'):
        require(payload[key].dtype in (torch.int32, torch.int64), f'Expected integer cache tensor: {key}')
    indices = payload['source_indices']
    require(bool(((indices >= 0) & (indices < N)).all()) and len(torch.unique(indices)) == len(indices),
            'Invalid original teacher source indices')
    require(len(torch.unique(payload['coordinates'], dim=0)) == len(indices), 'Duplicate input voxel coordinates')
    for key in ('representatives', 'set_statistics', 'cameras', 'intrinsics', 'targets', 'real'):
        payload[key] = payload[key].to(device=device)
    payload.update(pts=row['pts_seconds'], _source_cache_sha256=row['sha256'], _source_cache_path=row['path'])
    return payload
