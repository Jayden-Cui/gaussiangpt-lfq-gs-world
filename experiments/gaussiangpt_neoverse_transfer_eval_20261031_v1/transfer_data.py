"""Immutable, complete tower14 data for inference-only transfer diagnostics.

Importing this module does not import Torch, initialize CUDA, or read results.
The model-facing state dictionaries never contain target images or reconstructions.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
CLIP = 'buildingblock_tower_14_0'
PREPARATION_NAME = 'gaussiangpt_neoverse_tower14_transfer_20261031_v1'
PREPARATION_MANIFEST = '8c9381593b2a7bce009e38d73c07f9de1b0ab1a411cfa7feb9ce6729d268ef57'
INPUT_BINDING = 'a422de97bdfeb7b9dcae47a00478f43ddb63185b9b352b85e221cb914c7d1e8d'
CODEC_CHECKPOINT = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
SOURCE_VIDEO = '456d178c177d6790196f2e27f3bf5e1f5eebbb9fdaecc716794be90fbbab57f8'
V6_NAME = 'gaussiangpt_neoverse_native_z_20261031_v6'
V6_MANIFEST = '5eb6842dfbf9e65239cb72c83bec907b1abe21d2c2a1beca05dfce7562b94d2c'
FRAMES, HEIGHT, WIDTH, WINDOWS, FPS = 177, 252, 448, 168, 29.97
ROI_XYXY = (67, 38, 380, 227)
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr', 'motion_mae')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return value


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_normalization(normal):
    fixed = dict(center=[0.0, 0.0, 0.0], source_frame=0, fixed_for_all_frames=True,
                 uses_future_frames=False, changes_codec_output_std=False)
    require(isinstance(normal, dict) and all(normal.get(k) == v for k, v in fixed.items()),
            'Normalization must be fixed frame-zero scale only, without future calibration')
    scale = normal.get('scale')
    require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
            'Invalid normalization scale')
    require(math.log2(scale).is_integer(), 'Normalization scale is not dyadic')


def validate_mapping(mapping, pts):
    require(mapping.get('clip') == CLIP and mapping.get('source_sha256') == SOURCE_VIDEO,
            'Mapping source identity mismatch')
    require(mapping.get('source_frame_indices') == list(range(FRAMES))
            and mapping.get('model_timestamps') == list(range(FRAMES))
            and mapping.get('adjacent_pairs') == [[t, t + 1] for t in range(FRAMES - 1)],
            'Mapping must contain all177 consecutive real frames and176 adjacent pairs')
    require(mapping.get('source_pts_seconds') == pts and len(pts) == FRAMES
            and all(type(t) in (int, float) and math.isfinite(t)
                    and abs(t - frame / FPS) <= 1e-9 for frame, t in enumerate(pts)),
            'Stored PTS differ from the verified complete teacher sequence')


def load_prepared(preparation_root):
    """Read a successfully completed preparation, validate all bytes, return context.

    All output hashes are bound at admission from the completed producer's records;
    they were not known before the new teacher was exported. No result is invented
    when the preparation is incomplete. This function has no Torch dependency.
    """
    root = Path(preparation_root).resolve(strict=True)
    require(root.name == PREPARATION_NAME, 'Wrong preparation root')
    require(not (root / 'FAILED.json').exists() and not (root / 'queue/FAILED.json').exists(),
            'Preparation has a failure receipt')
    hashes, file_sizes = {}, {}

    def bind(path, expected=None, size=None):
        path = Path(path).resolve(strict=True)
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                'Input changed during hashing: ' + str(path))
        require(expected is None or digest == expected, 'Input SHA256 mismatch: ' + str(path))
        require(size is None or after.st_size == size, 'Input size mismatch: ' + str(path))
        require(str(path) not in hashes or hashes[str(path)] == digest, 'Conflicting input identity')
        hashes[str(path)], file_sizes[str(path)] = digest, after.st_size
        return digest

    bind(root / 'CODE_MANIFEST.json', PREPARATION_MANIFEST)
    for relative, digest in read(root / 'CODE_MANIFEST.json').items():
        path = (root / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root), 'Manifest path outside preparation')
        bind(path, digest)
    bind(root / 'input_binding.json', INPUT_BINDING)
    binding = read(root / 'input_binding.json')
    require(binding.get('schema') == 'gs_frozen_codec_transfer_binding_v1', 'Wrong input binding schema')
    static = {}
    for item in binding['files']:
        path = Path(item['path'])
        require(path.is_absolute() and str(path) not in static, 'Duplicate or relative static binding path')
        bind(path, item['sha256'], item['bytes'])
        static[str(path)] = item
    codec = [v for v in static.values() if v['path'].endswith('/state/codec_final.pt')]
    require(len(codec) == 1 and codec[0]['sha256'] == CODEC_CHECKPOINT, 'Frozen codec input differs from TRAIN')
    bind(root / 'CPU_VALIDATION.json')
    witness = read(root / 'CPU_VALIDATION.json')
    require(witness.get('status') == 'passed' and witness.get('gpu_executed') is False
            and witness.get('code_manifest_sha256') == PREPARATION_MANIFEST
            and witness.get('input_binding_sha256') == INPUT_BINDING, 'Wrong preparation CPU witness')
    bind(root / 'queue/DONE.json')
    done = read(root / 'queue/DONE.json')
    require(done.get('status') == 'complete' and done.get('preparation_contract_passed') is True
            and done.get('training_performed') is False and done.get('scientific_transfer_passed') is None
            and done.get('goal_complete') is False, 'Preparation queue is not successfully complete')
    require(done.get('admission') == dict(code_manifest_sha256=PREPARATION_MANIFEST,
        input_binding_sha256=INPUT_BINDING, cpu_validation_sha256=hashes[str(root / 'CPU_VALIDATION.json')]),
        'Preparation DONE belongs to different admitted source bytes')
    require([x.get('name') for x in done.get('outcomes', [])] == ['teacher_export', 'frozen_codec_preparation']
            and all(x.get('exit_code') == 0 and x.get('timed_out') is False for x in done['outcomes']),
            'Both producer stages must have exited successfully')
    bind(root / 'SUMMARY.json', done['summary_sha256'])
    summary = read(root / 'SUMMARY.json')
    queue = module_at('_transfer_preparation_contract', root / 'queue_prepare.py')
    queue.prepared_contract(summary, INPUT_BINDING)
    require(done['codec_state_sha256_before_and_after'] == summary['codec_state_sha256_before_and_after'],
            'Frozen full codec digest differs between receipts')
    require(summary.get('all_input_points_aggregated') is True
            and summary.get('source_points_per_frame') == 112896
            and summary.get('adjacent_pairs') == 176
            and summary.get('state_quantization') == 'Identity; continuous float32 Nx64',
            'Prepared full GS/continuous-state contract differs')
    bind(root / 'normalization.json')
    normal = read(root / 'normalization.json')
    validate_normalization(normal)
    require(summary['normalization'] == normal, 'Summary normalization differs')
    inputs_path = root / 'inputs_hashes.json'
    bind(inputs_path)
    inputs = read(inputs_path)
    require(inputs.get('schema') == 'gs_transfer_inputs_v1'
            and inputs.get('teacher_hashes_computed_after_export') is True
            and inputs['binding']['sha256'] == INPUT_BINDING, 'Preparation input provenance differs')
    declared_static = {x['path']: (x['bytes'], x['sha256']) for x in inputs['static_files']}
    require(declared_static == {p: (v['bytes'], v['sha256']) for p, v in static.items()},
            'Prepared static source list differs from pre-export binding')
    teacher = root / 'teacher/native448'
    teacher_entries = {}
    for item in inputs['teacher_and_source_files']:
        path = Path(item['path']).resolve(strict=True)
        allowed = path.is_relative_to(teacher) or str(path) in static
        require(allowed and str(path) not in teacher_entries, 'Wrong or duplicate teacher input path')
        bind(path, item['sha256'], item['bytes'])
        teacher_entries[str(path)] = item
    expected_teacher = {teacher / n for n in ('provenance.json', 'cameras.npz', 'input_rgb.npy', 'rendered_rgb.npy')}
    expected_teacher.update(teacher / 'gaussians' / f'native_group_{t:03d}.npz' for t in range(FRAMES))
    require(expected_teacher.issubset({Path(p) for p in teacher_entries}), 'Incomplete177-frame teacher source list')
    prepare = module_at('_transfer_prepared_teacher_contract', root / 'prepare_frozen_codec.py')
    pts = prepare.validate_provenance(read(teacher / 'provenance.json'), binding['expected_teacher'], teacher)
    mapping_path = root / 'mappings' / (CLIP + '.json')
    bind(mapping_path)
    validate_mapping(read(mapping_path), pts)
    quality_path = root / 'quality' / (CLIP + '.json')
    bind(quality_path)
    quality = read(quality_path)
    rows = quality.get('frames')
    require(isinstance(rows, list) and len(rows) == FRAMES
            and [row.get('frame') for row in rows] == list(range(FRAMES)), 'Missing/reordered frame quality rows')
    cache_paths, state_paths, sources = [], [], []
    for frame, row in enumerate(rows):
        path = root / 'quality' / CLIP / f'{frame:03d}.json'
        bind(path)
        require(read(path) == row, 'Per-frame quality differs from aggregate file')
        require(row.get('clip') == CLIP and row.get('pts_seconds') == pts[frame]
                and row.get('source_count') == 112896, 'Frame source identity mismatch')
        native_path = teacher / 'gaussians' / f'native_group_{frame:03d}.npz'
        require(row.get('native_gs_sha256') == hashes[str(native_path)], 'Frame GS source SHA differs')
        for key in ('native_render_parity', 'similarity_parity'):
            require(row[key]['psnr'] >= 60 and row[key]['mae'] <= 1e-5, 'Producer native parity did not pass')
        require(0 < row['input_voxels'] <= 80000 and 0 < row['max_decoder_candidates'] <= 600000
                and 0 < row['decoded_gaussians'] <= 2400000 and row['latent_positions'] > 0,
                'Producer candidate/count guard failed')
        for kind, paths in (('cache', cache_paths), ('state', state_paths)):
            path = root / ('cache' if kind == 'cache' else 'states') / CLIP / f'{frame:03d}.pt'
            digest = bind(path, row[kind + '_sha256'], row[kind + '_bytes'])
            paths.append(path)
            sources.append(dict(path=str(path), bytes=row[kind + '_bytes'], sha256=digest))
    require(quality.get('summary') == summary.get('quality'), 'Prepared quality aggregate differs')
    for group, metrics in summary['quality'].items():
        for metric, value in metrics.items():
            expected = sum(row[group][metric] for row in rows) / FRAMES
            require(math.isfinite(value) and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12),
                    'Prepared quality mean is inconsistent')
    source_root = root.parent / V6_NAME
    bind(source_root / 'CODE_MANIFEST.json', V6_MANIFEST)
    for name in ('sampling.py', 'sequence_data.py'):
        path = source_root / 'code' / name
        bind(path, read(source_root / 'CODE_MANIFEST.json')['code/' + name])
    return dict(root=root, hashes=hashes, file_sizes=file_sizes, quality_rows=rows,
                normalization=normal, pts=pts, cache_paths=cache_paths, state_paths=state_paths,
                codec_checkpoint_sha256=CODEC_CHECKPOINT, sources=sources, source_root=source_root,
                summary=summary, input_binding_sha256=INPUT_BINDING)


def verify_unchanged(context):
    for name, digest in context['hashes'].items():
        require(Path(name).stat().st_size == context['file_sizes'][name] and sha(name) == digest,
                'Admitted transfer input changed: ' + name)
    root = context['root']
    require(not (root / 'FAILED.json').exists() and not (root / 'queue/FAILED.json').exists(),
            'Preparation failure appeared during evaluation')


def model_state(state, frame, device):
    """Explicit allow-list; all images, reconstruction and metadata stay outside the model."""
    return dict(coordinates=state['coordinates'].to(device), features=state['features'].to(device),
                pts=state['pts'], frame_index=frame, stride=state['stride'], grid=state['grid'])


class TransferData:
    def __init__(self, context, device='cuda'):
        import torch
        self.torch, self.context = torch, context
        code = context['source_root'] / 'code'
        for name in ('sampling', 'sequence_data'):
            path = (code / (name + '.py')).resolve()
            require(sha(path) == context['hashes'][str(path)], 'State validation helper changed')
        sys.path.insert(0, str(code))
        sampling = importlib.import_module('sampling')
        sequence = importlib.import_module('sequence_data')
        for module in (sampling, sequence):
            require(Path(module.__file__).resolve().parent == code, 'Conflicting state validation module')
        self.states, self.records, self.frozen_reconstructions, self.state_metadata = [], [], [], []
        self.sources, self.normalization, self.pts = context['sources'], context['normalization'], context['pts']
        segment = sampling.Segment(CLIP, 0, 176)
        for frame, (state_path, cache_path) in enumerate(zip(context['state_paths'], context['cache_paths'])):
            # Recheck exact bytes immediately before deserializing, never unrestricted pickle.
            require(sha(state_path) == context['hashes'][str(state_path)]
                    and sha(cache_path) == context['hashes'][str(cache_path)], 'Prepared cache changed before load')
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            raw = torch.load(cache_path, map_location='cpu', weights_only=True)
            sequence._validate_state(state, state_path, sampling.Window(segment, min(frame, 167)), frame)
            self._validate_frame(state, raw, frame)
            self.state_metadata.append(dict(state['metadata']))
            self.states.append(model_state(state, frame, device))
            self.frozen_reconstructions.append(state['reconstruction'][0].to(device))
            self.records.append({key: raw[key].to(device) if torch.is_tensor(raw[key]) else raw[key]
                                 for key in ('real', 'cameras', 'intrinsics', 'near', 'far')})
        self.roi = torch.zeros((HEIGHT, WIDTH), dtype=torch.bool, device=device)
        x0, y0, x1, y1 = ROI_XYXY
        self.roi[y0:y1, x0:x1] = True

    def _validate_frame(self, state, raw, frame):
        torch, context = self.torch, self.context
        row = context['quality_rows'][frame]
        require(state.get('schema') == 'gsworld_continuous_state_v1'
                and state.get('feature_dtype') == 'float32' and state['features'].dtype == torch.float32
                and state.get('stride') == 4 and state.get('grid') == .008
                and len(state['coordinates']) == row['latent_positions'], 'Continuous64 state layout differs')
        for obj in (state, raw, state.get('metadata', {})):
            require(obj.get('clip') == CLIP and obj.get('frame') == frame
                    and obj.get('pts') == context['pts'][frame]
                    and obj.get('normalization') == context['normalization'], 'State/cache frame metadata mismatch')
        metadata = state['metadata']
        require(metadata.get('source_cache_sha256') == row['cache_sha256']
                and state.get('source_cache_sha256') == row['cache_sha256']
                and metadata.get('native_gs_sha256') == row['native_gs_sha256']
                and metadata.get('codec_checkpoint_sha256') == CODEC_CHECKPOINT
                and metadata.get('input_binding_sha256') == INPUT_BINDING, 'State provenance differs')
        require(raw.get('n_input') == 112896 and int(raw['counts'].sum()) == 112896
                and len(raw['coordinates']) == row['input_voxels'], 'Source GS contributions were lost')
        require(state.get('baseline_real') == row['codec_real'], 'Frozen codec reference metrics differ')
        for tensor, shape in ((raw['real'], (HEIGHT, WIDTH, 3)), (raw['targets'], (2, HEIGHT, WIDTH, 3)),
                              (state['reconstruction'], (2, HEIGHT, WIDTH, 3)),
                              (raw['cameras'], (2, 4, 4)), (raw['intrinsics'], (4,))):
            require(torch.is_tensor(tensor) and tuple(tensor.shape) == shape and tensor.dtype == torch.float32
                    and bool(torch.isfinite(tensor).all()), 'Invalid native float32 data tensor')
        require(bool(((raw['real'] >= 0) & (raw['real'] <= 1)).all())
                and bool(((state['reconstruction'] >= 0) & (state['reconstruction'] <= 1)).all()),
                'Image/reference range differs')
        require(all(type(raw[k]) in (int, float) and math.isfinite(raw[k]) for k in ('near', 'far'))
                and 0 < raw['near'] < raw['far'], 'Invalid clipping planes')

    def window(self, start):
        require(type(start) is int and 0 <= start < WINDOWS, 'Incomplete window requested')
        return self.states[start:start + 10]

    def _target(self, start, horizon):
        require(type(start) is int and type(horizon) is int and 0 <= start < WINDOWS and 0 <= horizon <= 9,
                'Expected a complete ten-frame target identity')
        return self.records[start + horizon]['real']

    def _image(self, image):
        torch = self.torch
        require(torch.is_tensor(image) and image.shape == (HEIGHT, WIDTH, 3)
                and image.dtype == torch.float32 and bool(torch.isfinite(image).all()),
                'Metrics require complete native float32 RGB')

    def motion_weight(self, start, horizon):
        target = self._target(start, horizon)
        return ((target - self.records[start]['real']).abs().mean(-1) / .1).clamp(0, 1).detach()

    def _motion_mean(self, error, start, horizon):
        weight = self.motion_weight(start, horizon)
        total = weight.sum()
        return None if float(total) == 0 else float((error * weight).sum() / total)

    def metrics(self, image, start, horizon):
        self._image(image)
        error = image - self._target(start, horizon)
        mse, mae = error.square(), error.abs().mean(-1)
        torch = self.torch
        return dict(mae=float(mae.mean()), psnr=float(-10 * torch.log10(mse.mean().clamp_min(1e-12))),
            workspace_roi_mae=float(mae[self.roi].mean()),
            workspace_roi_psnr=float(-10 * torch.log10(mse[self.roi].mean().clamp_min(1e-12))),
            motion_mae=self._motion_mean(mae, start, horizon))

    def delta_metrics(self, image, own_h0, start, horizon):
        self._image(image)
        self._image(own_h0)
        target = self._target(start, horizon)
        error = ((image - own_h0) - (target - self.records[start]['real'])).abs().mean(-1)
        return dict(mae=float(error.mean()), workspace_roi_mae=float(error[self.roi].mean()),
                    motion_mae=self._motion_mean(error, start, horizon))
