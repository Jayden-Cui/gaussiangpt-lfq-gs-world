"""Bound, inference-only replay diagnosis. Importing this module never imports torch.

CPU helper tests are not evidence of CUDA/ME/renderer reproducibility.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CHECKPOINT_SHA = 'dcdcd0be5f999b09086bbc31c91aea62898600c123a962791845314898d6eef3'
FULLPOOL_MANIFEST_SHA = '01505377362bf47c89ae23aeaa5de6ca0c5663fe9119e35f6c01c03524351376'
FULLPOOL_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1'
CASES = [{'clip': 'buildingblock_tower_15_176', 'frame': i} for i in (113, 114)]
LANES = ('same_gs_render', 'same_sparse_decode', 'cpu_rebuild', 'disk_rebuild', 'historical_disk_rebuild')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(p):
    expected = {
        ('schema',): 'gs_replay_diagnostic_v1',
        ('source', 'fullpool_root'): FULLPOOL_ROOT,
        ('source', 'fullpool_manifest_sha256'): FULLPOOL_MANIFEST_SHA,
        ('source', 'checkpoint_sha256'): CHECKPOINT_SHA,
        ('cases',): CASES, ('repeats',): 3,
        ('data', 'height'): 252, ('data', 'width'): 448,
        ('data', 'source_gaussians'): 112896, ('data', 'latent_width'): 64,
        ('thresholds', 'mae_max'): 1e-5, ('thresholds', 'psnr_min'): 60,
        ('resources', 'cpu_threads'): 6, ('resources', 'peak_limit_gib'): 12,
        ('resources', 'max_seconds'): 1200, ('resources', 'outer_seconds'): 2460,
        ('state_quantization',): False, ('formal_updates',): 0,
        ('historical_saved_state',): True,
    }
    for keys, value in expected.items():
        actual = p
        for key in keys:
            actual = actual[key]
        require(actual == value and (not isinstance(value, bool) or type(actual) is bool),
                'Protocol drift: ' + '.'.join(keys))


def verify_manifest(root, expected_sha=None):
    root = Path(root)
    digest = sha(root / 'CODE_MANIFEST.json')
    require(expected_sha is None or digest == expected_sha, 'Source manifest SHA differs')
    manifest = read(root / 'CODE_MANIFEST.json')
    require(isinstance(manifest, dict) and manifest, 'Missing manifest entries')
    require({p.name for p in root.glob('*.py')} <= set(manifest), 'Unbound executable in run root')
    require('protocol.json' in manifest, 'Protocol must be manifest-bound')
    for name, expected in manifest.items():
        require(Path(name).name == name and sha(root / name) == expected, 'Manifest mismatch: ' + name)
    return manifest, digest


def verify_root(root):
    manifest, digest = verify_manifest(root)
    require({'probe.py', 'test_probe.py', 'PROBE_NOTES.md'} <= set(manifest), 'Probe sources must be bound')
    proof = read(root / 'CPU_VALIDATION.json')
    require(proof.get('status') == 'passed' and proof.get('gpu_executed') is False
            and proof.get('code_manifest_sha256') == digest, 'Fresh exact CPU validation required')
    p = read(root / 'protocol.json')
    validate_protocol(p)
    return p, {'code_manifest_sha256': digest, 'protocol_sha256': sha(root / 'protocol.json'),
               'cpu_validation_sha256': sha(root / 'CPU_VALIDATION.json')}


def np_array(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


def array_digest(value):
    a = np_array(value)
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)}, sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def canonical_order(coordinates, tied_arrays=()):
    """Exact coordinate sort; ties use complete row bytes, never approximate matching."""
    import numpy as np
    c = np_array(coordinates)
    require(c.ndim == 2 and np.isfinite(c).all(), 'Canonical coordinates must be finite NxD')
    order = np.lexsort(tuple(c[:, i] for i in reversed(range(c.shape[1]))))
    if len(order) < 2:
        return order
    sorted_c = c[order]
    boundaries = np.flatnonzero(np.any(sorted_c[1:] != sorted_c[:-1], axis=1)) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(c)]
    arrays = [np_array(a) for a in tied_arrays]
    require(all(len(a) == len(c) for a in arrays), 'Canonical tie arrays must share row count')
    for start, end in zip(starts, ends):
        if end - start > 1 and arrays:
            ids = order[start:end].tolist()
            ids.sort(key=lambda i: b''.join(a[i].tobytes() for a in arrays))
            order[start:end] = ids
    return order


def difference(a, b):
    import numpy as np
    a, b = np_array(a), np_array(b)
    result = {'reference_shape': list(a.shape), 'shape': list(b.shape),
              'reference_dtype': str(a.dtype), 'dtype': str(b.dtype),
              'reference_sha256': array_digest(a), 'sha256': array_digest(b)}
    result['same_shape'] = a.shape == b.shape
    result['exact_equal'] = result['reference_sha256'] == result['sha256']
    result['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if result['same_shape']:
        delta = a.astype(np.float64) - b.astype(np.float64)
        finite = np.isfinite(delta)
        result.update(changed_elements=int(np.count_nonzero(a != b)), nonfinite_differences=int((~finite).sum()),
                      max_abs=float(np.abs(delta).max()) if delta.size and finite.all() else (0. if not delta.size else None),
                      mean_abs=float(np.abs(delta).mean()) if delta.size and finite.all() else (0. if not delta.size else None),
                      rmse=float(np.sqrt(np.mean(delta * delta))) if delta.size and finite.all() else (0. if not delta.size else None))
    return result


def state_summary(coordinates, features):
    import numpy as np
    c, f = np_array(coordinates), np_array(features)
    require(c.ndim == 2 and c.shape[1] == 3 and c.dtype.kind in 'iu', 'State must have integer Nx3 coordinates')
    require(f.shape == (len(c), 64) and f.dtype == np.float32, 'State must have float32 Nx64 features')
    require(np.isfinite(f).all(), 'State contains nonfinite features')
    order = canonical_order(c, (f,))
    unique = len(c) - int(np.any(c[order][1:] != c[order][:-1], axis=1).sum()) - 1 if len(c) else 0
    return {'positions': len(c), 'channels': 64, 'duplicate_coordinate_rows': unique,
            'coordinates_sha256': array_digest(c), 'features_sha256': array_digest(f),
            'canonical_coordinates_sha256': array_digest(c[order]),
            'canonical_features_sha256': array_digest(f[order])}


def compare_states(a, b):
    ac, af = np_array(a['coordinates']), np_array(a['features'])
    bc, bf = np_array(b['coordinates']), np_array(b['features'])
    ao, bo = canonical_order(ac, (af,)), canonical_order(bc, (bf,))
    return {'reference': state_summary(ac, af), 'candidate': state_summary(bc, bf),
            'coordinates': difference(ac, bc), 'features': difference(af, bf),
            'canonical_coordinates': difference(ac[ao], bc[bo]),
            'canonical_features': difference(af[ao], bf[bo]),
            'stride_equal': a['stride'] == b['stride'], 'grid_equal': a['grid'] == b['grid']}


def gs_key_name(key):
    # PCKeys inherits str, Enum: str(PCKeys.SH) is 'PCKeys.SH', not 'sh'.
    # Feature heads retain configuration string keys, so normalize both forms.
    name = getattr(key, 'value', key)
    require(isinstance(name, str), 'GS attribute keys must normalize to strings')
    return name


def gs_schema(points):
    """Small, JSON-safe metadata only; does not transfer or serialize tensor values."""
    return {'container_type': type(points).__module__ + '.' + type(points).__qualname__,
            'attributes': [
                {'key_repr': repr(k), 'normalized_key': getattr(k, 'value', k),
                 'key_type': type(k).__module__ + '.' + type(k).__qualname__,
                 'value_type': type(v).__module__ + '.' + type(v).__qualname__,
                 'shape': list(v.shape), 'dtype': str(v.dtype), 'device': str(getattr(v, 'device', 'cpu'))}
                for k, v in points.items()]}


def gs_arrays(points, keys):
    """Validate the bound PCKeys dictionary without collapsing batch/scalar attributes."""
    require(isinstance(points, dict) and points, 'Decoder must return PCKeys dictionary')
    named = {gs_key_name(k): np_array(v) for k, v in points.items()}
    require(len(named) == len(points), 'Ambiguous normalized GS attribute keys')
    required = ('COORDS', 'COORD_OFFSET', 'SCALES', 'QUATS', 'OPACITIES', 'SH0', 'SH', 'BATCH')
    mapping = {name: gs_key_name(getattr(keys, name)) for name in required}
    require(set(mapping.values()) <= set(named), 'Bound GS packet is missing required attributes')
    xyz = named[mapping['COORDS']]
    require(xyz.ndim == 2 and xyz.shape[1] == 3, 'GS coordinates must be Nx3')
    n = len(xyz)
    for name, a in named.items():
        require(a.ndim >= 1 and len(a) == n, 'Unsupported non-row GS attribute: ' + name)
    for attr, width in [('COORD_OFFSET', 3), ('SCALES', 3), ('QUATS', 4), ('SH0', 3), ('SH', 9)]:
        require(named[mapping[attr]].shape == (n, width), 'Unexpected GS attribute shape: ' + attr)
    require(named[mapping['OPACITIES']].shape in ((n,), (n, 1)), 'Unexpected opacity shape')
    require(named[mapping['BATCH']].shape in ((n,), (n, 1)), 'Unexpected batch shape')
    return named, mapping


def gs_order(arrays, mapping):
    import numpy as np
    anchor = np.column_stack((arrays[mapping['BATCH']].reshape(-1), arrays[mapping['COORDS']]))
    return canonical_order(anchor, tuple(arrays[k] for k in sorted(arrays)))


def gs_summary(arrays, mapping):
    import numpy as np
    order = gs_order(arrays, mapping)
    attributes = {k: {'shape': list(a.shape), 'dtype': str(a.dtype), 'finite': bool(np.isfinite(a).all()),
                      'sha256': array_digest(a), 'canonical_sha256': array_digest(a[order])}
                  for k, a in arrays.items()}
    return {'count': len(order), 'keys': sorted(arrays), 'attributes': attributes}


def compare_gs(a, b, mapping):
    import numpy as np
    require(set(a) == set(b), 'GS schema changed between decodes')
    ao, bo = gs_order(a, mapping), gs_order(b, mapping)
    anchors_equal = all(array_digest(a[mapping[k]][ao]) == array_digest(b[mapping[k]][bo])
                        for k in ('COORDS', 'BATCH'))
    anchors = np.column_stack((a[mapping['BATCH']].reshape(-1), a[mapping['COORDS']]))[ao]
    duplicate_anchors = int(np.all(anchors[1:] == anchors[:-1], axis=1).sum())
    return {'reference_count': len(ao), 'count': len(bo), 'count_equal': len(ao) == len(bo),
            'canonical_anchor_exact_equal': anchors_equal,
            'reference_duplicate_anchor_rows': duplicate_anchors,
            'canonical_attribute_correspondence_valid': anchors_equal and duplicate_anchors == 0,
            'rowwise': {k: difference(a[k], b[k]) for k in sorted(a)},
            'canonical': {k: difference(a[k][ao], b[k][bo]) for k in sorted(a)},
            'canonical_alignment': 'exact batch/world-coordinate sort; complete attribute bytes break ties; no nearest-neighbor matching'}


def render_metrics(a, b):
    """CPU float64 companion metrics; GPU runtime additionally records legacy float32 metrics."""
    import numpy as np
    a, b = np_array(a), np_array(b)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[2] == 3, 'Image shape mismatch')
    d = a.astype(np.float64) - b.astype(np.float64)
    finite = bool(np.isfinite(d).all())
    mae = float(np.abs(d).mean()) if finite else None
    psnr = -10 * math.log10(max(float(np.mean(d * d)), 1e-12)) if finite else None
    return {'mae': mae, 'psnr': psnr, 'finite': finite,
            'passed': finite and mae <= 1e-5 and psnr >= 60,
            'mae_max': 1e-5, 'psnr_min': 60, 'max_abs': float(np.abs(d).max()) if finite else None}


def gpu_metrics(torch, legacy, reference, candidate):
    # Preserve the actual original threshold arithmetic/reduction on CUDA.
    result = legacy.image_metrics(reference.to(candidate.device), candidate)
    finite = all(math.isfinite(v) for v in result.values())
    return {**{k: v if math.isfinite(v) else None for k, v in result.items()},
            'finite': finite, 'passed': finite and result['mae'] <= 1e-5 and result['psnr'] >= 60,
            'mae_max': 1e-5, 'psnr_min': 60}


class Guard:
    def __init__(self, torch, seconds):
        self.torch, self.seconds, self.started = torch, seconds, time.monotonic()
        self.peak_process_mib = 0

    def check(self):
        t = self.torch
        t.cuda.synchronize()
        require(max(t.cuda.max_memory_allocated(), t.cuda.max_memory_reserved()) <= 12 * 2**30,
                '12 GiB torch memory guard exceeded; no downsampling allowed')
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                          '--format=csv,noheader,nounits'], text=True, timeout=10)
        rows = [line.split(',') for line in output.splitlines() if line.strip()]
        own = [int(parts[1].strip()) for parts in rows if parts[0].strip() == str(os.getpid())]
        require(len(own) == 1, 'Single physical GPU process memory observation required')
        self.peak_process_mib = max(self.peak_process_mib, own[0])
        require(self.peak_process_mib <= 12 * 1024, '12 GiB process GPU memory guard exceeded')
        require(time.monotonic() - self.started <= self.seconds, 'Phase wall-time budget exceeded')

    def evidence(self):
        return {'elapsed_seconds': time.monotonic() - self.started, 'max_seconds': self.seconds,
                'peak_torch_allocated_gib': self.torch.cuda.max_memory_allocated() / 2**30,
                'peak_torch_reserved_gib': self.torch.cuda.max_memory_reserved() / 2**30,
                'peak_observed_process_mib': self.peak_process_mib}


def load_worker(source_root):
    verify_manifest(source_root, FULLPOOL_MANIFEST_SHA)
    sys.dont_write_bytecode = True  # Do not create pycache in any frozen run.
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location('_replay_probe_bound_fullpool', source_root / 'worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sparse_state(z, packet):
    return {'coordinates': z.C[:, 1:].detach().cpu() // int(packet['stride']),
            'features': z.F.detach().cpu(), 'stride': int(packet['stride']), 'grid': float(packet['grid'])}


def historical_packet(torch, x, case, checkpoint_sha):
    path = x.root / 'evaluations' / 'decoder_only' / 'states' / case['clip'] / f"{case['frame']:03d}.pt"
    before = sha(path)
    p = torch.load(path, map_location='cpu', weights_only=True)
    metadata = p['metadata']
    require(p['schema'] == 'multisource_continuous_encoder_state_v1', 'Historical state schema differs')
    require(metadata['clip'] == case['clip'] and metadata['frame'] == case['frame']
            and metadata['arm'] == 'decoder_only' and metadata['checkpoint_sha256'] == checkpoint_sha
            and metadata['source'] == x.evidence and metadata['additional_codec_updates'] == 4426
            and metadata['decoder_target_key'] is None and metadata['fresh_encode'] is True,
            'Historical state binding/metadata differs')
    require(sha(path) == before, 'Historical packet changed while reading')
    return p, {'path': str(path), 'sha256': before, 'metadata': metadata}


def run_case(torch, worker, x, model, root, p, case, guard, log):
    from conf.dataclasses import PCKeys
    legacy, render = worker.legacy, x.old.train.render_prediction
    payload = worker.pool.load_frame(x.by_id[case['clip'], case['frame']], device='cuda')
    require(int(payload['n_input']) == 112896 and tuple(payload['real'].shape) == (252, 448, 3),
            'Full-frame geometry/resolution contract violated')
    guard.check()
    encoded = model.encode_state(payload['representatives'], payload['set_statistics'])
    metadata = {**case, 'checkpoint_sha256': CHECKPOINT_SHA, 'source_cache_sha256': payload['_source_cache_sha256']}
    packet = legacy.state_packet(encoded, metadata)
    old, historical = historical_packet(torch, x, case, CHECKPOINT_SHA)
    require(old['metadata']['source_cache_sha256'] == payload['_source_cache_sha256'], 'Historical source cache differs')
    case_dir = root / 'artifacts' / f"{case['clip']}_{case['frame']:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    disk = case_dir / 'fresh_state.pt'
    legacy.atomic_save(packet, disk)
    disk_sha = sha(disk)
    initial_latent = state_summary(packet['coordinates'], packet['features'])
    historical_comparison = compare_states(packet, old)
    baseline_gs, _ = model.ae.decode(encoded['sparse'], target_key=None)
    baseline_gs = model._world(baseline_gs)
    schema = {**case, 'stage': 'first_decode_after_world', **gs_schema(baseline_gs)}
    write(case_dir / 'GS_SCHEMA.json', schema)
    print(json.dumps({'event': 'gs_schema', **schema}, allow_nan=False), flush=True)
    reference, mapping = gs_arrays(baseline_gs, PCKeys)
    reference_images = {name: render(baseline_gs, payload, view).detach().cpu()
                        for view, name in enumerate(('real_view', 'nearby_view'))}
    guard.check()
    torch.save(reference_images, case_dir / 'reference_images.pt')
    rows = []
    for lane in LANES:
        first_gs, first_images = None, None
        for repeat in range(3):
            guard.check()
            started = time.monotonic()
            if lane == 'same_gs_render':
                points, actual_state = baseline_gs, sparse_state(encoded['sparse'], packet)
                supplied_state = packet
            elif lane == 'same_sparse_decode':
                supplied_state = packet
                actual_state = sparse_state(encoded['sparse'], packet)
                points, _ = model.ae.decode(encoded['sparse'], target_key=None)
                points = model._world(points)
            else:
                supplied_state = packet
                if lane == 'disk_rebuild':
                    require(sha(disk) == disk_sha, 'Fresh disk packet changed')
                    supplied_state = torch.load(disk, map_location='cpu', weights_only=True)
                elif lane == 'historical_disk_rebuild':
                    require(sha(historical['path']) == historical['sha256'], 'Historical packet changed')
                    supplied_state = torch.load(historical['path'], map_location='cpu', weights_only=True)
                # Exactly the bound decode_state operations, exposed to inspect ME's actual row order.
                z = model._sparse(supplied_state['coordinates'].to(payload['representatives'].device),
                                  supplied_state['features'].to(payload['representatives'].device), supplied_state['stride'])
                actual_state = sparse_state(z, supplied_state)
                points, _ = model.ae.decode(z, target_key=None)
                points = model._world(points)
                del z
            guard.check()
            arrays, current_mapping = gs_arrays(points, PCKeys)
            require(mapping == current_mapping, 'GS key schema changed')
            images, image_reports, within_lane = {}, {}, {}
            for view, name in enumerate(('real_view', 'nearby_view')):
                image = render(points, payload, view)
                require(tuple(image.shape) == (252, 448, 3), 'Renderer image resolution changed')
                image_reports[name] = {'legacy_float32': gpu_metrics(torch, legacy, reference_images[name], image),
                                       'cpu_float64': render_metrics(reference_images[name], image)}
                if first_images is not None:
                    within_lane[name] = gpu_metrics(torch, legacy, first_images[name], image)
                images[name] = image.detach().cpu()
                del image
            guard.check()
            record = {**case, 'lane': lane, 'repeat': repeat, 'elapsed_seconds': time.monotonic() - started,
                      'supplied_latent': state_summary(supplied_state['coordinates'], supplied_state['features']),
                      'actual_sparse_vs_supplied': compare_states(supplied_state, actual_state),
                      'gs': gs_summary(arrays, mapping), 'gs_vs_reference': compare_gs(reference, arrays, mapping),
                      'render_vs_reference': image_reports, 'render_vs_lane_first': within_lane,
                      'gs_vs_lane_first': compare_gs(first_gs, arrays, mapping) if first_gs is not None else None,
                      'input_sparse_after_decode': state_summary(encoded['sparse'].C[:, 1:] // packet['stride'], encoded['sparse'].F)}
            record['replay_thresholds_passed'] = all(v['legacy_float32']['passed'] for v in image_reports.values())
            torch.save(images, case_dir / f'{lane}_{repeat}_images.pt')
            rows.append(record)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            if first_images is None:
                first_images, first_gs = images, arrays
            del points, arrays, actual_state, supplied_state, images
            gc.collect()
        del first_images, first_gs
        if lane == 'same_gs_render':
            del baseline_gs
        torch.cuda.empty_cache()
    guard.check()
    require(sha(disk) == disk_sha and sha(historical['path']) == historical['sha256'], 'State packet changed')
    return {**case, 'source_cache_sha256': payload['_source_cache_sha256'],
            'fresh_packet_sha256': disk_sha, 'fresh_latent': initial_latent,
            'historical': historical, 'fresh_vs_historical': historical_comparison,
            'reference_gs': gs_summary(reference, mapping), 'records': rows,
            'replay_thresholds_passed': all(r['replay_thresholds_passed'] for r in rows)}


def main(root):
    root = Path(root).resolve()
    p, evidence = verify_root(root)
    require(not any((root / name).exists() for name in ('RESULT.json', 'FAILED.json', 'records.jsonl', 'STARTED.json')),
            'Probe already started; do not overwrite evidence')
    require(root != Path(p['source']['fullpool_root']).resolve(), 'Probe cannot write to the frozen source run')
    started = time.monotonic()
    write(root / 'STARTED.json', {'status': 'initializing', 'source': evidence, 'pid': os.getpid(),
                                'formal_updates': 0, 'gpu_executed': False})
    guard = None
    try:
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        # Leave allocator headroom for CUDA/ME/renderer allocations counted by nvidia-smi.
        torch.cuda.set_per_process_memory_fraction(min(1., 10 * 2**30 / torch.cuda.get_device_properties(0).total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
        def timeout_handler(signum, frame):
            raise TimeoutError('Bounded replay phase deadline exceeded')
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1200)
        initialization_guard = Guard(torch, 1200)
        worker = load_worker(Path(p['source']['fullpool_root']))
        x = worker.load_runtime(Path(p['source']['fullpool_root']))
        initialization_guard.check()
        model, checkpoint, step = worker.evaluation_model(x, 'decoder_only')
        require(step == 4426 and sha(checkpoint) == CHECKPOINT_SHA, 'Unexpected decoder-only checkpoint')
        require(not model.training and not model.ae.decoder.training, 'Inference requires eval mode')
        require(isinstance(model.ae.vq, torch.nn.Identity), 'Continuous state identity VQ required')
        initialization_guard.check()
        initialization = initialization_guard.evidence()
        parameter_before = worker.legacy.digest_snapshot(worker.legacy.snapshot(model))
        signal.alarm(1200)
        guard = Guard(torch, 1200)
        cases = []
        with torch.no_grad(), (root / 'records.jsonl').open('x', buffering=1, encoding='utf-8') as log:
            for case in p['cases']:
                cases.append(run_case(torch, worker, x, model, root, p, case, guard, log))
                gc.collect()
                torch.cuda.empty_cache()
        require(parameter_before == worker.legacy.digest_snapshot(worker.legacy.snapshot(model)), 'Model state changed')
        require(sha(checkpoint) == CHECKPOINT_SHA, 'Checkpoint changed during diagnosis')
        require(verify_root(root)[1] == evidence, 'Probe binding changed during diagnosis')
        verify_manifest(x.root, FULLPOOL_MANIFEST_SHA)
        guard.check()
        all_passed = all(c['replay_thresholds_passed'] for c in cases)
        write(root / 'RESULT.json', {'schema': p['schema'], 'status': 'completed', 'completed': True,
              'gpu_executed': True, 'source': evidence, 'fullpool_source': x.evidence,
              'checkpoint_sha256': CHECKPOINT_SHA, 'model_state_sha256': parameter_before,
              'model_state_unchanged': True, 'formal_updates': 0, 'state_quantization': False,
              'thresholds': p['thresholds'], 'replay_thresholds_passed': all_passed,
              'diagnostic_completion_is_replay_acceptance': False, 'repair_validated': False,
              'scope': 'Two fixed frames in a fresh process; no replay of the previous 2238 evaluation frames',
              'interpretation_limit': 'Passing here cannot exclude long-evaluation coordinate-manager, RNG or operation-order effects; no root cause is inferred.',
              'initialization': initialization, 'diagnostic': guard.evidence(),
              'total_elapsed_seconds': time.monotonic() - started, 'cases': cases})
        signal.alarm(0)
    except BaseException as error:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        write(root / 'FAILED.json', {'status': 'failed', 'completed': False, 'error': repr(error),
              'source': evidence, 'formal_updates': 0, 'total_elapsed_seconds': time.monotonic() - started,
              'diagnostic': guard.evidence() if guard is not None else None,
              'partial_records': 'records.jsonl; threshold failures alone do not cause this file'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
