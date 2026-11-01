"""Audit existing codec variants on TRAIN sources and align sparse states by XYZ."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
DATA = OUTPUTS / 'gaussiangpt_neoverse_multisource_action_data_20261102_v1'
EXPERIMENT = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
sys.path.insert(0, str(DATA))
from source_store import TRAIN, EVAL, PINS, CODEC_SHA

ARMS = ('baseline', 'decoder_only', 'full_codec')
CHECKPOINTS = {
    'baseline': (OUTPUTS / 'gaussiangpt_neoverse_state_motion_20261030_v1/state/codec_final.pt', CODEC_SHA),
    'decoder_only': (EXPERIMENT / 'arms/decoder_only/step_0300.pt',
                     '153724acd62bdb8c611e24680ed39e0b4fc38b366b466b640d40eef6190052af'),
    'full_codec': (EXPERIMENT / 'arms/full_codec/step_0300.pt',
                   '12a4b81b5fd01f9a8905ef6b24db93b95b39e31c6783f6f00e82fea6c4612f58')}
DECODER_PREFIXES = ('ae.decoder', 'ae.feature_heads', 'ae.gaussian_multiplier')
FRAMES = (0, 44, 88, 132, 176)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def in_decoder(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in DECODER_PREFIXES)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoder_snapshot(state):
    return {name: tensor_hash(value) for name, value in state.items() if not in_decoder(name)}


def snapshot_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical(coords, features):
    coords, features = np.asarray(coords), np.asarray(features)
    if (coords.ndim != 2 or coords.shape[1] != 3 or features.shape != (len(coords), 64)
            or coords.dtype != np.int32 or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError('Invalid sparse state')
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    ordered_coords, ordered_features = coords[order], features[order]
    if len(coords) > 1 and bool(np.any(np.all(ordered_coords[1:] == ordered_coords[:-1], axis=1))):
        raise ValueError('Duplicate world coordinate')
    return ordered_coords, ordered_features


def compare_states(coords_a, features_a, coords_b, features_b):
    """Align on world coordinates, never on the sparse array row index."""
    ca, fa = canonical(coords_a, features_a)
    cb, fb = canonical(coords_b, features_b)
    same = np.array_equal(ca, cb)
    raw_order_equal = np.array_equal(np.asarray(coords_a), np.asarray(coords_b))
    if same:
        left, right = fa, fb
    else:
        lookup = {tuple(row): i for i, row in enumerate(cb.tolist())}
        pairs = [(i, lookup[tuple(row)]) for i, row in enumerate(ca.tolist()) if tuple(row) in lookup]
        if not pairs:
            raise ValueError('No shared world coordinates')
        left, right = fa[[a for a, _ in pairs]], fb[[b for _, b in pairs]]
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {'baseline_nodes': len(ca), 'candidate_nodes': len(cb), 'matched_nodes': len(left),
            'same_coordinate_set': bool(same), 'same_raw_row_order': bool(raw_order_equal),
            'feature_mean_abs': float(delta.mean()), 'feature_max_abs': float(delta.max())}


def selected_rows(path, *, expected_clip_frame=True):
    result = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            clip = row.get('clip')
            if clip not in TRAIN:
                continue
            key = (clip, row['frame'])
            if key in result:
                raise RuntimeError('Duplicate training-source row')
            result[key] = row
    if expected_clip_frame and set(result) != {(clip, frame) for clip in TRAIN for frame in range(177)}:
        raise RuntimeError('Incomplete 177-frame TRAIN-source coverage')
    return result


def screenshot_rows(path):
    result = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('clip') not in TRAIN or row.get('frame') not in FRAMES:
                continue
            key = (row['clip'], row['frame'], row['kind'])
            if key in result:
                raise RuntimeError('Duplicate screenshot row')
            result[key] = row
    required = {(clip, frame, kind) for clip in TRAIN for frame in FRAMES
                for kind in ('real', 'codec', 'nearby', 'nearby_teacher')}
    if not required.issubset(result):
        raise RuntimeError('Missing fixed training-source screenshot')
    return result


def load_image(folder, entry):
    path = folder / entry['path']
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise RuntimeError('Screenshot bytes changed')
    with Image.open(path) as image:
        if image.size != (448, 252):
            raise RuntimeError('Unexpected screenshot resolution')
        return image.convert('RGB').copy()


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Audit CPU tests missing')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Audit source changed: ' + name)
    if sha(EXPERIMENT / 'protocol.json') != PINS['protocol.json'] or sha(EXPERIMENT / 'DATA_BINDING.json') != PINS['DATA_BINDING.json']:
        raise RuntimeError('Experiment protocol or data binding changed')
    protocol = read(EXPERIMENT / 'protocol.json')
    if protocol['data']['train_clips'] != list(TRAIN) or protocol['data']['eval_only_clips'] != list(EVAL):
        raise RuntimeError('Training/evaluation source split changed')
    summary = read(EXPERIMENT / 'SUMMARY.json')
    if summary.get('engineering_status') != 'complete' or summary.get('production_codec_selected') is not None:
        raise RuntimeError('Unexpected historical codec experiment state')
    evaluations, state_rows, metric_rows, screenshot_entries = {}, {}, {}, {}
    for arm in ARMS:
        folder = EXPERIMENT / 'evaluations' / arm
        evaluation = read(folder / 'EVALUATION.json')
        if (evaluation.get('status') != 'evaluated' or evaluation.get('arm') != arm
                or evaluation.get('frames') != 531 or evaluation.get('frames_per_clip') != 177
                or evaluation.get('checkpoint_sha256') != CHECKPOINTS[arm][1]
                or evaluation.get('source', {}).get('protocol_sha256') != PINS['protocol.json']
                or evaluation.get('source', {}).get('binding_sha256') != PINS['DATA_BINDING.json']
                or evaluation != summary['evaluations'][arm]['evaluation']
                or not evaluation.get('all_frames_freshly_encoded')):
            raise RuntimeError('Historic evaluation provenance differs: ' + arm)
        for name, key in (('state_manifest.jsonl', 'state_manifest_sha256'),
                          ('targets.jsonl', 'targets_sha256'), ('screenshot_manifest.jsonl', 'screenshot_manifest_sha256')):
            if sha(folder / name) != evaluation[key]:
                raise RuntimeError('Historic manifest changed: ' + arm + '/' + name)
        evaluations[arm] = evaluation
        state_rows[arm] = selected_rows(folder / 'state_manifest.jsonl')
        metric_rows[arm] = selected_rows(folder / 'targets.jsonl')
        screenshot_entries[arm] = screenshot_rows(folder / 'screenshot_manifest.jsonl')

    snapshots, models = {}, {}
    for arm in ARMS:
        path, expected_sha = CHECKPOINTS[arm]
        if sha(path) != expected_sha:
            raise RuntimeError('Codec checkpoint changed: ' + arm)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if arm != 'baseline':
            done = read(EXPERIMENT / 'arms' / arm / 'TRAIN_DONE.json')
            if (payload.get('arm') != arm or payload.get('step') != 300 or payload.get('failed') is not False
                    or done.get('status') != 'complete' or done.get('checkpoint_sha256') != expected_sha
                    or done.get('frames_seen') != 354):
                raise RuntimeError('Adapted codec training receipt differs: ' + arm)
        state = payload['model']
        snapshot = encoder_snapshot(state)
        digest = snapshot_digest(snapshot)
        if digest != evaluations[arm]['encoder_sha256']:
            raise RuntimeError('Checkpoint encoder digest differs: ' + arm)
        snapshots[arm] = snapshot
        models[arm] = {'checkpoint_sha256': expected_sha, 'encoder_sha256': digest,
                       'encoder_tensor_count': len(snapshot)}
        del payload, state
    for arm in ('decoder_only', 'full_codec'):
        left, right = snapshots['baseline'], snapshots[arm]
        models[arm]['changed_encoder_tensors'] = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
    if models['decoder_only']['changed_encoder_tensors']:
        raise RuntimeError('Decoder-only checkpoint actually changed an encoder-side tensor')

    results = {}
    for clip in TRAIN:
        frame_results = {'decoder_only': [], 'full_codec': []}
        for frame in range(177):
            states = {}
            source_sha = None
            for arm in ARMS:
                row = state_rows[arm][(clip, frame)]
                metric = metric_rows[arm][(clip, frame)]
                if (row.get('split') != 'TRAIN' or row.get('checkpoint_sha256') != CHECKPOINTS[arm][1]
                        or row.get('encoder_sha256') != evaluations[arm]['encoder_sha256']
                        or metric.get('checkpoint_sha256') != CHECKPOINTS[arm][1]):
                    raise RuntimeError('State or metric provenance differs')
                source_sha = row['source_cache_sha256'] if source_sha is None else source_sha
                if row['source_cache_sha256'] != source_sha:
                    raise RuntimeError('Variants do not use the same source cache')
                path = EXPERIMENT / 'evaluations' / arm / row['state_path']
                if path.stat().st_size != row['state_bytes'] or sha(path) != row['state_sha256']:
                    raise RuntimeError('State artifact changed')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                states[arm] = (payload['coordinates'].numpy(), payload['features'].numpy())
            for arm in ('decoder_only', 'full_codec'):
                frame_results[arm].append(compare_states(*states['baseline'], *states[arm]))
        source = {}
        for arm, values in frame_results.items():
            source[arm] = {'frames': len(values),
                'coordinate_set_mismatch_frames': sum(not x['same_coordinate_set'] for x in values),
                'raw_order_changed_frames': sum(not x['same_raw_row_order'] for x in values),
                'feature_mean_abs_frame_mean': statistics.fmean(x['feature_mean_abs'] for x in values),
                'feature_max_abs_over_frames': max(x['feature_max_abs'] for x in values)}
        source['codec_real_mae'] = {}
        for arm in ARMS:
            values = [float(metric_rows[arm][(clip, f)]['codec_real']['mae']) for f in range(177)]
            mean = statistics.fmean(values)
            if abs(mean - evaluations[arm]['summary_by_clip'][clip]['codec_real']['mae']) > 1e-10:
                raise RuntimeError('Per-frame codec mean differs from historic summary')
            source['codec_real_mae'][arm] = mean
        results[clip] = source

    output = ROOT / 'visuals'
    output.mkdir()
    visual_rows = []
    for clip in TRAIN:
        for kind, reference_kind in (('real_view', 'real'), ('nearby_view', 'nearby_teacher')):
            canvas = Image.new('RGB', (1792, 1260))
            inputs = []
            for index, frame in enumerate(FRAMES):
                reference_entry = screenshot_entries['baseline'][(clip, frame, reference_kind)]
                reference = load_image(EXPERIMENT / 'evaluations/baseline', reference_entry)
                canvas.paste(reference, (0, index * 252))
                inputs.append({'frame': frame, 'column': 'reference', 'sha256': reference_entry['sha256']})
                for column, arm in enumerate(ARMS, 1):
                    entry = screenshot_entries[arm][(clip, frame, 'codec' if kind == 'real_view' else 'nearby')]
                    image = load_image(EXPERIMENT / 'evaluations' / arm, entry)
                    canvas.paste(image, (column * 448, index * 252))
                    inputs.append({'frame': frame, 'column': arm, 'sha256': entry['sha256']})
            target = output / f'{clip}_{kind}.png'
            canvas.save(target)
            visual_rows.append({'clip': clip, 'kind': kind, 'file': target.name, 'sha256': sha(target), 'inputs': inputs})
    result = {'status': 'complete', 'train_sources': results, 'models': models,
              'visuals': visual_rows, 'evaluation_source_records_selected': False,
              'state_rows_per_train_source_per_arm': 177, 'checkpoint_tensor_comparison': 'exact',
              'feature_comparison': 'coordinate-aligned; raw sparse row order is not correspondence',
              'historic_summary_sha256': sha(EXPERIMENT / 'SUMMARY.json'),
              'protocol_sha256': PINS['protocol.json'], 'data_binding_sha256': PINS['DATA_BINDING.json'],
              'source_code_sha256': sha(ROOT / 'CODE.sha256'), 'model_training_performed': False,
              'scope': 'Existing codec reconstruction on two TRAIN videos only; no action or pristine-test claim'}
    with (ROOT / 'CODEC_REUSE_AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: actual checkpoint tensors, 354 aligned train states per arm, and fixed visual contacts')


if __name__ == '__main__':
    main()
