"""Bind the user-selected baseline to real state packets and native replay evidence.

This module only validates codec/state provenance and exposes state input views.
It does not construct a model, renderer, teacher data, optimizer or training run.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import torch

CLIP = 'buildingblock_tower_17_176'
CHECKPOINT_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
ENCODER_SHA = 'c2ddb2b1dd5a43ee4a86286ea09ef141f61661963fa1d5979edf3d4f5917737b'
DECODER_SHA = '3723b9c8eec37952c14739133b9edc35f78d3926449b705e32d0d6a9f3a02c97'
EVALUATION_SHA = '56d5fe877518be1285c8fe02552cfe85eee8c41f0ce0f044489cc84a82a35aa5'
MANIFEST_SHA = '6982106649970254057d463e1213c36c1257ed1e2e202c77af830587559d0e54'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_bytes(path, expected_sha, expected_size=None):
    content = Path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_sha, 'Source bytes SHA mismatch: ' + str(path))
    if expected_size is not None:
        require(len(content) == expected_size, 'Source bytes size mismatch: ' + str(path))
    return content


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def validate_state_view(state, row):
    """Return a model view without relabeling, copying or mutating saved tensors."""
    require(isinstance(state, dict) and state.get('schema') == 'multisource_continuous_encoder_state_v1', 'State schema')
    frame = row['frame']
    meta = state.get('metadata', {})
    require(type(frame) is int and 0 <= frame < 177 and row.get('clip') == CLIP
            and meta.get('clip') == CLIP and meta.get('frame') == frame and meta.get('arm') == 'baseline',
            'State clip/frame/arm mismatch')
    require(row.get('checkpoint_sha256') == CHECKPOINT_SHA and meta.get('checkpoint_sha256') == CHECKPOINT_SHA
            and row.get('encoder_sha256') == ENCODER_SHA and meta.get('encoder_sha256') == ENCODER_SHA
            and row.get('decoder_sha256') == DECODER_SHA and meta.get('decoder_sha256') == DECODER_SHA,
            'Encoder/decoder/checkpoint identity mismatch')
    require(row.get('source_cache_sha256') == meta.get('source_cache_sha256')
            and row.get('normalization') == meta.get('normalization'), 'Raw cache/normalization identity mismatch')
    pts = meta.get('pts')
    require(type(pts) in (int, float) and math.isfinite(pts) and pts == row.get('pts'), 'State PTS mismatch')
    require(state.get('stride') == 4 and state.get('grid') == .008, 'State grid/stride mismatch')
    coords, features = state.get('coordinates'), state.get('features')
    require(isinstance(coords, torch.Tensor) and coords.device.type == 'cpu' and coords.ndim == 2
            and coords.shape[1] == 3 and coords.dtype in (torch.int32, torch.int64) and len(coords) > 0
            and len(torch.unique(coords, dim=0)) == len(coords), 'Invalid/duplicate state coordinates')
    require(isinstance(features, torch.Tensor) and features.device.type == 'cpu' and features.dtype == torch.float32
            and tuple(features.shape) == (len(coords), 64) and bool(torch.isfinite(features).all())
            and row.get('shape') == list(features.shape), 'Invalid state features')
    require(row.get('replay_passed') is True and row.get('cold_manager_replay_verified') is True,
            'State latent replay was not verified')
    return dict(coordinates=coords, features=features, stride=4, grid=.008, pts=pts,
                frame_index=frame, clip=CLIP, state_sha256=row['state_sha256'])


class BaselineStates:
    """Complete exact 177-frame source; this alone does not authorize training."""
    def __init__(self, arm_root):
        self.root = Path(arm_root).resolve()
        checked_bytes(self.root/'EVALUATION.json', EVALUATION_SHA)
        checked_bytes(self.root/'state_manifest.jsonl', MANIFEST_SHA)
        evaluation = _json(self.root/'EVALUATION.json')
        require(evaluation['arm'] == 'baseline' and evaluation['checkpoint_sha256'] == CHECKPOINT_SHA
                and evaluation['encoder_sha256'] == ENCODER_SHA and evaluation['decoder_sha256'] == DECODER_SHA
                and evaluation['frames'] == 9027 and evaluation['replay_failed_frames'] == 0
                and evaluation['cold_manager_replay_verified_all'] is True, 'Full evaluation identity mismatch')
        rows = [row for row in _jsonl(self.root/'state_manifest.jsonl') if row.get('clip') == CLIP]
        rows.sort(key=lambda row: row['frame'])
        require([row['frame'] for row in rows] == list(range(177)), '177 unique states required')
        for row in rows:
            expected = f'states/{CLIP}/{row["frame"]:03d}.pt'
            require(row['state_path'] == expected, 'State path mismatch')
        self.rows = rows

    def load(self, frame, device='cpu'):
        require(type(frame) is int and 0 <= frame < 177, 'Frame outside 0..176')
        row = self.rows[frame]
        raw = checked_bytes(self.root/row['state_path'], row['state_sha256'], row['state_bytes'])
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        view = validate_state_view(state, row)
        device = torch.device(device)
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in view.items()}

    def verify_all(self):
        times = []
        for frame in range(177):
            view = self.load(frame)
            times.append(float(view['pts']))
        require(all(b > a for a, b in zip(times, times[1:])), '177 source times must be strictly increasing')
        require(all(row['normalization'] == self.rows[0]['normalization'] for row in self.rows),
                'Normalization changed across the sequence')
        return dict(clip=CLIP, frames=177, state_manifest_sha256=MANIFEST_SHA,
                    checkpoint_sha256=CHECKPOINT_SHA, encoder_sha256=ENCODER_SHA, decoder_sha256=DECODER_SHA,
                    first_pts=times[0], last_pts=times[-1], frame_gap=1,
                    state_values_verified=True, state_views_do_not_mutate_packets=True,
                    training_ready=False)

    def window(self, start, device='cpu'):
        require(type(start) is int and 0 <= start < 168, 'Complete adjacent T10 window required')
        states = [self.load(frame, device=device) for frame in range(start, start+10)]
        require(all(float(a['pts']) < float(b['pts']) for a, b in zip(states, states[1:])), 'Window PTS order')
        return states


def verify_native_replay(states, native_results_root):
    """Validate independent replay receipts and actual packet/reference bytes.

    This is a codec-state evidence gate, not a teacher/data/optimizer admission.
    """
    require(isinstance(states, BaselineStates), 'Bound baseline states required')
    root = Path(native_results_root).resolve()
    result, produced = _json(root/'RESULT.json'), _json(root/'PRODUCED.json')
    require(result.get('status') == 'complete' and result.get('replay_passed_all') is True
            and result.get('separate_process') is True and result.get('encoder_or_codec_loaded') is False
            and result.get('frames') == 177 and result.get('views') == 354 and result.get('failed_frames') == []
            and result.get('checkpoint_sha256') == CHECKPOINT_SHA
            and result.get('thresholds') == {'mae_max': 1e-5, 'psnr_min': 60}, 'Native replay did not pass')
    require(produced.get('status') == 'complete' and produced.get('frames') == 177
            and produced.get('clip') == CLIP and produced.get('checkpoint_sha256') == CHECKPOINT_SHA
            and produced.get('encoder_sha256') == ENCODER_SHA and produced.get('state_manifest_sha256') == MANIFEST_SHA
            and produced.get('evaluation_sha256') == EVALUATION_SHA and produced.get('model_updated') is False
            and result.get('producer_pid') == produced.get('pid') and result.get('replay_pid') != produced.get('pid')
            and result.get('produced_sha256') == sha(root/'PRODUCED.json'), 'Native producer identity differs')
    manifest_path, replay_path = root/'PACKETS.jsonl', root/'REPLAY.jsonl'
    require(sha(manifest_path) == produced.get('packets_manifest_sha256') == result.get('packets_manifest_sha256')
            and sha(replay_path) == result.get('replay_rows_sha256'), 'Native replay rows changed')
    packets, records = _jsonl(manifest_path), _jsonl(replay_path)
    require([row.get('frame') for row in packets] == list(range(177))
            and [row.get('frame') for row in records] == list(range(177)), 'Native replay coverage differs')
    all_views = []
    for frame, (packet, record, state) in enumerate(zip(packets, records, states.rows)):
        require(packet['state_sha256'] == state['state_sha256']
                and packet['source_cache_sha256'] == state['source_cache_sha256']
                and packet['pts'] == state['pts'] and record['pts'] == packet['pts']
                and record['gs_count'] == packet['gs_count'] and record['passed'] is True,
                'Native packet/source frame mismatch')
        for dirname, key in (('packets', 'packet'), ('references', 'reference')):
            entry = packet[key]
            checked_bytes(root/dirname/f'{frame:03d}.pt', entry['sha256'], entry['bytes'])
        views = record['views']
        require([v.get('view') for v in views] == [0, 1], 'Both replay views required')
        for view in views:
            require(view.get('passed') is True and type(view.get('mae')) in (int, float)
                    and type(view.get('psnr')) in (int, float)
                    and math.isfinite(view['mae']) and math.isfinite(view['psnr'])
                    and view['mae'] <= 1e-5 and view['psnr'] >= 60, 'Replay threshold failure')
            all_views.append(view)
    require(result['max_mae'] == max(v['mae'] for v in all_views)
            and result['min_psnr'] == min(v['psnr'] for v in all_views), 'Replay summary differs from rows')
    return dict(native_replay_sha256=sha(root/'RESULT.json'), produced_sha256=sha(root/'PRODUCED.json'),
                passed=True, frames=177, views=354, selected_checkpoint_sha256=CHECKPOINT_SHA,
                teacher_data_verified=False, training_ready=False)
