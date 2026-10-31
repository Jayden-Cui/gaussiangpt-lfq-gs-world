"""CPU-only planning, actual-action packets and evaluation control mappings.

No codec/model import, admission inference, training, budget freeze or GPU call.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random

import torch

MODES = ('continuous16', 'lfq16')
CODE_CONTROLS = ('correct', 'fixed_bundle', 'fixed_single', 'shuffle42', 'shuffle84', 'shuffle126')
SOURCE_KEYS = ('model_checkpoint_sha256', 'codec_binding_sha256', 'state_manifest_sha256', 'adapter_sha256')
CANDIDATE_PLAN_SHA = 'd80f965a9fbd7c4906327691e0a671488b68d5c8d32f62a8842107163c58c984'
PACKET_SCHEMA = 'z_ablation_actual_actions_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _valid_digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _starts(values, *, complete=False):
    require(isinstance(values, (list, tuple)) and values, 'Nonempty explicit window starts required')
    require(all(type(s) is int and 0 <= s < 168 for s in values), 'Window starts must be integers in 0..167')
    require(len(set(values)) == len(values), 'Duplicate window starts')
    if complete:
        require(set(values) == set(range(168)), 'All 168 evaluation starts required')
    return list(values)


def plan_hash(plan):
    return _sha_bytes(_json_bytes(plan))


def candidate_plan():
    """Exact v7 seed+17/B4 schedule, explicitly an unfrozen 800-step candidate."""
    rng, starts = random.Random(20261031 + 17), []
    while len(starts) < 800 * 4:
        cycle = list(range(168))
        rng.shuffle(cycle)
        starts.extend(cycle)
    batches = [starts[i:i+4] for i in range(0, 800*4, 4)]
    require(plan_hash(batches) == CANDIDATE_PLAN_SHA, 'Candidate sampler changed')
    counts = Counter(s for batch in batches for s in batch)
    return dict(status='candidate', budget_frozen=False, seed=20261031, sampler_seed=20261031+17,
                steps=800, batch_size=4, frames_per_window=10, batches=batches,
                plan_sha256=CANDIDATE_PLAN_SHA, window_exposures=3200, frame_exposures=32000,
                future_exposures=28800, full_cycles=19, remaining_updates=2,
                exposures_per_start={str(s): counts[s] for s in range(168)})


def _source(source):
    require(isinstance(source, dict) and all(key in source for key in SOURCE_KEYS), 'Required source identities missing')
    require(all(_valid_digest(source[key]) for key in SOURCE_KEYS), 'Source SHA256 must be lowercase hex')
    require(all(type(key) is str for key in source), 'Source metadata keys must be strings')
    # Detach JSON metadata from caller ownership, while rejecting NaN and objects.
    try:
        encoded = _json_bytes(source)
        clean = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError('Source metadata must be finite JSON data') from error
    require(clean == source, 'Source metadata must round-trip without type changes')
    return clean


def _codes(codes, mode):
    require(mode in MODES, 'Unknown action mode')
    require(isinstance(codes, torch.Tensor) and codes.device.type == 'cpu', 'Actual codes must be CPU tensors')
    require(codes.dtype == torch.float32 and codes.ndim == 3 and codes.shape[0] > 0
            and tuple(codes.shape[1:]) == (9, 16), 'Actual codes must be float32 [N,9,16]')
    require(bool(torch.isfinite(codes).all()), 'Actual codes must be finite')
    if mode == 'lfq16':
        require(bool(((codes == -1) | (codes == 1)).all()), 'LFQ actual codes must be binary signs')
    return codes


def tensor_digest(tensor):
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu', 'Digest requires a CPU tensor')
    value = tensor.detach().contiguous()
    header = _json_bytes(dict(dtype=str(value.dtype), shape=list(value.shape)))
    return _sha_bytes(header + b'\n' + value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _packet_digest(packet):
    return _sha_bytes(_json_bytes({key: value for key, value in packet.items()
                                  if key not in ('codes', 'packet_sha256')}))


def make_action_packet(codes, *, mode, starts, s_ref, source):
    """Wrap actual float codes; no diagnostic ID is accepted as an action."""
    _codes(codes, mode)
    starts = _starts(starts)
    require(len(starts) == len(codes), 'Action rows and explicit starts differ')
    require(type(s_ref) in (float, int) and math.isfinite(s_ref) and s_ref > 1e-6, 'Invalid fixed action scale')
    source = _source(source)
    actual = codes.detach().contiguous().clone()
    packet = dict(schema=PACKET_SCHEMA, action_mode=mode, starts=starts, horizons=list(range(1, 10)),
                  code_width=16, s_ref=float(s_ref), source=source, source_sha256=_sha_bytes(_json_bytes(source)),
                  codes=actual, codes_sha256=tensor_digest(actual), codec_admission_verified=False)
    packet['packet_sha256'] = _packet_digest(packet)
    return packet


def validate_action_packet(packet, *, expected_mode, expected_source, expected_s_ref, expected_starts=None):
    require(isinstance(packet, dict), 'Action packet must be a dictionary')
    expected_keys = {'schema', 'action_mode', 'starts', 'horizons', 'code_width', 's_ref', 'source',
                     'source_sha256', 'codes', 'codes_sha256', 'codec_admission_verified', 'packet_sha256'}
    require(set(packet) == expected_keys and packet['schema'] == PACKET_SCHEMA, 'Action packet schema differs')
    require(expected_mode in MODES and packet['action_mode'] == expected_mode, 'Action mode mismatch')
    _codes(packet['codes'], expected_mode)
    starts = _starts(packet['starts'])
    require(len(starts) == len(packet['codes']) and packet['horizons'] == list(range(1, 10))
            and packet['code_width'] == 16, 'Action axes differ')
    if expected_starts is not None:
        require(starts == _starts(expected_starts), 'Action start order mismatch')
    require(type(expected_s_ref) in (float, int) and math.isfinite(expected_s_ref) and expected_s_ref > 1e-6
            and type(packet['s_ref']) is float and packet['s_ref'] == float(expected_s_ref), 'Action scale mismatch')
    expected = _source(expected_source)
    require(packet['source'] == expected and packet['source_sha256'] == _sha_bytes(_json_bytes(expected)),
            'Action source mismatch')
    require(packet['codes_sha256'] == tensor_digest(packet['codes'])
            and packet['packet_sha256'] == _packet_digest(packet), 'Action packet content digest mismatch')
    require(packet['codec_admission_verified'] is False, 'CPU packet validation cannot assert codec admission')
    return packet


def write_action_packet(path, packet):
    validate_action_packet(packet, expected_mode=packet['action_mode'], expected_source=packet['source'],
                           expected_s_ref=packet['s_ref'])
    stream = io.BytesIO()
    torch.save(packet, stream)
    payload = stream.getvalue()
    path = Path(path)
    with path.open('xb') as handle:
        handle.write(payload)
    return dict(path=str(path), bytes=len(payload), file_sha256=_sha_bytes(payload),
                packet_sha256=packet['packet_sha256'])


def read_action_packet(path, *, expected_file_sha256, expected_mode, expected_source, expected_s_ref,
                       expected_starts=None):
    require(_valid_digest(expected_file_sha256), 'An explicit packet file SHA256 is required')
    payload = Path(path).read_bytes()
    require(_sha_bytes(payload) == expected_file_sha256, 'Action file SHA256 mismatch')
    packet = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    return validate_action_packet(packet, expected_mode=expected_mode, expected_source=expected_source,
                                  expected_s_ref=expected_s_ref, expected_starts=expected_starts)


def control_mapping(control, recipient_starts):
    require(control in CODE_CONTROLS, 'Control requires actual-code mapping, not posterior/held/target generation')
    starts = _starts(recipient_starts)
    rows = []
    for start in starts:
        donor = start
        if control.startswith('shuffle'):
            donor = (start + int(control.removeprefix('shuffle'))) % 168
        elif control in ('fixed_bundle', 'fixed_single'):
            donor = 0
        rows.append(dict(control=control, recipient_start=start, recipient_frames=list(range(start, start+10)),
                         donor_start=donor, donor_horizons=[1]*9 if control == 'fixed_single' else list(range(1, 10))))
    return rows


def select_control_codes(packet, control, recipient_starts, *, expected_mode, expected_source, expected_s_ref):
    validate_action_packet(packet, expected_mode=expected_mode, expected_source=expected_source,
                           expected_s_ref=expected_s_ref)
    mapping = control_mapping(control, recipient_starts)
    index = {start: i for i, start in enumerate(packet['starts'])}
    require(all(row['donor_start'] in index for row in mapping), 'Required donor absent; no implicit padding')
    selected = []
    for row in mapping:
        bundle = packet['codes'][index[row['donor_start']]]
        selected.append(bundle[torch.tensor(row['donor_horizons'], dtype=torch.long) - 1])
    return dict(codes=torch.stack(selected), mapping=mapping, action_mode=expected_mode,
                source_packet_sha256=packet['packet_sha256'])


def _time(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and value.ndim == 0 and value.dtype != torch.bool
                and not value.is_complex(), 'PTS must be a real CPU scalar')
        numeric = float(value)
    else:
        require(type(value) in (int, float), 'PTS must be a real scalar')
        numeric = float(value)
    require(math.isfinite(numeric), 'PTS must be finite')
    return numeric


def blind_window(window):
    """Synthetic control: copy only S0 geometry/features; retain recipient times.

    Unknown fields are intentionally excluded so future geometry cannot hide in
    nested metadata. Outputs are explicitly controls, never relabeled real caches.
    """
    require(isinstance(window, (list, tuple)) and len(window) == 10, 'Blinding requires a full T10 window')
    first = window[0]
    require(isinstance(first, dict), 'State must be a dictionary')
    coordinates, features = first['coordinates'], first['features']
    require(isinstance(coordinates, torch.Tensor) and coordinates.device.type == 'cpu'
            and coordinates.dtype in (torch.int32, torch.int64) and coordinates.ndim == 2
            and coordinates.shape[1] == 3 and len(coordinates) > 0, 'Initial coordinates must be CPU integer [M,3]')
    require(len(torch.unique(coordinates, dim=0)) == len(coordinates), 'Duplicate initial coordinates')
    require(isinstance(features, torch.Tensor) and features.device.type == 'cpu' and features.dtype == torch.float32
            and tuple(features.shape) == (len(coordinates), 64) and bool(torch.isfinite(features).all()),
            'Initial features must be finite CPU float32 [M,64]')
    start = first['frame_index']
    require(type(start) is int and 0 <= start <= 167 and type(first['clip']) is str and first['clip'],
            'Invalid initial frame identity')
    require(type(first['stride']) is int and first['stride'] > 0
            and type(first['grid']) in (float, int) and math.isfinite(first['grid']) and first['grid'] > 0,
            'Invalid initial grid/stride')
    previous, blinded = float('-inf'), []
    for horizon, state in enumerate(window):
        require(isinstance(state, dict) and type(state['frame_index']) is int
                and state['frame_index'] == start+horizon and state['clip'] == first['clip'],
                'Recipient window frame identity differs')
        require(state['stride'] == first['stride'] and state['grid'] == first['grid'], 'Recipient grid/stride differs')
        pts = _time(state['pts'])
        require(pts > previous, 'Recipient PTS must be strictly increasing')
        previous = pts
        result = dict(coordinates=coordinates.detach().clone(), features=features.detach().clone(),
                      pts=copy.deepcopy(state['pts']), frame_index=state['frame_index'], clip=state['clip'],
                      stride=first['stride'], grid=first['grid'], blinded_posterior=True,
                      geometry_source_frame_index=start, state_semantics='synthetic_initial_repeated')
        if 'source_pts' in state:
            _time(state['source_pts'])
            result['source_pts'] = copy.deepcopy(state['source_pts'])
        blinded.append(result)
    return blinded


def evaluation_coverage(starts=None):
    """Count exposure and distinct physical frame identities, without dropping overlaps."""
    starts = _starts(list(range(168)) if starts is None else starts, complete=True)
    rows = [dict(start=start, horizon=h, frame=start+h) for start in starts for h in range(1, 10)]
    frames = Counter(row['frame'] for row in rows)
    horizons = Counter(row['horizon'] for row in rows)
    require(len(rows) == 1512 and set(frames) == set(range(1, 177))
            and all(horizons[h] == 168 for h in range(1, 10)), 'Evaluation coverage differs')
    return dict(starts=starts, future_rows=rows, future_exposures=len(rows), physical_future_frame_count=len(frames),
                physical_future_frames=sorted(frames), exposures_per_physical_frame=dict(sorted(frames.items())),
                exposures_per_horizon=dict(sorted(horizons.items())), initial_exposures=168,
                initial_frames=sorted(starts), all_physical_frames=list(range(177)),
                reduction_rule='average overlapping-window observations within each physical frame first')
