"""Strict CPU-readable native GS packet validation, independent of the codec."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_checked(path, size, digest):
    content = Path(path).read_bytes()
    require(len(content) == size and sha_bytes(content) == digest, 'File bytes/SHA mismatch')
    return content


def finite_tensor(value, shape):
    return (isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == torch.float32 and tuple(value.shape) == tuple(shape)
            and bool(torch.isfinite(value).all()))


def validate_packet(packet, frame, checkpoint_sha, state_sha):
    require(isinstance(packet, dict) and packet.get('schema') == 'native_codec_gs_packet_v1', 'Packet schema')
    require(packet.get('clip') == 'buildingblock_tower_17_176' and packet.get('frame') == frame,
            'Packet frame identity')
    require(packet.get('checkpoint_sha256') == checkpoint_sha and packet.get('state_sha256') == state_sha,
            'Packet codec/state identity')
    gs = packet.get('gs')
    require(isinstance(gs, torch.Tensor) and gs.ndim == 2 and 0 < len(gs) <= 2400000
            and finite_tensor(gs, (len(gs), 14)), 'GS must be finite full float32 [N,14]')
    require(finite_tensor(packet.get('sh'), (len(gs), 4, 3)), 'SH must be finite float32 [N,4,3]')
    require(finite_tensor(packet.get('cameras'), (2, 4, 4)), 'Two full cameras required')
    require(finite_tensor(packet.get('intrinsics'), (4,)), 'Intrinsics required')
    for key in ('near', 'far', 'pts'):
        require(type(packet.get(key)) is float and math.isfinite(packet[key]), key + ' must be finite float')
    require(0 < packet['near'] < packet['far'], 'Camera clipping planes')
    require(packet.get('resolution_hw') == [252, 448] and isinstance(packet.get('normalization'), dict),
            'Resolution/normalization')
    require(packet.get('source_cache_sha256') and packet.get('decoded_from_actual_state') is True,
            'Packet source missing')
    return packet


def load_packet(content, frame, checkpoint_sha, state_sha):
    packet = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    return validate_packet(packet, frame, checkpoint_sha, state_sha)


def write_new_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def save_new(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=sha(path))
