"""Verify existing teacher assets against every selected baseline-state source.

Runs on CPU; it does not create or modify teachers, choose action codes or train.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import torch
from codec_binding import BaselineStates, CLIP, checked_bytes, require, sha

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
STATE_ROOT = BASE/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
TEACHER_ROOT = BASE/'gaussiangpt_neoverse_gs_supervision_20261031_v10b'
NATIVE_ROOT = BASE/'neoverse_tower177_fps30_20261029_v1/native448/gaussians'
ASSETS_SHA = 'de1c59b6f3d4658af9a17a3ded8fed39fda9c2f199f69f99255dcdad653f42a6'
MASKS_SHA = '6ac20be3dc866c8c276b279bba0338b2a249b25215847161bedc93af9ce999be'
PREPARE_SHA = '7b63c8c50eb9a3e7f3e2f4b55576894f272c6c0f72de3502b6b7636c56dd702a'
RENDERER_SHA = '38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'
TEACHER_PROTOCOL_SHA = 'da372231cd1c44520bf5d3d390019c32092e31e09fd28873a0671eda9d48f8a5'


def validate_raw(raw, frame, state_row):
    require(isinstance(raw, dict) and raw.get('clip') == CLIP and raw.get('frame') == frame
            and raw.get('normalization') == state_row['normalization'] and raw.get('n_input') == 112896,
            'Raw frame identity/normalization/source count')
    require(raw.get('metrics', {}).get('pts_seconds') == state_row['pts'], 'Raw/state PTS mismatch')
    require(type(raw.get('near')) is float and type(raw.get('far')) is float
            and 0 < raw['near'] < raw['far'] and math.isfinite(raw['far']), 'Raw camera clipping')
    shapes = {'cameras': (2,4,4), 'intrinsics': (4,), 'targets': (2,252,448,3), 'real': (252,448,3)}
    for key, shape in shapes.items():
        value = raw.get(key)
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), 'Invalid raw '+key)
    counts = raw.get('counts')
    require(isinstance(counts, torch.Tensor) and int(counts.sum()) == 112896 and bool((counts > 0).all()),
            'Raw source GS were not fully aggregated')
    return raw


class BoundTeachers:
    def __init__(self, states, *, state_root=STATE_ROOT, teacher_root=TEACHER_ROOT, native_root=NATIVE_ROOT):
        require(isinstance(states, BaselineStates), 'The selected baseline state binding is required')
        self.states = states
        self.state_root, self.teacher_root, self.native_root = map(Path, (state_root, teacher_root, native_root))
        assets_raw = checked_bytes(self.teacher_root/'TEACHER_ASSETS.json', ASSETS_SHA)
        self.assets = json.loads(assets_raw)
        a = self.assets
        require(a['status'] == 'complete' and a['frames'] == 177 and a['source_points_per_frame'] == 112896
                and a['teachers_sha256'] == MASKS_SHA and a['protocol_sha256'] == TEACHER_PROTOCOL_SHA
                and a['alpha_threshold'] == .9 and a['erosion_kernel'] == 3
                and a['model_updates'] == 0 and a['prediction_dependent_mask'] is False,
                'Teacher asset receipt mismatch')
        checked_bytes(self.teacher_root/'protocol.json', TEACHER_PROTOCOL_SHA)
        prep_raw = checked_bytes(self.state_root/'prepare'/f'{CLIP}.json', PREPARE_SHA)
        checked_bytes(self.state_root/'code/common.py', RENDERER_SHA)
        self.preparation = json.loads(prep_raw)
        require(len(a['records']) == len(self.preparation['rows']) == 177, 'Teacher receipt coverage')
        masks_data = checked_bytes(self.teacher_root/'teachers.pt', MASKS_SHA)
        loaded = torch.load(io.BytesIO(masks_data), map_location='cpu', weights_only=True)
        require(set(loaded) == {'masks'} and loaded['masks'].dtype == torch.bool
                and tuple(loaded['masks'].shape) == (177,252,448), 'Teacher mask shape/dtype')
        require(bool(loaded['masks'].flatten(1).any(dim=1).all()), 'Empty teacher valid region')
        self.masks = loaded['masks']

    def load_raw(self, frame):
        require(type(frame) is int and 0 <= frame < 177, 'Raw frame outside 0..176')
        state_row, teacher, prepared = self.states.rows[frame], self.assets['records'][frame], self.preparation['rows'][frame]
        raw_path = self.state_root/'cache'/CLIP/f'{frame:03d}.pt'
        native_path = self.native_root/f'native_group_{frame:03d}.npz'
        require(teacher['frame'] == frame and prepared['frame'] == frame
                and teacher['cache_path'] == str(raw_path) and teacher['source_path'] == str(native_path)
                and teacher['cache_sha256'] == prepared['cache_sha256'] == state_row['source_cache_sha256']
                and teacher['source_sha256'] == prepared['native_gs_sha256']
                and prepared['source_count'] == 112896, 'Teacher/state/raw GS identity mismatch')
        require(teacher['mae'] <= 1e-5 and teacher['psnr'] >= 60, 'Prior raw GS teacher RGB parity failed')
        content = checked_bytes(raw_path, teacher['cache_sha256'])
        raw = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
        return validate_raw(raw, frame, state_row)

    def verify_all(self):
        for frame in range(177):
            self.load_raw(frame)
            entry = self.assets['records'][frame]
            native = self.native_root/f'native_group_{frame:03d}.npz'
            require(sha(native) == entry['source_sha256'], 'Native teacher GS bytes changed')
        return dict(clip=CLIP, frames=177, native_source_gs_per_frame=112896,
                    teacher_assets_sha256=ASSETS_SHA, teacher_masks_sha256=MASKS_SHA,
                    preparation_sha256=PREPARE_SHA, renderer_sha256=RENDERER_SHA,
                    raw_state_cache_identity_verified=True, native_source_files_verified=True,
                    masks_are_prediction_independent=True, model_updated=False, training_ready=False)

    def load_record(self, frame, device='cpu'):
        raw = self.load_raw(frame)
        device = torch.device(device)
        record = {key: raw[key] for key in ('real','cameras','intrinsics','near','far')}
        record = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in record.items()}
        return dict(record=record, teacher_targets=raw['targets'].to(device), mask=self.masks[frame].to(device))
