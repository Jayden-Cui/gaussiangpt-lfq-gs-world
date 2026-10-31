"""One real-state native bridge forward/backward witness; no optimizer update."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch
from codec_binding import BaselineStates, checked_bytes, require, sha, verify_native_replay
from teacher_binding import BoundTeachers
from runtime_bridge import BoundNativeBridge

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
ARM = BASE/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline'
NATIVE = BASE/'gaussiangpt_neoverse_native_codec_replay_20261031_v1/results'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    root = Path(__file__).resolve().parent
    out = root/'bridge_smoke'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    write(out/'STARTED.json',dict(pid=os.getpid(), frame=44, optimizer_updates=0))
    try:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one GPU required')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        free,total = torch.cuda.mem_get_info()
        require(free >= 20 * 2**30, 'Insufficient GPU memory; preserve other processes')
        torch.cuda.set_per_process_memory_fraction(12 * 2**30 / total)
        torch.cuda.reset_peak_memory_stats()
        states = BaselineStates(ARM)
        replay = verify_native_replay(states,NATIVE)
        teachers = BoundTeachers(states)
        raw = teachers.load_raw(44)
        state = states.load(44,device='cuda')
        state['features'] = state['features'].detach().clone().requires_grad_(True)
        record = {k:(raw[k].cuda() if torch.is_tensor(raw[k]) else raw[k])
                  for k in ('real','cameras','intrinsics','near','far')}
        rows = [json.loads(line) for line in (NATIVE/'PACKETS.jsonl').read_text().splitlines()]
        reference_info = rows[44]['reference']
        ref = torch.load(io.BytesIO(checked_bytes(NATIVE/'references/044.pt', reference_info['sha256'],
                                                  reference_info['bytes'])), map_location='cpu',weights_only=True).cuda()
        bridge = BoundNativeBridge(partial=True,device='cuda').train(True)
        bridge.assert_frozen()
        images = bridge.render_views(state,record)
        delta = images.detach()-ref
        mae = [float(delta[v].abs().mean()) for v in range(2)]
        psnr = [float(-10*torch.log10(delta[v].square().mean().clamp_min(1e-12))) for v in range(2)]
        require(all(v <= 1e-5 for v in mae) and all(v >= 60 for v in psnr), 'Bridge vs accepted GS reference failed')
        target = raw['targets'][1].cuda()
        mask = teachers.masks[44].cuda()
        loss = (images[0]-record['real']).abs().mean() + .25*(images[1]-target).abs().mean(-1)[mask].mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite real native loss')
        loss.backward()
        latent_grad = state['features'].grad
        require(latent_grad is not None and bool(torch.isfinite(latent_grad).all())
                and float(latent_grad.norm()) > 0, 'No finite nonzero latent gradient')
        groups = {}
        for group, module in (('decoder_out',bridge.codec.ae.decoder.out),('feature_heads',bridge.codec.ae.feature_heads),
                              ('multiplier',bridge.codec.ae.gaussian_multiplier)):
            grads = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
            groups[group] = float(torch.stack(grads).sum().sqrt()) if grads else 0.
        require(all(math.isfinite(value) and value >= 0 for value in groups.values()) and sum(groups.values()) > 0,
                'No native trainable gradient')
        bridge.assert_frozen()
        require(torch.cuda.max_memory_reserved() <= 12 * 2**30, 'GPU allocator budget exceeded')
        write(out/'RESULT.json',dict(status='passed',pid=os.getpid(),clip='buildingblock_tower_17_176',frame=44,
            source_state_sha256=states.rows[44]['state_sha256'], checkpoint_sha256=states.rows[44]['checkpoint_sha256'],
            native_replay_sha256=replay['native_replay_sha256'], bridge_source_sha256=sha(root/'runtime_bridge.py'),
            reference_sha256=reference_info['sha256'], real_view_and_nearby_mae=mae,
            real_view_and_nearby_psnr=psnr, decoded_gs_count=bridge.last_count,
            latent_gradient_norm=float(latent_grad.norm()), native_gradient_norms=groups,
            decoder_precision='float64',encoder_state_dtype='float32',codec_remained_eval=True,
            frozen_parameters_and_buffers_unchanged=True, optimizer_updates=0, formal_training=False,
            elapsed_seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
