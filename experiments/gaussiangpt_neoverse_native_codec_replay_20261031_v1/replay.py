"""Independent process: only saved native packets, references and renderer are loaded."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from native_packet import require, sha, read_checked, load_packet, write_new_json

COMMON = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/common.py')
COMMON_SHA = '38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'
CHECKPOINT = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'


def main():
    root = Path(__file__).resolve().parent
    out = root / 'results'
    started = time.monotonic()
    try:
        produced = json.loads((out / 'PRODUCED.json').read_text())
        require(produced['status'] == 'complete' and produced['frames'] == 177
                and produced['checkpoint_sha256'] == CHECKPOINT and produced['pid'] != os.getpid(),
                'Separate complete producer required')
        require(sha(out / 'PACKETS.jsonl') == produced['packets_manifest_sha256'], 'Manifest changed')
        require(sha(COMMON) == COMMON_SHA and produced['common_sha256'] == COMMON_SHA, 'Renderer source changed')
        require('MinkowskiEngine' not in sys.modules and 'codec' not in sys.modules, 'Codec module already loaded')
        spec = importlib.util.spec_from_file_location('_native_replay_common', COMMON)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One assigned GPU required')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., 6 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        free, _ = torch.cuda.mem_get_info()
        require(free >= 16 * 2**30, 'Shared GPU has insufficient free memory')
        torch.cuda.reset_peak_memory_stats()
        rows = [json.loads(line) for line in (out / 'PACKETS.jsonl').read_text().splitlines()]
        require([r['frame'] for r in rows] == list(range(177)), 'Missing/duplicated packet frames')
        results, failures = [], []
        with (out / 'REPLAY.jsonl').open('x', encoding='utf-8', buffering=1) as stream, torch.no_grad():
            for row in rows:
                frame = row['frame']
                require(time.monotonic() - started < 3600, 'Replay time budget exceeded')
                raw = read_checked(out / 'packets' / f'{frame:03d}.pt', row['packet']['bytes'], row['packet']['sha256'])
                packet = load_packet(raw, frame, CHECKPOINT, row['state_sha256'])
                require(len(packet['gs']) == row['gs_count'] and packet['pts'] == row['pts']
                        and packet['source_cache_sha256'] == row['source_cache_sha256'], 'Manifest/packet mismatch')
                raw_ref = read_checked(out / 'references' / f'{frame:03d}.pt', row['reference']['bytes'], row['reference']['sha256'])
                reference = torch.load(io.BytesIO(raw_ref), map_location='cpu', weights_only=True)
                require(reference.dtype == torch.float32 and tuple(reference.shape) == (2, 252, 448, 3)
                        and bool(torch.isfinite(reference).all()), 'Reference images invalid')
                gs, sh = packet['gs'].cuda(), packet['sh'].cuda()
                cameras, intrinsics = packet['cameras'].cuda(), packet['intrinsics'].cuda()
                views = []
                for view in range(2):
                    image = common.render(gs, cameras[view], intrinsics, packet['near'], packet['far'], sh)[0].clamp(0, 1)
                    delta = image - reference[view].cuda()
                    mae = float(delta.abs().mean())
                    mse = float(delta.square().mean())
                    psnr = -10 * __import__('math').log10(max(mse, 1e-12))
                    passed = mae <= 1e-5 and psnr >= 60
                    views.append(dict(view=view, mae=mae, psnr=psnr, passed=passed))
                item = dict(frame=frame, pts=row['pts'], gs_count=row['gs_count'], views=views,
                            passed=all(view['passed'] for view in views))
                results.append(item)
                if not item['passed']:
                    failures.append(frame)
                stream.write(json.dumps(item, allow_nan=False) + '\n')
                del raw, packet, raw_ref, reference, gs, sh, cameras, intrinsics, image, delta
                if frame in (0, 44, 88, 132, 176):
                    print('NATIVE_REPLAY', frame, '/176', flush=True)
        require('MinkowskiEngine' not in sys.modules and 'codec' not in sys.modules,
                'Replay unexpectedly imported the codec')
        require(torch.cuda.max_memory_reserved() <= 8 * 2**30, 'Replay GPU memory guard exceeded')
        write_new_json(out / 'RESULT.json', dict(status='complete' if not failures else 'threshold_failed',
            producer_pid=produced['pid'], replay_pid=os.getpid(), separate_process=True,
            input_scope='saved native GS/SH/cameras and reference images plus bound renderer only',
            encoder_or_codec_loaded=False, frames=177, views=354, failed_frames=failures,
            replay_passed_all=not failures, thresholds=dict(mae_max=1e-5, psnr_min=60),
            max_mae=max(v['mae'] for r in results for v in r['views']),
            min_psnr=min(v['psnr'] for r in results for v in r['views']),
            packets_manifest_sha256=produced['packets_manifest_sha256'], produced_sha256=sha(out/'PRODUCED.json'),
            replay_rows_sha256=sha(out/'REPLAY.jsonl'), checkpoint_sha256=CHECKPOINT,
            elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            codec_quality_accepted=False, action_Z_trained=False))
        if failures:
            raise SystemExit(2)
    except Exception as exc:
        write_new_json(out / 'REPLAY_FAILED.json', dict(status='failed', error=repr(exc),
                       traceback=traceback.format_exc(), action_Z_trained=False))
        raise


if __name__ == '__main__':
    main()
