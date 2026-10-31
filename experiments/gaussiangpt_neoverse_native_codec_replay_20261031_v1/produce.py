"""Decode all 177 selected saved states to GS packets and original-view references."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from native_packet import require, sha, read_checked, save_new, validate_packet, write_new_json

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
EVAL = BASE / 'gaussiangpt_neoverse_codec_complete_eval_20261031_v1'
ARM = EVAL / 'evaluations/baseline'
OLD = BASE / 'gaussiangpt_neoverse_fullpool_codec_20261031_v1'
CLIP = 'buildingblock_tower_17_176'
CHECKPOINT = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
ENCODER = 'c2ddb2b1dd5a43ee4a86286ea09ef141f61661963fa1d5979edf3d4f5917737b'
STATE_MANIFEST = '6982106649970254057d463e1213c36c1257ed1e2e202c77af830587559d0e54'
EVALUATION = '56d5fe877518be1285c8fe02552cfe85eee8c41f0ce0f044489cc84a82a35aa5'
QUEUE_DONE = 'e63c4fb321f19d9915de1bbadafb05e39f573ad29e01159eef541bd750eedd1b'
COMMON_SHA = '38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff'


def main():
    root = Path(__file__).resolve().parent
    out = root / 'results'
    out.mkdir(exist_ok=False)
    (out / 'packets').mkdir()
    (out / 'references').mkdir()
    start = time.monotonic()
    write_new_json(out / 'STARTED.json', dict(pid=os.getpid(), stage='produce', frames=177, action_Z_trained=False))
    try:
        require(sha(ARM / 'EVALUATION.json') == EVALUATION and sha(EVAL / 'queue/DONE.json') == QUEUE_DONE,
                'Evaluation evidence SHA changed')
        require(sha(ARM / 'state_manifest.jsonl') == STATE_MANIFEST, 'State manifest changed')
        evaluation = json.loads((ARM / 'EVALUATION.json').read_text())
        require(evaluation['arm'] == 'baseline' and evaluation['checkpoint_sha256'] == CHECKPOINT
                and evaluation['encoder_sha256'] == ENCODER and evaluation['frames'] == 9027
                and evaluation['replay_failed_frames'] == 0 and evaluation['cold_manager_replay_verified_all'],
                'Full evaluation evidence differs')
        rows = [json.loads(line) for line in (ARM / 'state_manifest.jsonl').read_text().splitlines()]
        rows = [row for row in rows if row['clip'] == CLIP]
        rows.sort(key=lambda row: row['frame'])
        require([row['frame'] for row in rows] == list(range(177)), '177 unique ordered states required')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One assigned GPU required')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., 10 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        free, _ = torch.cuda.mem_get_info()
        require(free >= 16 * 2**30, 'Shared GPU has insufficient free memory')
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(EVAL))
        import worker as complete
        import inference_profile as inference
        p, evidence = complete.verify_root(EVAL)
        require(evidence['code_manifest_sha256'] == '2e1fab1844bba17931c79e85f36ea3a5cb0664e19f71a1b2fe5832634a8e837b',
                'Evaluation code manifest differs')
        probe, inference_binding = inference.load_diagnostic(p['inference'])
        old = complete.load_old(OLD)
        x = old.load_runtime(OLD)
        old.gate(x)
        import MinkowskiEngine as ME
        inference.verify_actual_sources(probe, ME, inference_binding)
        inference.entry_state(ME)
        model, checkpoint, checkpoint_sha, _, _ = complete.evaluation_model(torch, old, x, 'baseline', None)
        require(checkpoint_sha == CHECKPOINT and sha(checkpoint) == CHECKPOINT, 'Selected checkpoint changed')
        precision = inference.load_precision_helper(p['inference']).DecoderPrecision(model, 'float64')
        inference.validate_conversion(precision.report)
        require(sha(Path(x.old.codec_root) / 'code/common.py') == COMMON_SHA, 'Native renderer source changed')
        completed = []
        previous_pts = float('-inf')
        manifest = out / 'PACKETS.jsonl'
        with manifest.open('x', encoding='utf-8', buffering=1) as stream, torch.no_grad():
            for row in rows:
                frame = row['frame']
                require(time.monotonic() - start < 3600, 'Producer time budget exceeded')
                require(row['checkpoint_sha256'] == CHECKPOINT and row['encoder_sha256'] == ENCODER
                        and row['replay_passed'] is True and row['cold_manager_replay_verified'] is True,
                        'State source or latent replay differs')
                state_path = ARM / row['state_path']
                require(state_path == ARM / 'states' / CLIP / f'{frame:03d}.pt', 'State path differs')
                raw_state = read_checked(state_path, row['state_bytes'], row['state_sha256'])
                state = torch.load(io.BytesIO(raw_state), map_location='cpu', weights_only=True)
                meta = state['metadata']
                require(state['schema'] == 'multisource_continuous_encoder_state_v1'
                        and state['stride'] == 4 and state['grid'] == .008
                        and meta['clip'] == CLIP and meta['frame'] == frame
                        and meta['checkpoint_sha256'] == CHECKPOINT and meta['encoder_sha256'] == ENCODER,
                        'State packet provenance differs')
                coordinates, features = state['coordinates'], state['features']
                require(coordinates.ndim == 2 and coordinates.shape[1] == 3
                        and coordinates.dtype in (torch.int32, torch.int64) and len(coordinates) > 0
                        and len(torch.unique(coordinates, dim=0)) == len(coordinates)
                        and features.dtype == torch.float32 and tuple(features.shape) == (len(coordinates), 64)
                        and bool(torch.isfinite(features).all()), 'State coordinate/feature data invalid')
                payload = old.pool.load_frame(x.by_id[CLIP, frame], device='cuda')
                require(meta['pts'] == payload['pts'] and meta['pts'] > previous_pts
                        and meta['normalization'] == payload['normalization']
                        and meta['source_cache_sha256'] == payload['_source_cache_sha256']
                        and payload['n_input'] == 112896, 'State/raw camera source or time differs')
                previous_pts = meta['pts']
                sparse = model._sparse(coordinates.to('cuda'), features.to('cuda'), state['stride'])
                prepared = precision.prepare_sparse(sparse, ME)
                points, _ = model.ae.decode(prepared, target_key=None)
                points = precision.to_render_points(model._world(points))
                gs, sh = x.old.train.prediction_gs(points)
                require(len(gs) == meta['decoded_gaussians'], 'Decoded GS count differs from evaluated state')
                references = []
                for view in range(2):
                    native = x.old.train.render(gs, payload['cameras'][view], payload['intrinsics'],
                                                payload['near'], payload['far'], sh)[0].clamp(0, 1)
                    original = x.old.train.render_prediction(points, payload, view)
                    difference = (native-original).abs().mean().item()
                    require(difference <= 1e-5, 'GS packet renderer diverges from codec renderer')
                    references.append(original.detach().cpu().contiguous())
                packet = dict(schema='native_codec_gs_packet_v1', clip=CLIP, frame=frame,
                    checkpoint_sha256=CHECKPOINT, state_sha256=row['state_sha256'],
                    source_cache_sha256=payload['_source_cache_sha256'], gs=gs.detach().cpu().contiguous(),
                    sh=sh.detach().cpu().contiguous(), cameras=payload['cameras'].detach().cpu().contiguous(),
                    intrinsics=payload['intrinsics'].detach().cpu().contiguous(), near=float(payload['near']),
                    far=float(payload['far']), pts=float(payload['pts']), resolution_hw=[252, 448],
                    normalization=payload['normalization'], decoded_from_actual_state=True,
                    action_Z_prediction=False)
                validate_packet(packet, frame, CHECKPOINT, row['state_sha256'])
                packet_path = out / 'packets' / f'{frame:03d}.pt'
                reference_path = out / 'references' / f'{frame:03d}.pt'
                packet_info = save_new(packet_path, packet)
                reference_info = save_new(reference_path, torch.stack(references))
                record = dict(frame=frame, pts=float(payload['pts']), state_sha256=row['state_sha256'],
                              source_cache_sha256=payload['_source_cache_sha256'], gs_count=len(gs),
                              packet=packet_info, reference=reference_info)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                completed.append(frame)
                del payload, state, sparse, prepared, points, gs, sh, packet, references
                if frame in (0, 44, 88, 132, 176):
                    print('NATIVE_PACKET', frame, '/176', flush=True)
        old.recheck(x)
        precision_check = precision.assert_unchanged()
        inference.validate_conversion(precision.report, precision_check)
        require(completed == list(range(177)) and sha(checkpoint) == CHECKPOINT, 'Incomplete/changed source')
        require(torch.cuda.max_memory_reserved() <= 12 * 2**30, 'GPU memory guard exceeded')
        write_new_json(out / 'PRODUCED.json', dict(status='complete', pid=os.getpid(), frames=177,
            clip=CLIP, checkpoint_sha256=CHECKPOINT, encoder_sha256=ENCODER,
            evaluation_sha256=EVALUATION, state_manifest_sha256=STATE_MANIFEST,
            packets_manifest_sha256=sha(manifest), common_sha256=COMMON_SHA,
            elapsed_seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, model_updated=False,
            action_Z_trained=False, native_replay_performed=False))
    except BaseException as exc:
        write_new_json(out / 'PRODUCER_FAILED.json', dict(status='failed', error=repr(exc),
                       traceback=traceback.format_exc(), action_Z_trained=False))
        raise


if __name__ == '__main__':
    main()
