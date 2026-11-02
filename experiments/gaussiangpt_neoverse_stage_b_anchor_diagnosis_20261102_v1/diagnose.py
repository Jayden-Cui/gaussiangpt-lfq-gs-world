"""Diagnose stage-B support and feature reachability on two motion windows.

Target support is a post-inference oracle used only for diagnosis. It is never
provided to the decoder and the program never trains or updates model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
B_PREP = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
EVAL_SHA = '09b76517d2db913c3aabecb405bd00cf1bcb4a1a54c42859063b99fcc62d58df'
CHECKPOINT_SHA = 'e1613f6688dd2dda1ac6b6e8870da99ac96ecc586fe23ccb930a13303b320a69'
DECODER_DIGEST = '83d4ef081a3e72ad80a24b5f48ca33aa5d0fba528bebd1d64b023c19de86cd29'
CODEC_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
CACHE_SHA = 'cb5c2b30f1b4fdac4db642046a46773cc0b96a4dabdab4917e0d3c79ff6eaa4a'
STARTS = (35, 123)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def oracle_support(condition, prediction, horizon, indices):
    if (indices.ndim != 1 or indices.dtype != torch.int64 or not len(indices)
            or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0 or int(indices.max()) >= len(condition.coordinates)):
        raise ValueError('Invalid target-support indices')
    return dict(coordinates=condition.coordinates[indices], features=prediction['features'][horizon-1, indices])


def save_panels(path, images):
    arrays = []
    for value in images:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = (np.clip(value, 0, 1) * 255).round().astype(np.uint8)
        if array.shape != (252, 448, 3):
            raise RuntimeError('Unexpected review image dimensions')
        arrays.append(array)
    Image.fromarray(np.concatenate(arrays, axis=1)).save(path)


def ensure_sources():
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Diagnostic CPU witness changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Diagnostic source changed: ' + name)
    protocol = read(ROOT / 'protocol.json')
    if (protocol.get('schema') != 'stage_b_two_window_support_feature_diagnostic_v1'
            or protocol.get('starts') != list(STARTS) or protocol.get('horizons') != list(range(1, 10))
            or protocol.get('checkpoint_sha256') != CHECKPOINT_SHA or protocol.get('evaluation_sha256') != EVAL_SHA
            or protocol.get('decoder_digest') != DECODER_DIGEST or protocol.get('original_codec_sha256') != CODEC_SHA
            or protocol.get('cache_manifest_sha256') != CACHE_SHA or protocol.get('training_performed') is not False
            or protocol.get('oracle_support_after_inference_only') is not True
            or protocol.get('ground_truth_target_features_used_in_oracle') is not False
            or protocol.get('time_limit_seconds') != 900 or protocol.get('manual_visual_review_required') is not True):
        raise RuntimeError('Diagnostic protocol differs')
    summary = read(B_RUN / 'SUMMARY.json')
    if (summary.get('status') != 'complete' or summary.get('stage') != 'B'
            or summary.get('manual_dynamic_review_required') is not True
            or summary.get('action_codec_complete') is not False):
        raise RuntimeError('Expected completed, not automatically accepted stage B')
    evaluation_path = B_RUN / 'training/sweep_20/free_z/evaluation/EVALUATION.json'
    checkpoint = B_RUN / 'training/sweep_20.pt'
    if sha(evaluation_path) != EVAL_SHA or sha(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError('Stage-B evaluation or model checkpoint changed')
    row = next((r for r in summary['results'] if r['sweep'] == 20), None)
    replay_path = evaluation_path.parent / 'REPLAY.json'
    if row is None or row['evaluation_sha256'] != EVAL_SHA or row['replay']['sha256'] != sha(replay_path):
        raise RuntimeError('Stage-B summary and independent replay differ')
    evaluation, replay = read(evaluation_path), read(replay_path)
    if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
            or evaluation['source'].get('decoder_digest') != DECODER_DIGEST
            or evaluation['source'].get('codes_sha256') != CHECKPOINT_SHA
            or evaluation['source'].get('codec_sha256') != CODEC_SHA
            or evaluation['source'].get('cache_manifest_sha256') != CACHE_SHA
            or evaluation['source'].get('quantized') is not False
            or replay.get('status') != 'passed' or replay.get('packets') != 104 or replay.get('views') != 208
            or replay.get('encoder_or_codec_loaded') is not False
            or replay.get('packets_sha256') != evaluation.get('packets_sha256')
            or sha(evaluation_path.parent / 'PACKETS.json') != evaluation['packets_sha256']):
        raise RuntimeError('Stage-B evaluation, codec, or native GS replay differs')
    return evaluation, checkpoint, sha(B_RUN / 'SUMMARY.json'), sha(replay_path)


def main(output):
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    evaluation, checkpoint, summary_sha, replay_sha = ensure_sources()
    output = Path(output)
    if not output.is_absolute() or output.exists() or output.parent.resolve() != ROOT.resolve():
        raise RuntimeError('A new absolute diagnostic output directory is required')
    output.mkdir()
    phase, began = 'loading_models', time.monotonic()
    try:
        sys.path.insert(0, str(B_PREP))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle = load_bound_sources('cpu')
        binding, p, fm, runtime, store = (bundle[k] for k in ('binding', 'p', 'fm', 'runtime', 'store'))
        if p['codec_checkpoint_sha256'] != CODEC_SHA or binding.sha(bundle['paths']['cache_manifest']) != CACHE_SHA:
            raise RuntimeError('Loaded original codec or state cache differs')
        item = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (item.get('schema') != 'stage_b_shared_decoder_free_codes_v1' or item.get('sweep') != 20
                or item.get('starts') != list(range(168)) or tuple(item['codes'].shape) != (168, 9, 16)
                or item.get('decoder_digest') != DECODER_DIGEST):
            raise RuntimeError('Stage-B checkpoint tensor ordering differs')
        decoder = bundle['decoder']
        decoder.load_state_dict(item['decoder'], strict=True)
        if runtime.digest(decoder) != DECODER_DIGEST or fm.frozen_digest(decoder) != bundle['frozen_digest']:
            raise RuntimeError('Decoder or initial-condition encoder differs')
        decoder.requires_grad_(False).eval()
        setup_gpu_once(p)
        decoder = decoder.cuda()
        from runtime_bridge import BoundNativeBridge
        from runtime import render
        from support_decoder import select_state
        from data import evaluate_metrics
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        std = store.frames['std'].cuda()
        if not torch.allclose(std, decoder.base.output_std, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Feature residual scale differs')
        old_rows = {(r['start'], r['horizon']): r for r in evaluation['rows'] if r['horizon'] > 0}
        if len(old_rows) != 1512:
            raise RuntimeError('Stage-B evaluation coverage differs')
        rows, bound = [], 3 * std
        phase = 'rendering_post_inference_oracle'
        with torch.no_grad():
            for start in STARTS:
                folder = output / ('s%03d' % start)
                folder.mkdir()
                window = store.get(start)
                prediction = decoder(window['condition'], item['codes'][start].cuda())
                for horizon in range(1, 10):
                    if time.monotonic() - began > 900:
                        raise RuntimeError('Diagnostic time budget exceeded')
                    sample = store.sample(start, horizon)
                    idx = window['targets']['indices'][horizon-1]
                    if idx.numel() != int(window['targets']['occupancy'][horizon-1].sum()):
                        raise RuntimeError('Target support and indices differ')
                    target_features = window['targets']['features'][horizon-1]
                    initial_features = window['condition'].features[idx]
                    outside = (target_features - initial_features).abs() > bound[None] + 1e-6
                    raw = prediction['raw_residual'][horizon-1, idx].tanh().abs()
                    state = select_state(window['condition'].coordinates, prediction['features'][horizon-1],
                                         prediction['logits'][horizon-1], p['max_predicted_latent_nodes'])
                    oracle = oracle_support(window['condition'], prediction, horizon, idx)
                    images, _, _ = render(bridge, state, sample['record'], views=(0, 1))
                    oracle_images, _, _ = render(bridge, oracle, sample['record'], views=(0, 1))
                    actual_images, _, _ = render(bridge, window['states'][horizon], sample['record'], views=(0, 1))
                    values = {name: evaluate_metrics(image, window, sample, p)
                              for name, image in (('predicted_support', images),
                                                  ('oracle_target_support', oracle_images),
                                                  ('actual_target', actual_images))}
                    original = old_rows[(start, horizon)]
                    for arm, key in (('predicted_support', 'correct'), ('actual_target', 'actual_target')):
                        before = original['metrics'][key]['metrics']['motion_mae']
                        after = values[arm]['metrics']['motion_mae']
                        if not math.isfinite(after) or abs(before - after) > 1e-5:
                            raise RuntimeError('Existing stage-B rendering did not reproduce')
                    path = folder / ('frame_%02d.png' % horizon)
                    save_panels(path, [sample['record']['real'], images[0], oracle_images[0], actual_images[0]])
                    rows.append(dict(start=start, horizon=horizon, frame=start+horizon, metrics=values,
                        target_nodes=len(idx), target_feature_elements=outside.numel(),
                        out_of_range_elements=int(outside.sum()),
                        target_cells_with_any_out_of_range=int(outside.any(-1).sum()),
                        target_tanh_abs_above_095=int((raw > .95).sum()),
                        target_tanh_abs_above_099=int((raw > .99).sum()),
                        review_frame=str(path), review_frame_sha256=sha(path)))
        if len(rows) != 18 or {(r['start'], r['horizon']) for r in rows} != {(s, h) for s in STARTS for h in range(1, 10)}:
            raise RuntimeError('Anchor diagnostic coverage differs')
        if runtime.digest(decoder) != DECODER_DIGEST or fm.frozen_digest(decoder) != bundle['frozen_digest']:
            raise RuntimeError('Frozen decoder changed during diagnosis')
        bridge.assert_frozen()
        per_window = {}
        for start in STARTS:
            selected = [r for r in rows if r['start'] == start]
            arms = ('predicted_support', 'oracle_target_support', 'actual_target')
            per_window[str(start)] = {name: dict(
                mean_motion_mae=sum(r['metrics'][name]['metrics']['motion_mae'] for r in selected)/len(selected),
                mean_object_proxy_iou=sum(r['metrics'][name]['object_proxy']['iou'] for r in selected)/len(selected),
                color_proxy_detected=sum(r['metrics'][name]['object_proxy']['prediction']['valid'] for r in selected))
                for name in arms}
            elements = sum(r['target_feature_elements'] for r in selected)
            nodes = sum(r['target_nodes'] for r in selected)
            per_window[str(start)]['feature_reachability'] = dict(
                out_of_range_feature_fraction=sum(r['out_of_range_elements'] for r in selected)/elements,
                cells_with_any_out_of_range_fraction=sum(r['target_cells_with_any_out_of_range'] for r in selected)/nodes,
                tanh_abs_above_095_fraction=sum(r['target_tanh_abs_above_095'] for r in selected)/elements,
                tanh_abs_above_099_fraction=sum(r['target_tanh_abs_above_099'] for r in selected)/elements)
        result = dict(status='complete', starts=list(STARTS), horizons=list(range(1, 10)), rows=rows,
            per_window=per_window, source=dict(stage_b_summary_sha256=summary_sha, evaluation_sha256=EVAL_SHA,
                checkpoint_sha256=CHECKPOINT_SHA, decoder_digest=DECODER_DIGEST, replay_sha256=replay_sha,
                codec_sha256=CODEC_SHA, cache_sha256=CACHE_SHA, code_sha256=sha(ROOT/'CODE.sha256')),
            oracle_is_post_inference_only=True, target_features_used_by_oracle=False,
            training_performed=False, decoder_and_codec_frozen=True,
            interpretation='Two training windows, not a full-168 diagnosis; oracle true support is unavailable at inference; latent bound is not a rendering impossibility proof',
            visual_review_required=True, elapsed_seconds=time.monotonic()-began)
        write(output/'DIAGNOSIS.json', result)
        print(json.dumps(dict(status='complete', per_window=per_window, output=str(output)), ensure_ascii=False))
    except BaseException as error:
        write(output/'FAILED.json', dict(status='failed', phase=phase, error=repr(error),
                                        traceback=traceback.format_exc(), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
