"""Render two regression windows from a stage-B checkpoint on the spare GPU.

This is an early diagnostic only. The official 168-window evaluation and
independent native replay remain the responsibility of the B coordinator.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
RUN = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PARENT_EVAL_SHA = '74692f5757622aa150e6617304fbb8786060ec0ceb7f0dd7b105d898cc4d53d7'
sys.path.insert(0, str(B_ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def verify_sources():
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Preview CPU tests are missing or changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Preview source changed: ' + name)
    b_witness = read(B_ROOT / 'CPU_TESTS.json')
    if b_witness.get('status') != 'passed' or b_witness.get('code_sha256') != sha(B_ROOT / 'CODE.sha256'):
        raise RuntimeError('Stage-B source binding changed')
    for line in (B_ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(B_ROOT / name) != digest:
            raise RuntimeError('Stage-B source changed: ' + name)


def validate_checkpoint(item, metadata, expected_source, sweep, file_sha):
    codes = item.get('codes')
    if (metadata.get('status') != 'complete' or metadata.get('sweep') != sweep
            or metadata.get('sha256') != file_sha
            or item.get('schema') != 'stage_b_shared_decoder_free_codes_v1'
            or item.get('sweep') != sweep or item.get('starts') != list(range(168))
            or item.get('source') != expected_source or item.get('decoder_digest') != metadata.get('decoder_digest')
            or not isinstance(codes, torch.Tensor) or codes.dtype != torch.float32
            or tuple(codes.shape) != (168, 9, 16) or not bool(torch.isfinite(codes).all())):
        raise RuntimeError('Incomplete, changed or mismatched B checkpoint')
    return codes


def rgb_array(tensor):
    value = tensor.detach().float().cpu().numpy()
    if value.shape != (252, 448, 3) or not np.isfinite(value).all():
        raise RuntimeError('Unexpected rendered/source image')
    return np.rint(value.clip(0, 1) * 255).astype(np.uint8)


def save_pair(path, target, prediction):
    left, right = rgb_array(target), rgb_array(prediction)
    image = Image.fromarray(np.concatenate((left, right), axis=1), mode='RGB')
    image.save(path)


def anchor_mean(rows):
    future = [r for r in rows if r['horizon'] > 0]
    if (len(future) != 18 or {(r['start'], r['horizon']) for r in future}
            != {(s, h) for s in (35, 123) for h in range(1, 10)}):
        raise RuntimeError('Two-window, nine-future-frame coverage differs')
    values = [r['metrics']['correct']['metrics']['motion_mae'] for r in future]
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError('Non-finite anchor motion metric')
    return sum(values) / len(values)


def run(label):
    # Exactly one interop setup in this process, before touching the model.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if label not in ('warm', '05', '10'):
        raise ValueError('Only warm, sweep 5 and sweep 10 are previewed')
    verify_sources()
    from b_core import load_bound_sources
    from b_worker_v3 import load_initial_codes
    from gpu_runtime import setup_gpu_once

    bundle = load_bound_sources('cpu')
    runtime, fm, p = bundle['runtime'], bundle['fm'], bundle['p']
    config = read(RUN / 'CONFIG.json')
    binding = read(RUN / 'BINDING.json')
    preflight = read(RUN / 'training/PREFLIGHT.json')
    if (preflight.get('status') != 'passed' or binding['config_snapshot_sha256'] != sha(RUN / 'CONFIG.json')
            or preflight['source']['config_sha256'] != sha(RUN / 'CONFIG.json')
            or not preflight['rgb_to_free_code_and_decoder_verified']
            or not preflight['codec_and_initial_encoder_frozen']):
        raise RuntimeError('Stage-B GPU preflight or config binding differs')
    decoder = bundle['decoder']
    if label == 'warm':
        codes, initial_source = load_initial_codes(config, bundle)
        if (initial_source != preflight['source']['initial_codes']
                or runtime.digest(decoder) != preflight['source']['warm_decoder_digest']):
            raise RuntimeError('Warm decoder/codes differ from the real B initialization')
        checkpoint_sha = None
    else:
        sweep = int(label)
        checkpoint = RUN / 'training' / f'sweep_{sweep:02d}.pt'
        metadata = read(RUN / 'training' / f'sweep_{sweep:02d}.json')
        item = torch.load(checkpoint, map_location='cpu', weights_only=True)
        checkpoint_sha = sha(checkpoint)
        codes = validate_checkpoint(item, metadata, preflight['source'], sweep, checkpoint_sha)
        decoder.load_state_dict(item['decoder'], strict=True)
        if runtime.digest(decoder) != item['decoder_digest']:
            raise RuntimeError('Checkpoint decoder tensor digest differs')
    if fm.frozen_digest(decoder) != bundle['frozen_digest']:
        raise RuntimeError('Frozen initial encoder/context changed')
    decoder.requires_grad_(False).eval()
    model_digest = runtime.digest(decoder)
    parent = Path(p['parent_root']) / 'free16/evaluation/EVALUATION.json'
    if sha(parent) != PARENT_EVAL_SHA:
        raise RuntimeError('Two-window original reference changed')
    historical = read(parent)
    original = anchor_mean(historical['rows'])
    output = ROOT / ('warm_start' if label == 'warm' else f'sweep_{label}')
    output.mkdir()
    phase = 'gpu_preflight'
    write(output / 'STATUS.json', {'status': 'running', 'phase': phase, 'label': label})
    try:
        setup_gpu_once(p)
        decoder = decoder.cuda()
        from runtime_bridge import BoundNativeBridge
        from data import evaluate_metrics
        from support_decoder import select_state, support_metrics
        from runtime import render
        from b_evaluate_v2 import anchor_metrics
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        store = bundle['store']
        rows = []
        phase = 'rendering_two_anchor_windows'
        write(output / 'STATUS.json', {'status': 'running', 'phase': phase, 'label': label}, replace=True)
        with torch.no_grad():
            for start in (35, 123):
                window = store.get(start)
                code = codes[start].cuda()
                prediction = decoder(window['condition'], code)
                altered = code.clone()
                altered[-1] += 2.5
                changed = decoder(window['condition'], altered)
                if max(float((prediction['features'][:-1] - changed['features'][:-1]).abs().max()),
                       float((prediction['logits'][:-1] - changed['logits'][:-1]).abs().max())) > 2e-5:
                    raise RuntimeError('Decoder prefix causality check failed')
                del changed, altered
                frames = output / f's{start:03d}'
                frames.mkdir()
                sample = store.sample(start, 0)
                image, _, _ = render(bridge, window['states'][0], sample['record'], views=(0, 1))
                save_pair(frames / 'frame_00.png', sample['record']['real'], image[0])
                for h in range(1, 10):
                    sample = store.sample(start, h)
                    state = select_state(window['condition'].coordinates, prediction['features'][h - 1],
                                         prediction['logits'][h - 1], p['max_predicted_latent_nodes'])
                    image, _, _ = render(bridge, state, sample['record'], views=(0, 1))
                    metrics = evaluate_metrics(image, window, sample, p)
                    support = support_metrics(prediction['logits'][h - 1], window['targets']['occupancy'][h - 1])
                    rows.append({'start': start, 'horizon': h, 'frame': start + h,
                                 'metrics': {'correct': metrics}, 'support': {'correct': support}})
                    save_pair(frames / f'frame_{h:02d}.png', sample['record']['real'], image[0])
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-i',
                                str(frames / 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                str(output / f's{start:03d}_source_vs_free_5fps.mp4')], check=True, timeout=120)
                del window, code, prediction
        bridge.assert_frozen()
        if runtime.digest(decoder) != model_digest or fm.frozen_digest(decoder) != bundle['frozen_digest']:
            raise RuntimeError('Preview changed the decoder or frozen initial context')
        current = anchor_mean(rows)
        if label == 'warm' and abs(current - original) > 1e-4:
            raise RuntimeError('Warm-start anchor metric differs from the original two-window reference')
        result = {'status': 'complete', 'label': label, 'windows': [35, 123], 'future_predictions': 18,
                  'anchor_motion_mae': current, 'original_two_window_anchor_motion_mae': original,
                  'difference_from_original': current - original, 'dynamic_anchors': anchor_metrics(rows, store, p),
                  'rows': rows, 'decoder_digest': model_digest, 'checkpoint_sha256': checkpoint_sha,
                  'config_sha256': sha(RUN / 'CONFIG.json'), 'review_sha256': binding['review']['review_sha256'],
                  'causal_prefix_test': 'passed', 'codec_and_initial_context_frozen': True,
                  'visual_dynamic_review': 'pending', 'independent_native_replay': 'not performed for preview',
                  'scope': 'Two known windows only; full 168-window evaluation and replay happen after training'}
        write(output / 'PREVIEW.json', result)
        write(output / 'STATUS.json', {'status': 'complete', 'phase': 'awaiting_visual_review', 'label': label},
              replace=True)
        print('PASS:', label, 'anchor motion MAE', current, 'original two-window', original)
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'label': label, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(output / 'FAILED.json', failure)
        write(output / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=('warm', '05', '10'), required=True)
    run(parser.parse_args().label)
