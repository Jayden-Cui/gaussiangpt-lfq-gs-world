"""Optimize only static free codes on two frozen, historically trained windows."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
BROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PREVIEW = OUTPUTS / 'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
STATIC = OUTPUTS / 'gaussiangpt_neoverse_qformer_static_control_20261102_v1'
HISTORICAL = OUTPUTS / 'gaussiangpt_neoverse_qformer_support_20261101_v1/qformer_continuous/evaluation'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def verify_self():
    import hashlib
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        return h.hexdigest()
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise ValueError('CPU test witness changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise ValueError('Bound source changed: ' + name)
    return sha


def camera_only(tree, record):
    # The optimizer can never read target RGB, teacher images, or target states.
    keys = ('cameras', 'intrinsics', 'near', 'far')
    return {name: tree(record[name], 'cuda') for name in keys}


def predicted_node_limit(bundle):
    # This belongs to the bound model config, not this probe's protocol.
    limit = bundle['p']['max_predicted_latent_nodes']
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError('Invalid bound model node limit')
    return limit


def train_step(decoder, bridge, item, code, optimizer, std, p, horizon, *, probe=False):
    import torch
    from support_decoder import select_state, state_loss
    from runtime import render
    if any(x.requires_grad for x in decoder.parameters()) or any(x.requires_grad for x in bridge.parameters()):
        raise RuntimeError('Decoder or native codec unexpectedly trainable')
    actual = [x for group in optimizer.param_groups for x in group['params']]
    if not any(x is code for x in actual) or any(x.requires_grad for x in decoder.parameters()):
        raise RuntimeError('Optimizer parameter scope differs')
    optimizer.zero_grad(set_to_none=True)
    prediction = decoder(item['condition'], code)
    latent, terms = state_loss(prediction, item['labels'], std)
    state = select_state(item['condition'].coordinates, prediction['features'][horizon-1],
                         prediction['logits'][horizon-1], p['max_predicted_latent_nodes'])
    images, _, _ = render(bridge, state, item['records'][horizon], views=(0,))
    diff = (images[0] - item['static_images'][horizon]).abs().mean(-1)
    full, roi = diff.mean(), diff[item['roi']].mean()
    rgb = full + roi
    total = rgb + 0.1 * latent
    if not bool(torch.isfinite(total)):
        raise RuntimeError('Nonfinite loss')
    rgb_grad = None
    if probe:
        rgb_grad = float(torch.autograd.grad(rgb, code, retain_graph=True)[0].norm())
        if not math.isfinite(rgb_grad) or rgb_grad <= 0:
            raise RuntimeError('Static RGB gradient did not reach the free code')
    total.backward()
    if code.grad is None or not bool(torch.isfinite(code.grad).all()):
        raise RuntimeError('Missing or nonfinite code gradient')
    if any(x.grad is not None for x in decoder.parameters()) or any(x.grad is not None for x in bridge.parameters()):
        raise RuntimeError('Frozen model received a gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(actual, p['grad_clip']))
    optimizer.step()
    if any(not bool(torch.isfinite(x).all()) for x in actual):
        raise RuntimeError('Nonfinite code after optimizer update')
    return {'loss': float(total.detach()), 'static_rgb_full': float(full.detach()),
            'static_rgb_roi': float(roi.detach()), 'latent': float(latent.detach()),
            'occupancy': float(terms['occupancy'].detach()), 'feature': float(terms['feature'].detach()),
            'preclip_gradient_norm': norm, 'rgb_to_code_gradient_norm': rgb_grad}


def evaluate(step, bank, correct, historical_static, train, bundle, bridge, p, control, sha):
    import numpy as np
    from PIL import Image
    import torch
    from support_decoder import select_state, support_metrics
    from runtime import render
    import static_probe
    import preview
    store = bundle['store']
    max_nodes = predicted_node_limit(bundle)
    rows = []
    old = {(r['start'], r['horizon']): r['metrics'] for r in control['rows']}
    differences = []
    out = ROOT / 'result' / f'step_{step:05d}'
    out.mkdir()
    save_frames = step in (0, p['updates'])
    with torch.no_grad():
        for i, start in enumerate(p['windows']):
            item = train[start]
            # Future states and RGB are used only in this offline evaluation.
            w = store.get(start)
            predictions = [bundle['decoder'](item['condition'], x) for x in
                           (correct[i].cuda(), historical_static[i].cuda(), bank[i])]
            folder = out / f's{start:03d}'
            if save_frames:
                folder.mkdir()
            for h in p['horizons']:
                record = item['records'][h]
                images = []
                for prediction in predictions:
                    state = select_state(item['condition'].coordinates, prediction['features'][h-1],
                                         prediction['logits'][h-1], max_nodes)
                    im, _, _ = render(bridge, state, record, views=(0,))
                    images.append(im[0])
                moving, _, _ = render(bridge, w['states'][h], record, views=(0,))
                still = item['static_images'][h]
                values = static_probe.weighted_controls(images[0], images[2], moving[0], still)
                if step == 0:
                    differences.extend(abs(values[k] - old[(start, h)][k]) for k in values if k != 'dynamic_weight_sum')
                support = support_metrics(predictions[2]['logits'][h-1], item['labels']['occupancy'][h-1])
                rows.append({'start': start, 'horizon': h, 'frame': start+h, 'metrics': values,
                    'static_full_rgb_mae': float((images[2]-still).abs().mean()), 'static_support': support})
                if save_frames:
                    real = store.sample(start, h)['record']['real']
                    panels = [real, moving[0], still, images[1], images[2]]
                    combined = np.concatenate([preview.rgb_array(x) for x in panels], axis=1)
                    Image.fromarray(combined, mode='RGB').save(folder / f'frame_{h:02d}.png')
            if save_frames:
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-start_number', '1',
                    '-i', str(folder / 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    str(out / f's{start:03d}_five_column_5fps.mp4')], check=True, timeout=90)
    if step == 0 and (len(differences) != 108 or max(differences) > 5e-5):
        raise RuntimeError('Step-zero historical static control does not reproduce')
    summary = static_probe.aggregate(rows)
    summary['frames_static_closer_than_moving'] = sum(
        r['metrics']['static_code_to_static_reference'] < r['metrics']['static_code_to_motion_reference'] for r in rows)
    summary['static_full_rgb_mae'] = sum(r['static_full_rgb_mae'] for r in rows) / len(rows)
    summary['static_support_iou'] = sum(r['static_support']['iou'] for r in rows) / len(rows)
    write(out / 'EVALUATION.json', {'status': 'complete', 'step': step, 'summary': summary, 'rows': rows,
        'step0_reproduction_max_abs_error': max(differences) if differences else None,
        'future_states_used_only_for_offline_evaluation': True, 'visual_review': 'pending',
        'independent_native_gs_replay': 'not performed'})
    return summary


def run():
    import torch
    from schedule import make_schedule
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sha = verify_self()
    p = read(ROOT / 'protocol.json')
    if (p['schema'] != 'two_window_frozen_decoder_static_free_code_v1' or p['windows'] != [35, 123]
            or p['horizons'] != list(range(1, 10)) or p['updates'] != 540 or p['epochs'] != 30
            or p['checkpoint_steps'] != [0, 180, 360, 540]):
        raise ValueError('Predeclared protocol differs')
    result = ROOT / 'result'
    result.mkdir()
    begun = time.monotonic()
    phase = 'binding_sources'
    write(result / 'STARTED.json', {'pid': os.getpid(), 'code_sha256': sha(ROOT / 'CODE.sha256'),
          'protocol_sha256': sha(ROOT / 'protocol.json'), 'main_training_signalled': False})
    try:
        sys.path.insert(0, str(STATIC))
        import static_probe
        static_probe.verify_self()
        control_path = STATIC / 'result/STATIC_CONTROL.json'
        codes_path, evaluation_path = HISTORICAL / 'action_codes.pt', HISTORICAL / 'EVALUATION.json'
        for path, expected in ((control_path, p['static_control_sha256']),
                (codes_path, p['historical_codes_sha256']), (evaluation_path, p['historical_evaluation_sha256'])):
            if sha(path) != expected:
                raise RuntimeError('Historical source differs: ' + str(path))
        control, historical = read(control_path), read(evaluation_path)
        payload = torch.load(codes_path, map_location='cpu', weights_only=True)
        if (control['status'] != 'complete' or payload['schema'] != 'qformer_continuous_codes_v1'
                or payload['shape'] != p['code_shape'] or payload['quantized'] is not False
                or payload['source'] != historical['source'] or historical['source']['codec_sha256'] != p['codec_sha256']
                or historical['source']['teacher_tensor_digest'] != p['decoder_digest']):
            raise RuntimeError('Historical controls, codec or decoder binding differ')
        correct, initial = payload['controls']['correct'], payload['controls']['initial_only']
        if any(x.dtype != torch.float32 or list(x.shape) != p['code_shape'] or not bool(torch.isfinite(x).all())
               for x in (correct, initial)):
            raise RuntimeError('Invalid historical continuous codes')
        sys.path.insert(0, str(PREVIEW))
        import preview
        preview.verify_sources()
        sys.path.insert(0, str(BROOT))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle = load_bound_sources('cpu')
        runtime, store, decoder = bundle['runtime'], bundle['store'], bundle['decoder']
        if (sha(bundle['paths']['cache_manifest']) != p['cache_manifest_sha256']
                or runtime.digest(decoder) != p['decoder_digest']):
            raise RuntimeError('Wrong fixed cache or decoder')
        for s in p['windows']:
            if store.manifest['windows'][str(s)]['state_sources'] != historical['source']['states'][str(s)]:
                raise RuntimeError('State cache differs')
        phase = 'gpu_preflight'
        setup_gpu_once(dict(bundle['p'], allocator_limit_gib=p['allocator_limit_gib'],
                            minimum_free_gpu_gib=p['minimum_free_gpu_gib']))
        decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import target_labels
        from runtime import render
        from data import tree
        from torch import nn
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        std = store.frames['std'].cuda()
        train = {}
        with torch.no_grad():
            for start in p['windows']:
                w = store.get(start)
                initial_state = {'coordinates': w['states'][0]['coordinates'].detach(),
                                 'features': w['states'][0]['features'].detach()}
                records = {h: camera_only(tree, store.frames['samples'][start+h]['record']) for h in p['horizons']}
                static_images = {h: render(bridge, initial_state, records[h], views=(0,))[0][0].detach()
                                 for h in p['horizons']}
                train[start] = {'condition': w['condition'], 'roi': w['roi'], 'records': records,
                    'static_images': static_images, 'labels': target_labels(w['condition'], [initial_state]*9)}
                del w
        test_code = nn.Parameter(initial[0].cuda().clone())
        test_optimizer = torch.optim.Adam([test_code], lr=p['learning_rate'],
            betas=tuple(bundle['p']['betas']), eps=bundle['p']['eps'])
        before = test_code.detach().clone()
        sample = train_step(decoder, bridge, train[35], test_code, test_optimizer, std, bundle['p'], 1, probe=True)
        if not bool((test_code.detach() != before).any()):
            raise RuntimeError('GPU preflight failed to change the test code')
        bridge.assert_frozen()
        if runtime.digest(decoder) != p['decoder_digest']:
            raise RuntimeError('GPU preflight changed frozen decoder')
        write(result / 'PREFLIGHT.json', {'status': 'passed', 'rgb_to_code_gradient_norm': sample['rgb_to_code_gradient_norm'],
             'separate_test_code_update_discarded': True, 'decoder_and_codec_frozen': True,
             'true_future_used_for_preflight_update': False})
        del test_code, test_optimizer, before
        bank = nn.ParameterList([nn.Parameter(initial[i].cuda().clone()) for i in range(2)])
        optimizer = torch.optim.Adam(list(bank.parameters()), lr=p['learning_rate'],
            betas=tuple(bundle['p']['betas']), eps=bundle['p']['eps'])
        if {id(x) for group in optimizer.param_groups for x in group['params']} != {id(x) for x in bank.parameters()}:
            raise RuntimeError('Optimizer must contain only free codes')
        schedule = make_schedule(p['windows'], p['horizons'], p['epochs'], p['seed'])
        if len(schedule) != p['updates']:
            raise RuntimeError('Unexpected training schedule length')
        phase = 'evaluating_step_0'
        summaries = {'0': evaluate(0, bank, correct, initial, train, bundle, bridge, p, control, sha)}
        for step, (start, horizon) in enumerate(schedule, 1):
            phase = f'train_{step}'
            i = p['windows'].index(start)
            info = train_step(decoder, bridge, train[start], bank[i], optimizer, std, bundle['p'], horizon)
            if step % 18 == 0:
                print(json.dumps({'step': step, 'last_loss': info['loss']}), flush=True)
            if step in p['checkpoint_steps']:
                phase = f'evaluating_step_{step}'
                bridge.assert_frozen()
                if runtime.digest(decoder) != p['decoder_digest']:
                    raise RuntimeError('Decoder changed during code optimization')
                checkpoint = result / f'codes_step_{step:05d}.pt'
                torch.save({'schema': 'two_window_static_free_codes_v1', 'step': step, 'starts': p['windows'],
                    'codes': torch.stack([v.detach().cpu() for v in bank]), 'optimizer': optimizer.state_dict(),
                    'decoder_digest': p['decoder_digest'], 'protocol_sha256': sha(ROOT/'protocol.json')}, checkpoint)
                summaries[str(step)] = evaluate(step, bank, correct, initial, train, bundle, bridge, p, control, sha)
        bridge.assert_frozen()
        if runtime.digest(decoder) != p['decoder_digest']:
            raise RuntimeError('Decoder changed by experiment')
        write(result / 'SUMMARY.json', {'status': 'complete', 'steps': p['updates'], 'summaries': summaries,
            'checkpoints': {str(s): sha(result / f'codes_step_{s:05d}.pt') for s in (180, 360, 540)},
            'training_future_state_or_image_used': False, 'main_training_signalled': False,
            'elapsed_seconds': time.monotonic()-begun, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'visual_review': 'pending', 'limitations': p['limitations']})
        print('PASS: frozen decoder static code optimization completed; visual review pending', flush=True)
    except BaseException as error:
        write(result / 'FAILED.json', {'status': 'failed', 'phase': phase, 'error': repr(error),
              'traceback': traceback.format_exc(), 'main_training_signalled': False, 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    run()
