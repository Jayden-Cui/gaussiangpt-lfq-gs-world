"""Train one decoder and two free-code futures for each of two initial states."""
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
BROOT = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PREVIEW = OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
STATIC_CONTROL = OUTPUTS/'gaussiangpt_neoverse_qformer_static_control_20261102_v1'
STATIC_OPT = OUTPUTS/'gaussiangpt_neoverse_static_code_probe_20261102_v3'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_self():
    witness = read(ROOT/'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT/'CODE.sha256'):
        raise ValueError('CPU test witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT/name) != expected:
            raise ValueError('Bound source changed: ' + name)


def camera_only(tree, record):
    return {k: tree(record[k], 'cuda') for k in ('cameras', 'intrinsics', 'near', 'far')}


def make_optimizer(decoder, bank, p, cfg):
    import torch
    trained = [v for v in decoder.parameters() if v.requires_grad]
    if not trained or len(list(bank.parameters())) != 4:
        raise RuntimeError('Expected trainable decoder and four free-code tensors')
    return torch.optim.Adam([{'params': trained, 'lr': cfg['decoder_learning_rate']},
                             {'params': list(bank.parameters()), 'lr': cfg['code_learning_rate']}],
                            betas=tuple(p['betas']), eps=p['eps'])


def step(decoder, bridge, store, item, bank, optimizer, std, p, start_index, horizon, *, probe=False):
    import torch
    from core import image_losses
    from support_decoder import select_state, state_loss
    from runtime import render
    if any(x.requires_grad for x in bridge.parameters()):
        raise RuntimeError('Native GS codec must be frozen')
    expected = [x for x in decoder.parameters() if x.requires_grad] + list(bank.parameters())
    actual = [x for group in optimizer.param_groups for x in group['params']]
    if len(expected) != len(actual) or {id(x) for x in expected} != {id(x) for x in actual}:
        raise RuntimeError('Optimizer must contain decoder and only the four free codes')
    optimizer.zero_grad(set_to_none=True)
    w = item['window']
    sample = store.sample(w['start'], horizon)
    record = item['records'][horizon]
    rows = {}
    for branch, code, target, labels, weight in (
        ('motion', bank[start_index], sample['record']['real'], w['targets'], sample['motion']),
        ('static', bank[start_index+2], item['static_images'][horizon], item['static_labels'], item['weights'][horizon])):
        prediction = decoder(item['condition'], code)
        latent, latent_terms = state_loss(prediction, labels, std)
        state = select_state(item['condition'].coordinates, prediction['features'][horizon-1],
                             prediction['logits'][horizon-1], p['max_predicted_latent_nodes'])
        images, _, _ = render(bridge, state, record, views=(0,))
        rgb, terms = image_losses(images[0], target, item['roi'], weight)
        loss = rgb + .1*latent
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite paired loss')
        grads = None
        if probe:
            grads = [float(x.norm()) for x in torch.autograd.grad(
                rgb, (code, decoder.base.features.weight), retain_graph=True)]
            if any(not math.isfinite(x) or x <= 0 for x in grads):
                raise RuntimeError('RGB gradient failed to reach ' + branch + ' code or decoder')
        (.5*loss).backward()
        rows[branch] = {'loss': float(loss.detach()), 'rgb': float(rgb.detach()),
            'motion_weighted_mae': float(terms['motion_mae'].detach()),
            'latent': float(latent.detach()), 'occupancy': float(latent_terms['occupancy'].detach()),
            'feature': float(latent_terms['feature'].detach()), 'rgb_to_code_decoder_gradients': grads}
        del prediction, latent, latent_terms, state, images, rgb, terms, loss
    if any(x.grad is not None and not bool(torch.isfinite(x.grad).all()) for x in expected):
        raise RuntimeError('Nonfinite paired training gradient')
    if any(x.grad is not None for x in decoder.parameters() if not x.requires_grad):
        raise RuntimeError('Frozen initial encoder or internal warm codes received gradients')
    if any(x.grad is not None for x in bridge.parameters()):
        raise RuntimeError('Frozen native codec received gradients')
    norm = float(torch.nn.utils.clip_grad_norm_(expected, p['grad_clip']))
    optimizer.step()
    if any(not bool(torch.isfinite(x).all()) for x in expected):
        raise RuntimeError('Nonfinite paired training parameter')
    return {'branches': rows, 'preclip_gradient_norm': norm}


def evaluate(step_number, decoder, bridge, store, items, bank, p, cfg, old_motion, old_static):
    import numpy as np
    from PIL import Image
    import torch
    from data import evaluate_metrics
    from runtime import render
    from support_decoder import select_state, support_metrics
    import preview
    import static_probe
    out = ROOT/'result'/f'step_{step_number:05d}'
    out.mkdir()
    rows = []
    motion_diffs, static_diffs = [], []
    old_motion_rows = {(r['start'], r['horizon']): r for r in old_motion['rows'] if r['horizon'] > 0}
    old_static_rows = {(r['start'], r['horizon']): r for r in old_static['rows']}
    frames = step_number in (0, cfg['updates'])
    decoder.eval()
    with torch.no_grad():
        for i, start in enumerate(cfg['windows']):
            item = items[start]
            w = item['window']
            predictions = [decoder(item['condition'], bank[i]), decoder(item['condition'], bank[i+2])]
            folder = out/f's{start:03d}'
            if frames:
                folder.mkdir()
            for h in cfg['horizons']:
                sample = store.sample(start, h)
                record = item['records'][h]
                images = []
                for pred in predictions:
                    state = select_state(item['condition'].coordinates, pred['features'][h-1],
                                         pred['logits'][h-1], p['max_predicted_latent_nodes'])
                    rendered, _, _ = render(bridge, state, record, views=(0, 1))
                    images.append(rendered)
                moving, still = item['moving_images'][h], item['static_images'][h]
                controls = static_probe.weighted_controls(images[0][0], images[1][0], moving, still)
                motion_metrics = evaluate_metrics(images[0], w, sample, p)['metrics']
                support = support_metrics(predictions[1]['logits'][h-1], item['static_labels']['occupancy'][h-1])
                if step_number == 0:
                    before = old_motion_rows[(start, h)]['metrics']['correct']['metrics']['motion_mae']
                    motion_diffs.append(abs(motion_metrics['motion_mae']-before))
                    previous = old_static_rows[(start, h)]['metrics']
                    for key in ('static_code_to_static_reference', 'static_code_to_motion_reference', 'reference_motion_vs_static'):
                        static_diffs.append(abs(controls[key]-previous[key]))
                rows.append({'start': start, 'horizon': h, 'frame': start+h, 'metrics': controls,
                    'motion_source_metrics': motion_metrics, 'static_support': support,
                    'static_full_rgb_mae': float((images[1][0]-still).abs().mean())})
                if frames:
                    panels = [sample['record']['real'], moving, images[0][0], still, images[1][0]]
                    combined = np.concatenate([preview.rgb_array(x) for x in panels], axis=1)
                    Image.fromarray(combined, mode='RGB').save(folder/f'frame_{h:02d}.png')
            if frames:
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1',
                    '-i',str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                    str(out/f's{start:03d}_paired_5fps.mp4')], check=True, timeout=90)
    if step_number == 0 and (len(motion_diffs) != 18 or len(static_diffs) != 54
            or max(motion_diffs+static_diffs) > 5e-5):
        raise RuntimeError('Starting motion or optimized static predictions did not reproduce')
    summary = static_probe.aggregate(rows)
    summary['motion_source_motion_mae'] = sum(r['motion_source_metrics']['motion_mae'] for r in rows)/18
    summary['static_closer_frames'] = sum(r['metrics']['static_code_to_static_reference']
        < r['metrics']['static_code_to_motion_reference'] for r in rows)
    summary['static_full_rgb_mae'] = sum(r['static_full_rgb_mae'] for r in rows)/18
    summary['static_support_iou'] = sum(r['static_support']['iou'] for r in rows)/18
    write(out/'EVALUATION.json', {'status': 'complete', 'step': step_number, 'summary': summary, 'rows': rows,
        'initial_motion_reproduction_max_abs': max(motion_diffs) if motion_diffs else None,
        'initial_static_reproduction_max_abs': max(static_diffs) if static_diffs else None,
        'same_decoder_initial_condition_and_camera': True, 'visual_review': 'pending',
        'independent_native_gs_replay': 'not performed'})
    return summary


def run():
    import torch
    from torch import nn
    from schedule import schedule
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    verify_self()
    cfg = read(ROOT/'protocol.json')
    if (cfg['schema'] != 'two_window_same_initial_motion_static_free_codes_v1'
            or cfg['windows'] != [35, 123] or cfg['horizons'] != list(range(1, 10))
            or cfg['epochs'] != 20 or cfg['updates'] != 360 or cfg['quantized'] is not False):
        raise ValueError('Fixed paired protocol differs')
    result = ROOT/'result'
    result.mkdir()
    began = time.monotonic()
    phase = 'binding'
    write(result/'STARTED.json', {'pid': os.getpid(), 'protocol_sha256': sha(ROOT/'protocol.json'),
          'code_sha256': sha(ROOT/'CODE.sha256'), 'main_training_signalled': False})
    try:
        sys.path.insert(0, str(STATIC_CONTROL))
        import static_probe
        static_probe.verify_self()
        sys.path.insert(0, str(PREVIEW))
        import preview
        preview.verify_sources()
        sys.path.insert(0, str(BROOT))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle = load_bound_sources('cpu')
        runtime, store, decoder, fm, p = (bundle[k] for k in ('runtime', 'store', 'decoder', 'fm', 'p'))
        if (runtime.digest(decoder) != cfg['warm_decoder_digest']
                or sha(bundle['paths']['cache_manifest']) != cfg['cache_manifest_sha256']
                or p['codec_checkpoint_sha256'] != cfg['codec_sha256']):
            raise RuntimeError('Wrong original decoder, codec or state cache')
        summary_path = STATIC_OPT/'result/SUMMARY.json'
        checkpoint_path = STATIC_OPT/'result/codes_step_00540.pt'
        static_eval_path = STATIC_OPT/'result/step_00540/EVALUATION.json'
        motion_eval_path = Path(p['parent_root'])/'free16/evaluation/EVALUATION.json'
        if (sha(summary_path) != cfg['static_probe_summary_sha256']
                or sha(checkpoint_path) != cfg['static_probe_checkpoint_sha256']
                or sha(static_eval_path) != cfg['static_probe_evaluation_sha256']
                or sha(motion_eval_path) != cfg['warm_motion_evaluation_sha256']):
            raise RuntimeError('Prior motion or static source changed')
        previous = read(summary_path)
        if previous['status'] != 'complete' or previous['checkpoints']['540'] != sha(checkpoint_path):
            raise RuntimeError('Static probe did not finish or checkpoint differs')
        static_eval, motion_eval = read(static_eval_path), read(motion_eval_path)
        if static_eval['status'] != 'complete' or static_eval['summary'] != previous['summaries']['540']:
            raise RuntimeError('Static evaluation and summary differ')
        static_payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        static_codes = static_payload['codes']
        motion_codes = decoder.codes.detach().cpu().clone()
        if (static_payload['schema'] != 'two_window_static_free_codes_v1' or static_payload['step'] != 540
                or static_payload['starts'] != cfg['windows']
                or static_payload['decoder_digest'] != cfg['warm_decoder_digest']
                or static_payload['protocol_sha256'] != sha(STATIC_OPT/'protocol.json')
                or any(x.dtype != torch.float32 or tuple(x.shape) != (2, 9, 16) or not bool(torch.isfinite(x).all())
                       for x in (motion_codes, static_codes))):
            raise RuntimeError('Paired code initialization differs')
        phase = 'gpu_preflight'
        setup_gpu_once(dict(p, allocator_limit_gib=cfg['allocator_limit_gib'],
                            minimum_free_gpu_gib=cfg['minimum_free_gpu_gib']))
        decoder = decoder.cuda()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import target_labels
        from runtime import render
        from data import tree
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        std = store.frames['std'].cuda()
        items = {}
        with torch.no_grad():
            for start in cfg['windows']:
                w = store.get(start)
                s0 = {'coordinates': w['states'][0]['coordinates'].detach(),
                      'features': w['states'][0]['features'].detach()}
                records, static_images, moving_images, weights = {}, {}, {}, {}
                for h in cfg['horizons']:
                    record = camera_only(tree, store.frames['samples'][start+h]['record'])
                    still = render(bridge, s0, record, views=(0,))[0][0].detach()
                    moving = render(bridge, w['states'][h], record, views=(0,))[0][0].detach()
                    weight = ((moving-still).abs().mean(-1)/.1).clamp(0, 1)
                    if float(weight.sum()) <= 1e-6:
                        raise RuntimeError('No visible difference for paired loss weighting')
                    records[h], static_images[h], moving_images[h], weights[h] = record, still, moving, weight
                items[start] = {'window': w, 'condition': w['condition'], 'roi': w['roi'],
                    'records': records, 'static_images': static_images, 'moving_images': moving_images,
                    'weights': weights, 'static_labels': target_labels(w['condition'], [s0]*9)}
        initial_decoder = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
        initial_digest, frozen = runtime.digest(decoder), fm.frozen_digest(decoder)
        def bank_new():
            return nn.ParameterList([nn.Parameter(x.cuda().clone()) for x in
                                     (motion_codes[0], motion_codes[1], static_codes[0], static_codes[1])])
        def check_frozen():
            bridge.assert_frozen()
            if fm.frozen_digest(decoder) != frozen:
                raise RuntimeError('Frozen initial-state encoder or context changed')
        bank = bank_new()
        opt = make_optimizer(decoder, bank, p, cfg)
        decoder.train()
        trial = step(decoder, bridge, store, items[35], bank, opt, std, p, 0, 1, probe=True)
        check_frozen()
        decoder.load_state_dict(initial_decoder, strict=True)
        if runtime.digest(decoder) != initial_digest:
            raise RuntimeError('Preflight decoder reset failed')
        del bank, opt, initial_decoder
        bank = bank_new()
        opt = make_optimizer(decoder, bank, p, cfg)
        if opt.state:
            raise RuntimeError('Preflight optimizer state not reset')
        write(result/'PREFLIGHT.json', {'status': 'passed', 'discarded_paired_update': True,
            'decoder_codes_optimizer_restored': True, 'codec_and_initial_encoder_frozen': True,
            'motion_and_static_rgb_to_code_decoder_gradients': {
                k: trial['branches'][k]['rgb_to_code_decoder_gradients'] for k in ('motion', 'static')},
            'future_only_for_motion_target_and_static_loss_weight': True})
        order = schedule(cfg['windows'], cfg['horizons'], cfg['epochs'], cfg['seed'])
        if len(order) != cfg['updates']:
            raise RuntimeError('Paired schedule length differs')
        phase = 'evaluating_initial'
        summaries = {'0': evaluate(0, decoder, bridge, store, items, bank, p, cfg, motion_eval, static_eval)}
        with (result/'train.jsonl').open('x', encoding='utf8', buffering=1) as log:
            for number, (start, horizon) in enumerate(order, 1):
                phase = f'training_{number}'
                if time.monotonic()-began > cfg['maximum_gpu_seconds']:
                    raise TimeoutError('Fixed paired diagnostic budget exceeded')
                decoder.train()
                row = step(decoder, bridge, store, items[start], bank, opt, std, p,
                           cfg['windows'].index(start), horizon)
                log.write(json.dumps({'step': number, 'start': start, 'horizon': horizon, **row}, allow_nan=False)+'\n')
                if number in cfg['checkpoint_steps']:
                    phase = f'evaluating_{number}'
                    check_frozen()
                    path = result/f'checkpoint_{number:05d}.pt'
                    with path.open('xb') as stream:
                        torch.save({'schema': cfg['schema'], 'step': number, 'decoder': decoder.state_dict(),
                            'decoder_digest': runtime.digest(decoder), 'windows': cfg['windows'],
                            'motion_codes': torch.stack([bank[0].detach().cpu(), bank[1].detach().cpu()]),
                            'static_codes': torch.stack([bank[2].detach().cpu(), bank[3].detach().cpu()]),
                            'optimizer': opt.state_dict(), 'protocol_sha256': sha(ROOT/'protocol.json')}, stream)
                    summaries[str(number)] = evaluate(number, decoder, bridge, store, items, bank, p, cfg,
                                                       motion_eval, static_eval)
        check_frozen()
        initial, final = summaries['0'], summaries[str(cfg['updates'])]
        moving_ok = final['motion_source_motion_mae'] <= 1.1*initial['motion_source_motion_mae']
        static_count_ok = final['static_closer_frames'] >= 16
        s = final['overall']
        static_mean_ok = s['static_code_to_static_reference'] <= .5*s['reference_motion_vs_static']
        write(result/'SUMMARY.json', {'status': 'complete', 'updates': cfg['updates'], 'summaries': summaries,
            'exploratory_numerical_gate': {'motion_no_more_than_110pct': moving_ok,
                'static_closer_at_least_16_of_18': static_count_ok,
                'static_error_at_most_half_reference_difference': static_mean_ok,
                'all_numeric_conditions': moving_ok and static_count_ok and static_mean_ok,
                'visual_review_required': True}, 'decoder_and_codes_jointly_trained': True,
            'true_future_never_in_decoder_condition': True, 'main_training_signalled': False,
            'elapsed_seconds': time.monotonic()-began, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'checkpoints': {str(s): sha(result/f'checkpoint_{s:05d}.pt') for s in (90,180,360)},
            'visual_review': 'pending', 'limits': cfg['limits']})
        print('PASS: paired free-code diagnostic completed; numerical and visual acceptance remain distinct', flush=True)
    except BaseException as error:
        write(result/'FAILED.json', {'status': 'failed', 'phase': phase, 'error': repr(error),
            'traceback': traceback.format_exc(), 'main_training_signalled': False, 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    run()
