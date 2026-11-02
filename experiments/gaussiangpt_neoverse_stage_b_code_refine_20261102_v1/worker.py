"""Optimize two continuous code sequences with the stage-B decoder frozen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT.parent
B_PREP = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
DIAG = OUTPUTS / 'gaussiangpt_neoverse_stage_b_anchor_diagnosis_20261102_v1'
STARTS = (35, 123)
CHECKPOINTS = (90, 180, 360)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, replace=False):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)


def schedule(sweeps, seed):
    if type(sweeps) is not int or sweeps <= 0 or type(seed) is not int:
        raise ValueError('Expected a positive sweep count and integer seed')
    rng, steps = random.Random(seed), []
    for sweep in range(1, sweeps + 1):
        horizons = list(range(1, 10))
        rng.shuffle(horizons)
        for horizon in horizons:
            steps.append(dict(step=len(steps)+1, sweep=sweep, horizon=horizon))
    return steps


def audit_schedule(steps, sweeps):
    if len(steps) != 9*sweeps or [x['step'] for x in steps] != list(range(1, len(steps)+1)):
        raise RuntimeError('Unexpected schedule length or order')
    for sweep in range(1, sweeps+1):
        horizons = [x['horizon'] for x in steps if x['sweep'] == sweep]
        if sorted(horizons) != list(range(1, 10)):
            raise RuntimeError('A future horizon was omitted or duplicated')
    return dict(status='passed', windows=list(STARTS), sweeps=sweeps,
                updates=len(steps), per_window_horizon_visits=sweeps,
                sampler_sha256=hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest())


def validate_config(config):
    if (config.get('schema') != 'fixed_b20_anchor_code_refinement_v1'
            or config.get('starts') != list(STARTS) or config.get('sweeps') != 40
            or config.get('logical_batch_size') != 2 or config.get('code_learning_rate') != .003
            or config.get('checkpoint_updates') != list(CHECKPOINTS)
            or config.get('max_seconds') != 3600 or config.get('quantized') is not False
            or config.get('decoder_frozen') is not True or config.get('codec_frozen') is not True
            or config.get('optimizer_state') != 'fresh_adam'):
        raise RuntimeError('Fixed-budget anchor refinement protocol differs')
    steps = schedule(config['sweeps'], config['sampler_seed'])
    return steps, audit_schedule(steps, config['sweeps'])


def verify_sources():
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Anchor refinement witness changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Anchor refinement source changed: ' + name)
    diag_witness = read(DIAG / 'CPU_TESTS.json')
    if diag_witness.get('status') != 'passed' or diag_witness.get('code_sha256') != sha(DIAG / 'CODE.sha256'):
        raise RuntimeError('Stage-B diagnosis witness changed')
    for line in (DIAG / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(DIAG / name) != digest:
            raise RuntimeError('Stage-B diagnosis source changed: ' + name)
    diagnosis = read(DIAG / 'result/DIAGNOSIS.json')
    if (sha(DIAG/'result/DIAGNOSIS.json') != 'ca5d042e622a9dd32e0bc6b7b37d7bde6c2ad1fe75dbf28f8f8679d8f510d589'
            or diagnosis.get('status') != 'complete' or diagnosis.get('oracle_is_post_inference_only') is not True
            or diagnosis.get('training_performed') is not False):
        raise RuntimeError('Completed post-inference B diagnosis changed')
    spec = importlib.util.spec_from_file_location('_bound_b_anchor_diagnosis', DIAG / 'diagnose.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluation, checkpoint, summary_sha, replay_sha = module.ensure_sources()
    config = read(ROOT / 'protocol.json')
    steps, sampler = validate_config(config)
    return module, evaluation, checkpoint, summary_sha, replay_sha, config, steps, sampler


def image_array(tensor):
    image = (np.clip(tensor.detach().cpu().numpy(), 0, 1)*255).round().astype(np.uint8)
    if image.shape != (252, 448, 3):
        raise RuntimeError('Unexpected review image shape')
    return image


@torch.no_grad()
def evaluate(step, decoder, bank, bridge, store, p, reference, runtime, digest, out):
    from data import evaluate_metrics
    from runtime import render
    from support_decoder import select_state
    rows = []
    folder = out / ('step_%05d' % step)
    folder.mkdir()
    for index, start in enumerate(STARTS):
        window = store.get(start)
        prediction = decoder(window['condition'], bank[index])
        for horizon in range(1, 10):
            sample = store.sample(start, horizon)
            state = select_state(window['condition'].coordinates, prediction['features'][horizon-1],
                                 prediction['logits'][horizon-1], p['max_predicted_latent_nodes'])
            images, _, _ = render(bridge, state, sample['record'], views=(0, 1))
            metrics = evaluate_metrics(images, window, sample, p)
            old = reference[(start, horizon)]
            if step == 0 and abs(metrics['metrics']['motion_mae'] - old['metrics']['correct']['metrics']['motion_mae']) > 1e-5:
                raise RuntimeError('Initial frozen decoder and code do not reproduce B evaluation')
            if step == 360 or horizon in (1, 3, 6, 9):
                path = folder / ('s%03d_h%02d.png' % (start, horizon))
                Image.fromarray(np.concatenate([image_array(sample['record']['real']), image_array(images[0])], 1)).save(path)
            rows.append(dict(start=start, horizon=horizon, metrics=metrics,
                             b20_motion_mae=old['metrics']['correct']['metrics']['motion_mae'],
                             codec_reference=old['metrics']['actual_target']))
    if runtime.digest(decoder) != digest or any(x.requires_grad for x in decoder.parameters()):
        raise RuntimeError('Frozen decoder changed during evaluation')
    bridge.assert_frozen()
    per_window = {}
    for start in STARTS:
        selected = [r for r in rows if r['start'] == start]
        if len(selected) != 9:
            raise RuntimeError('Incomplete anchor evaluation')
        per_window[str(start)] = dict(
            motion_mae=sum(r['metrics']['metrics']['motion_mae'] for r in selected)/9,
            b20_motion_mae=sum(r['b20_motion_mae'] for r in selected)/9,
            actual_target_motion_mae=sum(r['codec_reference']['metrics']['motion_mae'] for r in selected)/9,
            object_proxy_iou=sum(r['metrics']['object_proxy']['iou'] for r in selected)/9,
            color_proxy_detected=sum(r['metrics']['object_proxy']['prediction']['valid'] for r in selected))
    result = dict(status='complete', step=step, rows=rows, per_window=per_window,
        source='same B20 decoder and codec; only two continuous code tensors are optimized',
        oracle_target_support_used=False, training_window_evaluation=True, manual_visual_review_required=True)
    write(folder / 'EVALUATION.json', result)
    return result


def train_step(bank, decoder, bridge, store, std, optimizer, horizon, p, *, probe=False):
    from core import image_losses
    from runtime import render
    from support_decoder import select_state, state_loss
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise RuntimeError('The fixed decoder became trainable')
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for index, start in enumerate(STARTS):
        window = store.get(start)
        prediction = decoder(window['condition'], bank[index])
        latent, terms = state_loss(prediction, window['targets'], std)
        state = select_state(window['condition'].coordinates, prediction['features'][horizon-1],
                             prediction['logits'][horizon-1], p['max_predicted_latent_nodes'])
        sample = store.sample(start, horizon)
        images, gs, _ = render(bridge, state, sample['record'])
        rgb, terms_rgb = image_losses(images[0], sample['record']['real'], window['roi'], sample['motion'])
        if probe:
            gradient = torch.autograd.grad(rgb, bank, retain_graph=True)[0][index]
            norm = float(gradient.norm())
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError('RGB loss does not reach continuous code')
        loss = rgb + .1*latent
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite anchor loss')
        (loss/len(STARTS)).backward()
        rows.append(dict(start=start, horizon=horizon, loss=float(loss.detach()),
            motion_mae=float(terms_rgb['motion_mae'].detach()), latent_loss=float(latent.detach()),
            occupancy_loss=float(terms['occupancy'].detach()), feature_loss=float(terms['feature'].detach()),
            gs_count=len(gs), rgb_to_code_gradient_norm=norm if probe else None))
        del window, prediction, latent, terms, state, sample, images, gs, rgb, terms_rgb, loss
    if bank.grad is None or not bool(torch.isfinite(bank.grad).all()):
        raise RuntimeError('Missing or nonfinite code gradient')
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError('Frozen decoder received a gradient')
    norm = float(torch.nn.utils.clip_grad_norm_([bank], p['grad_clip']))
    optimizer.step()
    if not bool(torch.isfinite(bank).all()):
        raise RuntimeError('Nonfinite updated codes')
    return dict(windows=rows, preclip_gradient_norm=norm)


def main(output):
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    diagnostic, evaluation, checkpoint, summary_sha, replay_sha, config, steps, sampler = verify_sources()
    output = Path(output)
    if not output.is_absolute() or output.exists() or output.parent.resolve() != ROOT.resolve():
        raise RuntimeError('A new absolute output directory under this experiment is required')
    output.mkdir()
    began, phase = time.monotonic(), 'initializing'
    write(output/'STARTED.json', dict(pid=os.getpid(), schema=config['schema'], gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        planned_updates=len(steps), training_scope='two anchor windows; not full-168 training'))
    try:
        sys.path.insert(0, str(B_PREP))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle = load_bound_sources('cpu')
        binding, p, fm, runtime, store = (bundle[k] for k in ('binding', 'p', 'fm', 'runtime', 'store'))
        if p['codec_checkpoint_sha256'] != diagnostic.CODEC_SHA or binding.sha(bundle['paths']['cache_manifest']) != diagnostic.CACHE_SHA:
            raise RuntimeError('Codec or cache binding changed')
        item = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (item.get('schema') != 'stage_b_shared_decoder_free_codes_v1' or item.get('sweep') != 20
                or item.get('starts') != list(range(168)) or tuple(item['codes'].shape) != (168,9,16)
                or item.get('decoder_digest') != diagnostic.DECODER_DIGEST):
            raise RuntimeError('Stage-B checkpoint ordering changed')
        decoder = bundle['decoder']
        decoder.load_state_dict(item['decoder'], strict=True)
        if runtime.digest(decoder) != diagnostic.DECODER_DIGEST or fm.frozen_digest(decoder) != bundle['frozen_digest']:
            raise RuntimeError('Initial decoder or context differs')
        decoder.requires_grad_(False).eval()
        setup_gpu_once(p)
        decoder = decoder.cuda()
        from runtime_bridge import BoundNativeBridge
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        if any(x.requires_grad for x in bridge.parameters()):
            raise RuntimeError('GS codec is not frozen')
        std = store.frames['std'].cuda()
        initial = item['codes'][list(STARTS)].clone().cuda()
        bank = nn.Parameter(initial.clone())
        def optimizer():
            return torch.optim.Adam([bank], lr=config['code_learning_rate'], betas=tuple(p['betas']), eps=p['eps'])
        original_rows = {(r['start'], r['horizon']): r for r in evaluation['rows'] if r['horizon'] > 0}
        if len(original_rows) != 1512:
            raise RuntimeError('B reference coverage differs')
        baseline = evaluate(0, decoder, bank, bridge, store, p, original_rows, runtime, diagnostic.DECODER_DIGEST, output)
        phase = 'gpu_preflight'
        opt = optimizer()
        probe = train_step(bank, decoder, bridge, store, std, opt, 9, p, probe=True)
        with torch.no_grad():
            bank.copy_(initial)
        if not torch.equal(bank.detach(), initial) or runtime.digest(decoder) != diagnostic.DECODER_DIGEST:
            raise RuntimeError('Preflight reset did not restore codes or decoder')
        opt = optimizer()
        if opt.state:
            raise RuntimeError('Optimizer state was not reset after preflight')
        bridge.assert_frozen()
        write(output/'PREFLIGHT.json', dict(status='passed', discarded_updates=1,
            rgb_to_each_code_gradient_verified=True, codes_and_optimizer_restored=True,
            decoder_and_codec_frozen=True, probe=probe))
        source = dict(B_summary_sha256=summary_sha, B_evaluation_sha256=diagnostic.EVAL_SHA,
            B_checkpoint_sha256=diagnostic.CHECKPOINT_SHA, B_native_replay_sha256=replay_sha,
            diagnosis_sha256=sha(DIAG/'result/DIAGNOSIS.json'), decoder_digest=diagnostic.DECODER_DIGEST,
            codec_sha256=diagnostic.CODEC_SHA, cache_sha256=diagnostic.CACHE_SHA,
            code_sha256=sha(ROOT/'CODE.sha256'), sampler=sampler, quantized=False)
        phase = 'training'
        with (output/'train.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            for item in steps:
                if time.monotonic()-began > config['max_seconds']:
                    raise RuntimeError('Fixed training time budget exceeded')
                row = train_step(bank, decoder, bridge, store, std, opt, item['horizon'], p)
                log.write(json.dumps(dict(**item, elapsed_seconds=time.monotonic()-began, **row), allow_nan=False)+'\n')
                if item['step'] == 1 or item['step'] % 30 == 0:
                    write(output/'STATUS.json', dict(status='running', phase=phase, completed_updates=item['step'],
                        total_updates=len(steps), elapsed_seconds=time.monotonic()-began), replace=True)
                if item['step'] in CHECKPOINTS:
                    bridge.assert_frozen()
                    if runtime.digest(decoder) != diagnostic.DECODER_DIGEST:
                        raise RuntimeError('Frozen B decoder changed')
                    path = output/('codes_%05d.pt' % item['step'])
                    with path.open('xb') as stream:
                        torch.save(dict(schema='two_anchor_continuous_codes_v1', starts=list(STARTS),
                            step=item['step'], codes=bank.detach().cpu(), source=source), stream)
                    value = evaluate(item['step'], decoder, bank, bridge, store, p, original_rows,
                                     runtime, diagnostic.DECODER_DIGEST, output)
                    write(output/('checkpoint_%05d.json' % item['step']), dict(status='complete', step=item['step'],
                        codes_sha256=sha(path), evaluation_sha256=sha(output/('step_%05d/EVALUATION.json' % item['step'])),
                        per_window=value['per_window']))
        if runtime.digest(decoder) != diagnostic.DECODER_DIGEST or fm.frozen_digest(decoder) != bundle['frozen_digest']:
            raise RuntimeError('Frozen decoder or initial-condition encoder changed')
        bridge.assert_frozen()
        write(output/'DONE.json', dict(status='complete', updates=len(steps), source=source,
            checkpoints=list(CHECKPOINTS), final_codes_sha256=sha(output/'codes_00360.pt'),
            decoder_and_codec_frozen=True, training_scope='two anchors only',
            manual_visual_review_required=True, full_action_codec_complete=False,
            elapsed_seconds=time.monotonic()-began))
        write(output/'STATUS.json', dict(status='complete', phase='awaiting_visual_review',
            completed_updates=len(steps), total_updates=len(steps)), replace=True)
    except BaseException as error:
        failure = dict(status='failed', phase=phase, error=repr(error), traceback=traceback.format_exc(),
                       no_automatic_retry=True)
        write(output/'FAILED.json', failure)
        write(output/'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.output)
