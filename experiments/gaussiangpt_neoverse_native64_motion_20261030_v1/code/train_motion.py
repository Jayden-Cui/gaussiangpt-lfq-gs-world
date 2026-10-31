"""True batch4 native64 motion training, with joint LFQ entropy statistics."""
import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from motion_common import BASE, read, write, sha, frozen_codec, load_pair, load_target, Vgg19, pair_ids
from qmotion import build_model
from losses_motion import all_losses
from contracts import verify_sources


def parameter_sha(module, exclude_quantizer=False):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if exclude_quantizer and name.startswith('quantizer.'):
            continue
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, **metadata):
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def batch_loss(model, codec, records, images, vgg, protocol, *, rendering, force_target):
    starts, ends, initials, conditions, target_indices = zip(*records)
    z = model.encode_motion_batch(starts, ends, [v.dt for v in initials])
    predictions = model.restore_batch(z['z_quantized'], conditions)
    weights = protocol['train']['render_loss'] if rendering else protocol['train']['warmup_loss']
    averaged = {}; details = []
    batch = len(records)
    for i, (prediction, target, idx) in enumerate(zip(predictions, ends, target_indices)):
        raw_keep = prediction['occupancy_logits'] > 0
        union = raw_keep.clone()
        if force_target: union[idx] = True
        terms, aux = all_losses(codec, prediction, target, idx, images[i] if rendering else None, vgg,
            force_target=force_target, rendering=rendering, max_tokens=protocol['support']['max_predicted_tokens'])
        for key, value in terms.items():
            averaged[key] = averaged.get(key, 0) + value / batch
        details.append({'candidate_count': len(raw_keep), 'predicted_count': int(raw_keep.sum()),
            'kept_count': int(union.sum()), 'target_count': len(idx),
            'forced_additions': int((union & ~raw_keep).sum()),
            'decoded_gaussians': codec._motion_decode_stats.get('gaussians') if rendering else None,
            'decoder_candidates': codec._motion_decode_stats.get('max_candidates') if rendering else None})
        del terms, aux
    # LFQ's returned scalar already includes entropy_loss_weight; do not multiply twice.
    # One joint call above pools all B*K=16 tokens.
    loss = sum(weights[k] * value for k, value in averaged.items()) + z['aux_loss']
    return loss, averaged, z, details


def train(root, arm):
    verify_sources(root)
    protocol = read(root/'protocol.json'); settings = protocol['train']
    config = next(a for a in protocol['arms'] if a['name'] == arm)
    assert settings['batch_size'] == 4 and settings['max_steps'] * 4 == settings['total_pair_exposures']
    torch.set_num_threads(6); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.60)
    random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
    codec = frozen_codec(root); frozen_before = parameter_sha(codec)
    model = build_model(config).cuda().train()
    common_initial = parameter_sha(model, exclude_quantizer=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    vgg = Vgg19(BASE/'torch_cache').cuda().eval().requires_grad_(False)
    folder = root/arm; folder.mkdir(parents=True, exist_ok=True)
    pairs = pair_ids(protocol); rng = random.Random(protocol['seed'])
    order = list(range(len(pairs))); rng.shuffle(order)
    cursor = 0; epochs = 0; visited = set(); exposures = 0; no_force_steps = 0
    max_kept = 0; max_predicted = 0; completed = 0; last = {}; event = 'initialization'; context = {}
    started = time.monotonic()

    def snapshot(path, resumable=True):
        save_checkpoint(path, model, optimizer, arm=arm, quantizer=config['quantizer'], width=64,
            motion_tokens=config['motion_tokens'], step=completed, batch_size=settings['batch_size'],
            protocol_sha256=sha(root/'protocol.json'), source_checkpoint_sha256=protocol['source_checkpoint_sha256'],
            source_binding_sha256=sha(root/'source_binding.json'), resumable=resumable,
            elapsed_seconds=time.monotonic()-started, pair_exposures=exposures,
            sampler={'order': order, 'cursor': cursor, 'rng': rng.getstate(), 'visited': sorted(visited), 'epochs': epochs},
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
            no_force_steps=no_force_steps, common_initial_parameter_sha256=common_initial)

    def terminate(signum, frame):
        raise RuntimeError(f'Training interrupted by signal {signum}')
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        with (folder/'train_metrics.jsonl').open('x', buffering=1) as log:
            for step in range(1, settings['max_steps']+1):
                elapsed = time.monotonic()-started
                if elapsed >= settings['max_seconds']: break
                selected = []
                for _ in range(settings['batch_size']):
                    if cursor == len(order):
                        rng.shuffle(order); cursor = 0; epochs += 1
                    selected.append(pairs[order[cursor]]); cursor += 1
                event = 'load_batch'; context = {'attempted_step': step, 'pairs': selected}
                records = [load_pair(root, c, s, e, codec) for c, s, e in selected]
                rendering = step > settings['warmup_steps']
                fraction = max((step-1)/settings['max_steps'], elapsed/settings['max_seconds'])
                force_target = fraction < .8
                images = [load_target(c, e) for c, s, e in selected] if rendering else None
                optimizer.zero_grad(set_to_none=True); event = 'forward'
                loss, terms, z, details = batch_loss(model, codec, records, images, vgg, protocol,
                    rendering=rendering, force_target=force_target)
                if not bool(torch.isfinite(loss)): raise RuntimeError(f'Nonfinite loss at step {step}')
                event = 'backward'; loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
                if any(p.grad is not None or p.requires_grad for p in codec.parameters()):
                    raise RuntimeError('Frozen state codec received a parameter gradient')
                event = 'optimizer_step'; optimizer.step(); completed = step; event = 'step_complete'
                visited.update((c, s) for c, s, e in selected); exposures += len(selected)
                if rendering and not force_target: no_force_steps += 1
                max_kept = max(max_kept, *(d['kept_count'] for d in details))
                max_predicted = max(max_predicted, *(d['predicted_count'] for d in details))
                last = {'step': step, 'max_steps': settings['max_steps'], 'batch_size': len(selected),
                    'pair_exposures': exposures, 'elapsed_seconds': time.monotonic()-started,
                    'stage': 'render' if rendering else 'latent_warmup', **context, 'batch_details': details,
                    'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                    'terms': {k: float(v.detach()) for k, v in terms.items()},
                    'quantizer_aux_loss': float(z['aux_loss'].detach()),
                    'per_sample_entropy': float(z['per_sample_entropy'].detach()),
                    'codebook_entropy': float(z['codebook_entropy'].detach()),
                    'motion_ids': z['indices'].detach().cpu().tolist(),
                    'target_support_forcing': force_target, 'unique_pairs_seen': len(visited),
                    'no_target_forcing_render_steps': no_force_steps,
                    'gpu_peak_gib': torch.cuda.max_memory_allocated()/2**30}
                if step == 1 or step % 10 == 0 or step == settings['max_steps']:
                    log.write(json.dumps(last, allow_nan=False)+'\n'); write(folder/'train_progress.json', last)
                    print('TRAIN', arm, step, 'loss', round(last['loss'], 6), 'pairs', exposures, flush=True)
                if step == 1 or step % settings['save_every'] == 0: snapshot(folder/'latest.pt')
                del records, images, loss, terms, z, details
        assert completed > 0
        frozen_after = parameter_sha(codec); assert frozen_after == frozen_before
        snapshot(folder/'motion_final.pt')
        result = {'status': 'complete', 'arm': arm, 'steps': completed, 'max_steps': settings['max_steps'],
            'batch_size': settings['batch_size'], 'pair_exposures': exposures,
            'stop_reason': 'step_budget' if completed == settings['max_steps'] else 'wall_time_budget',
            'elapsed_seconds': time.monotonic()-started, 'last': last,
            'unique_pairs_seen': len(visited), 'all_pairs_seen': len(visited) == len(pairs),
            'no_target_forcing_render_steps': no_force_steps, 'max_kept_tokens': max_kept,
            'max_predicted_tokens': max_predicted, 'common_initial_parameter_sha256': common_initial,
            'frozen_parameter_sha256_before': frozen_before, 'frozen_parameter_sha256_after': frozen_after,
            'checkpoint_sha256': sha(folder/'motion_final.pt'), 'completed_epochs': epochs,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write(folder/'train_done.json', result)
        print('TRAIN_COMPLETE', arm, completed, flush=True)
        return result
    except BaseException as error:
        checkpoint_error = None
        try: snapshot(folder/'exception.pt', resumable=event == 'step_complete')
        except BaseException as save_error: checkpoint_error = repr(save_error)
        write(folder/'training_failure.json', {'status': 'failed', 'error': repr(error),
            'last_completed_step': completed, 'event': event, 'context': context,
            'checkpoint_error': checkpoint_error})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True)
    a = p.parse_args(); train(a.root, a.arm)
