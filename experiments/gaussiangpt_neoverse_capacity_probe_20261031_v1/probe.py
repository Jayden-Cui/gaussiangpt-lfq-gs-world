"""Frozen v7 per-window capacity oracles; target fitting, never prediction/AR.

Only torch and the standard library are imported by CPU tests. GPU execution
reuses the immutable terminal loader and retains every initial coordinate row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

import torch

sys.dont_write_bytecode = True
BRANCHES = ('continuous_z', 'bounded_features', 'unbounded_features')
STEPS = (0, 25, 50, 75, 100)
HELPER_SHA = 'a3430beacb0ffb3eeff8bb5ce8988428ac418b4fc8c3fab5345e3cb49d5689db'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_manifest(root):
    manifest = read(root / 'CODE_MANIFEST.json')
    if not isinstance(manifest, dict) or not {'probe.py', 'protocol.json'} <= set(manifest):
        raise ValueError('Manifest must bind probe.py and protocol.json')
    for name, digest in manifest.items():
        if not isinstance(digest, str) or sha(root / name) != digest:
            raise ValueError('Code manifest mismatch: ' + name)
    return sha(root / 'CODE_MANIFEST.json')


def validate_protocol(cfg):
    expected = dict(schema='gs_capacity_probe_protocol_v1', source_arm='z_delta',
                    selected_step=400, starts=[35, 123], branches=list(BRANCHES))
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Unexpected fixed protocol field: ' + key)
    if cfg['optimization'] != dict(steps=100, lr=.03, optimizer='Adam', checkpoints=list(STEPS)):
        raise ValueError('Optimization budget/recipe differs')
    if cfg['resources'] != dict(cpu_threads=6, max_seconds=1500, peak_limit_gib=48):
        raise ValueError('Worker resource contract differs')
    if cfg['baseline_tolerances'] != dict(mae=1e-4, psnr=.02):
        raise ValueError('Baseline parity tolerances differ')


def normalized_residual(features, initial, units):
    """Recover the current posterior without replacing zero-std channel units."""
    if features.ndim != 3 or initial.shape != features.shape[1:] or units.shape != (features.shape[-1],):
        raise ValueError('Expected features [T,N,D], initial [N,D], units [D]')
    if not all(bool(torch.isfinite(v).all()) for v in (features, initial, units)) or bool((units < 0).any()):
        raise ValueError('Invalid residual inputs')
    positive = units > 0
    difference = features - initial[None]
    if bool((difference[..., ~positive] != 0).any()):
        raise ValueError('Zero-std channel cannot represent posterior difference')
    residual = torch.zeros_like(features)
    residual[..., positive] = difference[..., positive] / units[positive]
    return residual


@torch.no_grad()
def project_residual_(parameter):
    parameter.clamp_(-1., 1.)


def staged_backward(features, loss_for_leaf):
    """One native graph at a time, then one VJP; no retained graph or auxiliary."""
    features = tuple(features)
    if not features or any(not f.requires_grad for f in features):
        raise ValueError('All output features must belong to the oracle graph')
    gradients, value = [], 0.
    for index, feature in enumerate(features):
        leaf = feature.detach().requires_grad_(True)
        loss = loss_for_leaf(index, leaf)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite/nonscalar native loss')
        loss.backward()
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
            raise RuntimeError('Disconnected/nonfinite native feature gradient')
        gradients.append(leaf.grad.detach())
        value += float(loss.detach())
        del leaf, loss
    torch.autograd.backward(features, grad_tensors=tuple(gradients))
    return value, [float(g.norm()) for g in gradients]


def assert_frozen(*modules):
    for module in modules:
        if any(child.training for child in module.modules()):
            raise RuntimeError('A frozen module entered training mode')
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError('Frozen parameter acquired a gradient or was unfrozen')


def parity(rows, expected, tolerances):
    if len(rows) != 10 or len(expected) != 10:
        raise ValueError('Parity requires the complete ten-frame window')
    maximum = {}
    for actual, reference in zip(rows, expected):
        for key, value in reference.items():
            if key not in actual or not isinstance(value, (int, float)):
                raise ValueError('Missing source metric: ' + key)
            delta = abs(actual[key] - value)
            tolerance = tolerances['psnr' if 'psnr' in key else 'mae']
            if not math.isfinite(delta) or delta > tolerance:
                raise RuntimeError(f'Posterior parity failed: {key} difference {delta}')
            maximum[key] = max(maximum.get(key, 0.), delta)
    return maximum


def metric_summary(rows):
    keys = tuple(rows[0]['metrics'])
    reduce = lambda selected: {k: statistics.fmean(r['metrics'][k] for r in selected) for k in keys}
    return dict(per_horizon=rows, h0=rows[0]['metrics'], mean_all10=reduce(rows), mean_future9=reduce(rows[1:]),
                motion_metric_scope='Use mean_future9.motion_mae; h0 motion=0 is the inherited zero-weight convention')


def boundary_stats(features, initial, units):
    r = normalized_residual(features.detach(), initial, units)
    positive = units > 0
    if not bool(positive.any()):
        raise RuntimeError('No nonzero residual units')
    absolute = r[..., positive].abs()
    return dict(near_boundary_abs_r_ge_099=float((absolute >= .99).float().mean()),
                outside_abs_r_gt_1_plus_1e_minus6=float((absolute > 1.000001).float().mean()),
                max_abs_r=float(absolute.max()), zero_unit_channels=(~positive).nonzero().flatten().tolist(),
                per_horizon_near_boundary=(absolute >= .99).float().mean((1, 2)).cpu().tolist())


def run(root, start, out):
    began = time.monotonic()
    cfg = read(root / 'protocol.json'); validate_protocol(cfg)
    manifest_hash = verify_manifest(root)
    protocol_hash = sha(root / 'protocol.json')
    if start not in cfg['starts']:
        raise ValueError('Undeclared diagnostic window')
    def expired(signum, frame):
        raise TimeoutError('Fixed 1500-second worker wall budget expired')
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, expired); signal.alarm(cfg['resources']['max_seconds'])
    source = Path(cfg['source_root']).resolve()
    helper_path = source / 'postfit_code/render_six_panel.py'
    if sha(helper_path) != HELPER_SHA:
        raise ValueError('Pinned selected-checkpoint helper changed')
    spec = importlib.util.spec_from_file_location('_capacity_terminal_loader', helper_path)
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    context = helper.load_terminal(source)
    if context['step'] != cfg['selected_step']:
        raise ValueError('Expected selected step400')
    runtime = context['runner'].setup_runtime(context['evidence'])
    _, core, _, _, original_cfg, data = runtime
    if (original_cfg['resources']['cpu_threads'] != cfg['resources']['cpu_threads']
            or original_cfg['resources']['peak_limit_gib'] != cfg['resources']['peak_limit_gib']):
        raise ValueError('Original runtime resources differ')
    torch.set_num_threads(cfg['resources']['cpu_threads'])
    model, bridge = helper.restore_selected(context, cfg['source_arm'], core, torch, original_cfg, data)
    torch.cuda.reset_peak_memory_stats()
    def frozen_hashes():
        assert_frozen(model, bridge)
        return dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
    initial_hashes = frozen_hashes()
    def check():
        if time.monotonic() - began > cfg['resources']['max_seconds']:
            raise TimeoutError('Fixed worker wall budget exceeded')
        if torch.cuda.max_memory_allocated() / 2**30 > cfg['resources']['peak_limit_gib']:
            raise RuntimeError('Original48GiB GPU guard exceeded')
        assert_frozen(model, bridge)
    initial = data.states[start]
    coordinates = initial['coordinates']
    initial_f = initial['features'].detach()
    units = (model.output_std * model.residual_scale).detach()
    condition = [core.make_condition(initial)]
    # Future states are admitted ONLY to obtain the unchanged posterior baseline.
    with torch.no_grad():
        encoded = model.encode_windows([data.window(start)])
        z0, ids = encoded['z_quantized'].detach().clone(), encoded['indices'].detach().cpu()
        contexts = encoded['initial_contexts']
        baseline = model.decode_latents(z0, condition, initial_contexts=contexts)[0]
        baseline_f = torch.stack([p['features'] for p in baseline]).detach().clone()
    del encoded
    if z0.shape != (1, 9, 64) or ids.shape != (1, 9):
        raise RuntimeError('Posterior shape differs')
    evaluated = context['arms'][cfg['source_arm']]
    if ids[0].tolist() != evaluated['rows'][(start, 1)]['motion_ids']:
        raise RuntimeError('Baseline posterior IDs differ from selected EVAL')
    r0 = normalized_residual(baseline_f, initial_f, units)
    reconstructed = initial_f[None] + units * r0
    initialization_error = float((reconstructed - baseline_f).abs().max())
    if float(r0.abs().max()) > 1.00001 or not torch.allclose(reconstructed, baseline_f, rtol=1e-6, atol=1e-6):
        raise RuntimeError('Posterior residual initialization does not reproduce the baseline')
    from PIL import Image, ImageDraw
    galleries = {}
    def save_png(image, path):
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
    def observe(predictions, label=None):
        rows, images, loss_value = [], [], 0.
        if len(predictions) != 10:
            raise RuntimeError('All ten output frames are required')
        if label:
            (out / label).mkdir()
        with torch.no_grad():
            for h, prediction in enumerate(predictions):
                check()
                if label != 'teacher_state_reference' and not torch.equal(prediction['coordinates'], coordinates):
                    raise RuntimeError('Initial latent coordinates changed')
                image = bridge.render(prediction, data.records[start + h])
                metrics = data.metrics(image, start, h)
                loss_value += float(data.loss(image, start, h)[0])
                rows.append(dict(horizon=h, frame=start+h, metrics=metrics, gs_count=bridge.last_count))
                if label:
                    png = Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).round().astype('uint8'))
                    save_png(png, out / label / f'h{h:02d}_frame{start+h:04d}.png'); images.append(png)
        if label:
            galleries[label] = images
        return dict(**metric_summary(rows), loss=loss_value)
    baseline_metrics = observe(baseline, 'baseline')
    expected = [evaluated['initial'][start]['correct']] + [evaluated['rows'][(start, h)]['correct'] for h in range(1, 10)]
    parity_errors = parity([r['metrics'] for r in baseline_metrics['per_horizon']], expected, cfg['baseline_tolerances'])
    # This positive reference is never used by any oracle parameter or initializer.
    teacher_metrics = observe(data.states[start:start+10], 'teacher_state_reference')
    (out / 'real').mkdir(); galleries['real'] = []
    for h in range(10):
        png = Image.fromarray((data.records[start+h]['real'].clamp(0, 1).cpu().numpy()*255).round().astype('uint8'))
        save_png(png, out / 'real' / f'h{h:02d}_frame{start+h:04d}.png'); galleries['real'].append(png)
    write_new(out / 'BASELINE.json', dict(metrics=baseline_metrics, teacher_state_reference=teacher_metrics,
              parity_passed=True, metric_max_difference=parity_errors, initialization_max_feature_error=initialization_error,
              posterior_ids=ids.tolist(), source_checkpoint=context['selection']['checkpoints'][cfg['source_arm']]))
    branches = {}
    for branch in cfg['branches']:
        check(); before = frozen_hashes()
        if before != initial_hashes:
            raise RuntimeError('Frozen state changed before a branch')
        parameter = torch.nn.Parameter((z0 if branch == 'continuous_z' else r0).clone())
        optimizer = torch.optim.Adam([parameter], lr=cfg['optimization']['lr'])
        def predictions():
            if branch == 'continuous_z':
                return model.decode_latents(parameter, condition, initial_contexts=contexts)[0]
            f = initial_f[None] + units * parameter
            return [dict(coordinates=coordinates, features=f[h]) for h in range(10)]
        observations = {}
        with torch.no_grad():
            beginning = predictions()
            start_error = float((torch.stack([p['features'] for p in beginning])-baseline_f).abs().max())
            if not torch.allclose(torch.stack([p['features'] for p in beginning]), baseline_f, rtol=1e-6, atol=1e-6):
                raise RuntimeError('An oracle did not start from the posterior features')
            observations['0'] = observe(beginning)
        parity([r['metrics'] for r in observations['0']['per_horizon']],
               [r['metrics'] for r in baseline_metrics['per_horizon']], cfg['baseline_tolerances'])
        with (out / f'{branch}_trace.jsonl').open('x', encoding='utf-8') as trace:
            for step in range(1, cfg['optimization']['steps'] + 1):
                check(); optimizer.zero_grad(set_to_none=True); frames = predictions(); counts = []
                def loss_for_leaf(h, leaf):
                    check()
                    image = bridge.render(dict(coordinates=coordinates, features=leaf), data.records[start+h])
                    counts.append(bridge.last_count)
                    return data.loss(image, start, h)[0]
                loss, feature_gradients = staged_backward([p['features'] for p in frames], loss_for_leaf)
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError('Nonfinite/disconnected oracle parameter gradient')
                gradient_norm = float(parameter.grad.norm()); assert_frozen(model, bridge)
                if not math.isfinite(gradient_norm):
                    raise RuntimeError('Nonfinite oracle gradient norm')
                optimizer.step()
                if branch == 'bounded_features':
                    project_residual_(parameter)
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError('Nonfinite oracle parameters after update')
                del frames
                trace.write(json.dumps(dict(step=step, loss_before_update=loss, parameter_gradient_norm=gradient_norm,
                            feature_gradient_norms=feature_gradients, gs_count_min=min(counts), gs_count_max=max(counts)), allow_nan=False)+'\n')
                if step in cfg['optimization']['checkpoints']:
                    with torch.no_grad():
                        frames = predictions()
                        observations[str(step)] = observe(frames, branch if step == 100 else None)
                    trace.flush()
        with torch.no_grad():
            final_f = torch.stack([p['features'] for p in frames])
            boundary = boundary_stats(final_f, initial_f, units)
        after = frozen_hashes()
        if after != before:
            raise RuntimeError('Frozen model/codec hash changed')
        semantics = ('Free continuous64 vectors; not constrained to legal LFQ code vectors or IDs'
                     if branch == 'continuous_z' else 'Per-frame normalized64 features; bypasses action codes and motion decoder')
        with (out / f'{branch}_tensor.pt').open('xb') as stream:
            torch.save(dict(parameter=parameter.detach().cpu(), final_features=final_f.cpu(), coordinates=coordinates.cpu(),
                            output_units=units.cpu(), initial_features=initial_f.cpu(), baseline_posterior_ids=ids,
                            branch=branch, start=start, steps=100, semantics=semantics, legal_action_ids=False), stream)
        branches[branch] = dict(optimizer_updates=100, final_step=100, selected_by='fixed_step100_no_best_pick',
                    observations=observations, boundary=boundary, frozen_hashes_before=before, frozen_hashes_after=after,
                    initialization_max_feature_error=start_error, legal_action_ids=False, semantics=semantics)
        del optimizer, parameter, frames, final_f
    # Three columns and twenty rows: two three-panel groups per physical frame.
    labels = ('real', 'baseline', 'teacher_state_reference') + BRANCHES
    sheet = Image.new('RGB', (448*3, (252+22)*20), 'white'); draw = ImageDraw.Draw(sheet)
    for h in range(10):
        for i, label in enumerate(labels):
            x, y = (i % 3)*448, (h*2+i//3)*274
            draw.text((x+4, y+4), f'frame {start+h} h{h} | {label}', fill='black')
            sheet.paste(galleries[label][h], (x, y+22))
    save_png(sheet, out / 'contact_sheet.png')
    check()
    if frozen_hashes() != initial_hashes or verify_manifest(root) != manifest_hash:
        raise RuntimeError('End-of-worker frozen/code state changed')
    for path, digest in context['hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('Selected terminal source changed: ' + path)
    for item in data.sources:
        if Path(item['path']).stat().st_size != item['bytes'] or sha(item['path']) != item['sha256']:
            raise RuntimeError('Immutable data changed: ' + item['path'])
    artifacts = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    summary = dict(schema='gs_native_capacity_probe_window_v1', status='complete', start=start, frames=list(range(start,start+10)),
              source_arm=cfg['source_arm'], selected_step=400, checkpoint=context['selection']['checkpoints'][cfg['source_arm']],
              code_manifest_sha256=manifest_hash, protocol_sha256=protocol_hash, source_manifest_sha256=helper.MAIN_MANIFEST,
              source_artifact_hashes=context['hashes'], data_sources=data.sources, frozen_hashes=initial_hashes,
              baseline=baseline_metrics, teacher_state_reference=teacher_metrics, branches=branches, parity_passed=True,
              original_latent_rows=len(coordinates), original_resolution_hw=[252,448], all_initial_latent_rows_retained=True,
              model_optimizer_updates=0, oracle_optimizer_updates=300, goal_complete=False, scientific_passed=None,
              scope='Two predeclared failure windows; target-fitting capacity diagnostic, not generalization or legal action learning',
              comparison='Use fixed step100 only; continuous relaxation success does not establish LFQ capacity',
              elapsed_seconds=time.monotonic()-began, gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
              artifact_sha256=artifacts)
    check(); write_new(out / 'SUMMARY.json', summary)
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--start', type=int, choices=(35,123), required=True)
    args = parser.parse_args(); root = args.root.resolve(); out = root / 'windows' / f'start_{args.start:04d}'
    owned = False
    try:
        out.mkdir(parents=True, exist_ok=False); owned = True
        result = run(root, args.start, out)
        print(json.dumps(dict(status=result['status'], start=args.start, model_optimizer_updates=0, oracle_optimizer_updates=300)))
    except BaseException as exc:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        if owned and not (out / 'SUMMARY.json').exists():
            write_new(out / 'FAILED.json', dict(status='failed', start=args.start, error=repr(exc), traceback=traceback.format_exc(), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
