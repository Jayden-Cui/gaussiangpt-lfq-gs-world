"""Bounded real-GS B4 parity/speed witness; never persists optimizer updates.

One warm-up per method, then two independently restored one-step measurements
per method. Only the optimizer's pre-clip boundary is instrumented. The new
output directory is exclusive; the existing v6 experiment is read-only.
Both methods apply the same explicit, training-required fixed-null-key-bias
policy. The failed v1 experiment remains failed and is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time
import traceback
from unittest.mock import patch

sys.dont_write_bytecode = True

SCHEMA = 'gs_native_step_acceleration_benchmark_v1'
REQUIRED_POLICY = 'qformer_softmax_key_bias_fixed_v1'
STARTS = [0, 42, 84, 126]
REPEATS = 2
CPU_TESTED = {
    'staged_step.py': 'b11908c3d4d34e578b674fe969d56faaa5347843cafe9800e724fe7da40108c0',
    'test_staged_step.py': '55ed5fb5f36e0814e23f2062f6a882e41abc755ab69fdaf90dae1731863e886d',
}
TOLERANCES = dict(loss_rtol=1e-4, loss_atol=1e-6, gradient_relative_l2=1e-3,
                  gradient_maxabs_atol=1e-5, gradient_maxabs_rtol=1e-3,
                  weight_maxabs_atol=1e-6, weight_maxabs_rtol=1e-4,
                  rgb_norm_rtol=1e-3, rgb_norm_atol=1e-7, minimum_speedup=1.2)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def progress(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value, torch, np):
    """Canonical content hash for in-memory tensors, optimizer state and RNG."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(('tensor:' + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(('ndarray:' + str(item.dtype) + str(item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update((type(item).__name__ + ':' + str(len(item))).encode())
            for child in item:
                visit(child)
        else:
            digest.update((type(item).__name__ + ':' + repr(item)).encode())
        digest.update(b'\0')

    visit(value)
    return digest.hexdigest()


def compare_tensors(reference, candidate, kind, torch):
    require(set(reference) == set(candidate), f'{kind} parameter names differ')
    failures, checked = [], 0
    worst_absolute = dict(name=None, value=0.)
    worst_relative = dict(name=None, value=0.)
    for name, original in reference.items():
        other = candidate[name]
        checked += 1
        if original is None or other is None:
            if original is not None or other is not None:
                failures.append(dict(name=name, reason='None gradient mismatch'))
            continue
        if original.shape != other.shape or original.dtype != other.dtype:
            failures.append(dict(name=name, reason='shape/dtype mismatch'))
            continue
        if not bool(torch.isfinite(original).all() and torch.isfinite(other).all()):
            failures.append(dict(name=name, reason='nonfinite tensor'))
            continue
        original64, other64 = original.double(), other.double()
        difference = other64-original64
        maximum = float(difference.abs().max()) if difference.numel() else 0.
        reference_maximum = float(original64.abs().max()) if original64.numel() else 0.
        reference_norm = float(original64.norm())
        difference_norm = float(difference.norm())
        relative = difference_norm/reference_norm if reference_norm else (0. if difference_norm == 0 else None)
        if maximum > worst_absolute['value']:
            worst_absolute = dict(name=name, value=maximum)
        if relative is None or (worst_relative['value'] is not None and relative > worst_relative['value']):
            worst_relative = dict(name=name, value=relative)
        if kind == 'gradients':
            bound = TOLERANCES['gradient_maxabs_atol'] + TOLERANCES['gradient_maxabs_rtol']*reference_maximum
            passed = relative is not None and relative <= TOLERANCES['gradient_relative_l2'] and maximum <= bound
        else:
            bound = TOLERANCES['weight_maxabs_atol'] + TOLERANCES['weight_maxabs_rtol']*reference_maximum
            passed = maximum <= bound
        if not passed:
            failures.append(dict(name=name, max_abs=maximum, max_abs_bound=bound,
                                 reference_max_abs=reference_maximum, relative_l2=relative,
                                 reason='nonzero candidate with zero reference' if relative is None else 'fixed tolerance exceeded'))
    return dict(passed=not failures, checked_parameters=checked, failures=failures,
                worst_max_abs=worst_absolute, worst_relative_l2=worst_relative)


def compare_results(reference, candidate, torch):
    checks, failures = {}, []
    for key in ('full_l1', 'roi_l1', 'motion_l1', 'auxiliary'):
        a, b = reference['result'][key], candidate['result'][key]
        passed = math.isfinite(a) and math.isfinite(b) and math.isclose(
            a, b, rel_tol=TOLERANCES['loss_rtol'], abs_tol=TOLERANCES['loss_atol'])
        checks[key] = dict(passed=passed, reference=a, candidate=b, absolute_error=abs(a-b))
        if not passed:
            failures.append(key)
    for key in ('batch_ids', 'gs_count_min', 'gs_count_max'):
        passed = reference['result'][key] == candidate['result'][key]
        checks[key] = dict(passed=passed)
        if not passed:
            failures.append(key)
    for key, shape in [('rgb_quantized_grad_per_sample_h', (4, 9)),
                       ('rgb_quantized_grad_per_h', (9,)), ('rgb_query_grad_per_h', (9,))]:
        a, b = reference['result'][key], candidate['result'][key]
        if a is None or b is None:
            checks[key] = dict(passed=False, reason='Missing RGB-only gradient')
            failures.append(key)
            continue
        a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        valid = tuple(a.shape) == shape == tuple(b.shape)
        passed = valid and bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                                and (a > 0).all() and (b > 0).all()
                                and torch.allclose(a, b, rtol=TOLERANCES['rgb_norm_rtol'], atol=TOLERANCES['rgb_norm_atol']))
        checks[key] = dict(passed=passed, shapes=[list(a.shape), list(b.shape)],
                          max_abs=float((a-b).abs().max()) if valid else None,
                          nonzero_complete=valid and bool((a > 0).all() and (b > 0).all()))
        if not passed:
            failures.append(key)
    for key in ('restored_state_fingerprint', 'rng_after_fingerprint'):
        passed = reference[key] == candidate[key]
        checks[key] = dict(passed=passed)
        if not passed:
            failures.append(key)
    gradients = compare_tensors(reference['gradients'], candidate['gradients'], 'gradients', torch)
    weights = compare_tensors(reference['weights'], candidate['weights'], 'weights', torch)
    return dict(passed=not failures and gradients['passed'] and weights['passed'],
                result_checks=checks, failed_result_fields=failures, gradients=gradients, weights=weights)


def run(root, checkpoint_path, output, max_seconds):
    begun = time.monotonic()
    report = dict(schema=SCHEMA, status='running', adoption_eligible=False, goal_complete=False,
                  numerical_pass=False, integrity_pass=False, resource_pass=False,
                  required_fixed_null_key_bias_policy=REQUIRED_POLICY,
                  starts=STARTS, batch_size=4, frames_per_window=10, resolution_hw=[252, 448],
                  arm='partial', measured_repeats_per_mode=REPEATS, warmups_per_mode=1,
                  tolerances=TOLERANCES, max_seconds=max_seconds,
                  max_peak_gib=48., minimum_initial_free_gib=55., trials=[], comparisons=[],
                  optimizer_updates_executed=0, optimizer_updates_persisted=0,
                  performance_scope='CUDA-synchronized step including identical pre-clip GPU gradient-clone instrumentation; restoration and CPU comparison excluded')
    snapshots = {}
    stage = 'source_admission'

    def checked_hash(path):
        path = Path(path).resolve()
        value = sha(path)
        snapshots[str(path)] = value
        return value

    def budget():
        if time.monotonic()-begun >= max_seconds:
            raise TimeoutError('Whole benchmark budget exhausted')

    def interrupted(signum, frame):
        raise TimeoutError(f'Benchmark interrupted by signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    progress(output/'PROGRESS.json', dict(stage=stage, **report))
    try:
        here = Path(__file__).resolve().parent
        report['candidate_sources_sha256'] = {name: checked_hash(here/name)
            for name in ('benchmark.py', 'staged_step.py', 'test_staged_step.py',
                         'null_key_bias.py', 'test_null_key_bias.py', 'README.md')}
        require(all(report['candidate_sources_sha256'][name] == value for name, value in CPU_TESTED.items()),
                'Candidate does not match the already CPU-tested staged source')
        manifest = read(root/'CODE_MANIFEST.json')
        require(isinstance(manifest, dict) and 'code/native_experiment.py' in manifest,
                'Missing complete baseline code manifest')
        for relative, expected in manifest.items():
            path = (root/relative).resolve()
            require(path.is_relative_to(root) and checked_hash(path) == expected, 'Baseline manifest mismatch: '+relative)
        report['baseline_main_manifest_sha256'] = checked_hash(root/'CODE_MANIFEST.json')
        report['baseline_protocol_sha256'] = checked_hash(root/'protocol.json')
        report['baseline_preflight_sha256'] = checked_hash(root/'PREFLIGHT.json')
        preflight = read(root/'PREFLIGHT.json')
        require(preflight.get('status') == 'passed'
                and preflight.get('code_manifest_sha256') == report['baseline_main_manifest_sha256']
                and preflight.get('protocol_sha256') == report['baseline_protocol_sha256'],
                'Baseline preflight is not bound to this exact code/protocol')
        require(checkpoint_path == root/'partial/step_0100.pt', 'Only original partial/step_0100.pt is admitted')
        report['checkpoint_path'] = str(checkpoint_path)
        report['checkpoint_sha256'] = checked_hash(checkpoint_path)
        report['original_plan_sha256'] = checked_hash(root/'partial/PLAN.json')
        original_plan = read(root/'partial/PLAN.json')
        require(original_plan.get('partial_native') is True
                and original_plan.get('code_manifest_sha256') == report['baseline_main_manifest_sha256']
                and original_plan.get('protocol_sha256') == report['baseline_protocol_sha256'],
                'Original partial plan source binding differs')
        budget()

        stage = 'initialize_original_native_pipeline'
        import numpy as np
        import torch
        sys.path.insert(0, str(root/'code'))
        import native_experiment as native
        import staged_step as staged
        import null_key_bias as fixed_bias
        require(Path(native.__file__).resolve() == root/'code/native_experiment.py', 'Imported a different native implementation')
        require(Path(staged.__file__).resolve() == here/'staged_step.py', 'Imported a different staged implementation')
        require(Path(fixed_bias.__file__).resolve() == here/'null_key_bias.py'
                and fixed_bias.POLICY == REQUIRED_POLICY, 'Imported a different fixed-key-bias policy')
        cfg = native.setup(root)  # Includes original GPU55GiB and allocator guards.
        require(cfg['train']['batch_size'] == 4 and cfg['train']['grad_clip'] == 1.
                and cfg['resources']['minimum_free_gib'] >= 55 and cfg['resources']['peak_limit_gib'] <= 48,
                'Original B4/clip/resource contract differs')
        report['initial_free_gib'] = torch.cuda.mem_get_info()[0]/2**30
        require(report['initial_free_gib'] >= 55, 'Less than55GiB initially free; no shared-process intervention')
        report['device'] = dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                                cuda_version=torch.version.cuda)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        require(checkpoint.get('completed_steps') == 100 and checkpoint.get('config') == cfg
                and checkpoint.get('plan') == original_plan.get('starts'), 'Checkpoint step/config/full plan mismatch')
        require(checkpoint['plan'] == native.make_plan(cfg), 'Checkpoint plan differs from original seeded plan')
        require({'model','codec','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'} <= set(checkpoint),
                'Checkpoint lacks complete model/native/optimizer/RNG state')
        report['checkpoint_completed_steps'] = 100
        report['checkpoint_plan_fingerprint'] = fingerprint(checkpoint['plan'], torch, np)
        report['checkpoint_rng_fingerprint'] = fingerprint({key: checkpoint[key] for key in
            ('torch_rng','cuda_rng','numpy_rng','python_rng')}, torch, np)
        data = native.Data()
        model, bridge, optimizer = native.build(cfg, data, 'partial')
        require(bridge.partial and not bridge.codec.ae.decoder.training, 'Native partial/eval mode changed')
        report['source_binding'] = bridge.binding
        require(bridge.binding == preflight['source_binding'], 'Native source binding changed after baseline preflight')
        require(data.reference_hash == preflight['fixed_reference_metrics_sha256'], 'Frozen source reference changed')
        for dependency in bridge.binding['dependencies']:
            require(checked_hash(dependency['path']) == dependency['sha256'], 'Native dependency changed')
        require(checked_hash(native.STATE_ROOT/'codec_final.pt') == bridge.binding['state_checkpoint_sha256'],
                'Native source checkpoint changed')
        require(checked_hash(native.V4_ROOT/'cache_manifest.json') == bridge.binding['cache_manifest_sha256'],
                'Native cache manifest changed')
        require(checked_hash(bridge.binding['initializer_checkpoint_path']) == bridge.binding['initializer_checkpoint_sha256'],
                'Native initializer changed')
        frame_ids = sorted({start+h for start in STARTS for h in range(10)})
        used_paths = {str(base/f'{frame:03d}.pt') for frame in frame_ids for base in
                     (native.MOTION_ROOT/'states'/native.CLIP, native.STATE_ROOT/'cache'/native.CLIP)}
        selected_sources = [row for row in data.sources if row['path'] in used_paths]
        require({row['path'] for row in selected_sources} == used_paths, 'Missing actual benchmark input source')
        report['input_sources_sha256'] = {row['path']: row['sha256'] for row in selected_sources}
        snapshots.update(report['input_sources_sha256'])  # Data already verified all these bytes.
        report['input_fingerprint'] = fingerprint([(frame, data.states[frame], data.records[frame]) for frame in frame_ids], torch, np)
        report['initial_latent_counts'] = [len(data.states[start]['coordinates']) for start in STARTS]

        named = dict([('model.'+name, value) for name, value in model.named_parameters()] +
                     [('native.'+name, value) for name, value in bridge.named_parameters()])
        require(len({id(p) for p in named.values()}) == len(named), 'Shared parameters cross the staged boundary')
        optimized = [p for group in optimizer.param_groups for p in group['params']]
        require({id(p) for p in optimized} == {id(p) for p in named.values() if p.requires_grad}
                and len({id(p) for p in optimized}) == len(optimized), 'Optimizer parameter scope differs')
        expected_native = {id(p) for module in (bridge.codec.ae.decoder.out, bridge.codec.ae.feature_heads,
                           bridge.codec.ae.gaussian_multiplier) for p in module.parameters()}
        require({id(p) for p in bridge.parameters() if p.requires_grad} == expected_native,
                'Native partial unfreeze scope differs')
        frozen_encoder = bridge.codec.ae.encoder
        require(not any(p.requires_grad for p in frozen_encoder.parameters()), 'Encoder must remain frozen')
        report['parameter_count'] = len(named)
        report['optimized_parameter_tensors'] = len(optimized)
        common_restored_fingerprint = None
        common_encoder_hash = None
        null_bias_guard = None

        def rng_state():
            return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        numpy_rng=np.random.get_state(), python_rng=random.getstate())

        def restore():
            nonlocal common_restored_fingerprint, common_encoder_hash, null_bias_guard
            budget()
            optimizer.zero_grad(set_to_none=True)
            model.load_state_dict(checkpoint['model'], strict=True)
            bridge.codec.load_state_dict(checkpoint['codec'], strict=True)
            # load_state_dict may retain CPU step tensors; deepcopy prevents a
            # benchmark update from mutating the reusable source snapshot.
            optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
            model.train()
            bridge.train()
            require(not bridge.codec.training and not bridge.codec.ae.decoder.training, 'Native eval mode not retained')
            torch.set_rng_state(checkpoint['torch_rng'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            np.random.set_state(copy.deepcopy(checkpoint['numpy_rng']))
            random.setstate(checkpoint['python_rng'])
            # Apply only after complete restore. Existing optimizer membership,
            # historical moments and current bias values are retained.
            null_bias_guard = fixed_bias.freeze_null_key_bias(model, optimizer)
            policy_record = fixed_bias.assert_null_key_bias_unchanged(model, optimizer, null_bias_guard)
            require(policy_record['in_optimizer_count'] == 27, 'Posterior benchmark must retain27 original optimizer members')
            fixed_ids = {id(model.get_parameter(name)) for name in policy_record['parameter_names']}
            require({id(p) for p in optimized} == {id(p) for p in named.values() if p.requires_grad} | fixed_ids,
                    'Optimizer scope differs beyond the explicit27 fixed key biases')
            if 'fixed_null_key_bias' in report:
                require(policy_record == report['fixed_null_key_bias'], 'Loaded fixed-bias values/state changed between trials')
            report['fixed_null_key_bias'] = policy_record
            current = fingerprint(dict(model=model.state_dict(), codec=bridge.codec.state_dict(),
                                       optimizer=optimizer.state_dict(), rng=rng_state()), torch, np)
            encoder_hash = native.tensor_hash(frozen_encoder)
            if common_restored_fingerprint is None:
                common_restored_fingerprint, common_encoder_hash = current, encoder_hash
            require(current == common_restored_fingerprint and encoder_hash == common_encoder_hash,
                    'Methods did not restore identical full model/native/optimizer/RNG state')
            gc.collect()
            torch.cuda.synchronize()
            return current

        def trial(mode, measured, repetition):
            budget()
            restored = restore()
            captured = {}
            clip_calls = 0
            original_clip = torch.nn.utils.clip_grad_norm_

            def capture_clip(parameters, max_norm, *args, **kwargs):
                nonlocal clip_calls
                parameters = list(parameters)
                require(clip_calls == 0 and [id(p) for p in parameters] == [id(p) for p in optimized]
                        and max_norm == 1., 'Unexpected clipping scope/count')
                clip_calls += 1
                for name, parameter in named.items():
                    captured[name] = None if parameter.grad is None else parameter.grad.detach().clone()
                return original_clip(parameters, max_norm, *args, **kwargs)

            progress(output/'PROGRESS.json', dict(stage='trial', mode=mode, measured=measured,
                repetition=repetition, elapsed_seconds=time.monotonic()-begun, **report))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with patch.object(torch.nn.utils, 'clip_grad_norm_', capture_clip):
                result = (native.step if mode == 'original' else staged.step_native)(model, bridge, optimizer, data, STARTS, cfg)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            report['optimizer_updates_executed'] += 1
            peak = torch.cuda.max_memory_allocated()/2**30
            require(clip_calls == 1 and set(captured) == set(named), 'Missing full pre-clip gradient capture')
            require(peak <= 48, 'Benchmark peak exceeds48GiB; no truncation/retry')
            require(native.tensor_hash(frozen_encoder) == common_encoder_hash
                    and not any(p.grad is not None for p in frozen_encoder.parameters()), 'Frozen encoder changed or received gradients')
            policy_record = fixed_bias.assert_null_key_bias_unchanged(model, optimizer, null_bias_guard)
            info = dict(mode=mode, measured=measured, repetition=repetition, step_seconds=elapsed,
                        peak_allocated_gib=peak, result=result, frozen_encoder_unchanged=True,
                        fixed_null_key_bias=policy_record,
                        restored_state_fingerprint=restored,
                        rng_after_fingerprint=fingerprint(rng_state(), torch, np))
            report['trials'].append(info)
            gradients = {name: None if value is None else value.to('cpu', copy=True) for name, value in captured.items()}
            weights = {name: value.detach().to('cpu', copy=True) for name, value in named.items()}
            del captured
            budget()
            print(json.dumps(dict(event='trial_complete', mode=mode, measured=measured,
                                  repetition=repetition, step_seconds=elapsed, peak_allocated_gib=peak)), flush=True)
            return dict(**info, gradients=gradients, weights=weights)

        stage = 'warmup'
        for mode in ('original', 'staged'):
            warmup = trial(mode, False, -1)
            del warmup
        stage = 'measure'
        previous = None
        for repetition in range(REPEATS):
            pair = {}
            # Alternate measured order to reduce a systematic first/second bias.
            for mode in (('original','staged') if repetition == 0 else ('staged','original')):
                pair[mode] = trial(mode, True, repetition)
            comparison = compare_results(pair['original'], pair['staged'], torch)
            report['comparisons'].append(dict(kind='paired_methods', repetition=repetition, **comparison))
            if previous is not None:
                for mode in ('original','staged'):
                    report['comparisons'].append(dict(kind='same_method_repeatability', mode=mode,
                        **compare_results(previous[mode], pair[mode], torch)))
            previous = pair
            progress(output/'PROGRESS.json', dict(stage='measured_pair_complete', elapsed_seconds=time.monotonic()-begun, **report))
            budget()

        stage = 'final_integrity'
        report['fixed_null_key_bias'] = fixed_bias.assert_null_key_bias_unchanged(model, optimizer, null_bias_guard)
        require(fingerprint([(frame, data.states[frame], data.records[frame]) for frame in frame_ids], torch, np)
                == report['input_fingerprint'], 'In-memory benchmark inputs changed')
        require(fingerprint({key: checkpoint[key] for key in ('torch_rng','cuda_rng','numpy_rng','python_rng')}, torch, np)
                == report['checkpoint_rng_fingerprint'], 'Reusable checkpoint RNG snapshot changed')
        for path, expected in snapshots.items():
            budget()
            require(sha(path) == expected, 'Bound source changed during benchmark: '+path)
        report['verified_source_sha256'] = snapshots
        report['frozen_encoder_sha256'] = common_encoder_hash
        report['common_restored_state_fingerprint'] = common_restored_fingerprint
        seconds = {mode: [item['step_seconds'] for item in report['trials'] if item['measured'] and item['mode'] == mode]
                   for mode in ('original','staged')}
        require(all(len(values) == REPEATS for values in seconds.values()), 'Incomplete measured trials')
        report['step_seconds'] = seconds
        report['median_speedup'] = statistics.median(seconds['original'])/statistics.median(seconds['staged'])
        report['peak_allocated_gib'] = max(item['peak_allocated_gib'] for item in report['trials'])
        report['numerical_pass'] = all(item['passed'] for item in report['comparisons'])
        report['integrity_pass'] = True
        report['resource_pass'] = report['peak_allocated_gib'] <= 48 and time.monotonic()-begun < max_seconds
        report['adoption_eligible'] = (report['numerical_pass'] and report['integrity_pass'] and report['resource_pass']
                                      and report['fixed_null_key_bias']['policy'] == REQUIRED_POLICY
                                      and report['fixed_null_key_bias']['unchanged'] is True
                                      and report['median_speedup'] >= TOLERANCES['minimum_speedup'])
        report.update(status='complete', elapsed_seconds=time.monotonic()-begun,
                      limitation='Two independent one-step timings per method are a bounded engineering screen, not precise performance stability or scientific quality evidence.')
        budget()
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_new(output/'SUMMARY.json', report)
        print(json.dumps(dict(status='complete', adoption_eligible=report['adoption_eligible'],
                              numerical_pass=report['numerical_pass'], median_speedup=report['median_speedup'],
                              summary=str(output/'SUMMARY.json'))), flush=True)
        return 0
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)  # Preserve failure evidence before any cleanup.
        # The original step can raise its peak guard after optimizer.step; on
        # failure the exact number of in-memory updates is not inferred.
        report['completed_step_calls_before_failure'] = report['optimizer_updates_executed']
        report['optimizer_updates_executed'] = None
        report.update(status='failed', adoption_eligible=False, resource_pass=False,
                      failed_stage=stage, error=repr(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-begun)
        write_new(output/'FAILED.json', report)
        if not (output/'SUMMARY.json').exists():
            write_new(output/'SUMMARY.json', report)
        print(json.dumps(dict(status='failed', stage=stage, error=repr(error), adoption_eligible=False)), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=590.)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 600:
        parser.error('--max-seconds must be positive and at most600')
    if not hasattr(signal, 'setitimer'):
        parser.error('Real benchmark requires the Linux GPU host; CPU AST review is separate')
    root, output = args.root.resolve(), args.output.resolve()
    checkpoint = (args.checkpoint or root/'partial/step_0100.pt').resolve()
    here = Path(__file__).resolve().parent
    if not output.is_relative_to(here/'diagnostics'):
        parser.error('Output must be a new directory under this candidate directory/diagnostics')
    if output.exists():
        parser.error('Output already exists; no overwrite, resume, or automatic retry')
    output.mkdir(parents=True, exist_ok=False)
    raise SystemExit(run(root, checkpoint, output, args.max_seconds))


if __name__ == '__main__':
    main()
