"""Read-only h9 replay against hash-bound NeoVerse nearby teacher pseudo-views."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import time
import traceback

FRAME_COUNT, START_COUNT, HORIZON = 177, 168, 9
HEIGHT, WIDTH, MAX_SECONDS = 252, 448, 600
CONTROLS = ('posterior_correct', 'shuffle84', 'blinded', 'posterior_adapted_held',
            'matched_constant', 'teacher_state_reference')
METRICS = ('mae', 'psnr', 'hand_activity_mae', 'hand_activity_psnr', 'motion_mae',
           'hand_activity_motion_mae', 'delta_mae', 'hand_activity_delta_mae',
           'motion_delta_mae', 'hand_activity_motion_delta_mae')
SEMANTICS = 'Nearby teacher pseudo-view; not real multi-view GT, independent holdout, or AR rollout'


def nearby_record(record):
    """Only camera geometry crosses the renderer boundary; never targets/real RGB."""
    if tuple(record['cameras'].shape) != (2, 4, 4):
        raise ValueError('Expected the original two-camera record')
    return dict(cameras=record['cameras'][1:2], **{k: record[k] for k in ('intrinsics', 'near', 'far')})


def camera_error(torch, cameras):
    if tuple(cameras.shape) != (2, 4, 4) or not bool(torch.isfinite(cameras).all()):
        raise ValueError('Invalid native camera pair')
    expected = cameras[0].clone()
    expected[:3, 3] += cameras[0, :3, 0] * .02
    error = float((cameras[1] - expected).abs().max())
    if error > 1e-6:
        raise ValueError('Nearby camera is not right-axis +0.02 normalized units')
    return error


def measure(torch, prediction, target, prediction0, target0, roi):
    """Targets enter here only; each control owns its prediction0 baseline."""
    if any(tuple(x.shape) != (*roi.shape, 3) or not bool(torch.isfinite(x).all())
           for x in (prediction, target, prediction0, target0)) or not bool(roi.any()):
        raise ValueError('Invalid native metric tensors or empty fixed ROI')
    error = (prediction - target).abs().mean(-1)
    squared = (prediction - target).square().mean(-1)
    delta = ((prediction - prediction0) - (target - target0)).abs().mean(-1)
    weights = ((target - target0).abs().mean(-1) / .1).clamp(0, 1)

    def weighted(value, weight):
        denominator = float(weight.sum())
        return float((value * weight).sum()) / denominator if denominator > 0 else None

    return dict(mae=float(error.mean()), psnr=-10 * math.log10(max(float(squared.mean()), 1e-12)),
        hand_activity_mae=float(error[roi].mean()),
        hand_activity_psnr=-10 * math.log10(max(float(squared[roi].mean()), 1e-12)),
        motion_mae=weighted(error, weights), hand_activity_motion_mae=weighted(error[roi], weights[roi]),
        delta_mae=float(delta.mean()), hand_activity_delta_mae=float(delta[roi].mean()),
        motion_delta_mae=weighted(delta, weights), hand_activity_motion_delta_mae=weighted(delta[roi], weights[roi]))


def validate_coverage(rows):
    keys = [(r['start'], r['horizon'], r['target_frame']) for r in rows]
    if len(keys) != START_COUNT or set(keys) != {(s, HORIZON, s + HORIZON) for s in range(START_COUNT)}:
        raise ValueError('Require exactly 168 unique h9 targets, physical frames 9..176')


def averages(rows):
    values, valid = {}, {}
    for control in CONTROLS:
        values[control], valid[control] = {}, {}
        for metric in METRICS:
            observations = [r['metrics'][control][metric] for r in rows if r['metrics'][control][metric] is not None]
            values[control][metric] = statistics.fmean(observations) if observations else None
            valid[control][metric] = len(observations)
    return values, valid


def original_view_trends(context, loss_mode, nearby_means):
    """Signs are descriptive within-view rankings, never a new scientific gate."""
    controls = {'shuffle84': ('z_' + loss_mode, 'shuffle84'), 'blinded': ('z_' + loss_mode, 'blinded'),
        'posterior_adapted_held': ('z_' + loss_mode, 'adapted_held'),
        'matched_constant': ('constant_' + loss_mode, 'correct')}
    posterior = context['arms']['z_' + loss_mode]['rows']
    result = {}
    for control, (arm, key) in controls.items():
        result[control] = {}
        for metric in ('mae', 'psnr', 'hand_activity_mae', 'hand_activity_psnr', 'motion_mae'):
            old_difference = statistics.fmean(posterior[s, HORIZON]['correct'][metric]
                - context['arms'][arm]['rows'][s, HORIZON][key][metric] for s in range(START_COUNT))
            left, right = nearby_means['posterior_correct'][metric], nearby_means[control][metric]
            new_difference = left - right if left is not None and right is not None else None
            sign = lambda value: None if value is None else (1 if value > 0 else -1 if value < 0 else 0)
            result[control][metric] = dict(original_real_rgb_correct_minus_control=old_difference,
                nearby_teacher_correct_minus_control=new_difference,
                same_rank_direction=None if new_difference is None else sign(old_difference) == sign(new_difference))
    return result


def run(root, loss_mode, output, max_seconds):
    from render_six_panel import (load_terminal, restore_selected, validate_actions, GaussianRecorder,
                                  sha, require_equal, write_new, MAIN_MANIFEST)
    root, output = root.resolve(), output.resolve()
    if output != root / 'diagnostics' / 'nearby_view_v7' / loss_mode:
        raise ValueError('Output must be ROOT/diagnostics/nearby_view_v7/{rgb,delta}')
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError('Internal budget must not exceed 600 seconds')
    output.mkdir(parents=True, exist_ok=False)
    begun, previous_alarm, stage, rows = time.monotonic(), None, 'terminal_metadata', []

    def remaining():
        if time.monotonic() - begun >= max_seconds:
            raise TimeoutError('Nearby pseudo-view diagnostic exceeded its fixed time budget')

    def deadline(signum, frame):
        raise TimeoutError('Nearby pseudo-view diagnostic deadline reached')

    try:
        if hasattr(signal, 'setitimer'):
            previous_alarm = signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, max_seconds)
        context = load_terminal(root)  # Refuses incomplete/failed four-arm training or evaluation.
        for path in (Path(__file__).resolve(), Path(__file__).with_suffix('.md').resolve()):
            context['hashes'][str(path)] = sha(path)
        runner, evidence = context['runner'], context['evidence']
        posterior_arm, constant_arm = 'z_' + loss_mode, 'constant_' + loss_mode
        selected_arms = (posterior_arm, constant_arm)
        import torch
        actions = {arm: validate_actions(torch, torch.load(context['arms'][arm]['directory'] / 'actions.pt',
                   map_location='cpu', weights_only=True), arm, context) for arm in selected_arms}
        stage = 'runtime_and_hash_bound_pseudo_targets'
        torch, core, _, _, cfg, data = runner.setup_runtime(evidence)
        if (cfg['data']['frames'] != FRAME_COUNT or cfg['data']['resolution_hw'] != [HEIGHT, WIDTH]
                or cfg['data']['frame_gap'] != 1 or cfg['resources']['peak_limit_gib'] != 48
                or len(data.states) != FRAME_COUNT or len(data.records) != FRAME_COUNT):
            raise ValueError('Native frame/resolution/resource contract changed')
        torch.cuda.reset_peak_memory_stats()
        source_entries = {item['path']: item for item in data.sources}
        targets, cache_rows, normalized = [], [], None
        for frame in range(FRAME_COUNT):
            remaining()
            path = core.STATE_ROOT / 'cache' / core.CLIP / f'{frame:03d}.pt'
            entry = source_entries.get(str(path))
            if entry is None or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('Pseudo-view cache is not hash-bound to the native data')
            context['hashes'][str(path)] = entry['sha256']
            raw = torch.load(path, map_location='cpu', weights_only=True)
            if (raw['frame'] != frame or raw['clip'] != core.CLIP or tuple(raw['targets'].shape) != (2, HEIGHT, WIDTH, 3)
                    or not bool(torch.isfinite(raw['targets']).all())):
                raise ValueError('Pseudo-view target identity or native size changed')
            norm = json.dumps(raw['normalization'], sort_keys=True)
            normalized = norm if normalized is None else normalized
            require_equal(norm, normalized, 'Single fixed world normalization')
            if raw['normalization'].get('fixed_for_all_frames') is not True:
                raise ValueError('Normalization is not fixed across all frames')
            for key in ('cameras', 'intrinsics', 'near', 'far'):
                cached, loaded = raw[key], data.records[frame][key]
                if torch.is_tensor(cached):
                    if not torch.equal(cached, loaded.detach().cpu()):
                        raise ValueError('Runtime camera geometry differs from verified cache')
                else:
                    require_equal(cached, loaded, 'Runtime ' + key)
            offset_error = camera_error(torch, raw['cameras'])
            count = int(raw['n_input'])
            if count <= 0 or int(raw['counts'].sum()) != count or raw['metrics']['source_count'] != count:
                raise ValueError('Source GS voxelization dropped points')
            state = data.states[frame]
            if state['frame_index'] != frame or state['features'].shape != (len(state['coordinates']), 64):
                raise ValueError('Native 64-dimensional state identity changed')
            targets.append(raw['targets'][1].clamp(0, 1).to('cuda'))
            cache_rows.append(dict(frame=frame, path=str(path), sha256=entry['sha256'], bytes=entry['bytes'],
                source_gaussians=count, voxel_count=len(raw['coordinates']), latent_count=len(state['coordinates']),
                native_source_sha256=raw['metrics']['native_gs_sha256'], nearby_camera_max_absolute_error=offset_error))
            del raw
        runtimes = {arm: restore_selected(context, arm, core, torch, cfg, data) for arm in selected_arms}
        for arm, (model, bridge) in runtimes.items():
            require_equal(model.code_mode, actions[arm]['code_mode'], 'Runtime code mode')
            if model.code_mode == 'constant':
                require_equal(model.constant_id, actions[arm]['constant_id'], 'Runtime constant ID')
        before = {arm: dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
                  for arm, (model, bridge) in runtimes.items()}
        recorders = {arm: GaussianRecorder(bridge) for arm, (_, bridge) in runtimes.items()}
        render_calls = decode_calls = 0
        stage = 'complete168_h9_pseudo_view_replay'
        with torch.no_grad(), (output / 'h9_metrics.jsonl').open('x', encoding='utf-8') as log:
            for start in range(START_COUNT):
                remaining()
                target_frame, shuffled = start + HORIZON, (start + 84) % START_COUNT
                condition = [core.make_condition(data.states[start])]
                predictions = {}
                for control, arm, key, bundle in (
                    ('posterior_correct', posterior_arm, 'ids', start), ('shuffle84', posterior_arm, 'ids', shuffled),
                    ('blinded', posterior_arm, 'blinded_ids', start), ('matched_constant', constant_arm, 'ids', start)):
                    sequence = runtimes[arm][0].decode_ids(actions[arm][key][bundle:bundle + 1].to('cuda'), condition)[0]
                    decode_calls += 1
                    if len(sequence) != 10:
                        raise ValueError('Replay omitted an output time slot')
                    for prediction in sequence:
                        if (not torch.equal(prediction['coordinates'], data.states[start]['coordinates'])
                                or prediction['features'].shape != data.states[start]['features'].shape
                                or not bool(torch.isfinite(prediction['features']).all())):
                            raise ValueError('Replay changed complete initial support or native feature width')
                    predictions[control] = (sequence[0], sequence[HORIZON])
                    del sequence
                predictions['posterior_adapted_held'] = (data.states[start], data.states[start])
                predictions['teacher_state_reference'] = (data.states[start], data.states[target_frame])
                measured, audits, held0 = {}, {}, None
                for control in CONTROLS:
                    arm = constant_arm if control == 'matched_constant' else posterior_arm
                    bridge = runtimes[arm][1]
                    images, audits[control] = [], {}
                    for slot, frame in ((0, start), (1, target_frame)):
                        remaining()
                        if control == 'teacher_state_reference' and slot == 0:
                            image, audit = held0  # Identical codec, state, camera: reuse this exact h0.
                        else:
                            image = bridge.render(predictions[control][slot], nearby_record(data.records[frame]))
                            render_calls += 1
                            audit = dict(recorders[arm].last)
                            if audit['count'] != bridge.last_count or not 0 < audit['count'] <= cfg['resources']['max_decoded_gaussians']:
                                raise ValueError('Complete native GS count or original guard mismatch')
                            if torch.cuda.max_memory_allocated() / 2**30 > 48:
                                raise RuntimeError('Original 48GiB peak guard exceeded')
                        if control == 'posterior_adapted_held' and slot == 0:
                            held0 = (image, audit)
                        images.append(image)
                        audits[control]['h0' if slot == 0 else 'h9'] = dict(audit, arm=arm)
                    measured[control] = measure(torch, images[1], targets[target_frame], images[0], targets[start], data.roi)
                    del images
                row = dict(start=start, horizon=HORIZON, target_frame=target_frame, loss_mode=loss_mode,
                    selected_step=context['step'], source_pts=float(data.states[target_frame]['pts']),
                    initial_pts=float(data.states[start]['pts']), metrics=measured, native_gaussians=audits,
                    initial_latent_count=len(data.states[start]['coordinates']),
                    future_teacher_latent_count=len(data.states[target_frame]['coordinates']),
                    posterior_ids=actions[posterior_arm]['ids'][start].tolist(), shuffle84_bundle_start=shuffled,
                    shuffle84_ids=actions[posterior_arm]['ids'][shuffled].tolist(),
                    blinded_ids=actions[posterior_arm]['blinded_ids'][start].tolist(),
                    constant_ids=actions[constant_arm]['ids'][start].tolist())
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                rows.append(row)
                del predictions, held0, image
                if (start + 1) % 21 == 0:
                    print('NEARBY_PSEUDO_VIEW', loss_mode, start + 1, '/168 h9', flush=True)
        stage = 'final_source_and_coverage_verification'
        validate_coverage(rows)
        require_equal(decode_calls, START_COUNT * 4, 'All four ID controls replayed for every start')
        require_equal(render_calls, START_COUNT * 11, 'All complete native h0/h9 renders, with one declared reuse')
        for arm, (model, bridge) in runtimes.items():
            runner.ensure_eval_codec(bridge)
            require_equal(dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec)), before[arm],
                          'No model/native weights or buffers changed')
            require_equal(core.verify_native_sources(), bridge.binding, 'Final native source binding')
        for path, expected in context['hashes'].items():
            remaining()
            require_equal(sha(path), expected, 'Unchanged diagnostic input: ' + path)
        require_equal(runner.load_verified(root)['code_manifest_sha256'], MAIN_MANIFEST, 'Final v7 source verification')
        means, valid = averages(rows)
        write_new(output / 'cache_provenance.json', dict(normalization=json.loads(normalized), frames=cache_rows,
            camera_offset=dict(right_axis=.02, units='normalized world units, not meters', tolerance_absolute=1e-6)))
        summary = dict(schema='gs_v7_nearby_teacher_h9_v1', status='complete', semantics=SEMANTICS,
            loss_mode=loss_mode, selected_step=context['step'], optimizer_updates=0, goal_complete=False, not_ar_rollout=True,
            scope=dict(horizon=9, starts=168, unique_physical_targets=168, target_range=[9, 176],
                target_source='raw.targets[1].clamp(0,1)', camera_source='raw.cameras[1]', resolution_hw=[HEIGHT, WIDTH]),
            metrics=means, metric_valid_targets=valid, trend_comparison=original_view_trends(context, loss_mode, means),
            checkpoints={arm: context['selection']['checkpoints'][arm] for arm in selected_arms},
            native_binding=context['arms'][posterior_arm]['evaluation']['source_binding'],
            all_initial_latent_points_retained=True, native_internal_hard_pruning_retained=True,
            native_render_calls=render_calls, latent_window_decode_calls=decode_calls,
            teacher_reference_h0_reuses_identical_held_h0=True, no_bootstrap=True, no_new_scientific_threshold=True,
            input_sha256=context['hashes'], source_cache_entries_sha256=hashlib.sha256(
                json.dumps(data.sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            artifact_sha256={name: sha(output / name) for name in ('h9_metrics.jsonl', 'cache_provenance.json')},
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_seconds=time.monotonic() - begun,
            max_seconds=max_seconds, external_queue_timeout_seconds=650,
            limitations=['One TRAIN clip and a teacher-generated nearby view, not independent multi-view ground truth',
                'Fixed original image-space ROI is reused; it is not a new annotation for the pseudo-view',
                'Teacher-state reference directly decodes true future 64D states with the selected posterior codec; not a theoretical upper bound',
                'Each delta uses that control own predicted h0; weights use teacher h9 minus teacher h0, threshold 0.1',
                'Motion metrics with zero weight are null and excluded from their stated valid-target count',
                'PSNR uses max(MSE,1e-12), so exact matches report 120 dB',
                'Original-view trend signs use the same 168 h9 targets but real RGB; compare rankings, not cross-view metric magnitudes',
                'No 176-frame bootstrap reuse, action-semantic, independent holdout, or generalization claim'])
        remaining()
        write_new(output / 'SUMMARY.json', summary)
        print(json.dumps(dict(status='complete', loss_mode=loss_mode, selected_step=context['step'],
                              targets=168, horizon=9, output=str(output))), flush=True)
    except BaseException as error:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        write_new(output / 'FAILED.json', dict(status='failed', stage=stage, completed_targets=len(rows),
            optimizer_updates=0, no_retry=True, error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - begun))
        raise
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


def cpu_self_test():
    import torch
    target0 = torch.zeros(2, 2, 3)
    target = target0.clone()
    target[0, 0] = .2
    roi = torch.tensor([[True, False], [False, False]])
    perfect = measure(torch, target, target, target0, target0, roi)
    held = measure(torch, target0, target, target0, target0, roi)
    shifted_initial = measure(torch, target, target, torch.full_like(target0, .1), target0, roi)
    assert perfect['mae'] == perfect['delta_mae'] == 0 and perfect['psnr'] == 120
    assert math.isclose(held['motion_mae'], .2, abs_tol=1e-7)
    assert math.isclose(held['delta_mae'], .05, abs_tol=1e-7)
    assert shifted_initial['mae'] == 0 and math.isclose(shifted_initial['delta_mae'], .1, abs_tol=1e-7)
    assert measure(torch, target0, target0, target0, target0, roi)['motion_mae'] is None
    cameras = torch.eye(4).repeat(2, 1, 1)
    cameras[1, 0, 3] = .02
    assert camera_error(torch, cameras) == 0
    record = dict(cameras=cameras, intrinsics=torch.eye(3), near=.1, far=10,
                  targets=object(), real=object(), unrelated=object())
    view = nearby_record(record)
    assert set(view) == {'cameras', 'intrinsics', 'near', 'far'}
    assert torch.equal(view['cameras'][0], cameras[1]) and tuple(record['cameras'].shape) == (2, 4, 4)
    cameras[1, 1, 3] = .02
    try:
        camera_error(torch, cameras)
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect camera offset accepted')
    rows = [dict(start=s, horizon=9, target_frame=s + 9) for s in range(168)]
    validate_coverage(rows)
    for invalid in (rows[:-1], rows[:-1] + [rows[0]], [dict(r, horizon=8) for r in rows]):
        try:
            validate_coverage(invalid)
        except ValueError:
            continue
        raise AssertionError('Invalid coverage accepted')
    assert not torch.cuda.is_initialized()
    print('CPU self-test passed: metric mathematics, own-h0 delta, target isolation, camera offset, unique h9 coverage; no CUDA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--loss-mode', choices=('rgb', 'delta'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
    elif args.root is None or args.loss_mode is None or args.output is None:
        parser.error('--root, --loss-mode and --output are required')
    else:
        run(args.root, args.loss_mode, args.output, args.max_seconds)


if __name__ == '__main__':
    main()
