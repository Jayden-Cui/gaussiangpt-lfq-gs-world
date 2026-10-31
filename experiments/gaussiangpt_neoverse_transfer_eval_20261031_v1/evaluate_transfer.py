"""Frozen selected-checkpoint posterior reconstruction on one new complete GS clip.

No training, checkpoint selection, reduction of points/frames, or automatic retry.
The original TRAIN data constructs/restores the model before target data is loaded.
"""
from __future__ import annotations

import argparse
import gc
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

sys.dont_write_bytecode = True
ARMS = ('z_rgb', 'constant_rgb', 'z_delta', 'constant_delta')
CODE_CONTROLS = ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'blinded')
CONTROLS = CODE_CONTROLS + ('adapted_held', 'teacher_state_reference', 'frozen_teacher_reference')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr', 'motion_mae')
DELTA_METRICS = ('mae', 'workspace_roi_mae', 'motion_mae')
MAIN_SHA = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
POSTFIT_SHA = '1669fe20188831af3cb6cae31dd1f8e8a2419cf8ed72010ef1130f23bc492f10'
PREPARATION_SHA = '8c9381593b2a7bce009e38d73c07f9de1b0ab1a411cfa7feb9ce6729d268ef57'
BINDING_SHA = 'a422de97bdfeb7b9dcae47a00478f43ddb63185b9b352b85e221cb914c7d1e8d'
LOADER_SHA = 'a3430beacb0ffb3eeff8bb5ce8988428ac418b4fc8c3fab5345e3cb49d5689db'
CLIP = 'buildingblock_tower_14_0'
SNAPSHOTS = (0, 44, 88, 132, 176)
SEMANTICS = 'Complete H9 posterior windows with true initial states; NOT AR rollout or action-semantic proof'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def protocol_contract(cfg):
    fixed = dict(schema='gs_neoverse_cross_source_eval_v1', main_manifest_sha256=MAIN_SHA,
        postfit_manifest_sha256=POSTFIT_SHA, preparation_manifest_sha256=PREPARATION_SHA,
        preparation_binding_sha256=BINDING_SHA, arms=list(ARMS), controls=list(CONTROLS),
        max_seconds_per_arm=2400, resolution_hw=[252, 448], frames=177, windows=168,
        horizons=9, frame_gap=1, workspace_roi_xyxy=[67, 38, 380, 227])
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError('Fixed transfer protocol changed')


def verify_sources(root, expected=None):
    manifest = read(root / 'CODE_MANIFEST.json')
    if not {'protocol.json', 'evaluate_transfer.py', 'transfer_data.py'}.issubset(manifest):
        raise ValueError('Manifest must bind transfer code and protocol')
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.parent) or sha(path) != digest:
            raise ValueError('Transfer source changed: ' + relative)
    cfg = read(root / 'protocol.json')
    protocol_contract(cfg)
    main, preparation = Path(cfg['main_root']).resolve(), Path(cfg['preparation_root']).resolve()
    if (main != root.parent / 'gaussiangpt_neoverse_temporal_z_20261031_v7'
            or preparation != root.parent / 'gaussiangpt_neoverse_tower14_transfer_20261031_v1'):
        raise ValueError('Transfer must use the named sibling experiments')
    pinned = {main / 'CODE_MANIFEST.json': MAIN_SHA, main / 'POSTFIT_MANIFEST.json': POSTFIT_SHA,
              main / 'postfit_code/render_six_panel.py': LOADER_SHA,
              preparation / 'CODE_MANIFEST.json': PREPARATION_SHA,
              preparation / 'input_binding.json': BINDING_SHA}
    named = {(root / name).resolve(): digest for name, digest in manifest.items()}
    for path, digest in pinned.items():
        if named.get(path) != digest or sha(path) != digest:
            raise ValueError('Unbound runtime loader or source experiment: ' + str(path))
    current = dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
                   protocol_sha256=sha(root / 'protocol.json'))
    if expected is not None and current != expected:
        raise ValueError('Transfer source contract changed during evaluation')
    return cfg, current


def validate_rows(rows, initial=False):
    expected = {(start, 0) for start in range(168)} if initial else {
        (start, horizon) for start in range(168) for horizon in range(1, 10)}
    seen, bundles = set(), {}
    for row in rows:
        start, horizon = row['start'], row['horizon']
        key = start, horizon
        if (type(start) is not int or type(horizon) is not int or key not in expected
                or key in seen or row['end'] != start + horizon or row.get('clip') != CLIP):
            raise ValueError('Wrong/duplicate physical target identity')
        seen.add(key)
        masks = []
        for control in CONTROLS:
            families = ((row[control], METRICS),) if initial else (
                (row[control], METRICS), (row['delta'][control], DELTA_METRICS))
            for values, metrics in families:
                if set(values) != set(metrics):
                    raise ValueError('Metric keys differ')
                for metric in metrics:
                    value = values[metric]
                    if metric == 'motion_mae' and value is None:
                        continue
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('Missing/nonfinite/negative metric')
                masks.append(values['motion_mae'] is None)
        if len(set(masks)) != 1 or (initial and not all(masks)):
            raise ValueError('Motion null mask must depend only on the true target, with null h0')
        if not initial:
            current = []
            for name in ('motion_ids', 'blinded_motion_ids'):
                ids = row[name]
                if len(ids) != 9 or any(type(v) is not int or not 0 <= v < 65536 for v in ids):
                    raise ValueError('Invalid nine-code LFQ bundle')
                current.append(ids)
            if start in bundles and bundles[start] != current:
                raise ValueError('One window has inconsistent code bundles across horizons')
            bundles[start] = current
    if seen != expected:
        raise ValueError('Incomplete fixed full-window coverage')


def aggregate(rows, control, delta=False):
    keys = DELTA_METRICS if delta else METRICS
    grouped = {}
    for row in rows:
        grouped.setdefault(row['end'], []).append(row['delta'][control] if delta else row[control])
    frame_means, counts = {}, {}
    for key in keys:
        values = [statistics.fmean(item[key] for item in group if item[key] is not None)
                  for group in grouped.values() if any(item[key] is not None for item in group)]
        frame_means[key] = statistics.fmean(values) if values else None
        counts[key] = len(values)
    return dict(frame_uniform=frame_means, valid_physical_frames=counts)


def run(root, arm, max_seconds):
    begun = time.monotonic()
    output = root / 'evaluations' / arm
    output.mkdir(parents=True, exist_ok=False)
    stage, rows, initial_rows, previous_alarm = 'source_admission', [], [], None

    def check():
        if time.monotonic() - begun >= max_seconds:
            raise TimeoutError('Fixed transfer evaluation budget exceeded; no retry')

    def deadline(signum, frame):
        raise TimeoutError('Transfer evaluation deadline reached')

    try:
        if hasattr(signal, 'setitimer'):
            previous_alarm = signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, max_seconds)
        protocol, admission = verify_sources(root)
        main, prep = Path(protocol['main_root']), Path(protocol['preparation_root'])
        replay = module_at('_transfer_selected_runtime', main / 'postfit_code/render_six_panel.py')
        adapter = module_at('_transfer_data_adapter', root / 'transfer_data.py')
        context = replay.load_terminal(main)
        prepared = adapter.load_prepared(prep)
        target_context_sha = hashlib.sha256(json.dumps(prepared['hashes'], sort_keys=True,
                                                     separators=(',', ':')).encode()).hexdigest()
        stage = 'restore_from_training_data'
        torch, core, _, _, cfg, train_data = context['runner'].setup_runtime(context['evidence'])
        torch.cuda.reset_peak_memory_stats()
        model, bridge = replay.restore_selected(context, arm, core, torch, cfg, train_data)
        selected = context['selection']['checkpoints'][arm]
        checkpoint = torch.load(selected['path'], map_location='cpu', weights_only=False)
        original_std = checkpoint['model']['output_std']
        if (not torch.equal(model.output_std.detach().cpu(), original_std)
                or not torch.equal(model.output_calibrated.detach().cpu(), checkpoint['model']['output_calibrated'])
                or not bool(model.output_calibrated)):
            raise ValueError('TRAIN output units were not restored from the selected checkpoint')
        std_hash = replay.tensor_digest(('output_std', original_std))
        del original_std, checkpoint, train_data
        gc.collect()
        data = adapter.TransferData(prepared, device='cuda')
        if len(data.states) != 177 or len(data.records) != 177 or len(data.frozen_reconstructions) != 177:
            raise ValueError('Target clip must contain all177 original frames')
        if cfg['resources']['peak_limit_gib'] != 48:
            raise ValueError('Native peak limit changed')
        if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in bridge.parameters()):
            raise ValueError('Transfer evaluation parameters must all be frozen')
        before = dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
        std_before = replay.tensor_digest(('output_std', model.output_std))
        if std_before != std_hash:
            raise ValueError('Loading target data changed the TRAIN output units')
        recorder = replay.GaussianRecorder(bridge)
        mapping = {row['source_frame']: row for row in context['timing'].frame_mapping()
                   if row['source_frame'] in SNAPSHOTS}
        snapshot_keys = {(row['start'], row['horizon']): frame for frame, row in mapping.items()}
        (output / 'snapshots').mkdir()
        snapshots, render_calls, decode_calls = {}, 0, 0

        def render(prediction, frame):
            nonlocal render_calls
            check()
            image = bridge.render(prediction, data.records[frame])
            render_calls += 1
            audit = dict(recorder.last)
            if (audit['count'] != bridge.last_count or not 0 < audit['count'] <= cfg['resources']['max_decoded_gaussians']
                    or torch.cuda.max_memory_allocated() / 2**30 > 48):
                raise RuntimeError('Complete GS count or original native memory guard failed')
            return image, audit

        def save_snapshots(start, horizon, images):
            key = start, horizon
            if key not in snapshot_keys:
                return
            from PIL import Image
            frame = snapshot_keys[key]
            if frame in snapshots or set(images) != set(CONTROLS):
                raise ValueError('Incomplete or duplicate fixed snapshot')
            files = {}
            for name, image in dict(real=data.records[frame]['real'], **images).items():
                if tuple(image.shape) != (252, 448, 3) or not bool(torch.isfinite(image).all()):
                    raise ValueError('Snapshot must use a complete native image')
                path = output / 'snapshots' / f'frame_{frame:03d}_{name}.png'
                pixels = (image.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                Image.fromarray(pixels, 'RGB').save(path)
                files[name] = path.relative_to(output).as_posix()
            snapshots[frame] = dict(mapping=mapping[frame], source_pts=float(data.states[frame]['pts']),
                                    semantics=SEMANTICS, images=files)

        stage = 'posterior_ids_and_native_evaluation'
        with torch.no_grad():
            ids, blind = [], []
            constant = model.code_mode == 'constant'
            for start in range(0, 168, 4):
                check()
                windows = [data.window(s) for s in range(start, start + 4)]
                ids.append(model.encode_windows(windows)['indices'])
                if not constant:
                    blind.append(model.encode_windows([context['probe'].blind_window(w) for w in windows])['indices'])
            ids = torch.cat(ids, 0)
            blinded_ids = ids if constant else torch.cat(blind, 0)
            if (tuple(ids.shape) != (168, 9) or tuple(blinded_ids.shape) != (168, 9)
                    or ids.dtype not in (torch.int32, torch.int64) or blinded_ids.dtype not in (torch.int32, torch.int64)
                    or not bool(((ids >= 0) & (ids < 65536)).all())
                    or not bool(((blinded_ids >= 0) & (blinded_ids < 65536)).all())
                    or constant != arm.startswith('constant_')
                    or (constant and not bool((ids == model.constant_id).all()))):
                raise ValueError('Invalid target posterior or constant LFQ IDs')
            ids_list, blind_list = ids.cpu().tolist(), blinded_ids.cpu().tolist()
            with (output / 'targets.jsonl').open('x', encoding='utf-8') as log:
                for start in range(168):
                    check()
                    condition = [core.make_condition(data.states[start])]
                    controls = [('correct', ids[start:start + 1])]
                    if not constant:
                        controls += [(f'shuffle{offset}', ids[(start + offset) % 168:(start + offset) % 168 + 1])
                                     for offset in (42, 84, 126)]
                        controls.append(('blinded', blinded_ids[start:start + 1]))
                    predictions = {}
                    for name, bundle in controls:
                        sequence = model.decode_ids(bundle, condition)[0]
                        decode_calls += 1
                        if len(sequence) != 10 or any(
                            not torch.equal(frame['coordinates'], data.states[start]['coordinates'])
                            or frame['features'].shape != data.states[start]['features'].shape
                            or not bool(torch.isfinite(frame['features']).all()) for frame in sequence):
                            raise ValueError('LDM changed the complete initial support, feature width or output count')
                        predictions[name] = sequence
                    initial, audit0 = {}, {}
                    for name, frames in predictions.items():
                        initial[name], audit0[name] = render(frames[0], start)
                    if constant:
                        for name in CODE_CONTROLS[1:]:
                            initial[name], audit0[name] = initial['correct'], dict(audit0['correct'], reused_from='correct')
                    initial['adapted_held'], audit0['adapted_held'] = render(data.states[start], start)
                    initial['teacher_state_reference'] = initial['adapted_held']
                    audit0['teacher_state_reference'] = dict(audit0['adapted_held'], reused_from='adapted_held')
                    initial['frozen_teacher_reference'] = data.frozen_reconstructions[start]
                    audit0['frozen_teacher_reference'] = dict(source='frozen_codec_cached_reconstruction', actual_render=False)
                    initial_row = dict(clip=CLIP, start=start, end=start, horizon=0,
                        native_gaussians=audit0, **{name: data.metrics(image, start, 0) for name, image in initial.items()})
                    initial_rows.append(initial_row)
                    save_snapshots(start, 0, initial)
                    for h in range(1, 10):
                        target = start + h
                        images, audits = {}, {}
                        for name, frames in predictions.items():
                            images[name], audits[name] = render(frames[h], target)
                        if constant:
                            for name in CODE_CONTROLS[1:]:
                                images[name], audits[name] = images['correct'], dict(audits['correct'], reused_from='correct')
                        images['adapted_held'], audits['adapted_held'] = render(data.states[start], target)
                        images['teacher_state_reference'], audits['teacher_state_reference'] = render(data.states[target], target)
                        images['frozen_teacher_reference'] = data.frozen_reconstructions[target]
                        audits['frozen_teacher_reference'] = dict(source='frozen_codec_cached_reconstruction', actual_render=False)
                        row = dict(clip=CLIP, start=start, end=target, horizon=h,
                            motion_ids=ids_list[start], blinded_motion_ids=blind_list[start], native_gaussians=audits,
                            **{name: data.metrics(image, start, h) for name, image in images.items()},
                            delta={name: data.delta_metrics(image, initial[name], start, h) for name, image in images.items()})
                        rows.append(row)
                        log.write(json.dumps(row, allow_nan=False) + '\n')
                        save_snapshots(start, h, images)
                        del images
                    log.flush()
                    del predictions, initial
                    if start % 21 == 0:
                        print(arm, 'TRANSFER', start + 1, '/168', flush=True)
        validate_rows(rows)
        validate_rows(initial_rows, initial=True)
        if set(snapshots) != set(SNAPSHOTS):
            raise ValueError('Missing fixed physical-frame snapshots')
        expected_calls = (168, 4872) if constant else (840, 11592)
        if (decode_calls, render_calls) != expected_calls:
            raise ValueError('Complete native work count differs from fixed scope')
        after = dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
        std_after = replay.tensor_digest(('output_std', model.output_std))
        if before != after or std_after != std_hash:
            raise ValueError('Frozen model/codec/TRAIN output units changed')
        write_new(output / 'initial_reconstruction.json', dict(rows=initial_rows, unique_initial_frames=168))
        write_new(output / 'snapshots.json', dict(frames=snapshots, resolution_hw=[252, 448], semantics=SEMANTICS))
        actions = dict(schema='gs_tower14_transfer_actions_v1', arm=arm, ids=ids.cpu(), blinded_ids=blinded_ids.cpu(),
            starts=list(range(168)), horizons=list(range(1, 10)), selected_step=context['step'],
            checkpoint_path=selected['path'], checkpoint_sha256=selected['sha256'], target_context_sha256=target_context_sha,
            code_mode=model.code_mode, constant_id=model.constant_id if constant else None, semantics=SEMANTICS, **admission)
        with (output / 'actions.pt').open('xb') as stream:
            torch.save(actions, stream)
        adapter.verify_unchanged(prepared)
        for path, digest in context['hashes'].items():
            check()
            if sha(path) != digest:
                raise ValueError('Main selected-source artifact changed: ' + path)
        verify_sources(root, admission)
        artifacts = {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
        check()
        result = dict(schema='gs_tower14_transfer_evaluation_v1', status='complete', arm=arm, clip=CLIP,
            selected_step=context['step'], checkpoint_path=selected['path'], checkpoint_sha256=selected['sha256'],
            main_manifest_sha256=MAIN_SHA, postfit_manifest_sha256=POSTFIT_SHA, **admission,
            preparation_manifest_sha256=PREPARATION_SHA, preparation_binding_sha256=BINDING_SHA,
            target_context_sha256=target_context_sha, target_input_sha256=prepared['hashes'], main_input_sha256=context['hashes'],
            windows=168, future_exposures=1512, unique_future_frames=176, initial_frames=168,
            optimizer_updates=0, all_parameters_frozen=True, all_initial_latent_points_retained=True,
            native_internal_hard_pruning_retained=True, model_state_sha256_before=before['model'],
            model_state_sha256_after=after['model'], codec_state_sha256_before=before['codec'], codec_state_sha256_after=after['codec'],
            output_std_sha256_source=std_hash, output_std_sha256_before=std_before, output_std_sha256_after=std_after,
            native_render_calls=render_calls, latent_window_decode_calls=decode_calls, code_mode=model.code_mode,
            constant_id=model.constant_id if constant else None, constant_identical_code_controls_reused=constant,
            artifact_sha256=artifacts, targets_sha256=artifacts['targets.jsonl'], actions_sha256=artifacts['actions.pt'],
            initial_reconstruction_sha256=artifacts['initial_reconstruction.json'],
            metrics={name: aggregate(rows, name) for name in CONTROLS},
            delta_metrics={name: aggregate(rows, name, delta=True) for name in CONTROLS},
            blinded_id_statistics=context['probe'].id_statistics(ids_list, blind_list),
            resolution_hw=[252, 448], workspace_roi_xyxy=[67, 38, 380, 227],
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_seconds=time.monotonic() - begun,
            max_seconds=max_seconds, goal_complete=False, scientific_transfer_passed=None, semantics=SEMANTICS,
            limitations=['One project-defined cross-source clip; historical exposure outside this experiment is unknown',
                'Posterior code sees future GS; decoder sees only true initial state and quantized IDs',
                'Workspace crop is fixed, not semantic hand annotation; motion weights include camera and illumination',
                'Each temporal difference uses that control own h0; zero-weight motion metrics are null',
                'Teacher-state reconstruction is a reference path, not a theoretical upper bound',
                'Snapshots use fixed physical-frame mapping and are not an AR rollout'])
        if result['gpu_peak_gib'] > 48:
            raise RuntimeError('Original native peak-memory guard exceeded')
        write_new(output / 'EVALUATION.json', result)
        print(json.dumps(dict(status='complete', arm=arm, selected_step=context['step'], output=str(output),
                              goal_complete=False)), flush=True)
    except BaseException as error:
        if hasattr(signal, 'setitimer'):
            signal.setitimer(signal.ITIMER_REAL, 0)
        write_new(output / 'FAILED.json', dict(status='failed', arm=arm, stage=stage, error=repr(error),
            traceback=traceback.format_exc(), completed_future_exposures=len(rows), completed_initial_frames=len(initial_rows),
            optimizer_updates=0, automatic_retry=False, goal_complete=False))
        raise
    finally:
        if hasattr(signal, 'setitimer'):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_alarm is not None:
                signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--max-seconds', type=int, default=2400)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 2400:
        parser.error('--max-seconds must be in1..2400')
    run(args.root.resolve(), args.arm, args.max_seconds)
