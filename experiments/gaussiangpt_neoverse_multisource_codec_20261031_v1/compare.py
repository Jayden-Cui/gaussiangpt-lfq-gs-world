"""CPU-only, fail-closed comparison of the fixed multisource codec experiment.

Reads hashes and worker receipts; never imports torch/ME, deserializes checkpoints,
trains, or replays GPU rendering. Only SUMMARY.json / SUMMARY.md are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import sources

ARMS = ('decoder_only', 'full_codec')
LABELS = ('baseline', *ARMS)
CLIPS = sources.ALL_CLIPS
TRAIN_CLIPS = sources.TRAIN_CLIPS
INITIAL_SHA256 = sources.CODEC_SHA
GROUPS = ('codec_real', 'teacher_real', 'codec_teacher', 'pseudo_view_teacher')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr')
LOSS = dict(real_l1=1., real_vgg19=.1, teacher_l1=.5, pseudo_view_l1=.25,
            workspace_roi_l1=.25, occupancy_bce=1.)
ROI = [67, 38, 380, 227]
SNAPSHOTS = (0, 44, 88, 132, 176)
PICTURES = ('real', 'teacher', 'codec', 'nearby', 'nearby_teacher')
KEYS = {(clip, frame) for clip in CLIPS for frame in range(177)}
UNSPECIFIED_HASH = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def read(path):
    value = sources.read(path)
    require(isinstance(value, dict), f'{path}: top level must be an object')
    return value


def rows(path):
    result = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'{path}:{line_number}: empty line')
            row = json.loads(line)
            require(isinstance(row, dict), f'{path}:{line_number}: row must be an object')
            result.append(row)
    return result


def check_hash(path, expected=UNSPECIFIED_HASH, size=None):
    if expected is not UNSPECIFIED_HASH:
        require(sources.valid_sha(expected), f'{path}: missing/invalid SHA256')
    else:
        expected = None
    return sources.fingerprint(path, expected=expected, size=size)


def child(folder, relative):
    require(isinstance(relative, str) and relative, 'Missing relative artifact path')
    path = (folder / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(folder.resolve()),
            f'Artifact escapes its evaluation directory: {relative}')
    return path


def fields(document, expected, context):
    for key, value in expected.items():
        actual = document.get(key)
        require(type(actual) is type(value) and actual == value, f'{context}.{key} differs or is missing')


def metric_values(value, context):
    require(isinstance(value, dict), f'{context}: missing metrics')
    require(all(numeric(value.get(metric)) for metric in METRICS), f'{context}: missing/nonfinite metric')
    require(0 <= value['mae'] <= 1 and 0 <= value['workspace_roi_mae'] <= 1,
            f'{context}: MAE outside normalized image range')


def check_replay(value, context):
    require(isinstance(value, dict) and set(value) == {'real_view', 'nearby_view'},
            f'{context}: both saved-packet replay views required')
    for view, scores in value.items():
        metric_values(scores, f'{context}/{view}')
        require(scores['mae'] <= 1e-5 and scores['psnr'] >= 60., f'{context}/{view}: replay failed')


def mean_metrics(selected):
    return {group: {metric: float(np.mean([row[group][metric] for row in selected]))
                    for metric in METRICS} for group in GROUPS}


def compare_means(actual, stored, context):
    require(isinstance(stored, dict), f'{context}: missing summary')
    for group in GROUPS:
        metric_values(stored.get(group), f'{context}/{group}')
        for metric in METRICS:
            require(math.isclose(actual[group][metric], stored[group][metric], rel_tol=1e-7, abs_tol=1e-9),
                    f'{context}/{group}/{metric}: summary does not match physical frames')


def load_contract(root):
    cfg = read(root / 'protocol.json')
    sources.validate_protocol(cfg)
    fields(cfg, dict(seed=20261031, goal_complete=False), 'protocol')
    fields(cfg['train'], dict(steps=300, seed=20261031, batch_size=2,
                             one_frame_per_train_clip_each_step=True, learning_rate=.0001,
                             betas=[.9, .95], weight_decay=0., grad_clip=1., save_steps=[150, 300],
                             force_target_steps=240, no_force_steps=60, loss=LOSS), 'train')
    fields(cfg['evaluation'], dict(baseline_once=True, records_per_model=531, frames_per_clip=177,
                                  clips=3, snapshot_frames=list(SNAPSHOTS), save_reencoded_states=True,
                                  saved_state_replay_psnr_min=60., saved_state_replay_mae_max=1e-5,
                                  views=[0, 1], workspace_roi_xyxy=ROI, target_forcing=False), 'evaluation')
    fields(cfg['arms']['decoder_only'], dict(encoder_frozen=True,
           trainable_prefixes=['ae.decoder.', 'ae.feature_heads.', 'ae.gaussian_multiplier.']), 'decoder arm')
    fields(cfg['arms']['full_codec'], dict(encoder_frozen=False, trainable_prefixes=['*']), 'full arm')
    binding = read(root / 'DATA_BINDING.json')
    fields(binding, dict(schema='gs_multisource_codec_data_binding_v1', frames_per_clip=177,
                        source_points_per_frame=112896, height=252, width=448,
                        train_clips=list(TRAIN_CLIPS), eval_only_clips=[sources.EVAL_CLIP],
                        historical_roles=sources.HISTORICAL_ROLES, official_independent_holdout=False,
                        historical_latents_reused=False, initial_codec_sha256=INITIAL_SHA256,
                        state_quantization='Identity; continuous float32 Nx64'), 'DATA_BINDING')
    require(isinstance(binding.get('files'), list) and binding['files'], 'No bound input files')
    bound, caches = {}, {}
    for item in binding['files']:
        path = Path(item['path']).resolve()
        require(path not in bound, f'Duplicate bound file: {path}')
        require(type(item.get('bytes')) is int and item['bytes'] >= 0, f'Invalid bound size: {path}')
        bound[path] = sources.fingerprint(path, expected=item['sha256'], size=item['bytes'],
                                          allow_empty=path.suffix == '.py')
        for clip in CLIPS:
            if path.parent.name == clip and path.parent.parent.name == 'cache':
                require(path.suffix == '.pt' and path.stem.isdigit(), f'Unexpected frame cache {path}')
                key = (clip, int(path.stem))
                require(key not in caches, f'Duplicate bound cache {key}')
                caches[key] = item['sha256']
    require(set(caches) == KEYS, 'Binding does not include all 531 full-frame source caches')
    require(any(item['sha256'] == INITIAL_SHA256 for item in bound.values()),
            'Shared original checkpoint is absent from verified input binding')
    for name in ('protocol.json', 'worker.py', 'sources.py', 'compare.py', 'CODE_MANIFEST.json'):
        require((root / name).resolve() in bound, f'Binding omits current {name}')
    code_manifest = read(root / 'CODE_MANIFEST.json')
    require(bool(code_manifest), 'Current code manifest is empty')
    require({'protocol.json', 'worker.py', 'sources.py', 'compare.py'} <= set(code_manifest),
            'Code manifest omits required current protocol/worker/source/comparison code')
    for relative, expected in code_manifest.items():
        path = (root / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root.parent),
                f'Code manifest path escapes run parent: {relative}')
        require(path in bound and bound[path]['sha256'] == expected,
                f'Current code manifest differs from verified DATA_BINDING: {relative}')
    plan = sources.balanced_plan(20261031, 300)
    source = dict(binding_sha256=check_hash(root / 'DATA_BINDING.json')['sha256'],
                  initial_checkpoint_sha256=INITIAL_SHA256,
                  protocol_sha256=check_hash(root / 'protocol.json')['sha256'],
                  worker_sha256=check_hash(root / 'worker.py')['sha256'],
                  sources_sha256=check_hash(root / 'sources.py')['sha256'],
                  plan_sha256=digest(plan), inherited_codec_steps=3000, additional_codec_updates=300,
                  seed=20261031, batch_frames=2, forcing_updates=240, unforced_updates=60,
                  goal_complete=False)
    preflight = read(root / 'PREFLIGHT.json')
    fields(preflight, dict(status='passed', initial_arms_identical=True, goal_complete=False), 'PREFLIGHT')
    fields(preflight.get('source', {}), source, 'PREFLIGHT.source')
    require(set(preflight.get('arms', {})) == set(ARMS), 'Preflight must contain both arms')
    initial_models = [preflight['arms'][arm].get('initial_model_sha256') for arm in ARMS]
    require(all(sources.valid_sha(value) for value in initial_models) and len(set(initial_models)) == 1,
            'Preflight initial models differ or are missing')
    for arm in ARMS:
        fields(preflight['arms'][arm], dict(passed=True, discarded_updates=2, production_updates=0,
               frozen_encoder_parameters_and_buffers_unchanged=arm == 'decoder_only'), f'PREFLIGHT/{arm}')
        before, after = (preflight['arms'][arm].get(name) for name in
                         ('initial_encoder_sha256', 'final_encoder_sha256'))
        require(sources.valid_sha(before) and sources.valid_sha(after) and
                ((before == after) if arm == 'decoder_only' else (before != after)),
                f'PREFLIGHT/{arm}: encoder update/freeze evidence is missing or contradictory')
    return dict(source=source, plan=plan, caches=caches, binding=binding,
                preflight=preflight, preflight_sha256=check_hash(root / 'PREFLIGHT.json')['sha256'])


def verify_trace(trace, arm, plan):
    require(len(trace) == 300, f'{arm}: trace must contain exactly 300 updates')
    seen, encoder_gradient_seen = set(), False
    for step, row in enumerate(trace, 1):
        fields(row, dict(step=step, arm=arm, force_target=step <= 240), f'{arm} trace {step}')
        samples = row.get('frames')
        require(isinstance(samples, list) and len(samples) == 2, f'{arm}/{step}: incomplete B2 batch')
        pair = [(sample.get('clip'), sample.get('frame')) for sample in samples]
        require(pair == list(plan[step - 1]) and all(type(sample['frame']) is int for sample in samples),
                f'{arm}/{step}: EVAL leakage or wrong balanced TRAIN plan')
        for sample in samples:
            require(numeric(sample.get('loss')), f'{arm}/{step}: nonfinite sample loss')
            terms = sample.get('terms', {})
            require(set(terms) == set(LOSS) and all(numeric(v) for v in terms.values()),
                    f'{arm}/{step}: incorrect/nonfinite loss terms')
            expected = sum(LOSS[k] * terms[k] for k in LOSS)
            require(math.isclose(sample['loss'], expected, rel_tol=2e-5, abs_tol=1e-6),
                    f'{arm}/{step}: sample loss does not match fixed weights')
            for count in ('latent_count', 'output_count'):
                require(type(sample.get(count)) is int and sample[count] > 0,
                        f'{arm}/{step}: missing complete-frame geometry count')
        require(numeric(row.get('loss')) and math.isclose(row['loss'],
                sum(sample['loss'] for sample in samples) / 2, rel_tol=1e-7, abs_tol=1e-9),
                f'{arm}/{step}: B2 arithmetic mean is inconsistent')
        require(set(row.get('terms', {})) == set(LOSS), f'{arm}/{step}: missing averaged loss terms')
        for name in LOSS:
            require(numeric(row['terms'][name]) and math.isclose(row['terms'][name],
                    sum(sample['terms'][name] for sample in samples) / 2, rel_tol=1e-7, abs_tol=1e-9),
                    f'{arm}/{step}: incorrect averaged term {name}')
        require(numeric(row.get('gradient_norm')) and row['gradient_norm'] > 0,
                f'{arm}/{step}: missing finite nonzero gradient')
        gradients = row.get('gradient_groups', {})
        require(isinstance(gradients, dict) and gradients, f'{arm}/{step}: missing gradient evidence')
        if arm == 'decoder_only':
            require(set(gradients) <= {'ae.decoder', 'ae.feature_heads', 'ae.gaussian_multiplier'},
                    f'{arm}/{step}: encoder has trainable gradients')
        for group, counts in gradients.items():
            require(all(type(counts.get(k)) is int for k in ('parameters', 'with_grad', 'nonzero_grad'))
                    and 0 <= counts['nonzero_grad'] <= counts['with_grad'] <= counts['parameters']
                    and counts['parameters'] > 0, f'{arm}/{step}/{group}: invalid gradient counts')
            if group not in ('ae.decoder', 'ae.feature_heads', 'ae.gaussian_multiplier') and counts['nonzero_grad'] > 0:
                encoder_gradient_seen = True
        seen.update(pair)
        fields(row, dict(seen_frames=len(seen)), f'{arm}/{step}')
        if step == 177:
            require(len(seen) == 354, f'{arm}: first 177 updates did not cover all 354 TRAIN frames')
    require(arm != 'full_codec' or encoder_gradient_seen, 'full_codec: no recorded nonzero encoder gradients')


def load_training(root, arm, contract):
    folder = root / 'arms' / arm
    require(not (folder / 'FAILURE.json').exists(), f'{arm}: explicit failure receipt exists')
    source = read(folder / 'SOURCE.json')
    fields(source, {**contract['source'], 'arm': arm, 'preflight_sha256': contract['preflight_sha256']},
           f'{arm}/SOURCE')
    done = read(folder / 'TRAIN_DONE.json')
    fields(done, dict(status='complete', arm=arm, steps=300, inherited_codec_steps=3000,
                      additional_codec_updates=300, frames_seen=354, first_epoch_coverage_at_step=177,
                      no_target_forcing_steps=60, frozen_encoder_verified=arm == 'decoder_only',
                      fresh_optimizer_verified=True, goal_complete=False), f'{arm}/TRAIN_DONE')
    fields(done.get('source', {}), contract['source'], f'{arm}/TRAIN_DONE.source')
    require(done.get('initial_model_sha256') == contract['preflight']['arms'][arm]['initial_model_sha256'],
            f'{arm}: initial model differs from shared preflight')
    check_hash(folder / 'step_0300.pt', done.get('checkpoint_sha256'))
    check_hash(folder / 'step_0150.pt', done.get('checkpoint_0150_sha256'))
    check_hash(folder / 'trace.jsonl', done.get('trace_sha256'))
    for name in ('initial_encoder_sha256', 'final_encoder_sha256',
                 'initial_encoder_parameters_sha256', 'final_encoder_parameters_sha256'):
        require(sources.valid_sha(done.get(name)), f'{arm}: missing {name}')
    require(done['initial_encoder_sha256'] == contract['preflight']['arms'][arm]['initial_encoder_sha256'],
            f'{arm}: initial encoder differs from preflight')
    if arm == 'decoder_only':
        require(done['initial_encoder_sha256'] == done['final_encoder_sha256'] and
                done['initial_encoder_parameters_sha256'] == done['final_encoder_parameters_sha256'],
                'decoder_only: frozen encoder parameters or buffers changed')
    else:
        require(done['initial_encoder_sha256'] != done['final_encoder_sha256'] and
                done['initial_encoder_parameters_sha256'] != done['final_encoder_parameters_sha256'],
                'full_codec: encoder parameters did not update')
    verify_trace(rows(folder / 'trace.jsonl'), arm, contract['plan'])
    return done


def index_frames(records, context):
    index = {}
    for row in records:
        key = (row.get('clip'), row.get('frame'))
        require(type(key[1]) is int and key in KEYS and key not in index,
                f'{context}: invalid/duplicate physical clip/frame {key}')
        require(row.get('split') == ('TRAIN' if key[0] in TRAIN_CLIPS else 'EVAL'),
                f'{context}: wrong role for {key}')
        require(numeric(row.get('pts')), f'{context}: missing/nonfinite presentation time for {key}')
        index[key] = row
    require(set(index) == KEYS and len(records) == 531, f'{context}: incomplete 531-frame coverage')
    return index


def load_evaluation(root, label, contract, training):
    folder = root / 'evaluations' / label
    require(not (folder / 'FAILURE.json').exists(), f'{label}: evaluation failure receipt exists')
    fields(read(folder / 'SOURCE.json'), {**contract['source'], 'arm': label}, f'{label}/SOURCE')
    evaluation = read(folder / 'EVALUATION.json')
    step = 0 if label == 'baseline' else 300
    checkpoint = INITIAL_SHA256 if label == 'baseline' else training[label]['checkpoint_sha256']
    fields(evaluation, dict(status='evaluated', arm=label, frames=531, frames_per_clip=177,
           additional_codec_updates=step, inherited_codec_steps=3000, checkpoint_sha256=checkpoint,
           roi_xyxy=ROI, every_packet_saved_replay_passed=True, all_frames_freshly_encoded=True,
           decoder_eval=True, decoder_target_key=None, goal_complete=False), f'{label}/EVALUATION')
    fields(evaluation.get('source', {}), contract['source'], f'{label}/EVALUATION.source')
    encoder = evaluation.get('encoder_sha256')
    require(sources.valid_sha(encoder), f'{label}: missing encoder hash')
    if label != 'baseline':
        require(encoder == training[label]['final_encoder_sha256'], f'{label}: evaluation encoder differs')
    for name in ('targets', 'state_manifest', 'screenshot_manifest'):
        check_hash(folder / f'{name}.jsonl', evaluation.get(f'{name}_sha256'))
    targets = index_frames(rows(folder / 'targets.jsonl'), f'{label}/targets')
    packets = index_frames(rows(folder / 'state_manifest.jsonl'), f'{label}/packets')
    for key, row in targets.items():
        for group in GROUPS:
            metric_values(row.get(group), f'{label}/{key}/{group}')
        fields(row, dict(checkpoint_sha256=checkpoint, encoder_sha256=encoder,
                         additional_codec_updates=step, latent_channels=64, input_gaussians=112896),
               f'{label}/{key}')
        for name, maximum in (('input_voxels', 80000), ('decoded_gaussians', 2400000), ('latent_positions', 80000)):
            require(type(row.get(name)) is int and 0 < row[name] <= maximum,
                    f'{label}/{key}: invalid complete-frame {name}')
        check_replay(row.get('replay'), f'{label}/{key}')
        packet = packets[key]
        fields(packet, dict(arm=label, checkpoint_sha256=checkpoint, encoder_sha256=encoder,
                           binding_sha256=contract['source']['binding_sha256'], inherited_codec_steps=3000,
                           additional_codec_updates=step, decoder_target_key=None, fresh_encode=True,
                           frame_count=177, shape=[row['latent_positions'], 64],
                           decoded_gaussians=row['decoded_gaussians'], source_cache_sha256=contract['caches'][key]),
               f'{label}/{key}/packet')
        require(packet['pts'] == row['pts'] and packet.get('normalization') is not None,
                f'{label}/{key}: packet time/normalization is missing or differs')
        require(packet.get('state_path') == f'states/{key[0]}/{key[1]:03d}.pt',
                f'{label}/{key}: wrong saved state identity')
        require(type(packet.get('state_bytes')) is int and packet['state_bytes'] > 0,
                f'{label}/{key}: invalid state size')
        check_hash(child(folder, packet['state_path']), packet.get('state_sha256'), packet['state_bytes'])
        check_replay(packet.get('saved_packet_replay'), f'{label}/{key}/packet')
        require(packet['saved_packet_replay'] == row['replay'], f'{label}/{key}: replay receipts disagree')
    means = {clip: mean_metrics([targets[(clip, frame)] for frame in range(177)]) for clip in CLIPS}
    for clip in CLIPS:
        compare_means(means[clip], evaluation.get('summary_by_clip', {}).get(clip), f'{label}/{clip}')
    all_mean = mean_metrics(list(targets.values()))
    compare_means(all_mean, evaluation.get('summary_all'), f'{label}/all')
    screenshots = rows(folder / 'screenshot_manifest.jsonl')
    required_pictures = {(clip, frame, kind) for clip in CLIPS for frame in SNAPSHOTS for kind in PICTURES}
    seen = set()
    for picture in screenshots:
        key = (picture.get('clip'), picture.get('frame'), picture.get('kind'))
        require(key in required_pictures and key not in seen, f'{label}: invalid/duplicate screenshot {key}')
        seen.add(key)
        require(picture.get('path') == f'screenshots/{key[0]}/{key[1]:03d}_{key[2]}.png',
                f'{label}: screenshot path has wrong clip/frame/view identity')
        require(type(picture.get('bytes')) is int and picture['bytes'] > 0,
                f'{label}: screenshot bytes missing or invalid')
        path = child(folder, picture.get('path'))
        check_hash(path, picture.get('sha256'), picture.get('bytes'))
        with path.open('rb') as stream:
            header = stream.read(24)
        require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
                and struct.unpack('>II', header[16:24]) == (448, 252),
                f'{label}: screenshot is not native 448x252 PNG: {path}')
    require(seen == required_pictures, f'{label}: missing fixed-frame visual evidence')
    return dict(evaluation=evaluation, by_clip=means, all_frames_descriptive=all_mean,
                screenshots=screenshots, _targets=targets, _packets=packets)


def moving_block_bootstrap(differences):
    """v6 compare_arms algorithm adapted from frames1..176 to all frames0..176.

    The old function hard-codes future-only coverage and cannot be called for this
    codec's complete-frame task. Same non-circular blocks, RNG and percentiles.
    """
    require(set(differences) == set(range(177)), 'Bootstrap needs all 177 physical frames of one clip')
    values = np.asarray([differences[t] for t in range(177)], dtype=np.float64)
    require(bool(np.isfinite(values).all()), 'Nonfinite paired differences')
    generator = np.random.default_rng(20261031)
    starts = generator.integers(0, len(values) - 16 + 1, size=(2000, math.ceil(len(values) / 16)))
    indices = (starts[:, :, None] + np.arange(16)).reshape(2000, -1)[:, :len(values)]
    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return dict(mean_improvement=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                positive_interval=bool(low > 0), physical_frames=177, block_length=16,
                replicates=2000, seed=20261031,
                interpretation='reference MAE minus candidate MAE; positive is lower candidate error',
                scope='Descriptive paired interval for this single source clip; no population inference')


def scientific_report(evaluations):
    comparisons = {}
    for candidate, reference in (('decoder_only', 'baseline'), ('full_codec', 'baseline'), ('full_codec', 'decoder_only')):
        by_clip = {}
        for clip in CLIPS:
            by_clip[clip] = {
                group: {metric: moving_block_bootstrap({frame:
                    evaluations[reference]['_targets'][(clip, frame)][group][metric] -
                    evaluations[candidate]['_targets'][(clip, frame)][group][metric]
                    for frame in range(177)}) for metric in ('mae', 'workspace_roi_mae')}
                for group in GROUPS}
        comparisons[f'{candidate}_vs_{reference}'] = by_clip
    quality = {}
    for label, evaluation in evaluations.items():
        quality[label] = {}
        for clip, metrics in evaluation['by_clip'].items():
            teacher = metrics['codec_teacher']['psnr']
            full_gap = metrics['teacher_real']['psnr'] - metrics['codec_real']['psnr']
            roi_gap = metrics['teacher_real']['workspace_roi_psnr'] - metrics['codec_real']['workspace_roi_psnr']
            quality[label][clip] = dict(codec_teacher_psnr=teacher, full_real_psnr_gap_to_teacher=full_gap,
                workspace_roi_real_psnr_gap_to_teacher=roi_gap,
                reference_passed=teacher >= 25 and full_gap <= 1 and roi_gap <= 1)
    return comparisons, quality


def build_summary(root):
    root = Path(root).resolve()
    issues, training, evaluations = [], {}, {}
    contract = None
    def attempt(context, operation):
        try:
            return operation()
        except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
            issues.append(f'{context}: {error}')
            return None
    if (root / 'FAILURE.json').exists():
        issues.append('Root FAILURE.json exists; failure cannot be overwritten by successful receipts')
    contract = attempt('contract', lambda: load_contract(root))
    if contract is not None:
        for arm in ARMS:
            result = attempt(arm, lambda arm=arm: load_training(root, arm, contract))
            if result is not None:
                training[arm] = result
        for label in LABELS:
            if label != 'baseline' and label not in training:
                continue
            result = attempt(label, lambda label=label: load_evaluation(root, label, contract, training))
            if result is not None:
                evaluations[label] = result
    if len(training) == 2:
        for key in ('initial_model_sha256', 'initial_encoder_sha256', 'initial_encoder_parameters_sha256'):
            if training[ARMS[0]][key] != training[ARMS[1]][key]:
                issues.append(f'Two arms differ in {key}')
    if len(evaluations) == 3:
        baseline_encoder = evaluations['baseline']['evaluation']['encoder_sha256']
        if any(done['initial_encoder_sha256'] != baseline_encoder for done in training.values()):
            issues.append('Shared baseline encoder differs from training initialization')
        for key in KEYS:
            reference = evaluations['baseline']['_targets'][key]
            packet = evaluations['baseline']['_packets'][key]
            for arm in ARMS:
                row, other_packet = evaluations[arm]['_targets'][key], evaluations[arm]['_packets'][key]
                if row['pts'] != reference['pts'] or row['teacher_real'] != reference['teacher_real']:
                    issues.append(f'{arm}/{key}: physical time or unchanged teacher reference differs')
                if other_packet['normalization'] != packet['normalization']:
                    issues.append(f'{arm}/{key}: per-clip original normalization differs')
    complete = not issues and len(training) == 2 and len(evaluations) == 3
    comparisons, quality = scientific_report(evaluations) if complete else ({}, {})
    visual = []
    if complete:
        # Fixed frames are not selected by observed metric outcomes.
        for clip in CLIPS:
            for frame in SNAPSHOTS:
                visual.append(dict(clip=clip, frame=frame, role='TRAIN' if clip in TRAIN_CLIPS else 'EVAL-only',
                    status='pending_human_review', selection='fixed a priori, not selected by outcome',
                    screenshots={label: [dict(picture, path=str(root / 'evaluations' / label / picture['path']))
                        for picture in evaluations[label]['screenshots']
                        if picture['clip'] == clip and picture['frame'] == frame] for label in LABELS}))
    return dict(schema='gs_multisource_codec_comparison_v1',
                engineering_status='complete' if complete else 'incomplete_or_invalid',
                scientific_status='descriptive_results_available' if complete else 'inconclusive',
                comparison_eligibility=dict(eligible=complete, reasons=issues), goal_complete=False,
                production_codec_selected=None, action_z_trained=False, state_lfq_enabled=False,
                scope='Continuous state-codec reconstruction only; no learned action Z or motion/rollout success',
                data_roles=dict(train=list(TRAIN_CLIPS), eval_only=[sources.EVAL_CLIP],
                    historical_roles=sources.HISTORICAL_ROLES, pristine_holdout=False,
                    warning='tower14 historical TEST was already exposed; never use it to select a checkpoint'),
                statistical_scope=dict(unit='one physical frame within one source clip',
                    method='paired non-circular moving-block percentile', block_length=16, draws=2000, seed=20261031,
                    algorithm_origin='v6/code/compare_arms.py; 176-future-frame API adapted to 177 full frames',
                    warning='Three clips are not 531 independent generalization samples; no pooled CI or population claim'),
                checkpoint_policy='Shared pinned original checkpoint once; only fixed additional step300 per arm',
                verification_scope='CPU hashes and complete worker receipts, not independent GPU replay or checkpoint deserialization',
                training=training,
                evaluations={label: {key: value for key, value in result.items() if not key.startswith('_')}
                             for label, result in evaluations.items()},
                paired_comparisons=comparisons,
                absolute_quality_reference=dict(scope='Per-clip mean codec reconstruction reference only, not action success',
                    thresholds=dict(codec_teacher_psnr_min=25., full_real_psnr_gap_max=1., workspace_roi_real_psnr_gap_max=1.),
                    results=quality),
                visual_acceptance=dict(status='pending_human_review', fixed_frames=list(SNAPSHOTS),
                    checks=['object geometry and stacking', 'hand/object fidelity', 'workspace continuity',
                            'nearby teacher view consistency', 'rendering artifacts and camera effects'], cases=visual))


def markdown(summary):
    eligibility = summary['comparison_eligibility']
    lines = ['# 多来源状态 codec 对照', '', f"工程状态：{summary['engineering_status']}；科学状态：{summary['scientific_status']}。",
             '', '只评估连续状态 codec 的重建；未训练动作 Z，不启用状态 LFQ，不自动替换生产 codec，goal_complete=false。',
             '', 'TRAIN 为 tower17 和 tower10；tower14 仅评估，但历史 TEST 已暴露，不是未见留出集。仅采用预先固定追加第 300 步。',
             '', '每 clip 以 177 个物理帧配对，移动块长度 16、2,000 次抽样、种子 20261031。区间仅描述当前 clip，不能将三 clip 混作 531 个独立泛化样本。',
             '', '核验范围：CPU 文件哈希及完整 worker 回执；本比较脚本不重新执行 GPU 渲染，也不反序列化模型。']
    if not eligibility['eligible']:
        lines += ['', '证据不完整，拒绝强比较：', ''] + [f'- {issue}' for issue in eligibility['reasons']]
        return '\n'.join(lines) + '\n'
    lines += ['', '| 对照 | clip | full MAE 改善 [95% CI] | ROI MAE 改善 [95% CI] |',
              '|---|---|---:|---:|']
    def interval(value):
        return f"{value['mean_improvement']:.6f} [{value['ci95_low']:.6f}, {value['ci95_high']:.6f}]"
    for comparison, clips in summary['paired_comparisons'].items():
        for clip, groups in clips.items():
            lines.append(f"| {comparison} | {clip} | {interval(groups['codec_real']['mae'])} | {interval(groups['codec_real']['workspace_roi_mae'])} |")
    lines += ['', '改善量为参考 MAE 减候选 MAE，正值表示候选误差降低；teacher 与附近视角指标完整保存在 JSON 中。',
              '', '| 模型 | clip | codec/teacher PSNR | full 相对 teacher 差值 dB | ROI 差值 dB | 绝对参考 |',
              '|---|---|---:|---:|---:|---|']
    for label, clips in summary['absolute_quality_reference']['results'].items():
        for clip, metrics in clips.items():
            lines.append(f"| {label} | {clip} | {metrics['codec_teacher_psnr']:.3f} | {metrics['full_real_psnr_gap_to_teacher']:.3f} | {metrics['workspace_roi_real_psnr_gap_to_teacher']:.3f} | {'通过' if metrics['reference_passed'] else '未通过'} |")
    lines += ['', '绝对参考要求 codec 对 teacher 的 PSNR ≥25 dB，full/ROI 相对 teacher 的真实图像 PSNR 落差均 ≤1 dB。它只描述重建质量；科学质量较差仍可工程 complete。',
              '', '以下固定 5 帧/clip 留待人工视觉验收；不是按结果挑出的失败样本，也不视为已完成验收：', '']
    for case in summary['visual_acceptance']['cases']:
        links = []
        for label, pictures in case['screenshots'].items():
            links.extend(f"[{label}/{picture['kind']}](<{picture['path']}>)" for picture in pictures)
        lines.append(f"- {case['clip']} 帧 {case['frame']:03d}：" + ' · '.join(links))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(root.is_dir(), 'Run root must already exist')
    summary = build_summary(root)
    for name, content in (('SUMMARY.json', json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n'),
                          ('SUMMARY.md', markdown(summary))):
        temporary = root / (name + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(root / name)
    print(summary['engineering_status'])
    return 0 if summary['comparison_eligibility']['eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
