"""Selected-v7 six-panel posterior replay; no training, retry, or AR claim.

The standard-library --self-test reads only the pinned v6 timing helper and
uses memory fixtures. Actual rendering requires all four selected evaluations
and the main queue to have completed. This script never calls backward/step.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = 'gs_native_v7_six_panel_render_v1'
MAIN_MANIFEST = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
TIMING_HELPER = 'ad43827c53817edc18c02a62be6cae527dce0919cb89a702f8caf6b42c7ecd0f'
ARMS = ('z_rgb', 'constant_rgb', 'z_delta', 'constant_delta')
FRAME_COUNT, WIDTH, HEIGHT = 177, 448, 252
HEADER, PANEL_HEADER = 52, 28
VIDEO_SIZE = (3 * WIDTH, HEADER + 2 * (HEIGHT + PANEL_HEADER))
SNAPSHOTS = (0, 44, 88, 132, 176)
MAX_SECONDS = 600
METRICS = ('mae', 'hand_activity_mae', 'motion_mae', 'psnr', 'hand_activity_psnr')
PANELS = ('real_rgb', 'posterior_adapted_held', 'matched_constant',
          'posterior_correct', 'shuffle84', 'blinded')
SEMANTICS = 'Periodic true-initial reset posterior window reconstruction; NOT AR rollout'
ACTION_SEMANTICS = ('Whole nine-code posterior bundles at each start, or marked constant control; '
                    'not adjacent independent AR actions')
BLINDING = ('All10 posterior states use S0 coordinates/features with original nominal times; '
            'same current S0 decoded')


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
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timing_helper(source):
    path = source / 'analysis_code/render_comparison.py'
    if sha(path) != TIMING_HELPER:
        raise ValueError('Previously verified timing/encoding helper changed')
    return module_at('_v6_six_panel_timing', path), path


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + ' mismatch')


def numeric_tree_equal(actual, expected, label):
    """Only summary arithmetic roundoff is tolerated; no metric threshold change."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + ': dictionary keys differ')
        for key, value in expected.items():
            numeric_tree_equal(actual[key], value, label + '.' + str(key))
    elif type(expected) in (int, float):
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(label + ': numeric aggregate differs')
    else:
        require_equal(actual, expected, label)


def terminal_contract(selection, summary, done, protocol_hash):
    """Pure metadata gate, also exercised by the no-files memory fixtures."""
    step = selection.get('selected_step')
    if (selection.get('schema') != 'gs_temporal_four_arm_selection_v1'
            or selection.get('status') != 'selected' or type(step) is not int
            or step not in (200, 300, 400)
            or selection.get('code_manifest_sha256') != MAIN_MANIFEST
            or selection.get('protocol_sha256') != protocol_hash
            or set(selection.get('checkpoints', {})) != set(ARMS)
            or set(selection.get('original_completed_steps', {})) != set(ARMS)):
        raise ValueError('Invalid common-step/source-bound SELECTION')
    for arm in ARMS:
        checkpoint = selection['checkpoints'][arm]
        if (Path(checkpoint.get('path', '')).name != f'step_{step:04d}.pt'
                or type(selection['original_completed_steps'][arm]) is not int
                or selection['original_completed_steps'][arm] < step):
            raise ValueError('Selected checkpoint must be an existing saved step, never a final alias')
    if (summary.get('schema') != 'gs_temporal_four_arm_comparison_v7'
            or summary.get('engineering_status') != 'complete'
            or summary.get('comparison_eligible') is not True
            or summary.get('selected_step') != step
            or summary.get('code_manifest_sha256') != MAIN_MANIFEST
            or summary.get('checkpoint_selection') != selection
            or set(summary.get('arms', {})) != set(ARMS)):
        raise ValueError('Main SUMMARY is incomplete or belongs to another selection')
    if (done.get('status') != 'complete' or done.get('selected_step') != step
            or done.get('summary') != 'SUMMARY.json'):
        raise ValueError('Main queue has no matching successful DONE receipt')
    return step


def evaluation_contract(evaluation, arm, step, selected, main_hash, protocol_hash):
    if (evaluation.get('schema') != 'gs_native_v7_common_checkpoint_evaluation_v1'
            or evaluation.get('status') != 'complete'
            or evaluation.get('scope') != 'single_clip_TRAIN_common_checkpoint_posterior_reconstruction'
            or evaluation.get('arm') != arm or evaluation.get('selected_step') != step
            or evaluation.get('checkpoint_step') != step
            or evaluation.get('checkpoint_path') != selected['path']
            or evaluation.get('checkpoint_sha256') != selected['sha256']
            or evaluation.get('code_manifest_sha256') != main_hash
            or evaluation.get('protocol_sha256') != protocol_hash
            or evaluation.get('optimizer_updates') != 0
            or evaluation.get('windows') != 168 or evaluation.get('real_future_exposures') != 1512
            or evaluation.get('unique_future_frames') != 176
            or evaluation.get('physical_frames_with_some_reconstruction') != FRAME_COUNT):
        raise ValueError('Selected evaluation schema/step/source/coverage mismatch: ' + arm)


def load_terminal(root):
    """Read successful terminal artifacts only; never reads live STATUS or processes."""
    if (root / 'queue/FAILED.json').exists():
        raise ValueError('Main queue contains a failure receipt')
    require_equal(sha(root / 'CODE_MANIFEST.json'), MAIN_MANIFEST, 'Pinned v7 main manifest')
    manifest = read(root / 'CODE_MANIFEST.json')
    require_equal(sha(root / 'runner.py'), manifest['runner.py'], 'runner source')
    runner = module_at('_v7_six_panel_runner', root / 'runner.py')
    evidence = runner.load_verified(root)
    runner.check_witness(evidence)
    selection, summary, done = (read(root / name) for name in
                                ('SELECTION.json', 'SUMMARY.json', 'queue/DONE.json'))
    step = terminal_contract(selection, summary, done, evidence['protocol_sha256'])
    helper, helper_path = timing_helper(evidence['source'])
    comparison = module_at('_v6_six_panel_comparison', evidence['source'] / 'code/compare_arms.py')
    probe = module_at('_v6_six_panel_parity', evidence['source'] / 'analysis_code/future_information_probe.py')
    hashes = {}

    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None:
            require_equal(actual, expected, 'Artifact hash: ' + str(path))
        hashes[str(path)] = actual
        return actual

    for name in ('CODE_MANIFEST.json', 'protocol.json', 'PREFLIGHT.json',
                 'SELECTION.json', 'SUMMARY.json', 'queue/DONE.json'):
        bind(root / name)
    bind(helper_path, TIMING_HELPER)
    bind(Path(__file__).resolve())
    bind(Path(__file__).with_suffix('.md').resolve())
    arms, references = {}, None
    for arm in ARMS:
        for mode in ('train', 'eval'):
            receipt_path = root / 'queue' / f'{arm}_{mode}_PROCESS.json'
            receipt = read(receipt_path)
            if (receipt.get('arm') != arm or receipt.get('mode') != mode
                    or receipt.get('exit_code') != 0 or receipt.get('timed_out') is not False):
                raise ValueError('Main process did not finish successfully: ' + arm + '/' + mode)
            bind(receipt_path)
        directory = root / arm
        eval_directory = root / 'evaluations' / f'step_{step:04d}' / arm
        if (directory / 'FAILED.json').exists() or (eval_directory / 'FAILED.json').exists():
            raise ValueError('Selected arm/evaluation contains a failure receipt: ' + arm)
        selected = selection['checkpoints'][arm]
        path = Path(selected['path']).resolve()
        require_equal(path, directory / f'step_{step:04d}.pt', 'Exact selected checkpoint path')
        bind(path, selected['sha256'])
        train, source, plan = (read(directory / name) for name in ('TRAIN_DONE.json', 'SOURCE.json', 'PLAN.json'))
        evaluation = read(eval_directory / 'EVALUATION.json')
        evaluation_contract(evaluation, arm, step, selected, MAIN_MANIFEST, evidence['protocol_sha256'])
        source_arm = evidence['protocol']['arms'][arm]['source_arm']
        original = evidence['source_records'][source_arm]
        if (train.get('schema') != 'gs_native_v7_train_done_v1' or train.get('status') != 'complete'
                or train.get('arm') != arm or train.get('steps') != train.get('completed_steps')
                or train.get('completed_steps') != selection['original_completed_steps'][arm]
                or train.get('completed_steps') != evaluation.get('original_completed_steps')
                or train.get('checkpoint_sha256', {}).get(str(step)) != selected['sha256']
                or train.get('encoder_unchanged') is not True
                or train.get('full_window_coverage') is not True
                or train.get('target_frames_seen') != list(range(FRAME_COUNT))):
            raise ValueError('Invalid original training completion receipt: ' + arm)
        if (source.get('schema') != 'gs_native_v7_continuation_source_v1' or source.get('arm') != arm
                or source.get('source_arm') != source_arm or source.get('source_steps') != 100
                or source.get('source_checkpoint_path') != original['path']
                or source.get('source_checkpoint_sha256') != original['sha256']
                or source.get('source_plan_sha256') != original['plan_sha256']
                or source.get('source_train_done_sha256') != original['train_done_sha256']
                or source.get('original_initial_model_hash') != original['plan']['initial_model_hash']
                or source.get('source_manifest_sha256') != evidence['protocol']['source_manifest_sha256']
                or source.get('source_protocol_sha256') != evidence['protocol']['source_protocol_sha256']
                or source.get('delta_weight') != evidence['protocol']['arms'][arm]['delta_weight']
                or source.get('source_binding') != evaluation.get('source_binding')
                or source.get('frozen_reference_sha256') != evaluation.get('frozen_reference_sha256')
                or evaluation.get('source_checkpoint_sha256') != original['sha256']):
            raise ValueError('Continuation source provenance mismatch: ' + arm)
        for document in (train, source, plan):
            require_equal(document.get('code_manifest_sha256'), MAIN_MANIFEST, arm + ' source manifest')
            require_equal(document.get('protocol_sha256'), evidence['protocol_sha256'], arm + ' protocol')
        require_equal(plan.get('starts'), evidence['plan'], arm + ' complete sampling plan')
        require_equal(plan.get('schema'), 'gs_native_v7_plan_v1', arm + ' plan schema')
        require_equal(plan.get('full_plan_sha256'), runner.plan_hash(evidence['plan']), arm + ' full plan hash')
        require_equal(plan.get('first100_plan_sha256'), runner.plan_hash(evidence['plan'][:100]), arm + ' first100 plan')
        require_equal(plan.get('original_initial_model_hash'), source['original_initial_model_hash'], 'Initial model')
        require_equal(runner.plan_hash(plan['starts'][:step]), selection['plan_prefix_sha256'], 'Selected plan prefix')
        fixed = source.get('fixed_null_key_bias', {})
        if (fixed.get('policy') != runner.NULL_KEY_BIAS_POLICY or fixed.get('unchanged') is not True
                or fixed.get('count') != 27 or fixed.get('scalar_count') != 864
                or fixed.get('in_optimizer_count') != (0 if source_arm == 'constant' else 27)):
            raise ValueError('Invalid null-key-bias policy evidence')
        for document in (train, evaluation):
            require_equal(document.get('fixed_null_key_bias'), fixed, arm + ' fixed key bias')
        for document in (source, train, evaluation):
            require_equal(document.get('null_key_bias_source_sha256'), evidence['benchmark']['null_key_bias_sha256'],
                          arm + ' fixed key bias helper')
        bindings = {'source_record_sha256': directory / 'SOURCE.json',
                    'plan_sha256': directory / 'PLAN.json', 'train_done_sha256': directory / 'TRAIN_DONE.json',
                    'targets_sha256': eval_directory / 'targets.jsonl',
                    'initial_reconstruction_sha256': eval_directory / 'initial_reconstruction.json',
                    'actions_sha256': eval_directory / 'actions.pt'}
        for key, file in bindings.items():
            bind(file, evaluation[key])
        require_equal(train.get('source_sha256'), hashes[str(directory / 'SOURCE.json')], 'Train source hash')
        require_equal(train.get('plan_sha256'), hashes[str(directory / 'PLAN.json')], 'Train plan hash')
        public = summary['arms'][arm]
        require_equal(bind(eval_directory / 'EVALUATION.json'), public.get('evaluation_sha256'), 'SUMMARY EVAL hash')
        require_equal(public.get('targets_sha256'), evaluation['targets_sha256'], 'SUMMARY row hash')
        require_equal(public.get('checkpoint'), selected, 'SUMMARY checkpoint')
        issues = []
        rows = comparison.read_rows(eval_directory / 'targets.jsonl', issues)
        if issues:
            raise ValueError('Invalid evaluated target rows: ' + str(issues[:3]))
        runner.validate_targets(rows)
        for control in runner.ALL_CONTROLS:
            reduced = comparison.reduce_rows(rows, control)
            numeric_tree_equal(public['metrics'][control], reduced, arm + '/' + control + '/SUMMARY')
            numeric_tree_equal(evaluation['metrics'][control],
                               {key: reduced[key] for key in ('exposure_mean', 'frame_uniform', 'per_horizon')},
                               arm + '/' + control + '/EVAL')
        initial_document = read(eval_directory / 'initial_reconstruction.json')
        initial = initial_document.get('rows', [])
        if (len(initial) != 168 or {row['start'] for row in initial} != set(range(168))
                or initial_document.get('unique_initial_frames') != 168
                or any(row['end'] != row['start'] or row['horizon'] != 0 for row in initial)):
            raise ValueError('Incomplete selected initial reconstruction rows')
        for control in runner.LIVE_CONTROLS:
            for row in initial:
                probe.validate_metrics(row[control])
            numeric_tree_equal(initial_document['metrics'][control],
                               {key: statistics.fmean(row[control][key] for row in initial)
                                for key in ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr')},
                               arm + '/' + control + '/h0')
        current_references = {(row['start'], row['horizon']):
                              {key: row[key] for key in ('held_initial', 'frozen_target')} for row in rows}
        if references is not None:
            require_equal(current_references, references, 'Common frozen baseline rows')
        references = current_references
        arms[arm] = dict(path=path, directory=eval_directory, source=source, train=train, plan=plan,
                         evaluation=evaluation, rows={(row['start'], row['horizon']): row for row in rows},
                         initial={row['start']: row for row in initial})
    source_bindings = [item['source']['source_binding'] for item in arms.values()]
    if any(binding != source_bindings[0] for binding in source_bindings):
        raise ValueError('Arms do not share the same native input source')
    eligible = [candidate for candidate in evidence['protocol']['train']['save_steps']
                if all(arms[arm]['train']['completed_steps'] >= candidate
                       and (root / arm / f'step_{candidate:04d}.pt').is_file() for arm in ARMS)]
    if not eligible or step != max(eligible):
        raise ValueError('SELECTION is not the largest common predeclared saved step')
    return dict(runner=runner, evidence=evidence, selection=selection, summary=summary, step=step,
                timing=helper, probe=probe, arms=arms, hashes=hashes)


def validate_actions(torch, actions, arm, context):
    evidence, item, step = context['evidence'], context['arms'][arm], context['step']
    ev = item['evaluation']
    constant = arm.startswith('constant_')
    required = dict(schema='gs_native_v7_common_checkpoint_actions_v1', arm=arm, selected_step=step,
                    checkpoint_path=str(item['path']), checkpoint_sha256=ev['checkpoint_sha256'],
                    source_checkpoint_sha256=ev['source_checkpoint_sha256'],
                    code_manifest_sha256=MAIN_MANIFEST, protocol_sha256=evidence['protocol_sha256'],
                    starts=list(range(168)), horizons=list(range(1, 10)), source_frames=177, frame_gap=1,
                    all_windows_complete=True, semantics=ACTION_SEMANTICS, blinding=BLINDING,
                    numerical_stability_policy=context['runner'].NULL_KEY_BIAS_POLICY,
                    null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
                    future_information_probe_sha256=evidence['protocol']['future_information_probe_sha256'],
                    cache_manifest_sha256=ev['source_binding']['cache_manifest_sha256'],
                    code_mode='constant' if constant else 'posterior')
    for key, value in required.items():
        require_equal(actions.get(key), value, arm + ' actions.' + key)
    for name in ('ids', 'blinded_ids'):
        tensor = actions.get(name)
        if (not torch.is_tensor(tensor) or tensor.shape != (168, 9)
                or tensor.dtype not in (torch.int32, torch.int64)
                or not bool(((tensor >= 0) & (tensor < 65536)).all())):
            raise ValueError('Invalid exported complete nine-ID bundles')
    if constant:
        if (type(actions.get('constant_id')) is not int
                or not bool((actions['ids'] == actions['constant_id']).all())
                or not torch.equal(actions['ids'], actions['blinded_ids'])):
            raise ValueError('Matched constant actions are not constant')
    elif actions.get('constant_id') is not None:
        raise ValueError('Posterior actions mislabeled as constant')
    ids, blinded = actions['ids'].tolist(), actions['blinded_ids'].tolist()
    for (start, horizon), row in item['rows'].items():
        if (row.get('motion_ids') != ids[start] or row.get('blinded_motion_ids') != blinded[start]
                or row.get('blinded_horizon_id_changed') != (ids[start][horizon - 1] != blinded[start][horizon - 1])):
            raise ValueError('Action bundle differs from selected evaluated row')
    require_equal(context['probe'].id_statistics(ids, blinded), ev['blinded_id_statistics'], 'Blinded ID statistics')
    return actions


def restore_selected(context, arm, core, torch, cfg, data):
    item, evidence = context['arms'][arm], context['evidence']
    saved = torch.load(item['path'], map_location='cpu', weights_only=False)
    metadata = saved.get('continuation', {})
    source_arm = evidence['protocol']['arms'][arm]['source_arm']
    expected = dict(schema='gs_native_v7_checkpoint_v1', arm=arm, source_arm=source_arm,
                    source_steps=100, intended_total_steps=400, source_recipe_config_preserved=True,
                    source_checkpoint_sha256=evidence['source_records'][source_arm]['sha256'],
                    code_manifest_sha256=MAIN_MANIFEST, protocol_sha256=evidence['protocol_sha256'],
                    fixed_null_key_bias=item['source']['fixed_null_key_bias'],
                    null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
                    delta_weight=evidence['protocol']['arms'][arm]['delta_weight'])
    for key, value in expected.items():
        require_equal(metadata.get(key), value, arm + ' checkpoint.' + key)
    if (saved.get('completed_steps') != context['step'] or saved.get('config') != cfg
            or saved.get('plan') != evidence['plan']):
        raise ValueError('Selected checkpoint internal step/config/plan mismatch')
    model, bridge, optimizer = core.build(cfg, data, source_arm)
    del optimizer
    model.load_state_dict(saved['model'], strict=True)
    bridge.codec.load_state_dict(saved['codec'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    bridge.eval().requires_grad_(False)
    context['runner'].ensure_eval_codec(bridge)
    require_equal(bridge.binding, item['evaluation']['source_binding'], 'Runtime native source')
    require_equal(core.tensor_hash(bridge.codec.ae.encoder), item['source']['frozen_encoder_sha256'], 'Frozen encoder')
    require_equal(data.reference_hash, item['evaluation']['frozen_reference_sha256'], 'Runtime frozen baseline')
    if (bridge.codec.protocol['max_decoder_candidates'] != cfg['resources']['max_decoder_candidates']
            or bridge.codec.protocol['max_decoded_gaussians'] != cfg['resources']['max_decoded_gaussians']
            or len(bridge.codec._guards) != len(bridge.codec.ae.decoder.ups) + 1):
        raise ValueError('Original native candidate guards changed')
    return model, bridge


def tensor_digest(*named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        if tensor is None:
            digest.update(b'None')
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class GaussianRecorder:
    """Wrap only this instance's renderer callable; preserve every GS and argument."""
    def __init__(self, bridge):
        self.original = bridge.render_fn
        self.last = None
        bridge.render_fn = self

    def __call__(self, gs, *args, **kwargs):
        # NativeBridge passes SH as the final positional argument. Nothing is
        # detached on the rendering path; only the separate audit hash is CPU.
        sh = args[4] if len(args) == 5 else kwargs.get('sh')
        if len(args) != 5 and 'sh' not in kwargs:
            raise RuntimeError('Native render signature changed')
        self.last = dict(count=len(gs), gs_and_sh_sha256=tensor_digest(('gs', gs), ('sh', sh)))
        return self.original(gs, *args, **kwargs)


def panel_layout():
    return [(i % 3 * WIDTH, HEADER + i // 3 * (HEIGHT + PANEL_HEADER) + PANEL_HEADER,
             WIDTH, HEIGHT) for i in range(6)]


def compose_panel(images, row, loss_mode, step, fps, Image, ImageDraw, font):
    if len(images) != 6:
        raise ValueError('Exactly six native-size panels are required')
    canvas = Image.new('RGB', VIDEO_SIZE, (18, 20, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), 'POSTERIOR RECONSTRUCTION | periodic TRUE-initial reset | NOT AR ROLLOUT',
              font=font, fill=(255, 220, 130))
    draw.text((8, 25), f'{loss_mode} | selected step {step} | frame {row["source_frame"]:03d}/176 | '
              f'initial {row["start"]:03d} | h={row["horizon"]} | PTS {row["source_pts"]:.6f}s | '
              f'{fps.numerator}/{fps.denominator} fps' + (' | RESET' if row['reset'] else ''),
              font=font, fill=(235, 235, 235))
    labels = ('REAL RGB', 'HOLD INITIAL | posterior adapted codec', 'MATCHED CONSTANT | own adapted codec',
              'POSTERIOR | correct ID bundle',
              f'SHUFFLE84 | bundle {row["shuffle84_bundle_start"]:03d} | same h',
              'BLINDED | future-state access removed')
    for image, label, (x, y, width, height) in zip(images, labels, panel_layout()):
        if image.size != (width, height):
            raise ValueError('Panel resized or native image dimensions changed')
        draw.text((x + 8, y - PANEL_HEADER + 6), label, font=font, fill=(230, 235, 242))
        canvas.paste(image, (x, y))
    return canvas


def run(root, loss_mode, output, max_seconds):
    begun = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    stage, completed_frames, process, previous_alarm = 'terminal_metadata', 0, None, None
    streams = []

    def remaining():
        seconds = max_seconds - (time.monotonic() - begun)
        if seconds <= 0:
            raise TimeoutError('Six-panel replay exceeded its fixed time budget')
        return seconds

    def deadline(signum, frame):
        raise TimeoutError('Six-panel replay deadline reached')

    try:
        if hasattr(signal, 'setitimer'):
            previous_alarm = signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, max_seconds)
        context = load_terminal(root)
        runner, evidence, timing = context['runner'], context['evidence'], context['timing']
        posterior_arm, constant_arm = 'z_' + loss_mode, 'constant_' + loss_mode
        selected_arms = (posterior_arm, constant_arm)
        ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        if ffmpeg is None or ffprobe is None:
            raise FileNotFoundError('System ffmpeg and ffprobe are required; no installation attempted')
        version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True,
                                 timeout=min(5., remaining()), check=True).stdout.splitlines()[0]
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default()
        import torch
        actions = {}
        for arm in ARMS:
            actions[arm] = validate_actions(torch, torch.load(context['arms'][arm]['directory'] / 'actions.pt',
                                            map_location='cpu', weights_only=True), arm, context)
        stage = 'runtime_and_selected_checkpoints'
        torch, core, _, _, cfg, data = runner.setup_runtime(evidence)
        if (cfg['data']['frames'] != FRAME_COUNT or cfg['data']['resolution_hw'] != [HEIGHT, WIDTH]
                or cfg['data']['frame_gap'] != 1 or cfg['resources']['peak_limit_gib'] != 48
                or len(data.states) != FRAME_COUNT or len(data.records) != FRAME_COUNT):
            raise ValueError('Native frame/resolution/resource contract changed')
        torch.cuda.reset_peak_memory_stats()
        timing_protocol = core.MOTION_ROOT / 'protocol.json'
        binding_path = core.V4_ROOT / 'source_binding.json'
        binding = read(binding_path)
        dependency = next((row for row in binding['dependencies'] if row['path'] == str(timing_protocol)), None)
        if dependency is None or sha(timing_protocol) != dependency['sha256']:
            raise ValueError('Source timing protocol is not hash-bound to the original native data')
        context['hashes'][str(binding_path)] = sha(binding_path)
        context['hashes'][str(timing_protocol)] = sha(timing_protocol)
        fps = timing.rational_fps(read(timing_protocol)['nominal_delta_seconds'])
        times = timing.validate_times([float(state['pts']) for state in data.states], fps)
        for frame, state in enumerate(data.states):
            if state.get('frame_index') != frame:
                raise ValueError('Physical frame identity changed')
            if 'source_pts' in state and not math.isclose(float(state['source_pts']), float(state['pts']),
                                                         rel_tol=0, abs_tol=1e-7):
                raise ValueError('Source PTS and cached PTS differ')
        runtimes = {arm: restore_selected(context, arm, core, torch, cfg, data) for arm in selected_arms}
        for arm, (model, bridge) in runtimes.items():
            require_equal(model.code_mode, actions[arm]['code_mode'], 'Runtime code mode')
            if model.code_mode == 'constant':
                require_equal(model.constant_id, actions[arm]['constant_id'], 'Runtime constant ID')
        before = {arm: dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
                  for arm, (model, bridge) in runtimes.items()}
        recorders = {arm: GaussianRecorder(bridge) for arm, (_, bridge) in runtimes.items()}
        data_source_hash = hashlib.sha256(json.dumps(data.sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        stem = f'{loss_mode}_step{context["step"]:04d}_posterior_reset_177f_fps{fps.numerator}_{fps.denominator}_six_panel'
        video_path = output / (stem + '.mp4')
        command = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'warning',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
                   f'{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}', '-framerate', times['ffmpeg_framerate'],
                   '-i', 'pipe:0', '-an', '-frames:v', str(FRAME_COUNT), '-c:v', 'libx264',
                   '-preset', 'fast', '-threads', '6', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-progress', 'pipe:1', str(video_path)]
        stderr = (output / 'ffmpeg.log').open('x', encoding='utf-8')
        progress = (output / 'encoding_progress.txt').open('x', encoding='utf-8')
        streams.extend([stderr, progress])
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=progress, stderr=stderr)
        mapping, rows, snapshots = timing.frame_mapping(), [], {}
        current_start, cached, decode_calls, render_calls = None, {}, 0, 0
        stage = 'render177_six_panels'
        model, posterior_bridge = runtimes[posterior_arm]
        constant_model, constant_bridge = runtimes[constant_arm]
        with torch.no_grad(), (output / 'frame_mapping_metrics.jsonl').open('x', encoding='utf-8') as log:
            for index, base in enumerate(mapping):
                remaining()
                start, h = base['start'], base['horizon']
                row = dict(base, source_pts=float(data.states[index]['pts']),
                           presentation_pts=times['presentation_pts'][index],
                           initial_source_pts=float(data.states[start]['pts']), clip=core.CLIP,
                           loss_mode=loss_mode, selected_step=context['step'], conditioning_uses_true_initial=True,
                           not_ar_rollout=True, posterior_ids=actions[posterior_arm]['ids'][start].tolist(),
                           constant_ids=actions[constant_arm]['ids'][start].tolist(),
                           shuffle84_ids=actions[posterior_arm]['ids'][base['shuffle84_bundle_start']].tolist(),
                           blinded_ids=actions[posterior_arm]['blinded_ids'][start].tolist())
                if start != current_start:
                    condition = [core.make_condition(data.states[start])]
                    for control, source_model, tensor, bundle in (
                        ('posterior_correct', model, actions[posterior_arm]['ids'], start),
                        ('shuffle84', model, actions[posterior_arm]['ids'], base['shuffle84_bundle_start']),
                        ('blinded', model, actions[posterior_arm]['blinded_ids'], start),
                        ('matched_constant', constant_model, actions[constant_arm]['ids'], start)):
                        cached[control] = source_model.decode_ids(tensor[bundle:bundle + 1].to('cuda'), condition)[0]
                        decode_calls += 1
                    for sequence in cached.values():
                        if len(sequence) != 10:
                            raise RuntimeError('Selected ID replay omitted a time slot')
                        for prediction in sequence:
                            if (not torch.equal(prediction['coordinates'], data.states[start]['coordinates'])
                                    or prediction['features'].shape != data.states[start]['features'].shape
                                    or not bool(torch.isfinite(prediction['features']).all())):
                                raise RuntimeError('Selected ID replay dropped/changed initial support or features')
                    current_start = start
                real = data.records[index]['real']
                if (real.shape != (HEIGHT, WIDTH, 3) or not bool(torch.isfinite(real).all())
                        or bool((real < 0).any()) or bool((real > 1).any())):
                    raise ValueError('Invalid native real RGB; no resize fallback')
                tensors, measured, gaussian_audit, parity = [real], {}, {}, {}
                for control, arm, metric_control, prediction in (
                    ('posterior_adapted_held', posterior_arm, 'adapted_held', data.states[start]),
                    ('matched_constant', constant_arm, 'correct', cached['matched_constant'][h]),
                    ('posterior_correct', posterior_arm, 'correct', cached['posterior_correct'][h]),
                    ('shuffle84', posterior_arm, 'shuffle84', cached['shuffle84'][h]),
                    ('blinded', posterior_arm, 'blinded', cached['blinded'][h])):
                    remaining()
                    bridge = runtimes[arm][1]
                    image = bridge.render(prediction, data.records[index])
                    render_calls += 1
                    measured[control] = data.metrics(image, start, h)
                    item = context['arms'][arm]
                    reference = item['initial'][start] if h == 0 else item['rows'][(start, h)]
                    parity[control] = context['probe'].check_rerender(measured[control], reference[metric_control])
                    audit = recorders[arm].last
                    if audit is None or audit['count'] != bridge.last_count or audit['count'] <= 0:
                        raise RuntimeError('Missing actual native GS audit')
                    gaussian_audit[control] = dict(audit, arm=arm,
                        checkpoint_sha256=item['evaluation']['checkpoint_sha256'])
                    tensors.append(image)
                    if torch.cuda.max_memory_allocated() / 2**30 > 48:
                        raise RuntimeError('Original 48GiB peak guard exceeded')
                row.update(metrics=measured, native_gaussians=gaussian_audit,
                           initial_latent_count=len(data.states[start]['coordinates']),
                           initial_support_sha256=tensor_digest(('coordinates', data.states[start]['coordinates'])),
                           replay_absolute_difference_from_selected_evaluation=parity)
                panels = [Image.fromarray(value.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
                          for value in tensors]
                canvas = compose_panel(panels, row, loss_mode, context['step'], fps, Image, ImageDraw, font)
                if index in SNAPSHOTS:
                    name = f'{stem}_frame_{index:03d}.png'
                    with (output / name).open('xb') as stream:
                        canvas.save(stream, format='PNG')
                    snapshots[index] = name
                remaining()
                process.stdin.write(canvas.tobytes())
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                rows.append(row)
                completed_frames += 1
                if index % 44 == 0:
                    print('SIX_PANEL', loss_mode, completed_frames, '/177', flush=True)
                del tensors, panels, canvas, image
        stage = 'finalize_and_verify'
        process.stdin.close()
        exit_code = process.wait(timeout=remaining())
        process = None
        for stream in streams:
            stream.close()
        streams.clear()
        if exit_code != 0:
            raise RuntimeError(f'ffmpeg exited {exit_code}; partial outputs retained')
        encoded = timing.encoding_progress((output / 'encoding_progress.txt').read_text())
        probe_result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                         '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration', '-of', 'json',
                         str(video_path)], capture_output=True, text=True, timeout=min(15., remaining()), check=True)
        video_probe = json.loads(probe_result.stdout)['streams'][0]
        if (video_probe['width'] != VIDEO_SIZE[0] or video_probe['height'] != VIDEO_SIZE[1]
                or int(video_probe['nb_frames']) != FRAME_COUNT or Fraction(video_probe['avg_frame_rate']) != fps
                or completed_frames != FRAME_COUNT or [row['source_frame'] for row in rows] != list(range(FRAME_COUNT))
                or decode_calls != 4 * len({row['start'] for row in mapping}) or render_calls != 5 * FRAME_COUNT):
            raise RuntimeError('Video geometry/frame/rate or complete cached-window replay mismatch')
        for arm, (loaded_model, bridge) in runtimes.items():
            runner.ensure_eval_codec(bridge)
            require_equal(dict(model=core.tensor_hash(loaded_model), codec=core.tensor_hash(bridge.codec)),
                          before[arm], 'No model/native weights or buffers changed during replay')
        require_equal(core.verify_native_sources(), context['arms'][posterior_arm]['evaluation']['source_binding'],
                      'Native dependencies/checkpoints/cache manifest remained unchanged')
        for path, expected in context['hashes'].items():
            remaining()
            require_equal(sha(path), expected, 'Input artifact remained unchanged: ' + path)
        require_equal(runner.load_verified(root)['code_manifest_sha256'], MAIN_MANIFEST, 'Final source verification')
        remaining()
        artifacts = {name: sha(output / name) for name in
                     [video_path.name, *snapshots.values(), 'frame_mapping_metrics.jsonl',
                      'ffmpeg.log', 'encoding_progress.txt']}
        def means(observations):
            return {control: {key: statistics.fmean(row['metrics'][control][key] for row in observations)
                              for key in METRICS} for control in PANELS[1:]}
        summary = dict(schema=SCHEMA, status='complete', loss_mode=loss_mode, selected_step=context['step'],
            parity_pass=True, parity_tolerances=dict(mae_absolute=1e-4, psnr_absolute_db=.02),
            maximum_parity_absolute_difference={control: {key: max(
                row['replay_absolute_difference_from_selected_evaluation'][control][key] for row in rows)
                for key in METRICS} for control in PANELS[1:]},
            optimizer_updates=0, goal_complete=False, not_ar_rollout=True, semantics=SEMANTICS,
            physical_frames=FRAME_COUNT, unique_physical_frames=FRAME_COUNT,
            panel_order=list(PANELS), panel_native_size_wh=[WIDTH, HEIGHT], video_size_wh=list(VIDEO_SIZE),
            checkpoints={arm: context['selection']['checkpoints'][arm] for arm in selected_arms},
            original_training_completed_steps={arm: context['arms'][arm]['train']['completed_steps'] for arm in selected_arms},
            selected_evaluation_hashes={arm: context['hashes'][str(context['arms'][arm]['directory'] / 'EVALUATION.json')]
                                        for arm in selected_arms},
            source_initials_shared=True, all_initial_latent_points_retained=True, native_internal_hard_pruning_retained=True,
            native_binding=context['arms'][posterior_arm]['evaluation']['source_binding'],
            native_gs_counts={control: dict(min=min(row['native_gaussians'][control]['count'] for row in rows),
                max=max(row['native_gaussians'][control]['count'] for row in rows)) for control in PANELS[1:]},
            native_gs_hash_scope='Per rendered frame: actual complete renderer gs and SH tensors, dtype/shape/bytes; no subsets',
            time_contract=times, resets=[row['source_frame'] for row in mapping if row['reset']],
            window_starts_used=sorted({row['start'] for row in mapping}), latent_window_decode_calls=decode_calls,
            native_render_calls=render_calls, metrics=dict(all177_physical_frames=means(rows),
                future176_physical_frames=means(rows[1:]), frame0=rows[0]['metrics']),
            metrics_scope='One declared start/h per physical frame; not the all168x9 evaluation aggregate',
            main_manifest_sha256=MAIN_MANIFEST, selection_sha256=context['hashes'][str(root / 'SELECTION.json')],
            main_summary_sha256=context['hashes'][str(root / 'SUMMARY.json')], input_sha256=context['hashes'],
            source_cache_entries_sha256=data_source_hash, timing_helper_sha256=TIMING_HELPER,
            ffmpeg_version=version, ffmpeg_command=command, encoder_final_progress=encoded, ffprobe=video_probe,
            artifact_sha256=artifacts, snapshots=snapshots, video=video_path.name,
            elapsed_seconds=time.monotonic() - begun, max_seconds=max_seconds,
            gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            limitations=['Single TRAIN clip; posterior reconstruction, no AR or generalization claim',
                         'Blinded replay is distribution shift, not independently proof of action semantics',
                         'Matched constant uses its own selected codec; held uses the posterior selected codec',
                         'Source timing is hash-bound cached nominal PTS, not independently checked MP4 container PTS'])
        description = (f'# v7 {loss_mode} 六格后验重建\n\n'
            f'统一 selected checkpoint：{context["step"]} 步。**定期重新给真实初态，不是 AR rollout。**\n\n'
            '顺序：真实 RGB、后验臂适配 codec 保持初态、匹配 constant 臂正确码、后验正确码、shuffle84、blinded。'
            '两模型均来自 SELECTION 指定并完成全量评测的同一步 checkpoint；没有使用各自 final。\n\n'
            f'177 个物理帧各出现一次，每格448×252，不缩放；视频1344×612，额外空间用于标题。'
            f'源帧率 {fps.numerator}/{fps.denominator}，首末 PTS 跨度 {times["source_endpoint_span_seconds"]:.9f}s，'
            f'含最后一帧的编码时长 {times["encoded_duration_seconds"]:.9f}s。'
            '播放 PTS=source PTS−source PTS0；尾部 frame172…176 使用 start167/h5…9。\n\n'
            '全部五种渲染控制逐帧与 selected EVAL 对应 start/h 指标复核：MAE绝差≤1e-4，PSNR绝差≤0.02dB。'
            '数值指标取编码前完整浮点RGB；当前单一时间映射均值不冒充168×9全量评测均值。'
            '`frame_mapping_metrics.jsonl` 含 source/start/h/PTS/reset、IDs、完整GS数量与hash。\n\n'
            f'视频 `{video_path.name}`，截图 frame0/44/88/132/176。所有输入与产物hash见 SUMMARY.json。'
            f'耗时{summary["elapsed_seconds"]:.2f}s，峰值{summary["gpu_peak_gib"]:.3f}GiB，参数更新0次。\n')
        with (output / 'SUMMARY.md').open('x', encoding='utf-8') as stream:
            stream.write(description)
        summary['artifact_sha256']['SUMMARY.md'] = sha(output / 'SUMMARY.md')
        remaining()
        summary['elapsed_seconds'] = time.monotonic() - begun
        write_new(output / 'SUMMARY.json', summary)
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps(dict(schema=SCHEMA, status='complete', loss_mode=loss_mode,
                              selected_step=context['step'], parity_pass=True, video=str(video_path))), flush=True)
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        for stream in streams:
            stream.close()
        write_new(output / 'FAILED.json', dict(schema=SCHEMA, status='failed', loss_mode=loss_mode,
            stage=stage, error=repr(error), traceback=failure_traceback, physical_frames_written=completed_frames,
            elapsed_seconds=time.monotonic() - begun, max_seconds=max_seconds, optimizer_updates=0,
            partial_outputs_retained=True, automatic_retry=False, goal_complete=False))
        raise
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


def cpu_self_test():
    # Read only the pinned helper source; no experiment result/data imports.
    source = Path(__file__).resolve().parents[2] / 'gaussiangpt_neoverse_native_z_20261031_v6'
    helper, _ = timing_helper(source)
    mapping = helper.frame_mapping()
    assert [row['source_frame'] for row in mapping] == list(range(177))
    assert [(row['start'], row['horizon']) for row in mapping[-5:]] == [(167, h) for h in range(5, 10)]
    assert len({row['start'] for row in mapping}) == 20
    assert VIDEO_SIZE == (1344, 612) and len(set(panel_layout())) == 6
    for x, y, width, height in panel_layout():
        assert (width, height) == (448, 252) and x + width <= VIDEO_SIZE[0] and y + height <= VIDEO_SIZE[1]
    selection = dict(schema='gs_temporal_four_arm_selection_v1', status='selected', selected_step=200,
        code_manifest_sha256=MAIN_MANIFEST, protocol_sha256='p',
        original_completed_steps={arm: 250 for arm in ARMS},
        checkpoints={arm: dict(path='/fixture/' + arm + '/step_0200.pt', sha256='c') for arm in ARMS})
    summary = dict(schema='gs_temporal_four_arm_comparison_v7', engineering_status='complete',
        comparison_eligible=True, selected_step=200, code_manifest_sha256=MAIN_MANIFEST,
        checkpoint_selection=selection, arms={arm: {} for arm in ARMS})
    done = dict(status='complete', selected_step=200, summary='SUMMARY.json')
    assert terminal_contract(selection, summary, done, 'p') == 200
    cases = []
    for mutation in ('incomplete', 'step', 'manifest', 'fake_final', 'missing_arm'):
        s, m, d = copy.deepcopy((selection, summary, done))
        if mutation == 'incomplete': d['status'] = 'running'
        elif mutation == 'step': m['selected_step'] = 300
        elif mutation == 'manifest': s['code_manifest_sha256'] = 'wrong'
        elif mutation == 'fake_final': s['checkpoints']['z_rgb']['path'] = '/fixture/z_rgb/final.pt'
        elif mutation == 'missing_arm': del s['checkpoints']['constant_rgb']
        cases.append((s, m, d))
    for s, m, d in cases:
        try:
            terminal_contract(s, m, d, 'p')
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete/mismatched source metadata was accepted')
    ev = dict(schema='gs_native_v7_common_checkpoint_evaluation_v1', status='complete',
        scope='single_clip_TRAIN_common_checkpoint_posterior_reconstruction', arm='z_rgb', selected_step=200,
        checkpoint_step=200, checkpoint_path=selection['checkpoints']['z_rgb']['path'], checkpoint_sha256='c',
        code_manifest_sha256=MAIN_MANIFEST, protocol_sha256='p', optimizer_updates=0,
        windows=168, real_future_exposures=1512, unique_future_frames=176, physical_frames_with_some_reconstruction=177)
    evaluation_contract(ev, 'z_rgb', 200, selection['checkpoints']['z_rgb'], MAIN_MANIFEST, 'p')
    for field, value in [('checkpoint_sha256', 'wrong'), ('real_future_exposures', 1511), ('checkpoint_step', 300)]:
        invalid = dict(ev, **{field: value})
        try:
            evaluation_contract(invalid, 'z_rgb', 200, selection['checkpoints']['z_rgb'], MAIN_MANIFEST, 'p')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid EVAL metadata was accepted')
    assert 'torch' not in sys.modules and 'PIL' not in sys.modules
    print('CPU_SELF_TEST_PASSED: pinned mapping177/tail; six native448x252 panels; '
          'incomplete/source/step/final-alias/coverage rejection; no torch/GPU/results/writes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--loss-mode', choices=('rgb', 'delta'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
        return
    if args.root is None or args.loss_mode is None or args.output is None:
        parser.error('--root, --loss-mode and --output are required')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('--max-seconds must be positive and at most600')
    root, output = args.root.resolve(), args.output.resolve()
    if output != root / 'diagnostics/render_comparison_v7' / args.loss_mode:
        parser.error('Output must equal ROOT/diagnostics/render_comparison_v7/{loss-mode}')
    if output.exists():
        parser.error('Output exists; no overwrite, resume or retry')
    run(root, args.loss_mode, output, args.max_seconds)


if __name__ == '__main__':
    main()
