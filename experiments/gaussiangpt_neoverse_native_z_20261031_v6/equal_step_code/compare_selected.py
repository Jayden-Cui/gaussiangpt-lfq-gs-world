"""CPU-only comparison of selected common checkpoints; never retrains.

The original compare_arms implementation supplies row loading, aggregation,
paired physical-frame bootstrap, eligibility and every scientific screen rule.
A local in-memory load_arm adapter changes the evaluation-step descriptor only;
original TRAIN_DONE documents and their final completed steps remain untouched.
No torch/checkpoint deserialization, GPU, fake experiment files or final aliases.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True

ARMS = ('frozen', 'partial', 'constant')
SOURCE_NAMES = ('TRAIN_DONE.json', 'EVALUATION.json', 'PLAN.json', 'final.pt',
                'actions.pt', 'targets.jsonl', 'initial_reconstruction.json')
MANIFEST_NAMES = {'main_manifest_sha256': 'CODE_MANIFEST.json',
                  'postfit_manifest_sha256': 'POSTFIT_MANIFEST.json',
                  'equal_step_manifest_sha256': 'EQUAL_STEP_MANIFEST.json'}
SCHEMA = 'gs_native_equal_steps_comparison_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prefix_sha(starts, step):
    return hashlib.sha256(json.dumps(starts[:step], sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'Expected object: {path}')
    return value


def selection_contract(selection, hashes, root):
    """Pure metadata validation, also exercised by in-memory CPU fixtures."""
    require(isinstance(selection, dict), 'SELECTION is missing or invalid')
    require(selection.get('schema') == 'gs_native_equal_steps_selection_v1'
            and selection.get('status') == 'selected', 'No legal selected SELECTION')
    step = selection.get('selected_step')
    require(type(step) is int and step in (50, 100, 150), 'Selected step must be50/100/150')
    for key in (*MANIFEST_NAMES, 'protocol_sha256'):
        require(is_sha(selection.get(key)) and selection[key] == hashes.get(key), f'SELECTION {key} mismatch')
    for key in ('initial_model_hash', 'plan_prefix_sha256'):
        require(is_sha(selection.get(key)), f'SELECTION missing {key}')
    entries = selection.get('arms')
    require(isinstance(entries, dict) and set(entries) == set(ARMS), 'SELECTION must contain exactly three arms')
    for arm, entry in entries.items():
        require(isinstance(entry, dict), f'Invalid selected arm {arm}')
        checkpoint = Path(entry.get('checkpoint_path', ''))
        require(checkpoint.is_absolute() and checkpoint.parent == root/arm,
                f'{arm} checkpoint must be an original absolute source path')
        require(checkpoint.name == f'step_{step:04d}.pt', f'{arm} checkpoint must be the original periodic step file')
        require(type(entry.get('checkpoint_steps')) is int and entry['checkpoint_steps'] == step,
                f'{arm} checkpoint step mismatch')
        require(type(entry.get('completed_steps')) is int and entry['completed_steps'] >= step,
                f'{arm} original final steps cannot precede selection')
        for key in ('checkpoint_sha256', 'train_done_sha256', 'plan_sha256'):
            require(is_sha(entry.get(key)), f'{arm} missing {key}')
        for key in ('initial_model_hash', 'plan_prefix_sha256'):
            require(entry.get(key) == selection[key], f'{arm} common {key} mismatch')
    require(len({entry['completed_steps'] for entry in entries.values()}) > 1,
            'Equal original final steps do not require this common-checkpoint evaluation')
    return step


def availability_contract(selection, available):
    """Validate the quality-independent largest-common-existing selection."""
    require(set(available) == set(ARMS), 'Missing checkpoint availability for an arm')
    choices = [step for step in (50, 100, 150)
               if all(step <= selection['arms'][arm]['completed_steps'] and step in available[arm] for arm in ARMS)]
    require(choices and selection['selected_step'] == max(choices), 'Selection is not the largest common existing periodic checkpoint')
    if 'available_checkpoint_steps' in selection:
        require(selection['available_checkpoint_steps'] == available, 'Checkpoint availability differs from central selection')


def original_contract(arm, entry, selection, documents, hashes, main_hash, protocol_hash):
    """Validate original final records without relabeling them selected_step."""
    train, plan, evaluation = (documents[name] for name in ('TRAIN_DONE.json', 'PLAN.json', 'EVALUATION.json'))
    step = selection['selected_step']
    require(train.get('status') == 'complete' and evaluation.get('status') == 'complete',
            f'{arm} original training/evaluation not complete')
    require(type(train.get('steps')) is int and train['steps'] == entry['completed_steps'],
            f'{arm} actual original completed steps changed')
    require(train.get('full_window_coverage') is True and train.get('windows_visited') == 168
            and train.get('target_frames_seen') == list(range(177)), f'{arm} original coverage is incomplete')
    require(hashes['TRAIN_DONE.json'] == entry['train_done_sha256']
            and hashes['PLAN.json'] == entry['plan_sha256'], f'{arm} original TRAIN_DONE/PLAN changed')
    require(hashes['final.pt'] == train.get('final_sha256') == evaluation.get('checkpoint_sha256'),
            f'{arm} original final checkpoint bytes disagree with original records')
    require(plan.get('initial_model_hash') == selection['initial_model_hash']
            and plan.get('code_manifest_sha256') == main_hash
            and plan.get('protocol_sha256') == protocol_hash, f'{arm} original initialization/code/protocol mismatch')
    require(plan.get('partial_native') is (arm != 'frozen'), f'{arm} original partial-decoder arm flag mismatch')
    starts = plan.get('starts')
    require(isinstance(starts, list) and len(starts) >= train['steps']
            and all(isinstance(batch, list) and len(batch) == 4
                    and all(type(start) is int and 0 <= start < 168 for start in batch)
                    for batch in starts[:train['steps']]), f'{arm} original B4 plan is invalid')
    require(prefix_sha(starts, step) == selection['plan_prefix_sha256'], f'{arm} selected B4 prefix hash mismatch')
    require({start for batch in starts[:step] for start in batch} == set(range(168)),
            f'{arm} selected prefix does not cover168 complete windows')
    return starts[:step]


def evaluation_contract(evaluation, arm, entry, selection, selection_hash, source_hashes):
    require(evaluation.get('schema') == 'gs_native_equal_step_evaluation_v1'
            and evaluation.get('status') == 'complete'
            and evaluation.get('evaluation_kind') == 'common_checkpoint_evaluation'
            and evaluation.get('scope') == 'single_clip_TRAIN_common_checkpoint_reconstruction'
            and evaluation.get('optimizer_updates') == 0 and evaluation.get('goal_complete') is False,
            f'{arm} common-checkpoint evaluation is incomplete/invalid')
    require(evaluation.get('arm') == arm and evaluation.get('selected_step') == selection['selected_step']
            and evaluation.get('original_final_steps') == entry['completed_steps'], f'{arm} evaluation step/arm mismatch')
    require(evaluation.get('checkpoint_path') == entry['checkpoint_path']
            and evaluation.get('checkpoint_sha256') == entry['checkpoint_sha256']
            and evaluation.get('selection_sha256') == selection_hash, f'{arm} selected checkpoint/selection binding mismatch')
    for key in (*MANIFEST_NAMES, 'protocol_sha256', 'initial_model_hash', 'plan_prefix_sha256'):
        require(evaluation.get(key) == selection[key], f'{arm} evaluation {key} mismatch')
    require(evaluation.get('windows') == 168 and evaluation.get('real_future_exposures') == 1512
            and evaluation.get('unique_future_frames') == 176, f'{arm} evaluation coverage counts invalid')
    require(evaluation.get('source_evidence_sha256') == source_hashes, f'{arm} original source-evidence hashes differ')
    require(is_sha(evaluation.get('frozen_reference_sha256')), f'{arm} missing frozen reference hash')


def metric_sections_match(recomputed, stored, label):
    require(isinstance(stored, dict), f'{label} summary missing')
    for section in ('exposure_mean', 'frame_uniform', 'per_horizon'):
        expected = recomputed[section]
        actual = stored.get(section)
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}/{section} keys differ')
        groups = expected.items() if section == 'per_horizon' else [(section, expected)]
        for group, values in groups:
            target = actual[group] if section == 'per_horizon' else actual
            require(isinstance(target, dict) and set(target) == set(values), f'{label}/{section}/{group} metric keys differ')
            for metric, value in values.items():
                require(type(target[metric]) in (float, int) and math.isfinite(target[metric])
                        and math.isclose(value, target[metric], rel_tol=1e-7, abs_tol=1e-9),
                        f'{label}/{section}/{group}/{metric} differs from per-row aggregation')


def initial_contract(initial, core):
    rows = initial.get('rows')
    require(isinstance(rows, list) and len(rows) == 168
            and all(isinstance(row, dict) and type(row.get('start')) is int for row in rows)
            and sorted(row['start'] for row in rows) == list(range(168))
            and initial.get('unique_initial_frames') == 168, 'Incomplete/duplicate h0 initial reconstruction')
    keys = ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr')
    for row in rows:
        require(row.get('end') == row['start'] and row.get('horizon') == 0, 'Invalid h0 source frame')
        for control in ('correct', 'adapted_held'):
            require(isinstance(row.get(control), dict) and all(core.numeric(row[control].get(k)) for k in keys),
                    'Missing/nonfinite h0 metric')
    metrics = {control: core.reduce_rows(rows, control) for control in ('correct', 'adapted_held')}
    for control, result in metrics.items():
        stored = initial.get('metrics', {}).get(control, {})
        for key in keys:
            require(core.numeric(stored.get(key)) and math.isclose(stored[key], result['frame_uniform'][key],
                    rel_tol=1e-7, abs_tol=1e-9), f'h0 stored {control}/{key} differs from rows')
    return dict(complete=True, frames=list(range(168)), metrics=metrics,
                frame0=next(row for row in rows if row['start'] == 0))


def original_module(root):
    spec = importlib.util.spec_from_file_location('equal_step_original_compare', root/'code/compare_arms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_rules(core, root, arms):
    # No artifact changes or fake TRAIN_DONE: the adapter exists only in this
    # local synchronous call. Every bootstrap and pass/fail rule is original.
    with patch.object(core, 'load_arm', side_effect=lambda unused_root, name: arms[name]):
        return core.build_summary(root)


def empty_arm(name, reason):
    return dict(name=name, engineering_status='incomplete_or_invalid', issues=[reason], steps=None,
                selected_checkpoint_steps=None, original_completed_steps=None, planned_steps=None,
                stop_reason=None, training_full_coverage=False, evaluation_full_coverage=False,
                metrics={}, train_done=None, evaluation=None, plan=None, initial_reconstruction=None,
                source_files={}, _rows=[], _plan_prefix=None)


def source_paths_from_bound_code(root):
    """Read literal path assignments without importing torch-backed gs_bridge."""
    result = {}
    tree = ast.parse((root/'code/gs_bridge.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('STATE_ROOT', 'V4_ROOT')):
            value = node.value
            require(isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path'
                    and len(value.args) == 1 and isinstance(value.args[0], ast.Constant)
                    and isinstance(value.args[0].value, str), 'Native source paths are no longer literal verified constants')
            result[node.targets[0].id] = Path(value.args[0].value)
    require(set(result) == {'STATE_ROOT', 'V4_ROOT'}, 'Missing native source path contract')
    return result


def build(root, evaluation_root):
    begun = time.monotonic()
    issues, snapshots, cached_hashes = [], {}, {}
    arms = {arm: empty_arm(arm, 'Selection/evaluation has not been validated') for arm in ARMS}
    core, selection, selection_hash, hashes, source_hashes = None, None, None, {}, {}

    def checked_hash(path):
        path = Path(path).resolve()
        if path not in cached_hashes:
            cached_hashes[path] = sha(path)
        snapshots[str(path)] = cached_hashes[path]
        return cached_hashes[path]

    def document(path):
        checked_hash(path)
        return read_object(path)

    def verify_manifest(name):
        manifest = document(root/name)
        for relative, expected in manifest.items():
            path = (root/relative).resolve()
            require(path.is_relative_to(root) and is_sha(expected) and checked_hash(path) == expected,
                    f'Manifest mismatch: {name}: {relative}')
        return checked_hash(root/name)

    try:
        hashes['main_manifest_sha256'] = verify_manifest('CODE_MANIFEST.json')
        core = original_module(root)
        for key, name in MANIFEST_NAMES.items():
            if key != 'main_manifest_sha256':
                hashes[key] = verify_manifest(name)
        protocol = document(root/'protocol.json')
        clip = protocol.get('data', {}).get('clip')
        require(isinstance(clip, str) and clip, 'Bound protocol lacks source clip')
        hashes['protocol_sha256'] = checked_hash(root/'protocol.json')
        selection = document(evaluation_root/'SELECTION.json')
        selection_hash = checked_hash(evaluation_root/'SELECTION.json')
        step = selection_contract(selection, hashes, root)
        availability_contract(selection, {arm: [candidate for candidate in (50, 100, 150)
                               if (root/arm/f'step_{candidate:04d}.pt').is_file()] for arm in ARMS})
        originals, prefixes = {}, {}
        for arm in ARMS:
            source = root/arm
            require(not (source/'FAILED.json').exists(), f'{arm} original source has FAILED.json')
            originals[arm] = {name: document(source/name) for name in ('TRAIN_DONE.json', 'PLAN.json', 'EVALUATION.json')}
            source_hashes[arm] = {name: checked_hash(source/name) for name in SOURCE_NAMES}
            entry = selection['arms'][arm]
            actual = Path(entry['checkpoint_path']).resolve()
            require(actual == Path(entry['checkpoint_path']) and actual.parent == source,
                    f'{arm} selected checkpoint is not the original file path')
            require(checked_hash(actual) == entry['checkpoint_sha256'], f'{arm} actual selected checkpoint bytes mismatch')
            prefixes[arm] = original_contract(arm, entry, selection, originals[arm], source_hashes[arm],
                                              hashes['main_manifest_sha256'], hashes['protocol_sha256'])
        require(all(prefix == prefixes['frozen'] for prefix in prefixes.values()), 'Actual three-arm selected plan prefixes differ')

        for arm in ARMS:
            try:
                folder = evaluation_root/arm
                require(not (folder/'FAILED.json').exists(), f'{arm} common evaluation has FAILED.json')
                evaluation = document(folder/'EVALUATION.json')
                evaluation_contract(evaluation, arm, selection['arms'][arm], selection, selection_hash, source_hashes)
                arm_issues = []
                target_hash = checked_hash(folder/'targets.jsonl')
                rows = core.read_rows(folder/'targets.jsonl', arm_issues)
                require(not arm_issues, '; '.join(arm_issues))
                require(len(rows) == 1512 and {(r['start'], r['horizon']) for r in rows} == core.REQUIRED_KEYS,
                        f'{arm} missing/duplicate future target rows')
                bundles = {}
                for row in rows:
                    ids = row.get('motion_ids')
                    require(isinstance(ids, list) and len(ids) == 9
                            and all(type(value) is int and 0 <= value < 65536 for value in ids), f'{arm} invalid row ID bundle')
                    if arm == 'constant':
                        require(ids == [0]*9, 'Constant control no longer uses the fixed valid ID0 bundle')
                    require(row['start'] not in bundles or bundles[row['start']] == ids, f'{arm} inconsistent IDs within window')
                    bundles[row['start']] = ids
                metrics = {control: core.reduce_rows(rows, control) for control in core.CONTROLS}
                require(isinstance(evaluation.get('metrics'), dict), f'{arm} evaluation metrics missing')
                for control, result in metrics.items():
                    metric_sections_match(result, evaluation['metrics'].get(control), f'{arm}/{control}')
                initial = document(folder/'initial_reconstruction.json')
                initial_summary = initial_contract(initial, core)
                for name, expected_key in [('targets.jsonl', 'targets_sha256'),
                                           ('initial_reconstruction.json', 'initial_reconstruction_sha256'),
                                           ('actions.pt', 'actions_sha256')]:
                    require(checked_hash(folder/name) == evaluation.get(expected_key), f'{arm} {name} output hash mismatch')
                original = originals[arm]
                arms[arm] = dict(name=arm, engineering_status='complete', issues=[], steps=step,
                    selected_checkpoint_steps=step, original_completed_steps=original['TRAIN_DONE.json']['steps'],
                    planned_steps=original['TRAIN_DONE.json'].get('planned_steps'),
                    stop_reason=original['TRAIN_DONE.json'].get('stop_reason'),
                    training_full_coverage=True, evaluation_full_coverage=True,
                    metrics=metrics, train_done=original['TRAIN_DONE.json'], evaluation=evaluation,
                    plan=original['PLAN.json'], initial_reconstruction=initial_summary,
                    selected_source=selection['arms'][arm],
                    source_files={name: dict(path=str(folder/name), sha256=checked_hash(folder/name))
                                  for name in ('EVALUATION.json', 'targets.jsonl', 'initial_reconstruction.json', 'actions.pt')},
                    original_source_sha256=source_hashes[arm], _rows=rows, _plan_prefix=prefixes[arm])
            except (OSError, KeyError, TypeError, ValueError) as error:
                arms[arm] = empty_arm(arm, str(error))
                arms[arm]['selected_checkpoint_steps'] = step
                arms[arm]['original_completed_steps'] = originals[arm]['TRAIN_DONE.json']['steps']

        # Per-row baselines must agree with one another AND the actual immutable
        # original reference, not merely carry the same self-reported hash.
        if all(arm['engineering_status'] == 'complete' for arm in arms.values()):
            paths = source_paths_from_bound_code(root)
            native = arms['frozen']['evaluation'].get('source_binding')
            require(isinstance(native, dict), 'Common evaluation lacks native source binding')
            for arm in ARMS:
                require(arms[arm]['evaluation'].get('source_binding') == native, 'Three native source bindings differ')
            require(checked_hash(paths['STATE_ROOT']/'codec_final.pt') == native.get('state_checkpoint_sha256'),
                    'Actual native source checkpoint hash mismatch')
            require(checked_hash(paths['V4_ROOT']/'cache_manifest.json') == native.get('cache_manifest_sha256'),
                    'Actual native cache manifest hash mismatch')
            require(isinstance(native.get('initializer_checkpoint_path'), str)
                    and checked_hash(native['initializer_checkpoint_path']) == native.get('initializer_checkpoint_sha256'),
                    'Actual native initializer checkpoint hash mismatch')
            for dependency in native.get('dependencies', []):
                require(checked_hash(dependency['path']) == dependency['sha256'], 'Native source dependency changed')
            reference_path = paths['V4_ROOT']/'vw1_h9_lfq/evaluation/targets.jsonl'
            reference_hash = checked_hash(reference_path)
            require(all(arm['evaluation']['frozen_reference_sha256'] == reference_hash for arm in arms.values()),
                    'Actual frozen reference hash mismatch')
            baseline = {}
            with reference_path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get('clip') != clip:
                        continue
                    key = (row.get('start'), row.get('horizon'))
                    if key in core.REQUIRED_KEYS:
                        require(key not in baseline and row.get('end') == sum(key), 'Invalid duplicate frozen source reference')
                        baseline[key] = row
            require(set(baseline) == core.REQUIRED_KEYS, 'Actual frozen source reference is incomplete')
            for arm in arms.values():
                for row in arm['_rows']:
                    for control in ('held_initial', 'frozen_target'):
                        require(row[control] == baseline[(row['start'], row['horizon'])][control],
                                f'{arm["name"]} per-target {control} differs from original source')
        for path, expected in snapshots.items():
            require(sha(path) == expected, f'Evidence changed while comparing: {path}')
    except (OSError, KeyError, TypeError, ValueError, ImportError) as error:
        issues.append(str(error))

    if issues:
        for arm in arms.values():
            arm['issues'] += ['Global binding: '+issue for issue in issues]
            arm['engineering_status'] = 'incomplete_or_invalid'
    if core is not None:
        summary = original_rules(core, root, arms)
    else:
        summary = dict(engineering_status='incomplete_or_failed', scientific_status='inconclusive',
                       z_initial_screen_passed=False, goal_complete=False,
                       comparison_eligibility=dict(eligible=False, reasons=issues, matched_steps=None),
                       arms={name: {k: v for k, v in arm.items() if not k.startswith('_')} for name, arm in arms.items()},
                       comparisons={}, screen_requirements=[], horizon_degradations=[])
    summary.update(schema=SCHEMA, evaluation_kind='common_checkpoint_evaluation_comparison', goal_complete=False,
                   selected_step=selection.get('selected_step') if selection else None,
                   selection_sha256=selection_hash, binding_issues=issues,
                   original_completed_steps={name: arm['original_completed_steps'] for name, arm in arms.items()},
                   manifest_hashes=hashes, original_source_evidence_sha256=source_hashes,
                   verified_input_sha256=snapshots, elapsed_seconds=time.monotonic()-begun,
                   original_comparison_implementation='code/compare_arms.py: build_summary via local in-memory load_arm adapter',
                   step_semantics='steps in comparison eligibility means selected evaluation checkpoint; original train_done is unchanged',
                   scope='共同checkpoint的单TRAIN片段后验重建补评；没有重新训练、截断原训练记录或改名final',
                   actions_validation='actions.pt bytes are bound by worker EVALUATION; no torch deserialization here; per-window row IDs validated')
    return summary, core


def markdown(summary, core=None):
    if core is not None and 'horizon_rule' in summary:
        text = core.markdown(summary)
        text = text.replace('# 三臂原生 GS / Z 比较', '# 共同checkpoint三臂原生GS / Z补评')
        text = text.replace('完成／计划步数', '本次评测checkpoint步数／原计划步数')
        text = text.replace('只有三臂步数相同、初始化和已完成采样计划一致、完整覆盖且冻结基线一致，才计算强比较。',
                            '只有三臂本次评测checkpoint步数相同、初始化和该步数之前的采样计划一致、完整覆盖且冻结基线一致，才计算强比较。')
    else:
        text = '# 共同checkpoint三臂原生GS / Z补评\n\n**尚不能公平比较（inconclusive）。**\n\n'
        text += '\n'.join('- '+reason for reason in summary.get('binding_issues', [])) + '\n'
    prefix = ('\n\n本次只对已有共同checkpoint补做评测，**没有重新训练，没有伪造TRAIN_DONE或final别名，'
              '没有覆盖主实验SUMMARY**。`goal_complete`始终为false。\n\n'
              f'中央选择步数：`{summary.get("selected_step")}`。下表列原实验实际最终训练步数；'
              '它们没有被本次补评改写。\n\n| 臂 | 原始实际训练最终步数 |\n|---|---:|\n')
    prefix += '\n'.join(f'| {arm} | {summary["original_completed_steps"].get(arm)} |' for arm in ARMS)
    prefix += ('\n\n初筛函数、阈值与物理帧移动块bootstrap直接复用绑定的原compare_arms；'
               '14个必需MAE区间下界均须大于0，规定对照的逐h全图MAE均不得恶化。'
               'h0/物理frame0单列，1512次重叠曝光不作为1512个独立样本。\n\n')
    heading, _, rest = text.partition('\n')
    return heading + prefix + rest


class MemoryRows:
    """Only for self-test: behaves like the tiny file API read_rows needs."""
    def __init__(self, rows=None):
        self.text = None if rows is None else '\n'.join(json.dumps(row) for row in rows)
    def exists(self):
        return self.text is not None
    def open(self, **kwargs):
        return io.StringIO(self.text)


def self_test(root):
    core = original_module(root)
    errors = []
    assert core.read_rows(MemoryRows(), errors) == [] and errors
    toy = [dict(end=1, horizon=1, correct=dict(mae=1.)), dict(end=1, horizon=2, correct=dict(mae=3.)),
           dict(end=2, horizon=1, correct=dict(mae=10.))]
    reduced = core.reduce_rows(toy, 'correct')
    assert reduced['frame_uniform']['mae'] == 6. and reduced['exposure_mean']['mae'] == 14/3
    def values(error):
        return dict(mae=error, hand_activity_mae=error, motion_mae=error, psnr=20., hand_activity_psnr=20.)
    rows = [dict(start=start, horizon=h, end=start+h, motion_ids=[0]*9,
                 **{control: values(2.) for control in core.CONTROLS})
            for start in range(168) for h in range(1,10)]
    errors = []
    assert len(core.read_rows(MemoryRows(rows), errors)) == 1512 and not errors
    prefix = [[(4*batch+i)%168 for i in range(4)] for batch in range(50)]
    fake_hash = 'a'*64
    fixture = {}
    for arm, final in zip(ARMS, (73, 100, 150)):
        selected_rows = copy.deepcopy(rows)
        if arm == 'partial':
            for row in selected_rows:
                row['correct'] = values(1.)
        fixture[arm] = dict(name=arm, engineering_status='complete', issues=[], steps=50,
            selected_checkpoint_steps=50, original_completed_steps=final, planned_steps=150,
            stop_reason='fixture', training_full_coverage=True, evaluation_full_coverage=True,
            metrics={control:core.reduce_rows(selected_rows,control) for control in core.CONTROLS},
            train_done=dict(steps=final), evaluation=dict(frozen_reference_sha256=fake_hash),
            plan={key:fake_hash for key in ('initial_model_hash','protocol_sha256','code_manifest_sha256')},
            initial_reconstruction=None, source_files={}, _rows=selected_rows, _plan_prefix=prefix)
    passed = original_rules(core, root, fixture)
    assert passed['scientific_status'] == 'initial_screen_passed' and len(passed['screen_requirements']) == 14
    assert all(item['physical_frames'] == 176 for result in passed['comparisons'].values() for item in result.values())
    assert fixture['frozen']['train_done']['steps'] == 73
    rendered = markdown({**passed, 'selected_step':50,
                         'original_completed_steps':{arm:fixture[arm]['original_completed_steps'] for arm in ARMS}}, core)
    assert '本次评测checkpoint步数／原计划步数' in rendered and '完成／计划步数' not in rendered
    assert '| frozen | 73 |' in rendered and '| partial | 100 |' in rendered and '| constant | 150 |' in rendered
    for row in fixture['partial']['_rows']:
        row['correct'] = values(3.)
    failed = original_rules(core, root, fixture)
    assert failed['scientific_status'] == 'initial_screen_not_passed' and not failed['z_initial_screen_passed']
    fixture['constant']['engineering_status'] = 'incomplete_or_invalid'
    missing = original_rules(core, root, fixture)
    assert missing['scientific_status'] == 'inconclusive' and not missing['comparisons']
    hashes = {key:fake_hash for key in (*MANIFEST_NAMES,'protocol_sha256')}
    selection = dict(schema='gs_native_equal_steps_selection_v1',status='selected',selected_step=50,
                     initial_model_hash=fake_hash,plan_prefix_sha256=prefix_sha(prefix,50),**hashes,
                     arms={arm:dict(checkpoint_path=str(root/arm/'step_0050.pt'),checkpoint_sha256=fake_hash,
                                    checkpoint_steps=50,completed_steps=final,initial_model_hash=fake_hash,
                                    plan_prefix_sha256=prefix_sha(prefix,50),train_done_sha256=fake_hash,plan_sha256=fake_hash)
                           for arm,final in zip(ARMS,(73,100,150))})
    assert selection_contract(selection, hashes, root) == 50
    availability_contract(selection, dict(frozen=[50], partial=[50,100], constant=[50,100,150]))
    for bad in (None, {**selection,'status':'inconclusive'}, {**selection,'selected_step':49},
                {**selection,'arms':{'partial':selection['arms']['partial']}},
                {**selection,'main_manifest_sha256':'b'*64}):
        try:
            selection_contract(bad, hashes, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid selection was accepted')
    def reject(call):
        try:
            call()
        except (ValueError, KeyError):
            return
        raise AssertionError('Invalid source/selection metadata was accepted')
    bad = copy.deepcopy(selection)
    bad['arms']['frozen'].update(checkpoint_path=str(root/'frozen/final.pt'), completed_steps=50)
    reject(lambda: selection_contract(bad, hashes, root))
    bad = copy.deepcopy(selection)
    for entry in bad['arms'].values():
        entry['completed_steps'] = 150
    reject(lambda: selection_contract(bad, hashes, root))
    newer = copy.deepcopy(selection)
    newer['arms']['frozen']['completed_steps'] = 125
    reject(lambda: availability_contract(newer, {arm:[50,100] for arm in ARMS}))
    reject(lambda: availability_contract(selection, dict(frozen=[],partial=[50],constant=[50])))

    # Source/hash contracts use dictionaries only: no fake artifact files.
    original_hashes = {name:fake_hash for name in SOURCE_NAMES}
    documents = {'TRAIN_DONE.json':dict(status='complete',steps=73,full_window_coverage=True,
                    windows_visited=168,target_frames_seen=list(range(177)),final_sha256=fake_hash),
                 'PLAN.json':dict(initial_model_hash=fake_hash,code_manifest_sha256=fake_hash,
                    protocol_sha256=fake_hash,partial_native=False,starts=prefix*3),
                 'EVALUATION.json':dict(status='complete',checkpoint_sha256=fake_hash)}
    args = ('frozen',selection['arms']['frozen'],selection,documents,original_hashes,fake_hash,fake_hash)
    assert original_contract(*args) == prefix
    wrong_hashes = {**original_hashes,'TRAIN_DONE.json':'b'*64}
    reject(lambda: original_contract(*args[:4],wrong_hashes,*args[5:]))
    wrong_documents = copy.deepcopy(documents)
    wrong_documents['TRAIN_DONE.json']['steps'] = 50
    reject(lambda: original_contract(*args[:3],wrong_documents,*args[4:]))
    source_hashes = {arm:dict(original_hashes) for arm in ARMS}
    evaluation = dict(schema='gs_native_equal_step_evaluation_v1',status='complete',
        evaluation_kind='common_checkpoint_evaluation',scope='single_clip_TRAIN_common_checkpoint_reconstruction',
        optimizer_updates=0,goal_complete=False,arm='frozen',selected_step=50,original_final_steps=73,
        checkpoint_path=selection['arms']['frozen']['checkpoint_path'],checkpoint_sha256=fake_hash,
        selection_sha256=fake_hash,initial_model_hash=fake_hash,plan_prefix_sha256=selection['plan_prefix_sha256'],
        windows=168,real_future_exposures=1512,unique_future_frames=176,
        source_evidence_sha256=source_hashes,frozen_reference_sha256=fake_hash,**hashes)
    ev_args = ('frozen',selection['arms']['frozen'],selection,fake_hash,source_hashes)
    evaluation_contract(evaluation,*ev_args)
    for invalid in ({**evaluation,'source_evidence_sha256':{}}, {**evaluation,'selection_sha256':'b'*64},
                    {**evaluation,'original_final_steps':50}, {**evaluation,'optimizer_updates':1}):
        reject(lambda invalid=invalid: evaluation_contract(invalid,*ev_args))
    assert 'torch' not in sys.modules
    print('CPU_SELF_TEST_PASSED: memory-only missing rows/selection/source/hash rejection; periodic-only largest-common selection; physical-frame reduction; '
          'original14-rule positive/negative/inconclusive directions; no torch/GPU/fake experiment files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evaluation-root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_test:
        self_test(root)
        return
    if args.evaluation_root is None:
        parser.error('--evaluation-root is required')
    target = args.evaluation_root.resolve()
    if target != (root/'diagnostics/equal_steps_v1').resolve() or not target.is_dir():
        parser.error('Evaluation root must be the existing ROOT/diagnostics/equal_steps_v1')
    if any((target/name).exists() for name in ('SUMMARY.json','SUMMARY.md')):
        parser.error('Equal-step summary already exists; refusing overwrite')
    summary, core = build(root, target)
    text = markdown(summary, core)
    for name, content in [('SUMMARY.json',json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)),
                          ('SUMMARY.md',text)]:
        with (target/name).open('x',encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps(dict(schema=SCHEMA,scientific_status=summary['scientific_status'],
                          selected_step=summary['selected_step'],goal_complete=False,
                          summary=str(target/'SUMMARY.md')),ensure_ascii=False))


if __name__ == '__main__':
    main()
