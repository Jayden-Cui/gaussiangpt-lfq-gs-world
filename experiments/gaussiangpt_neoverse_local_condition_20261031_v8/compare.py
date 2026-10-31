"""Read-only, fail-closed fixed600 comparison; no torch, training or GPU calls.

The four statistical functions are imported unchanged from the hash-bound v6
comparison. Only SUMMARY.json / SUMMARY.md are written. A statistical screen
never completes the research goal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ARMS = ('base_z', 'local_z', 'base_constant', 'local_constant')
LIVE = ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'blinded', 'adapted_held')
FROZEN = ('held_initial', 'frozen_target')
CONTROLS = LIVE + FROZEN
METRICS = ('mae', 'hand_activity_mae', 'motion_mae')
IMAGE_METRICS = METRICS + ('psnr', 'hand_activity_psnr')
INITIAL_METRICS = ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr')
DELTA = dict(mae='delta_mae', hand_activity_mae='hand_activity_delta_mae', motion_mae='motion_delta_mae')
COMPARISON_SHA256 = '14fa085be07f8ab3f6ef79ec671362f062d6b2726f3968b02f8c4d95ff91c1b8'
SOURCE_MANIFEST_SHA256 = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
KEYS = {(s, h) for s in range(168) for h in range(1, 10)}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


class Evidence:
    def __init__(self):
        self.issues, self.files = [], {}

    def check(self, condition, message):
        if not condition:
            self.issues.append(message)
        return bool(condition)

    def document(self, path):
        try:
            value = json.loads(Path(path).read_text(encoding='utf-8'))
            if not isinstance(value, dict):
                raise ValueError('顶层不是对象')
            self.file_hash(path)
            return value
        except (OSError, ValueError) as error:
            self.issues.append(f'{path}：{error}')
            return {}

    def file_hash(self, path):
        key = str(Path(path).resolve())
        if key not in self.files:
            try:
                self.files[key] = sha(path)
            except OSError as error:
                self.issues.append(f'{path} 无法计算 SHA256：{error}')
                return None
        return self.files[key]

    def binding(self, record, key, path):
        actual = self.file_hash(path)
        self.check(actual is not None and record.get(key) == actual, f'{path} 的 {key} 绑定缺失或不一致')


def load_statistics(path):
    if sha(path) != COMPARISON_SHA256:
        raise ValueError('旧比较程序 SHA256 与预声明来源不同')
    spec = importlib.util.spec_from_file_location('_v8_bound_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_protocol(p, evidence):
    evidence.check(p.get('schema') == 'gs_local_condition_v8' and p.get('seed') == 20261031,
                   '协议身份或随机种子改变')
    for section, expected in {
        'train': dict(source_step=400, total_steps=600, additional_steps=200, save_steps=[500, 600], max_seconds=1800),
        'evaluation': dict(max_seconds=2400, windows=168, future_exposures=1512, physical_future_frames=176),
        'preflight': dict(max_seconds=600, starts=[0, 42, 84, 126], parity_starts=[35, 123], updates_per_arm=1),
        'resources': dict(cpu_threads=6, peak_limit_gib=48),
        'baseline_tolerances': dict(mae=1e-4, psnr=.02),
    }.items():
        actual = p.get(section)
        evidence.check(isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items()),
                       f'协议 {section} 与固定预算/覆盖契约不一致')
    evidence.check(p.get('delta_weight') == 1 and p.get('automatic_retry') is False and p.get('goal_complete') is False,
                   '损失权重、自动重试或目标状态改变')
    expected_arms = {name: dict(source_arm='constant_delta' if name.endswith('constant') else 'z_delta',
                                local_skip_enabled=name.startswith('local_')) for name in ARMS}
    evidence.check(p.get('arms') == expected_arms, '四臂设计或投影启用规则改变')
    evidence.check(p.get('source_manifest_sha256') == SOURCE_MANIFEST_SHA256, '原400步源代码清单身份改变')


def validate_rows(rows, evidence):
    evidence.check(len(rows) == 1512 and {(r['start'], r['horizon']) for r in rows} == KEYS,
                   '未来评测必须完整覆盖168窗/1512目标/176真实未来帧')
    for row in rows:
        label = f"目标({row['start']},{row['horizon']})"
        valid = all(isinstance(row.get(c), dict) and all(numeric(row[c].get(m)) for m in IMAGE_METRICS)
                    for c in LIVE)
        delta = row.get('delta')
        valid = valid and isinstance(delta, dict) and all(
            isinstance(delta.get(c), dict) and all(numeric(delta[c].get(m)) for m in DELTA.values()) for c in LIVE)
        valid = valid and all(c in delta and delta[c] is None for c in FROZEN)
        for field in ('motion_ids', 'blinded_motion_ids'):
            ids = row.get(field)
            valid = valid and isinstance(ids, list) and len(ids) == 9 and all(
                type(i) is int and 0 <= i < 65536 for i in ids)
        if not evidence.check(valid, label + ' 缺完整图像/差分/动作ID证据或冻结差分不为null'):
            return False
        changed = row['motion_ids'][row['horizon']-1] != row['blinded_motion_ids'][row['horizon']-1]
        if not evidence.check(row.get('blinded_horizon_id_changed') is changed, label + ' blinded ID变化标志不符'):
            return False
    by_start = {}
    for row in rows:
        pair = (row['motion_ids'], row['blinded_motion_ids'])
        if row['start'] in by_start and by_start[row['start']] != pair:
            evidence.check(False, '同一窗口的九个未来目标没有复用同一动作bundle')
            return False
        by_start[row['start']] = pair
    return len(rows) == 1512


def delta_rows(rows):
    return [dict(start=r['start'], end=r['end'], horizon=r['horizon'],
                 **{c: {m: r['delta'][c][field] for m, field in DELTA.items()} for c in LIVE}) for r in rows]


def check_aggregate(actual, expected, label, evidence):
    if not isinstance(actual, dict):
        evidence.check(False, label + ' 缺汇总对象')
        return
    for key, value in expected.items():
        if isinstance(value, dict):
            check_aggregate(actual.get(key), value, label + '/' + str(key), evidence)
        elif not (numeric(actual.get(key)) and math.isclose(actual[key], value, rel_tol=1e-7, abs_tol=1e-9)):
            evidence.check(False, label + '/' + str(key) + ' 与逐目标重算不一致')


def stored_aggregate(reduced, delta=False):
    fields = ('exposure_mean', 'frame_uniform', 'per_horizon')
    result = {key: reduced[key] for key in fields}
    if delta:
        return {key: ({h: {DELTA[m]: v for m, v in values.items()} for h, values in value.items()}
                      if key == 'per_horizon' else {DELTA[m]: v for m, v in value.items()})
                for key, value in result.items()}
    return result


def validate_initial(initial, statistics, evidence):
    evidence.check(initial.get('schema') == 'gs_local_condition_initial_reconstruction_v1', 'h0证据schema不符')
    rows = initial.get('rows')
    valid = isinstance(rows, list) and len(rows) == 168 and all(isinstance(r, dict) for r in rows)
    valid = valid and all(type(r.get('start')) is int for r in rows)
    valid = valid and sorted(r['start'] for r in rows) == list(range(168))
    valid = valid and all(r.get('end') == r['start'] and type(r.get('horizon')) is int and r['horizon'] == 0
        and all(isinstance(r.get(c), dict) and all(numeric(r[c].get(m)) for m in INITIAL_METRICS) for c in LIVE)
        and all(c in r and r[c] is None for c in FROZEN) for r in rows)
    valid = evidence.check(valid and initial.get('unique_initial_frames') == 168,
                           'initial_reconstruction必须包含全部168窗h0、六个当前控制及两个冻结null')
    if not valid:
        return None
    reduced = {c: statistics.reduce_rows(rows, c) for c in LIVE}
    for c in LIVE:
        expected = {m: reduced[c]['frame_uniform'][m] for m in INITIAL_METRICS}
        check_aggregate(initial.get('metrics', {}).get(c), expected, 'initial/' + c, evidence)
    evidence.check(initial.get('uncomputed_controls') == list(FROZEN), 'h0冻结参考不可冒充当前codec输出')
    return dict(metrics=reduced, frame0=next(r for r in rows if r['start'] == 0), frames=list(range(168)))


def validate_plan(plan, source, selection, evidence):
    evidence.check(plan.get('schema') == 'gs_local_condition_plan_v1', 'PLAN schema不符')
    starts = plan.get('starts')
    valid = isinstance(starts, list) and len(starts) == 600 and all(
        isinstance(batch, list) and len(batch) == 4 and all(type(s) is int and 0 <= s < 168 for s in batch)
        for batch in starts)
    if not evidence.check(valid, 'PLAN必须是完整600步B4合法计划'):
        return None
    evidence.check(plan.get('source_steps') == 400 and plan.get('continued_batch_indices') == list(range(400, 600)),
                   'PLAN追加区间必须是400:600，共200步')
    for key, values in [('full_plan_sha256', starts), ('continuation_plan_sha256', starts[400:])]:
        evidence.check(plan.get(key) == digest(values) and source.get(key) == digest(values), key + ' 不一致')
    old_digest = hashlib.sha256(json.dumps(starts[:400], separators=(',', ':')).encode()).hexdigest()
    evidence.check(selection.get('plan_prefix_sha256') == old_digest, '前400步与实际选择源的原始计划不同')
    visited = {s for batch in starts[400:] for s in batch}
    evidence.check(visited == set(range(168)), '200追加计划没有覆盖全部168窗')
    return starts


def validate_train_log(path, starts, source, evidence):
    try:
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        evidence.check(len(rows) == 200, 'train.jsonl必须恰有200条已执行更新')
        for index, row in enumerate(rows, 400):
            valid = isinstance(row, dict) and row.get('step') == index + 1 and row.get('added_steps') == index - 399
            valid = valid and row.get('plan_index') == index and starts is not None and index < 600 and row.get('starts') == starts[index]
            valid = valid and row.get('frozen_encoder_sha256') == source.get('frozen_encoder_sha256')
            valid = valid and row.get('output_std_sha256') == source.get('output_std_sha256')
            valid = valid and row.get('fixed_null_key_bias') == source.get('fixed_null_key_bias')
            if not evidence.check(valid, f'train.jsonl更新{index+1}步数/计划/冻结状态无效'):
                return None
        return rows[-1] if rows else None
    except (OSError, ValueError, TypeError) as error:
        evidence.check(False, f'train.jsonl不能验证：{error}')
        return None


def load_arm(root, name, statistics, common):
    evidence = common['evidence']; before = len(evidence.issues)
    train_dir, eval_dir = root / 'arms' / name, root / 'evaluations' / name
    source = evidence.document(train_dir / 'SOURCE.json'); plan = evidence.document(train_dir / 'PLAN.json')
    train = evidence.document(train_dir / 'TRAIN_DONE.json'); evaluation = evidence.document(eval_dir / 'EVALUATION.json')
    initial = evidence.document(eval_dir / 'initial_reconstruction.json')
    for directory in (train_dir, eval_dir):
        evidence.check(not (directory / 'FAILED.json').exists(), str(directory) + ' 存在FAILED.json，不能覆盖失败')
    expected_source = 'constant_delta' if name.endswith('constant') else 'z_delta'
    expected = dict(schema='gs_local_condition_source_v1', arm=name, source_arm=expected_source, source_steps=400, local_skip_enabled=name.startswith('local_'),
                    source_manifest_sha256=SOURCE_MANIFEST_SHA256, delta_weight=1., goal_complete=False)
    for key, value in expected.items():
        evidence.check(source.get(key) == value, name + '/SOURCE/' + key + ' 与四臂契约不同')
    for key in ('data_hash', 'frozen_reference_sha256', 'frozen_encoder_sha256', 'output_std_sha256',
                'loaded_model_sha256', 'loaded_codec_sha256', 'zero_initial_projection_sha256'):
        value = source.get(key)
        evidence.check(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                       name + '/SOURCE/' + key + ' 缺少合法SHA256')
    for key in ('source_binding', 'original_config', 'fixed_null_key_bias'):
        evidence.check(isinstance(source.get(key), dict) and bool(source[key]), name + '/SOURCE/' + key + ' 缺失')
    evidence.check(isinstance(source.get('local_parameter_names'), list) and len(source['local_parameter_names']) == 1,
                   name + ' 未记录唯一新增64x64投影参数')
    for record in (source, train, evaluation):
        evidence.binding(record, 'code_manifest_sha256', root / 'CODE_MANIFEST.json')
        evidence.binding(record, 'protocol_sha256', root / 'protocol.json')
    evidence.binding(source, 'source_selection_sha256', common['source_root'] / 'SELECTION.json')
    selected = common['selection'].get('checkpoints', {}).get(expected_source, {})
    evidence.check(source.get('source_checkpoint_path') == selected.get('path') and
                   source.get('source_checkpoint_sha256') == selected.get('sha256') and bool(selected),
                   name + ' 源权重不匹配已选择的实际400步checkpoint')
    if source.get('source_checkpoint_path'):
        evidence.binding(source, 'source_checkpoint_sha256', Path(source['source_checkpoint_path']))
    else:
        evidence.check(False, name + ' 缺source400权重路径')
    if source.get('frozen_reference_path'):
        evidence.binding(source, 'frozen_reference_sha256', Path(source['frozen_reference_path']))
    else:
        evidence.check(False, name + ' 缺冻结参考原文件路径')
    for record, label in ((train, 'TRAIN_DONE'), (evaluation, 'EVALUATION')):
        evidence.check(all(record.get(k) == v for k, v in source.items() if k != 'schema') and bool(source), name + '/' + label + ' 来源字段不一致')
        evidence.check(record.get('status') == 'complete' and record.get('completed_steps') == 600 and record.get('steps') == 600,
                       name + '/' + label + ' 不是实际完成600步')
    evidence.check(train.get('schema') == 'gs_local_condition_train_v1' and train.get('additional_steps') == 200
                   and train.get('original_config_preserved') is True and train.get('encoder_unchanged') is True
                   and train.get('actual_wall_budget_seconds') == 1800, name + ' 训练完成契约不符')
    for key, path in dict(source_sha256=train_dir/'SOURCE.json', plan_sha256=train_dir/'PLAN.json',
                          train_jsonl_sha256=train_dir/'train.jsonl', preflight_sha256=root/'PREFLIGHT.json').items():
        evidence.binding(train, key, path)
    evidence.binding(train.get('checkpoint_sha256', {}), '600', train_dir/'step_0600.pt')
    starts = validate_plan(plan, source, common['selection'], evidence)
    last = validate_train_log(train_dir/'train.jsonl', starts, source, evidence)
    evidence.check(last is not None and train.get('last') == last, name + ' 完成收据last与最后一条实际更新不同')
    witness = common['preflight'].get('arms', {}).get(name, {})
    for key in ('loaded_model_sha256', 'loaded_codec_sha256', 'zero_initial_projection_sha256'):
        evidence.check(witness.get(key) == source.get(key) and bool(source.get(key)), name + ' 预检和正式训练初始化/' + key + ' 不符')
    evidence.check(witness.get('source_checkpoint_sha256') == source.get('source_checkpoint_sha256') and
                   witness.get('local_skip_enabled') is name.startswith('local_') and witness.get('optimizer_updates') == 1
                   and witness.get('frozen_encoder_unchanged') is True and witness.get('all_initial_points_retained') is True,
                   name + ' 预检来源、投影启用、冻结或完整support证据缺失')
    gradients = witness.get('projection_gradient_norm'); changed = witness.get('projection_changed')
    evidence.check(numeric(gradients) and ((gradients > 0 and changed is True) if name.startswith('local_')
                                         else (gradients == 0 and changed is False)), name + ' 预检投影更新行为不符合启用状态')
    evidence.check(witness.get('native_groups_updated') == dict(native_out=True, native_heads=True, native_multiplier=True),
                   name + ' 预检native训练组未全部更新')
    for key, value in dict(schema='gs_local_condition_evaluation_v1', selected_step=600, checkpoint_step=600, additional_steps=200,
        optimizer_updates=0, windows=168, real_future_exposures=1512, unique_future_frames=176,
        physical_frames_with_some_reconstruction=177, all_initial_latent_points_retained=True, all_ten_frames=True,
        full_native_resolution=[252, 448], uncomputed_delta_controls=list(FROZEN)).items():
        evidence.check(evaluation.get(key) == value, name + '/EVALUATION/' + key + ' 不符')
    evidence.check(evaluation.get('train') == train, name + ' EVALUATION嵌入训练收据不符')
    for key, path in dict(source_record_sha256=train_dir/'SOURCE.json', train_done_sha256=train_dir/'TRAIN_DONE.json',
        plan_sha256=train_dir/'PLAN.json', checkpoint_sha256=train_dir/'step_0600.pt', targets_sha256=eval_dir/'targets.jsonl',
        initial_reconstruction_sha256=eval_dir/'initial_reconstruction.json', actions_sha256=eval_dir/'actions.pt',
        snapshots_sha256=eval_dir/'snapshots.json').items():
        evidence.binding(evaluation, key, path)
    row_issues = []; rows = statistics.read_rows(eval_dir/'targets.jsonl', row_issues)
    evidence.issues += [name + '/' + issue for issue in row_issues]
    valid = validate_rows(rows, evidence)
    metrics, delta, changes = {}, {}, []
    if valid:
        changes = delta_rows(rows)
        metrics = {c: statistics.reduce_rows(rows, c) for c in CONTROLS}
        delta = {c: statistics.reduce_rows(changes, c) for c in LIVE}
        for c in CONTROLS:
            check_aggregate(evaluation.get('metrics', {}).get(c), stored_aggregate(metrics[c]), name + '/metrics/' + c, evidence)
            if c in LIVE:
                check_aggregate(evaluation.get('delta_metrics', {}).get(c), stored_aggregate(delta[c], True), name + '/delta/' + c, evidence)
            else:
                evidence.check(evaluation.get('delta_metrics', {}).get(c, 'missing') is None, name + '/' + c + ' 差分汇总必须null')
        if name.endswith('constant'):
            evidence.check(all(r[c] == r['correct'] and r['delta'][c] == r['delta']['correct']
                               for r in rows for c in LIVE[1:5]), name + ' constant的置换/盲化控制不等于正确码')
            evidence.check(len({tuple(r['motion_ids']) for r in rows}) == 1 and all(
                r['motion_ids'] == r['blinded_motion_ids'] and len(set(r['motion_ids'])) == 1 for r in rows),
                name + ' constant动作ID不恒定')
    initial_summary = validate_initial(initial, statistics, evidence)
    return dict(name=name, engineering_status='complete' if len(evidence.issues) == before else 'invalid_or_incomplete',
                issues=evidence.issues[before:], steps=train.get('completed_steps'), source=source,
                metrics=metrics, delta=delta, initial_reconstruction=initial_summary,
                blinded_id_statistics=evaluation.get('blinded_id_statistics'),
                artifacts=dict(train_directory=str(train_dir), evaluation_directory=str(eval_dir)),
                _rows=rows, _delta=changes, _plan=starts)


def paired_screens(arms, statistics):
    screens, comparisons = {}, {}
    for architecture in ('base', 'local'):
        z, constant = architecture + '_z', architecture + '_constant'
        rows, requirements, deteriorations = arms[z]['_rows'], [], []
        specs = [(c, rows, c, METRICS) for c in LIVE[1:4]]
        specs += [('matched_constant', arms[constant]['_rows'], 'correct', METRICS),
                  ('adapted_held', rows, 'adapted_held', METRICS[:2])]
        for label, other, control, metrics in specs:
            key = z + '_vs_' + label
            values = statistics.paired_comparison(rows, other, control, metrics); comparisons[key] = values
            requirements += [dict(comparison=key, metric=m, passed=v['positive_interval'], **v) for m, v in values.items()]
            deteriorations += [dict(comparison=key, horizon=int(h), full_mae_improvement=gain)
                               for h, gain in values['mae']['per_horizon_mean_improvement'].items() if gain < -1e-12]
        screens[architecture] = dict(passed=len(requirements) == 14 and all(x['passed'] for x in requirements) and not deteriorations,
            requirements=requirements, horizon_degradations=deteriorations, held_control='current_arm_step600_adapted_held')
        for label, control in [('blinded', 'blinded'), ('frozen_held_descriptive', 'held_initial'), ('frozen_target_descriptive', 'frozen_target')]:
            comparisons[z + '_vs_' + label] = statistics.paired_comparison(rows, rows, control, METRICS if control == 'blinded' else METRICS[:2])
        for control in LIVE[1:]:
            comparisons[z + '_delta_vs_' + control] = statistics.paired_comparison(arms[z]['_delta'], arms[z]['_delta'], control, METRICS)
        comparisons[z + '_delta_vs_matched_constant'] = statistics.paired_comparison(arms[z]['_delta'], arms[constant]['_delta'], 'correct', METRICS)
    for mode in ('z', 'constant'):
        key = 'local_' + mode + '_vs_base_' + mode
        for suffix, rows_key in [('', '_rows'), ('_delta', '_delta')]:
            comparisons[key + suffix] = statistics.paired_comparison(arms['local_' + mode][rows_key], arms['base_' + mode][rows_key], 'correct', METRICS)
    effects = comparisons['local_z_vs_base_z']
    initial = {a: arms[a]['initial_reconstruction']['metrics']['correct']['frame_uniform'] for a in ARMS}
    checks = dict(matched_action_screen=screens['local']['passed'],
                  motion_positive_interval=effects['motion_mae']['positive_interval'],
                  full_mae_nonworse=effects['mae']['mean_improvement'] >= 0,
                  roi_mae_nonworse=effects['hand_activity_mae']['mean_improvement'] >= 0,
                  h0_full_mae_nonworse=initial['local_z']['mae'] <= initial['base_z']['mae'],
                  h0_roi_mae_nonworse=initial['local_z']['hand_activity_mae'] <= initial['base_z']['hand_activity_mae'])
    return screens, comparisons, checks


def build_summary(root):
    root = Path(root).resolve(); evidence = Evidence(); arms = {}; screens = {}; comparisons = {}; checks = {}
    try:
        protocol = evidence.document(root/'protocol.json'); validate_protocol(protocol, evidence)
        manifest = evidence.document(root/'CODE_MANIFEST.json')
        evidence.check(all(k in manifest for k in ('worker.py', 'local_skip.py', 'compare.py', 'protocol.json')), 'v8代码清单缺必需文件')
        for name in manifest:
            evidence.binding(manifest, name, root/name)
        statistics_path = (root/protocol.get('comparison_source_relative', '../gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py')).resolve()
        statistics = load_statistics(statistics_path); evidence.file_hash(statistics_path)
        source_root = Path(protocol['source_root'])
        evidence.binding(protocol, 'source_manifest_sha256', source_root/'CODE_MANIFEST.json')
        source_manifest = evidence.document(source_root/'CODE_MANIFEST.json')
        for name in source_manifest:
            evidence.binding(source_manifest, name, source_root/name)
        selection = evidence.document(source_root/'SELECTION.json')
        evidence.check(selection.get('selected_step') == 400 and selection.get('status') == 'selected', '源SELECTION不是实际400步')
        evidence.binding(selection, 'code_manifest_sha256', source_root/'CODE_MANIFEST.json')
        evidence.binding(selection, 'protocol_sha256', source_root/'protocol.json')
        preflight = evidence.document(root/'PREFLIGHT.json')
        evidence.check(preflight.get('schema') == 'gs_local_condition_preflight_v1' and preflight.get('status') == 'passed'
                       and preflight.get('gpu_witness') is True and preflight.get('parameters_discarded') is True
                       and preflight.get('trial_optimizer_updates') == 4 and preflight.get('production_optimizer_updates') == 0
                       and set(preflight.get('arms', {})) == set(ARMS), '匹配四臂GPU预检未通过或参数未丢弃')
        evidence.binding(preflight, 'code_manifest_sha256', root/'CODE_MANIFEST.json')
        evidence.binding(preflight, 'protocol_sha256', root/'protocol.json')
        common = dict(evidence=evidence, source_root=source_root, selection=selection, preflight=preflight)
        for arm in ARMS:
            try:
                arms[arm] = load_arm(root, arm, statistics, common)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                evidence.check(False, arm + ' 证据不能完整解析：' + repr(error))
        if len(arms) == 4:
            plans = [a['_plan'] for a in arms.values()]
            evidence.check(plans[0] is not None and all(p == plans[0] for p in plans), '四臂完整600步/追加200步采样计划不同')
            for key in ('data_hash', 'source_binding', 'frozen_reference_sha256', 'original_config', 'local_parameter_names', 'zero_initial_projection_sha256'):
                values = [a['source'].get(key) for a in arms.values()]
                evidence.check(values[0] is not None and all(v == values[0] for v in values), '四臂' + key + ' 不一致')
            for mode in ('z', 'constant'):
                base, local = arms['base_' + mode]['source'], arms['local_' + mode]['source']
                for key in ('source_checkpoint_sha256', 'source_checkpoint_path', 'frozen_encoder_sha256', 'output_std_sha256',
                            'fixed_null_key_bias', 'loaded_model_sha256', 'loaded_codec_sha256'):
                    evidence.check(base.get(key) is not None and base.get(key) == local.get(key), mode + ' 同源配对初始化/' + key + ' 不同')
            references = [{(r['start'], r['horizon']): {c:r[c] for c in FROZEN} for r in a['_rows']} for a in arms.values()]
            evidence.check(all(ref == references[0] for ref in references), '逐目标冻结参考不一致')
        if not evidence.issues:
            screens, comparisons, checks = paired_screens(arms, statistics)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, ImportError) as error:
        evidence.check(False, '比较前提无法完整验证：' + repr(error))
    eligible = not evidence.issues and len(arms) == 4
    return dict(schema='gs_local_condition_comparison_v1', engineering_status='complete' if eligible else 'incomplete_or_invalid',
        comparison_eligible=eligible, comparison_eligibility=dict(eligible=eligible, reasons=evidence.issues),
        issues=evidence.issues, selected_step=600 if eligible else None, goal_complete=False,
        scientific_status=('local_skip_screen_passed' if all(checks.values()) else 'local_skip_screen_not_passed') if eligible else 'inconclusive',
        scope='单一TRAIN片段的固定600步后验重建；不构成AR、跨片段泛化或语义动作迁移证据',
        arms={a: {k:v for k,v in item.items() if not k.startswith('_')} for a,item in arms.items()},
        z_contribution_screens=screens, comparisons=comparisons, local_skip_checks=checks,
        local_skip_screen_passed=eligible and bool(checks) and all(checks.values()),
        source400_baselines={a: item['source'].get('source_checkpoint_path') for a,item in arms.items()},
        source400_note='仅列本轮绑定的400步源路径；原400步收益不能代替本轮匹配600步对照，未重新宣称旧结果。',
        source_files=evidence.files, comparison_source_sha256=COMPARISON_SHA256,
        bootstrap=dict(block_length=16, replicates=2000, seed=20261031, physical_frames=176,
                       aggregation='先对同一真实目标帧的重复窗口误差取均值，再对物理帧1..176等权',
                       interval='旧v6原函数的配对非循环移动块百分位区间；只描述当前TRAIN片段'),
        screen_definition='各架构14项正CI：正确码胜同臂3shuffle及匹配constant的full/ROI/motion，胜当前adapted_held的full/ROI；五组对照全部h的full均值不退化（容差1e-12）。',
        frozen_reference_note='旧frozen held/target仅独立描述，不冒充本轮current codec adapted_held门槛。',
        caution='人工检查原生分辨率大动作失败帧及跨片段证据仍必需；任何初筛通过都不自动完成研究目标。')


def fmt(value):
    return f'{value:.6f}' if numeric(value) else '—'


def markdown(summary):
    lines = ['# 逐点初始条件短连接：四臂固定600步比较', '',
        f"工程状态：`{summary['engineering_status']}`；可比较：`{summary['comparison_eligible']}`；目标完成：`false`。", '',
        summary['scope'] + '。', '', summary['screen_definition'], '', summary['frozen_reference_note'], '',
        '| 臂 | 步数 | PSNR ↑ | 全图 MAE ↓ | ROI MAE ↓ | 运动 MAE ↓ | 差分全图 MAE ↓ | h0全图 / ROI MAE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm, item in summary['arms'].items():
        m = (item['metrics'].get('correct') or {}).get('frame_uniform', {})
        d = (item['delta'].get('correct') or {}).get('frame_uniform', {})
        initial = item['initial_reconstruction']; h0 = initial['metrics']['correct']['frame_uniform'] if initial else {}
        lines.append(f"| {arm} | {item['steps']} | " + ' | '.join(fmt(m.get(k)) for k in ('psnr', *METRICS))
                     + f" | {fmt(d.get('mae'))} | {fmt(h0.get('mae'))} / {fmt(h0.get('hand_activity_mae'))} |")
    lines += ['', '先合并同一真实目标帧的重叠预测，再对176帧等权；h0单独汇总168帧。移动块16帧、2000次、种子20261031。', '']
    if summary['issues']:
        lines += ['公平比较被阻断，以下缺失/失败证据不会被当作通过：', ''] + ['- ' + x for x in summary['issues']]
    for name, screen in summary['z_contribution_screens'].items():
        lines += [f"- {name} 普通图像14项动作初筛：{screen['passed']}；正区间 {sum(r['passed'] for r in screen['requirements'])}/14；逐h退化 {len(screen['horizon_degradations'])}。"]
    lines += ['', '局部短连接专项检查：', '']
    lines += ['- ' + key + '：' + str(value) for key, value in summary['local_skip_checks'].items()]
    lines += ['', '| 比较 | 指标 | 改善量（对照−实验） | 95%区间 | 下界>0 |', '|---|---|---:|---|---|']
    for name, values in summary['comparisons'].items():
        for metric, item in values.items():
            lines.append(f"| {name} | {metric} | {fmt(item['mean_improvement'])} | [{fmt(item['ci95_low'])}, {fmt(item['ci95_high'])}] | {item['positive_interval']} |")
    lines += ['', summary['source400_note'], '', summary['caution'], '',
              '完整逐h、各控制及h0汇总、来源文件SHA256和失败原因见 SUMMARY.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    if not root.is_dir():
        parser.error('实验根目录不存在；不会新建实验')
    summary = build_summary(root)
    for name, contents in [('SUMMARY.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)),
                            ('SUMMARY.md', markdown(summary))]:
        path = root/name; temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(contents, encoding='utf-8'); temporary.replace(path)
    print(json.dumps({k: summary[k] for k in ('engineering_status', 'comparison_eligible', 'selected_step', 'goal_complete')}, ensure_ascii=False))
    return 0 if summary['comparison_eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
