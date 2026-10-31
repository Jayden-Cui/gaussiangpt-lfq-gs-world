"""CPU-only v9 comparison: two new arms and two immutable v8 references.

Reference receipts retain their original paths, names and schemas. Only
SUMMARY.json / SUMMARY.md are written. No torch, GPU, training or reference edits.
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
ARMS = ('spatial_z', 'spatial_constant')
ALL_ARMS = ('base_z', 'spatial_z', 'base_constant', 'spatial_constant')
LIVE = ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'blinded', 'adapted_held')
FROZEN = ('held_initial', 'frozen_target')
CONTROLS = LIVE + FROZEN
METRICS = ('mae', 'hand_activity_mae', 'motion_mae')
IMAGE_METRICS = METRICS + ('psnr', 'hand_activity_psnr')
INITIAL_METRICS = ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr')
DELTA = dict(mae='delta_mae', hand_activity_mae='hand_activity_delta_mae', motion_mae='motion_delta_mae')
COMPARISON_SHA256 = '14fa085be07f8ab3f6ef79ec671362f062d6b2726f3968b02f8c4d95ff91c1b8'
SOURCE_MANIFEST_SHA256 = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
REFERENCE_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_local_condition_20261031_v8'
REFERENCE_MANIFEST = '169764ae89e7bbc589b27b749beb2de7e1105fd9036d122d20eaeae2be8ad7cd'
REFERENCE_PROTOCOL = '38765c9f6d7538035746ed521aa3c41bc34ea80fd4f5f2a93dfed1897b8d0934'
REFERENCE_SUMMARY = '7c7ea3826ff477101308d93182d1cb764010eb531d995f06f49d17c1ff5c5123'
REFERENCE_COMPARISON = 'eea336f5ea955aa955dc7d7d37ac3a43b2dbbf8821d568e9a545083894b938fe'
DELTA_DEFINITION = '(pred_h-own_pred_0)-(real_h-real_0); each control has its own h0'
HELD_DEFINITION = 'Current step600 arm codec renders unchanged initial state; frozen old reference is separate'
SPATIAL_PARAMETERS = {'decoder_backbone.' + name for name in (
    'initial_norm.weight', 'initial_norm.bias', 'action_norm.weight', 'action_norm.bias',
    'Wf.weight', 'position.project.weight', 'Wz.weight', 'Wo.weight')}
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
    evidence.check(initial.get('schema') == 'gs_spatial_action_initial_reconstruction_v1', 'h0证据schema不符')
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
    evidence.check(plan.get('schema') == 'gs_spatial_action_plan_v1', 'PLAN schema不符')
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def fields(record, expected, label, evidence):
    for key, value in expected.items():
        evidence.check(type(record.get(key)) is type(value) and record.get(key) == value,
                       label + '/' + key + ' 缺失或改变')


def validate_protocol(p, evidence):
    fields(p, dict(schema='gs_spatial_action_v9', seed=20261031, delta_weight=1.,
                   automatic_retry=False, goal_complete=False, source_manifest_sha256=SOURCE_MANIFEST_SHA256),
           'protocol', evidence)
    for section, expected in {
        'train': dict(source_step=400, total_steps=600, additional_steps=200, save_steps=[500, 600], max_seconds=1800),
        'evaluation': dict(max_seconds=2400, windows=168, future_exposures=1512, physical_future_frames=176),
        'preflight': dict(max_seconds=600, starts=[0,42,84,126], parity_starts=[35,123], updates_per_arm=2),
        'resources': dict(cpu_threads=6, peak_limit_gib=48),
        'invariants': dict(B=4, T=10, frame_gap=1, source_frames=177, all_initial_latent_rows=True,
                           resolution_hw=[252,448], latent_width=64, lfq_bits=16, state_quantization=False,
                           original_native_pruning=True, native_encoder_frozen=True),
        'arms': {a: dict(source_arm='constant_delta' if a.endswith('constant') else 'z_delta',
                        spatial_action_enabled=True) for a in ARMS},
        'reference_baselines': dict(root=REFERENCE_ROOT, code_manifest_sha256=REFERENCE_MANIFEST,
            protocol_sha256=REFERENCE_PROTOCOL, summary_sha256=REFERENCE_SUMMARY,
            arms={'spatial_z':'base_z', 'spatial_constant':'base_constant'}, step=600,
            same_source400_and_optimizer_rng_plan_required=True),
    }.items():
        fields(p.get(section, {}), expected, 'protocol/' + section, evidence)


def manifest(root, evidence, required=()):
    value = evidence.document(root/'CODE_MANIFEST.json')
    require(value and set(required) <= set(value), '代码清单缺必需文件：' + str(root))
    for relative in value:
        path = (root/relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(root.parent), '清单路径越出运行父目录')
        evidence.binding(value, relative, path)
    return value


def load_reference(root, protocol, evidence):
    """Verify actual v8 files, then import their unchanged validator for v8 arms."""
    binding = protocol['reference_baselines']
    for key, filename in (('code_manifest_sha256','CODE_MANIFEST.json'), ('protocol_sha256','protocol.json'),
                          ('summary_sha256','SUMMARY.json')):
        evidence.binding(binding, key, root/filename)
    require(not evidence.issues, '参考身份或当前协议未通过，不能执行来源验证代码')
    manifest(root, evidence, ('compare.py', 'worker.py', 'local_skip.py', 'protocol.json'))
    require(evidence.file_hash(root/'compare.py') == REFERENCE_COMPARISON and not evidence.issues,
            'v8比较器或来源清单不符合固定SHA')
    spec = importlib.util.spec_from_file_location('_v9_immutable_v8_validation', root/'compare.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old_protocol = evidence.document(root/'protocol.json'); module.validate_protocol(old_protocol, evidence)
    summary = evidence.document(root/'SUMMARY.json')
    fields(summary, dict(schema='gs_local_condition_comparison_v1', engineering_status='complete',
                         comparison_eligible=True, selected_step=600, goal_complete=False), 'reference/SUMMARY', evidence)
    for section in ('train', 'evaluation', 'resources', 'baseline_tolerances'):
        evidence.check(old_protocol.get(section) == protocol.get(section), '参考与新运行预算/定义不同：' + section)
    evidence.check(old_protocol.get('source_root') == protocol.get('source_root') and
                   old_protocol.get('delta_weight') == protocol.get('delta_weight'), '参考source400或delta损失不同')
    preflight = evidence.document(root/'PREFLIGHT.json')
    fields(preflight, dict(schema='gs_local_condition_preflight_v1', status='passed', gpu_witness=True,
           parameters_discarded=True, trial_optimizer_updates=4, production_optimizer_updates=0), 'reference/PREFLIGHT', evidence)
    evidence.check(set(preflight.get('arms', {})) == {'base_z','base_constant','local_z','local_constant'},
                   '原v8四臂预检不完整')
    for key, filename in (('code_manifest_sha256','CODE_MANIFEST.json'), ('protocol_sha256','protocol.json')):
        evidence.binding(preflight, key, root/filename)
    return module, summary, preflight


def validate_optimizer(source, evidence):
    fields(source, dict(optimizer_rng_restored_from_source=True, source_recipe_config_preserved=True), 'optimizer/source', evidence)
    for name in ('original_optimizer_state_sha256', 'original_rng_state_sha256'):
        evidence.check(valid_sha(source.get(name)), '缺原400步优化器/RNG摘要：' + name)
    attachment = source.get('optimizer_attachment', {})
    fields(attachment, dict(original_membership_and_order_preserved=True, added_state_initially_empty=True),
           'optimizer/attachment', evidence)
    before = attachment.get('original_groups_before')
    evidence.check(bool(before) and before == attachment.get('original_groups_after'), '旧优化器group/state在挂接时改变')
    evidence.check(attachment.get('added_parameter_names') == source.get('spatial_parameter_names'), '新增优化器参数组不完整或错序')
    hyper = attachment.get('added_group_hyperparameters', {})
    cfg = source.get('original_config', {}).get('train', {})
    evidence.check(numeric(hyper.get('lr')) and hyper['lr'] == cfg.get('motion_lr'), '新增组未保留原motion_lr')
    for name in ('betas','weight_decay'):
        expected = cfg.get(name)
        evidence.check(expected is not None and hyper.get(name) == expected, '新增组优化器默认值改变：' + name)


def validate_training_receipt(train, evidence):
    fields(train, dict(schema='gs_spatial_action_train_v1', status='complete', completed_steps=600, steps=600,
        additional_steps=200, original_config_preserved=True, encoder_unchanged=True,
        actual_wall_budget_seconds=1800, goal_complete=False), 'TRAIN_DONE', evidence)


def validate_controls(evaluation, name, evidence):
    fields(evaluation, dict(delta_definition=DELTA_DEFINITION, adapted_held_definition=HELD_DEFINITION,
        uncomputed_delta_controls=list(FROZEN), constant_shuffle_render_reused=name.endswith('constant'),
        constant_blinded_render_reused=name.endswith('constant')), name + '/controls', evidence)


def load_spatial_arm(root, name, statistics, common):
    evidence = common['evidence']; before = len(evidence.issues)
    train_dir, eval_dir = root/'arms'/name, root/'evaluations'/name
    source = evidence.document(train_dir/'SOURCE.json'); plan = evidence.document(train_dir/'PLAN.json')
    train = evidence.document(train_dir/'TRAIN_DONE.json'); evaluation = evidence.document(eval_dir/'EVALUATION.json')
    initial = evidence.document(eval_dir/'initial_reconstruction.json')
    for directory in (train_dir, eval_dir):
        evidence.check(not (directory/'FAILED.json').exists(), str(directory) + ' 存在FAILED')
    parent = 'constant_delta' if name.endswith('constant') else 'z_delta'
    fields(source, dict(schema='gs_spatial_action_source_v1', arm=name, source_arm=parent, source_steps=400,
        spatial_action_enabled=True, zero_initial_Wo=True, source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        reference_baselines=common['protocol']['reference_baselines'], delta_weight=1., goal_complete=False), name + '/SOURCE', evidence)
    for key in ('data_hash','frozen_reference_sha256','frozen_encoder_sha256','output_std_sha256',
                'loaded_model_sha256','loaded_codec_sha256','initial_spatial_parameters_sha256'):
        evidence.check(valid_sha(source.get(key)), name + '/SOURCE缺有效SHA：' + key)
    for key in ('source_binding','original_config','fixed_null_key_bias'):
        evidence.check(isinstance(source.get(key),dict) and bool(source[key]), name + '/SOURCE缺失：' + key)
    names = source.get('spatial_parameter_names', [])
    evidence.check(isinstance(names,list) and len(names)==8 and set(names)==SPATIAL_PARAMETERS, '新增空间动作参数集合改变')
    validate_optimizer(source, evidence)
    for record in (source, train, evaluation):
        evidence.binding(record,'code_manifest_sha256',root/'CODE_MANIFEST.json')
        evidence.binding(record,'protocol_sha256',root/'protocol.json')
    evidence.binding(source,'source_selection_sha256',common['source_root']/'SELECTION.json')
    selected = common['selection'].get('checkpoints',{}).get(parent,{})
    evidence.check(source.get('source_checkpoint_path') == selected.get('path') and
                   source.get('source_checkpoint_sha256') == selected.get('sha256') and bool(selected), '新臂不来自实际已选400步')
    for key, path in (('source_checkpoint_sha256',source.get('source_checkpoint_path')),
                      ('frozen_reference_sha256',source.get('frozen_reference_path'))):
        require(isinstance(path,str) and path, '缺少原来源文件路径：' + key)
        evidence.binding(source,key,Path(path))
    for record, label in ((train,'TRAIN_DONE'),(evaluation,'EVALUATION')):
        evidence.check(all(record.get(k)==v for k,v in source.items() if k!='schema'), name + '/' + label + '来源改变')
    validate_training_receipt(train,evidence)
    for key,path in dict(source_sha256=train_dir/'SOURCE.json',plan_sha256=train_dir/'PLAN.json',
                         train_jsonl_sha256=train_dir/'train.jsonl',preflight_sha256=root/'PREFLIGHT.json').items():
        evidence.binding(train,key,path)
    for step in (500,600):
        evidence.binding(train.get('checkpoint_sha256',{}),str(step),train_dir/f'step_{step:04d}.pt')
    starts=validate_plan(plan,source,common['selection'],evidence)
    last=validate_train_log(train_dir/'train.jsonl',starts,source,evidence)
    evidence.check(last is not None and train.get('last')==last,'训练完成收据last与实际更新不符')
    witness=common['preflight'].get('arms',{}).get(name,{})
    for key in ('loaded_model_sha256','loaded_codec_sha256','initial_spatial_parameters_sha256',
                'original_optimizer_state_sha256','original_rng_state_sha256','source_checkpoint_sha256'):
        evidence.check(bool(source.get(key)) and witness.get(key)==source.get(key),'预检/正式训练初始化不同：'+key)
    fields(witness,dict(optimizer_updates=2,spatial_action_enabled=True,zero_initial_Wo=True,
           frozen_encoder_unchanged=True,all_initial_points_retained=True,
           native_groups_updated=dict(native_out=True,native_heads=True,native_multiplier=True)),name+'/preflight',evidence)
    require(isinstance(witness.get('measurements'),list) and len(witness['measurements'])==2,'缺两次真实预检测量')
    require(isinstance(witness.get('spatial_updates'),list) and len(witness['spatial_updates'])==2,'缺两次空间参数梯度/更新证据')
    for update, record in enumerate(witness['spatial_updates'],1):
        fields(record,dict(update=update),name+'/spatial_preflight',evidence)
        gradients=record.get('gradients',{}); short_names={n.removeprefix('decoder_backbone.') for n in SPATIAL_PARAMETERS}
        require(set(gradients)==short_names,'预检缺8个新增参数梯度')
        for parameter, values in gradients.items():
            fields(values,dict(requires_grad=True,has_grad=True,finite=True),parameter,evidence)
            evidence.check(numeric(values.get('norm')) and values['norm']>=0,'预检参数梯度不有限')
            before_sha=record.get('parameter_sha256_before',{}).get(parameter)
            after_sha=record.get('parameter_sha256_after',{}).get(parameter)
            evidence.check(valid_sha(before_sha) and valid_sha(after_sha),'预检缺参数前后hash')
            evidence.check(record.get('parameters_changed',{}).get(parameter) is (before_sha!=after_sha),'预检更新标志与hash矛盾')
        for group in (('Wo',) if update==1 else ('initial_norm','action_norm','Wf','position','Wz','Wo')):
            values=record.get('groups',{}).get(group,{})
            evidence.check(numeric(values.get('gradient_norm')) and values['gradient_norm']>0 and values.get('changed') is True,
                           '预检缺新增组有效更新：'+group)
    fields(evaluation,dict(schema='gs_spatial_action_evaluation_v1',status='complete',completed_steps=600,steps=600,
        selected_step=600,checkpoint_step=600,additional_steps=200,optimizer_updates=0,windows=168,
        real_future_exposures=1512,unique_future_frames=176,physical_frames_with_some_reconstruction=177,
        all_initial_latent_points_retained=True,all_ten_frames=True,full_native_resolution=[252,448]),name+'/EVALUATION',evidence)
    validate_controls(evaluation,name,evidence)
    evidence.check(evaluation.get('train')==train,'EVALUATION内嵌TRAIN_DONE不符')
    for key,path in dict(source_record_sha256=train_dir/'SOURCE.json',train_done_sha256=train_dir/'TRAIN_DONE.json',
        plan_sha256=train_dir/'PLAN.json',checkpoint_sha256=train_dir/'step_0600.pt',targets_sha256=eval_dir/'targets.jsonl',
        initial_reconstruction_sha256=eval_dir/'initial_reconstruction.json',actions_sha256=eval_dir/'actions.pt',
        snapshots_sha256=eval_dir/'snapshots.json').items(): evidence.binding(evaluation,key,path)
    row_issues=[]; rows=statistics.read_rows(eval_dir/'targets.jsonl',row_issues)
    evidence.issues.extend(name+'/'+issue for issue in row_issues)
    valid=validate_rows(rows,evidence); metrics,delta,changes={},{},[]
    if valid:
        changes=delta_rows(rows)
        metrics={c:statistics.reduce_rows(rows,c) for c in CONTROLS}
        delta={c:statistics.reduce_rows(changes,c) for c in LIVE}
        for control in CONTROLS:
            check_aggregate(evaluation.get('metrics',{}).get(control),stored_aggregate(metrics[control]),name+'/'+control,evidence)
            if control in LIVE:
                check_aggregate(evaluation.get('delta_metrics',{}).get(control),stored_aggregate(delta[control],True),name+'/delta/'+control,evidence)
            else: evidence.check(evaluation.get('delta_metrics',{}).get(control,'missing') is None,'冻结参考差分必须null')
        if name.endswith('constant'):
            evidence.check(all(r[c]==r['correct'] and r['delta'][c]==r['delta']['correct'] for r in rows for c in LIVE[1:5]),
                           'constant的shuffle/blinded控制改变')
            evidence.check(len({tuple(r['motion_ids']) for r in rows})==1 and all(r['motion_ids']==r['blinded_motion_ids']
                and len(set(r['motion_ids']))==1 for r in rows),'constant动作ID不恒定')
    initial_summary=validate_initial(initial,statistics,evidence)
    return dict(name=name,engineering_status='complete' if len(evidence.issues)==before else 'invalid_or_incomplete',
        issues=evidence.issues[before:],steps=train.get('completed_steps'),source=source,metrics=metrics,delta=delta,
        initial_reconstruction=initial_summary,artifacts=dict(train_directory=str(train_dir),evaluation_directory=str(eval_dir)),
        provenance_kind='new_v9_arm',_rows=rows,_delta=changes,_plan=starts,_evaluation=evaluation)


def validate_pairing(arms, evidence):
    require(set(arms)==set(ALL_ARMS),'必须是两新臂与两真实v8参考')
    plans=[arms[a]['_plan'] for a in ALL_ARMS]
    evidence.check(plans[0] is not None and all(p==plans[0] for p in plans),'原参考与新臂完整600步计划错位')
    for name in ALL_ARMS:
        evidence.check(arms[name].get('steps')==600,name+'不是固定600步')
    for key in ('data_hash','source_binding','frozen_reference_sha256','original_config','full_plan_sha256','continuation_plan_sha256'):
        values=[arms[a]['source'].get(key) for a in ALL_ARMS]
        evidence.check(values[0] is not None and all(v==values[0] for v in values),'四条证据的'+key+'不一致')
    for mode in ('z','constant'):
        old,new=arms['base_'+mode]['source'],arms['spatial_'+mode]['source']
        for key in ('source_arm','source_steps','source_checkpoint_path','source_checkpoint_sha256','source_selection_sha256',
                    'frozen_encoder_sha256','output_std_sha256','fixed_null_key_bias','loaded_model_sha256','loaded_codec_sha256'):
            evidence.check(old.get(key) is not None and old.get(key)==new.get(key),mode+'配对原400步/初始化不同：'+key)
        evidence.check(old.get('local_skip_enabled') is False and new.get('spatial_action_enabled') is True,
                       '不能用旧local臂冒充base或重标参考')
    a,b=(arms[name]['source'] for name in ARMS)
    evidence.check(a.get('initial_spatial_parameters_sha256')==b.get('initial_spatial_parameters_sha256') and
                   valid_sha(a.get('initial_spatial_parameters_sha256')),'两新臂新增分支初始化不一致')
    refs=[{(r['start'],r['horizon']):{c:r[c] for c in FROZEN} for r in arms[a]['_rows']} for a in ALL_ARMS]
    evidence.check(all(r==refs[0] for r in refs),'各臂逐目标冻结参考改变')


def verify_snapshots(folder, evidence):
    import struct
    receipt=evidence.document(folder/'snapshots.json'); rows=receipt.get('rows')
    require(isinstance(rows,list) and len(rows)==5,'必须保留五个固定帧截图清单')
    evidence.check(receipt.get('rerendered') is False,'截图必须来自本次固定评测')
    expected={0:(0,0),44:(35,9),88:(79,9),132:(123,9),176:(167,9)}
    # Only the two known failure windows and endpoints have mandated exact windows;
    # frame88 retains the original source timing mapper's valid window.
    seen=set()
    for row in rows:
        frame=row.get('source_frame'); require(frame in expected and frame not in seen,'重复/未声明截图帧')
        seen.add(frame); s,h=row.get('start'),row.get('horizon')
        evidence.check(type(s) is int and type(h) is int and 0<=s<168 and 0<=h<=9 and s+h==frame,'截图物理时序不符')
        if frame in (0,44,132,176): evidence.check((s,h)==expected[frame],'固定失败窗口截图被替换')
        require(set(row.get('files',{}))=={'real','correct','shuffle84','blinded','adapted_held'},'缺截图控制')
        for control, record in row['files'].items():
            path=Path(record['path']).resolve()
            evidence.check(path==folder.resolve()/'snapshots'/f'frame_{frame:03d}_{control}.png','截图路径不属于实际评测目录')
            evidence.binding(record,'sha256',path)
            with path.open('rb') as stream: header=stream.read(24)
            evidence.check(len(header)==24 and header[:8]==b'\x89PNG\r\n\x1a\n' and
                           struct.unpack('>II',header[16:24])==(448,252),'截图不是原生448x252 PNG')
    return rows




def paired_screens(arms, statistics):
    screens, comparisons = {}, {}
    for architecture in ('base', 'spatial'):
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
        key = 'spatial_' + mode + '_vs_base_' + mode
        for suffix, rows_key in [('', '_rows'), ('_delta', '_delta')]:
            comparisons[key + suffix] = statistics.paired_comparison(arms['spatial_' + mode][rows_key], arms['base_' + mode][rows_key], 'correct', METRICS)
    effects = comparisons['spatial_z_vs_base_z']
    initial = {a: arms[a]['initial_reconstruction']['metrics']['correct']['frame_uniform'] for a in ALL_ARMS}
    checks = dict(matched_action_screen=screens['spatial']['passed'],
                  motion_positive_interval=effects['motion_mae']['positive_interval'],
                  full_mae_nonworse=effects['mae']['mean_improvement'] >= 0,
                  roi_mae_nonworse=effects['hand_activity_mae']['mean_improvement'] >= 0,
                  h0_full_mae_nonworse=initial['spatial_z']['mae'] <= initial['base_z']['mae'],
                  h0_roi_mae_nonworse=initial['spatial_z']['hand_activity_mae'] <= initial['base_z']['hand_activity_mae'])
    return screens, comparisons, checks


def build_summary(root):
    root=Path(root).resolve(); evidence=Evidence(); arms={}; screens={}; comparisons={}; checks={}; visual={}
    reference_root=None
    try:
        protocol=evidence.document(root/'protocol.json'); validate_protocol(protocol,evidence)
        manifest(root,evidence,('compare.py','worker.py','spatial_action.py','protocol.json'))
        for path in (root/'FAILED.json',root/'preflight/FAILED.json'):
            evidence.check(not path.exists(),'存在失败标记：'+str(path))
        statistics_path=(root/protocol['comparison_source_relative']).resolve()
        statistics=load_statistics(statistics_path); evidence.file_hash(statistics_path)
        source_root=Path(protocol['source_root'])
        evidence.binding(protocol,'source_manifest_sha256',source_root/'CODE_MANIFEST.json')
        manifest(source_root,evidence)
        selection=evidence.document(source_root/'SELECTION.json')
        fields(selection,dict(status='selected',selected_step=400),'source/SELECTION',evidence)
        evidence.binding(selection,'code_manifest_sha256',source_root/'CODE_MANIFEST.json')
        evidence.binding(selection,'protocol_sha256',source_root/'protocol.json')
        reference_root=Path(protocol['reference_baselines']['root'])
        old,reference_summary,reference_preflight=load_reference(reference_root,protocol,evidence)
        common=dict(evidence=evidence,source_root=source_root,selection=selection,preflight=reference_preflight)
        for name in ('base_z','base_constant'):
            arm=old.load_arm(reference_root,name,statistics,common)
            stored=reference_summary.get('arms',{}).get(name,{})
            evidence.check(stored.get('source')==arm['source'] and stored.get('steps')==600,
                           name+'不符合已绑定v8 SUMMARY的原始来源')
            evidence.check(stored.get('metrics')==arm['metrics'] and stored.get('delta')==arm['delta'],
                           name+'重算指标与原SUMMARY不符')
            arm['provenance_kind']='immutable_v8_reference'
            arm['_evaluation']=evidence.document(reference_root/'evaluations'/name/'EVALUATION.json')
            validate_controls(arm['_evaluation'],name,evidence)
            arms[name]=arm
            visual[name]=verify_snapshots(reference_root/'evaluations'/name,evidence)
        preflight=evidence.document(root/'PREFLIGHT.json')
        fields(preflight,dict(schema='gs_spatial_action_preflight_v1',status='passed',gpu_witness=True,
               parameters_discarded=True,trial_optimizer_updates=4,production_optimizer_updates=0,goal_complete=False),
               'PREFLIGHT',evidence)
        evidence.check(set(preflight.get('arms',{}))==set(ARMS),'新运行必须仅包含两新臂预检')
        for key,filename in (('code_manifest_sha256','CODE_MANIFEST.json'),('protocol_sha256','protocol.json')):
            evidence.binding(preflight,key,root/filename)
        evidence.check(evidence.file_hash(root/'preflight/RESULT.json')==evidence.file_hash(root/'PREFLIGHT.json'),
                       '预检RESULT和PREFLIGHT不一致')
        common.update(preflight=preflight,protocol=protocol)
        for name in ARMS:
            arms[name]=load_spatial_arm(root,name,statistics,common)
            visual[name]=verify_snapshots(root/'evaluations'/name,evidence)
        validate_pairing(arms,evidence)
        if not evidence.issues:
            screens,comparisons,checks=paired_screens(arms,statistics)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,IndexError,ImportError) as error:
        evidence.check(False,'比较前提无法完整核验：'+repr(error))
    eligible=not evidence.issues and set(arms)==set(ALL_ARMS)
    passed=eligible and bool(checks) and all(checks.values())
    return dict(schema='gs_spatial_action_comparison_v1',engineering_status='complete' if eligible else 'incomplete_or_invalid',
        comparison_eligible=eligible,comparison_eligibility=dict(eligible=eligible,reasons=evidence.issues),issues=evidence.issues,
        selected_step=600 if eligible else None,goal_complete=False,
        scientific_status=('spatial_action_screen_passed' if passed else 'spatial_action_screen_not_passed') if eligible else 'inconclusive',
        scope='固定旧codec上的单TRAIN片段后验重建；CPU或图像初筛不代表可靠动作Z、AR或泛化成功',
        new_arms=list(ARMS),reference_arms=['base_z','base_constant'],reference_root=str(reference_root) if reference_root else None,
        reference_note='原v8第600步完成工件只读复用；并非新运行臂或额外独立随机种子。未采用多来源codec。',
        arms={a:{k:v for k,v in item.items() if not k.startswith('_')} for a,item in arms.items()},
        z_contribution_screens=screens,comparisons=comparisons,spatial_action_checks=checks,spatial_action_screen_passed=passed,
        source_files=evidence.files,comparison_source_sha256=COMPARISON_SHA256,reference_validator_sha256=REFERENCE_COMPARISON,
        bootstrap=dict(block_length=16,replicates=2000,seed=20261031,physical_frames=176,
                       aggregation='先合并物理帧的重叠预测，再对176个未来物理帧等权；单clip描述性区间'),
        screen_definition='每架构14项正CI：3shuffle与匹配constant的full/ROI/motion，以及当前adapted_held的full/ROI；逐h全图均值不退化。',
        frozen_reference_note='原frozen held/target仅描述，不替代当前第600步codec的adapted_held。',
        visual_acceptance=dict(status='pending_human_review',required_pairs=[dict(frame=44,start=35,horizon=9),
            dict(frame=132,start=123,horizon=9)],snapshots=visual,
            criterion='检查积木实际移动、位置和轮廓；拖影/缺失不能由平均PSNR补偿。'),
        verification_scope='只检查原始收据、文件SHA和完整指标；不在比较器中反序列化checkpoint或重新执行GPU渲染')


def markdown(summary):
    lines=['# GS空间动作调制：固定600步比较','',
           f"工程状态：{summary['engineering_status']}；可比较：{summary['comparison_eligible']}；goal_complete=false。",'',
           summary['scope']+'。','',summary['reference_note'],'',summary['screen_definition'],'',summary['frozen_reference_note'],'',
           '| 臂 | 来源 | PSNR | full MAE | ROI MAE | motion MAE |','|---|---|---:|---:|---:|---:|']
    for name,item in summary['arms'].items():
        metrics=(item['metrics'].get('correct') or {}).get('frame_uniform',{})
        values=[f'{metrics[k]:.6f}' if numeric(metrics.get(k)) else '—' for k in ('psnr',*METRICS)]
        lines.append('| '+name+' | '+item['provenance_kind']+' | '+' | '.join(values)+' |')
    lines+=['','科学状态：'+summary['scientific_status'],'','空间动作专项检查：','']
    lines += ['- '+key+'：'+str(value) for key,value in summary['spatial_action_checks'].items()]
    if summary['issues']: lines+=['','证据不足，拒绝合格比较：','']+['- '+issue for issue in summary['issues']]
    lines+=['','必须人工验收固定帧44（start35/h9）与132（start123/h9），当前状态：待验收。','']
    for arm,rows in summary['visual_acceptance']['snapshots'].items():
        for row in rows:
            if row['source_frame'] in (44,132):
                links=['['+control+'](<'+record['path']+'>)' for control,record in row['files'].items()]
                lines.append(f"- {arm} 帧{row['source_frame']}："+' · '.join(links))
    lines+=['','完整配对CI、逐h、h0、差分及来源SHA见SUMMARY.json；所有区间只描述当前片段。','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args(argv).root.resolve()
    if not root.is_dir(): parser.error('实验根目录不存在；不会创建新运行')
    summary=build_summary(root)
    for name,content in [('SUMMARY.json',json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)),('SUMMARY.md',markdown(summary))]:
        path=root/name;temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(content,encoding='utf-8');temporary.replace(path)
    print(json.dumps({key:summary[key] for key in ('engineering_status','comparison_eligible','scientific_status','goal_complete')}))
    return 0 if summary['comparison_eligible'] else 2


if __name__=='__main__':
    raise SystemExit(main())


