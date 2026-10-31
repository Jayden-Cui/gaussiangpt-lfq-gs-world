"""Post-training posterior-future ablation; never trains or changes live code.

All ten pseudo-window states reuse the actual initial coordinates/features.
Only the original equally spaced timestamps are retained. The resulting legal
LFQ IDs are decoded with the actual initial state and unchanged native GS
renderer. This is a distribution-shift ablation, not proof of action semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True

METRICS = ('mae', 'hand_activity_mae', 'motion_mae', 'psnr', 'hand_activity_psnr')
DELTA_METRICS = ('delta_mae', 'hand_activity_delta_mae', 'motion_delta_mae')
EXPECTED_KEYS = {(start, h) for start in range(168) for h in range(1, 10)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected object: {path}')
    return value


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_metrics(values, metric_keys=METRICS):
    if not isinstance(values, dict):
        raise ValueError('Missing metric object')
    for key in metric_keys:
        if type(values.get(key)) not in (int, float) or not math.isfinite(values[key]):
            raise ValueError(f'Missing/nonfinite metric: {key}')


def blind_window(window):
    """Reuse S0 tensors only; no true future tensor or metadata is propagated."""
    if len(window) != 10:
        raise ValueError('A complete ten-frame window is required')
    initial = window[0]
    times = [float(state['pts']) for state in window]
    if not all(math.isfinite(value) for value in times):
        raise ValueError('Nonfinite timestamps')
    differences = [b - a for a, b in zip(times, times[1:])]
    if (min(differences) <= 0
            or any(not math.isclose(value, differences[0], rel_tol=1e-5, abs_tol=1e-8)
                   for value in differences)):
        raise ValueError('Original timestamps must be strictly increasing and equally spaced')
    return [dict(coordinates=initial['coordinates'], features=initial['features'], pts=value)
            for value in times]


def paired_aggregate(rows, metric_keys=METRICS, error_keys=METRICS[:3]):
    """Compare after reducing repeated predictions of the same physical frame."""
    if not rows:
        raise ValueError('Cannot aggregate empty observations')
    by_frame, by_horizon = {}, {}
    for row in rows:
        validate_metrics(row['correct'], metric_keys)
        validate_metrics(row['blinded'], metric_keys)
        by_frame.setdefault(row['end'], []).append(row)
        by_horizon.setdefault(row['horizon'], []).append(row)

    def group_means(group):
        original = {key: statistics.fmean(row['correct'][key] for row in group) for key in metric_keys}
        blinded = {key: statistics.fmean(row['blinded'][key] for row in group) for key in metric_keys}
        # Positive MAE increase means removing true future information hurt.
        increase = {key: blinded[key] - original[key] for key in error_keys}
        return dict(correct=original, blinded=blinded, blinded_minus_correct_mae=increase,
                    observations=len(group))

    per_frame = {str(frame): group_means(group) for frame, group in sorted(by_frame.items())}
    frame_uniform = {
        control: {key: statistics.fmean(value[control][key] for value in per_frame.values())
                  for key in (error_keys if control == 'blinded_minus_correct_mae' else metric_keys)}
        for control in ('correct', 'blinded', 'blinded_minus_correct_mae')}
    return dict(frame_uniform=frame_uniform, per_physical_frame=per_frame,
                per_horizon={str(h): group_means(group) for h, group in sorted(by_horizon.items())},
                observations=len(rows), physical_frames=len(per_frame),
                unit='physical target frame, after averaging overlapping window observations')


def check_rerender(current, reference):
    """Fail closed if the unchanged correct IDs no longer reproduce old metrics."""
    validate_metrics(current)
    validate_metrics(reference)
    differences = {key: abs(current[key] - reference[key]) for key in METRICS}
    for key, difference in differences.items():
        tolerance = .02 if key in ('psnr', 'hand_activity_psnr') else 1e-4
        if difference > tolerance:
            raise RuntimeError(f'Correct-ID rerender parity failed: {key} abs_diff={difference:.9g} '
                               f'exceeds {tolerance}; current={current[key]}, old={reference[key]}')
    return differences


def temporal_difference_metrics(image, initial_image, data, start, horizon):
    """Tensor computation on native images; h0 is excluded by construction.

    This measures image changes including camera/illumination changes. It is
    neither semantic motion segmentation nor a 3D position error.
    """
    if not 1 <= horizon <= 9:
        raise ValueError('Temporal-difference metrics require a real future h1..9')
    real_delta = data.records[start + horizon]['real'] - data.records[start]['real']
    predicted_delta = image - initial_image
    error = (predicted_delta - real_delta).abs().mean(-1)
    weight = data.motion_weight(start, horizon)
    return dict(delta_mae=float(error.mean()),
                hand_activity_delta_mae=float(error[data.roi].mean()),
                motion_delta_mae=float((weight * error).sum() / weight.sum().clamp_min(1e-6)))


def id_statistics(original, blinded):
    if (len(original) != 168 or len(blinded) != 168
            or any(len(row) != 9 for row in original + blinded)
            or any(type(value) is not int or not 0 <= value < 65536
                   for row in original + blinded for value in row)):
        raise ValueError('Expected two complete valid [168,9] LFQ ID arrays')
    changed = [[a != b for a, b in zip(left, right)] for left, right in zip(original, blinded)]
    return dict(token_change_rate=sum(sum(row) for row in changed) / 1512,
                bundle_change_rate=sum(any(row) for row in changed) / 168,
                changed_tokens=sum(sum(row) for row in changed),
                changed_bundles=sum(any(row) for row in changed),
                per_horizon={str(h + 1): dict(
                    change_rate=sum(row[h] for row in changed) / 168,
                    original_distinct_ids=len({row[h] for row in original}),
                    blinded_distinct_ids=len({row[h] for row in blinded})) for h in range(9)})


def verify_main_manifest(root):
    manifest_path = root / 'CODE_MANIFEST.json'
    manifest = read_json(manifest_path)
    if 'protocol.json' not in manifest or 'code/native_experiment.py' not in manifest:
        raise ValueError('Main manifest does not bind protocol and native_experiment')
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not isinstance(expected, str) or sha(path) != expected:
            raise ValueError('Main manifest mismatch: ' + str(relative))
    return sha(manifest_path)


def load_completed_evidence(root, arm):
    directory = root / arm
    names = ('TRAIN_DONE.json', 'EVALUATION.json', 'PLAN.json', 'final.pt',
             'actions.pt', 'targets.jsonl', 'initial_reconstruction.json')
    if (directory / 'FAILED.json').exists():
        raise RuntimeError('Arm has FAILED.json; refusing to upgrade incomplete/failed evidence')
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    train = read_json(directory / 'TRAIN_DONE.json')
    evaluation = read_json(directory / 'EVALUATION.json')
    plan = read_json(directory / 'PLAN.json')
    if (train.get('status') != 'complete' or evaluation.get('status') != 'complete'
            or type(train.get('steps')) is not int or train['steps'] <= 0
            or train.get('full_window_coverage') is not True
            or train.get('target_frames_seen') != list(range(177))):
        raise RuntimeError('A completed, fully covered arm is required')
    final_hash = sha(directory / 'final.pt')
    if final_hash != train.get('final_sha256') or final_hash != evaluation.get('checkpoint_sha256'):
        raise ValueError('Final checkpoint hash differs from training/evaluation evidence')
    if (plan.get('protocol_sha256') != sha(root / 'protocol.json')
            or plan.get('code_manifest_sha256') != sha(root / 'CODE_MANIFEST.json')):
        raise ValueError('Completed arm is bound to different main code/protocol')
    correct = {}
    with (directory / 'targets.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['start'], row['horizon'])
            if key not in EXPECTED_KEYS or key in correct or row['end'] != sum(key):
                raise ValueError('Invalid/duplicate posterior target record')
            validate_metrics(row['correct'])
            correct[key] = row
    if set(correct) != EXPECTED_KEYS:
        raise ValueError('Correct-code metrics do not cover all 1512 future observations')
    initial_source = read_json(directory / 'initial_reconstruction.json')
    initial = {}
    for row in initial_source.get('rows', []):
        start = row.get('start')
        if (type(start) is not int or not 0 <= start < 168 or start in initial
                or row.get('horizon') != 0 or row.get('end') != start):
            raise ValueError('Invalid/duplicate initial reconstruction record')
        validate_metrics(row['correct'])
        initial[start] = row
    if set(initial) != set(range(168)):
        raise ValueError('h0 correct-code references are incomplete')
    hashes = {name: sha(directory / name) for name in names}
    return dict(directory=directory, train=train, evaluation=evaluation, plan=plan,
                final_hash=final_hash, correct=correct, initial=initial, input_hashes=hashes)


def report_markdown(summary):
    values = summary['future_metrics']['frame_uniform']
    lines = ['# 盲化未来后验诊断', '',
             f"臂：`{summary['arm']}`；checkpoint 步数：{summary['completed_steps']}。"
             '本诊断完成了推理，没有训练或更新参数。', '',
             '**这是分布偏移消融，不能独自证明动作语义。** '
             '必须与同 horizon 跨起点 shuffle 和训练后的 constant 对照联合解释。', '',
             '伪窗口十项都复用同一初态的坐标／64维特征，仅保留原始等间隔时间位置；'
             '后验得到合法 LFQ IDs 后，decoder 仍只接收真实初态及这些 IDs。', '',
             '| 未来帧等权指标 | 原正确码 | 盲化后验码 | 盲化 − 正确 |',
             '|---|---:|---:|---:|']
    for key, label in [('mae', '全图 MAE'), ('hand_activity_mae', '固定活动 ROI MAE'),
                       ('motion_mae', '运动加权 MAE')]:
        lines.append(f"| {label} | {values['correct'][key]:.6f} | {values['blinded'][key]:.6f} | "
                     f"{values['blinded_minus_correct_mae'][key]:+.6f} |")
    ids = summary['id_statistics']
    lines += ['', '正的 MAE 差值表示去掉真实未来后重建变差。'
              '先对同一真实目标帧的重复窗口预测取均值，再对176个未来帧等权；'
              '1512次曝光不是1512个独立样本。', '',
              f"ID 改变率：{ids['token_change_rate']:.2%}；九码束改变率：{ids['bundle_change_rate']:.2%}。", '',
              '| h | ID 改变率 | 全图 MAE 差 | ROI MAE 差 | 运动加权 MAE 差 |',
              '|---|---:|---:|---:|---:|']
    for h in range(1, 10):
        change = summary['future_metrics']['per_horizon'][str(h)]['blinded_minus_correct_mae']
        lines.append(f"| {h} | {ids['per_horizon'][str(h)]['change_rate']:.2%} | "
                     + ' | '.join(f'{change[key]:+.6f}' for key in METRICS[:3]) + ' |')
    delta = summary['temporal_difference_metrics']['frame_uniform']
    lines += ['', '## 帧间图像变化误差', '',
              '真实变化为 real[start+h]−real[start]，预测变化为各自预测[h]−各自预测[0]。'
              '正确码与盲化码各使用自己的初始预测图。固定静态偏差可在差分中抵消，'
              '完全静态的预测不会因此匹配非零真实变化。', '',
              '| 差分 MAE | 正确码 | 盲化码 | 盲化 − 正确 |',
              '|---|---:|---:|---:|']
    for key, label in zip(DELTA_METRICS, ('全图', '固定活动 ROI', '运动加权')):
        lines.append(f"| {label} | {delta['correct'][key]:.6f} | {delta['blinded'][key]:.6f} | "
                     f"{delta['blinded_minus_correct_mae'][key]:+.6f} |")
    lines += ['', '| h | 全图差分 MAE 差 | ROI 差分 MAE 差 | 运动加权差分 MAE 差 |',
              '|---|---:|---:|---:|']
    for h in range(1,10):
        change = summary['temporal_difference_metrics']['per_horizon'][str(h)]['blinded_minus_correct_mae']
        lines.append(f'| {h} | ' + ' | '.join(f'{change[key]:+.6f}' for key in DELTA_METRICS) + ' |')
    lines += ['', '差分也会包含相机与光照变化，因此这里只称帧间图像变化误差，'
              '不称语义运动、动作语义或3D位置误差；h0不计入差分指标。']
    initial = summary['initial_metrics']['frame_uniform']['blinded_minus_correct_mae']
    frame0 = summary['initial_metrics']['per_physical_frame']['0']['blinded_minus_correct_mae']
    lines += ['', f"h0 单列：168个初始帧的全图／ROI MAE 差为 {initial['mae']:+.6f} / "
              f"{initial['hand_activity_mae']:+.6f}；物理frame0为 {frame0['mae']:+.6f} / "
              f"{frame0['hand_activity_mae']:+.6f}。h0 的运动权重为零，不据此判断运动效果。", '',
              '盲化会把真实运动窗口变为重复静态状态，可能落在训练分布之外；性能下降也可能来自'
              '这种分布偏移。ID不变说明本次消融未改变离散码，ID改变但质量不降也不等于语义无用。'
              'decoder可能联合使用九个码，某h的ID未变不代表该h图像必然不变。', '',
              f"同checkpoint正确码已重渲染 {summary['rerender_parity']['verified_images']} 张图，"
              '全图／ROI PSNR绝差均不超过0.02 dB，所有MAE类绝差均不超过1e-4。'
              '主配对使用此次重渲染正确码，旧指标另保留用于核验。', '',
              f"耗时 {summary['elapsed_seconds']:.1f}s；GPU分配峰值 {summary['gpu_peak_gib']:.3f}GiB。"
              '原分辨率、全部初态支持和原生GS guard保持不变。', '',
              '全部输入hash、逐物理帧和逐h结果见 `SUMMARY.json`；本结果不修改主实验科学状态，'
              '不构成最终目标完成。', '']
    return '\n'.join(lines)


def run(root, arm, output, max_seconds):
    begun = time.monotonic()
    current_stage = 'verify_inputs'
    completed_rows = 0
    rerender_verified = 0
    rerender_max_difference = {key: 0. for key in METRICS}
    output.mkdir(parents=True, exist_ok=False)

    def check_budget():
        if time.monotonic() - begun >= max_seconds:
            raise TimeoutError(f'Diagnostic budget {max_seconds}s exhausted')

    try:
        if arm == 'constant':
            write_new(output / 'SKIPPED.json', dict(status='skipped', arm=arm, optimizer_updates=0,
                      reason='constant模式没有真实未来posterior；本诊断只适用于frozen/partial', goal_complete=False))
            return
        main_manifest_hash = verify_main_manifest(root)
        evidence = load_completed_evidence(root, arm)
        check_budget()
        # Import the verified existing environment only after CPU provenance checks.
        sys.path.insert(0, str(root / 'code'))
        import torch
        import native_experiment as core

        actions = torch.load(evidence['directory'] / 'actions.pt', map_location='cpu', weights_only=True)
        original_ids = actions.get('ids')
        if (actions.get('schema') != 'common_initial_h9_bundle_v1'
                or actions.get('code_mode') != 'posterior'
                or actions.get('checkpoint_sha256') != evidence['final_hash']
                or actions.get('starts') != list(range(168))
                or actions.get('horizons') != list(range(1,10))
                or actions.get('all_windows_complete') is not True
                or actions.get('source_frames') != 177 or actions.get('frame_gap') != 1
                or not torch.is_tensor(original_ids) or original_ids.shape != (168,9)
                or original_ids.dtype not in (torch.int32, torch.int64)
                or not bool(((original_ids >= 0) & (original_ids < 65536)).all())):
            raise ValueError('actions.pt is not the completed posterior arm export')
        original_list = original_ids.tolist()
        for (start, h), row in evidence['correct'].items():
            if row.get('motion_ids') != original_list[start]:
                raise ValueError('actions.pt IDs differ from the original evaluated IDs')
        cfg = core.setup(root)
        current_stage = 'load_native_data_and_model'
        torch.cuda.reset_peak_memory_stats()
        data = core.Data()
        model, bridge, optimizer = core.build(cfg, data, arm)
        del optimizer
        for action_key, binding_key in [('cache_manifest_sha256', 'cache_manifest_sha256'),
                                        ('state_source_sha256', 'state_checkpoint_sha256')]:
            if actions.get(action_key) != bridge.binding[binding_key]:
                raise ValueError('actions.pt and bound native state source disagree: ' + action_key)
        checkpoint = torch.load(evidence['directory'] / 'final.pt', map_location='cpu', weights_only=False)
        if (checkpoint.get('config') != cfg
                or checkpoint.get('completed_steps') != evidence['train']['steps']
                or checkpoint.get('plan') != evidence['plan'].get('starts')):
            raise ValueError('Checkpoint config/steps/plan disagree with completed evidence')
        model.load_state_dict(checkpoint['model'], strict=True)
        bridge.codec.load_state_dict(checkpoint['codec'], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        bridge.eval().requires_grad_(False)
        if model.code_mode != 'posterior':
            raise ValueError('Only posterior models may run this diagnostic')
        check_budget()
        current_stage = 'encode_blinded_windows'
        with torch.no_grad():
            batches = []
            for start in range(0,168,4):
                check_budget()
                windows = [blind_window(data.window(s)) for s in range(start,start+4)]
                encoded = model.encode_windows(windows)
                batches.append(encoded['indices'].detach().cpu())
                del encoded, windows
            blinded_ids = torch.cat(batches, dim=0)
            blinded_list = blinded_ids.tolist()
            stats = id_statistics(original_list, blinded_list)
            with (output / 'blinded_actions.pt').open('xb') as stream:
                torch.save(dict(schema='static_initial_posterior_ablation_v1', ids=blinded_ids,
                                original_ids=original_ids, starts=list(range(168)), horizons=list(range(1,10)),
                                checkpoint_sha256=evidence['final_hash'], source_arm=arm,
                                semantics='Legal LFQ IDs from repeated S0 at real temporal positions; diagnostic only'), stream)
            rows, initial_rows = [], []
            current_stage = 'native_correct_and_blinded_rgb_evaluation'
            with (output / 'targets.jsonl').open('x', encoding='utf-8') as target_log, \
                 (output / 'initial_reconstruction.jsonl').open('x', encoding='utf-8') as initial_log, \
                 (output / 'progress.jsonl').open('x', encoding='utf-8') as progress_log:
                for start in range(168):
                    check_budget()
                    condition = [core.make_condition(data.states[start])]
                    predictions = model.decode_ids(blinded_ids[start:start+1].to('cuda'), condition)[0]
                    correct_predictions = model.decode_ids(original_ids[start:start+1].to('cuda'), condition)[0]
                    if len(predictions) != 10 or len(correct_predictions) != 10:
                        raise RuntimeError('Missing full10 decoder outputs')
                    initial_correct_image = initial_blinded_image = None
                    for h, (prediction, correct_prediction) in enumerate(zip(predictions, correct_predictions)):
                        check_budget()
                        if (not torch.equal(prediction['coordinates'], data.states[start]['coordinates'])
                                or not torch.equal(correct_prediction['coordinates'], data.states[start]['coordinates'])):
                            raise RuntimeError('Diagnostic changed original initial support')
                        baseline = evidence['initial'][start] if h == 0 else evidence['correct'][(start,h)]
                        correct_image = bridge.render(correct_prediction, data.records[start+h])
                        correct_count = bridge.last_count
                        correct_metrics = data.metrics(correct_image, start, h)
                        parity = check_rerender(correct_metrics, baseline['correct'])
                        for key, difference in parity.items():
                            rerender_max_difference[key] = max(rerender_max_difference[key], difference)
                        rerender_verified += 1
                        if torch.cuda.max_memory_allocated()/2**30 > cfg['resources']['peak_limit_gib']:
                            raise RuntimeError('Original GPU allocation guard exceeded; no support reduction')
                        check_budget()
                        image = bridge.render(prediction, data.records[start+h])
                        blinded_metrics = data.metrics(image, start, h)
                        validate_metrics(blinded_metrics)
                        row = dict(start=start,end=start+h,horizon=h,correct=correct_metrics,
                                   reference_correct=baseline['correct'],rerender_absolute_difference=parity,
                                   blinded=blinded_metrics,decoded_gaussians=bridge.last_count,
                                   correct_decoded_gaussians=correct_count,
                                   bundle_changed=original_list[start] != blinded_list[start])
                        if h:
                            correct_delta = temporal_difference_metrics(
                                correct_image, initial_correct_image, data, start, h)
                            blinded_delta = temporal_difference_metrics(
                                image, initial_blinded_image, data, start, h)
                            validate_metrics(correct_delta, DELTA_METRICS)
                            validate_metrics(blinded_delta, DELTA_METRICS)
                            row['correct_delta'] = correct_delta
                            row['blinded_delta'] = blinded_delta
                            row['horizon_id_changed'] = original_list[start][h-1] != blinded_list[start][h-1]
                            rows.append(row); target_log.write(json.dumps(row,allow_nan=False)+'\n')
                            completed_rows += 1
                        else:
                            # Exactly these two initial images live beyond the
                            # current frame. No full window of RGB is retained.
                            initial_correct_image = correct_image
                            initial_blinded_image = image
                            initial_rows.append(row); initial_log.write(json.dumps(row,allow_nan=False)+'\n')
                        if torch.cuda.max_memory_allocated()/2**30 > cfg['resources']['peak_limit_gib']:
                            raise RuntimeError('Original GPU allocation guard exceeded; no support reduction')
                        del image, correct_image
                    del predictions, correct_predictions, initial_correct_image, initial_blinded_image
                    target_log.flush(); initial_log.flush()
                    progress_log.write(json.dumps(dict(windows=start+1,future_observations=completed_rows,
                                                       elapsed_seconds=time.monotonic()-begun))+'\n')
                    progress_log.flush()
                    if start % 21 == 0:
                        print('FUTURE_PROBE',arm,start+1,'/168',flush=True)
        current_stage = 'verify_and_aggregate'
        check_budget()
        if ({(row['start'],row['horizon']) for row in rows} != EXPECTED_KEYS
                or len(rows) != 1512 or len(initial_rows) != 168 or rerender_verified != 1680):
            raise RuntimeError('Incomplete diagnostic observations')
        if verify_main_manifest(root) != main_manifest_hash:
            raise RuntimeError('Main manifest changed during diagnostic')
        for name, expected in evidence['input_hashes'].items():
            if sha(evidence['directory']/name) != expected:
                raise RuntimeError('Completed arm source changed during diagnostic: '+name)
        delta_rows = [dict(start=row['start'],end=row['end'],horizon=row['horizon'],
                           correct=row['correct_delta'],blinded=row['blinded_delta']) for row in rows]
        summary = dict(status='complete',arm=arm,completed_steps=evidence['train']['steps'],
                       optimizer_updates=0,goal_complete=False,evidence_type='distribution_shift_ablation',
                       main_code_manifest_sha256=main_manifest_hash,
                       diagnostic_script_sha256=sha(Path(__file__)),input_hashes=evidence['input_hashes'],
                       checkpoint_sha256=evidence['final_hash'],id_statistics=stats,
                       future_metrics=paired_aggregate(rows),initial_metrics=paired_aggregate(initial_rows),
                       temporal_difference_metrics=paired_aggregate(delta_rows,DELTA_METRICS,DELTA_METRICS),
                       rerender_parity=dict(verified_images=rerender_verified,
                                            max_absolute_difference=rerender_max_difference,
                                            tolerance_psnr=.02,tolerance_mae=1e-4,
                                            comparison_baseline='same-run correct-ID rerender; old metrics retained per row'),
                       source_binding=bridge.binding,elapsed_seconds=time.monotonic()-begun,
                       gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,max_seconds=max_seconds,
                       physical_future_frames=176,future_observations=1512,initial_observations=168,
                       limitations=['单TRAIN片段、后验消融，不是AR预测或泛化',
                                    '重复静态状态可能落在训练分布之外，下降不能独自证明动作语义',
                                    '必须与同h shuffle及训练constant联合解释',
                                    'decoder可联合使用全部九码；逐h ID改变率不是孤立因果归因',
                                    '帧间差分含相机和光照变化，不是语义运动或3D位置误差',
                                    '计时在每个编码batch/渲染帧边界检查，不抢占单个正在执行的CUDA算子'])
        write_new(output/'SUMMARY.json',summary)
        with (output/'SUMMARY.md').open('x',encoding='utf-8') as stream:
            stream.write(report_markdown(summary))
        print(json.dumps(dict(status='complete',arm=arm,summary=str(output/'SUMMARY.md'),optimizer_updates=0),ensure_ascii=False))
    except BaseException as error:
        write_new(output/'FAILED.json',dict(status='failed',arm=arm,stage=current_stage,
                  error=repr(error),traceback=traceback.format_exc(),future_observations=completed_rows,
                  rerender_verified_images=rerender_verified,rerender_max_absolute_difference=rerender_max_difference,
                  elapsed_seconds=time.monotonic()-begun,optimizer_updates=0,goal_complete=False))
        raise


def cpu_self_test():
    # Standard-library-only checks: no torch import, GPU, experiment reads, or writes.
    coordinates, features = object(), object()
    source = [dict(coordinates=coordinates if h==0 else object(),
                   features=features if h==0 else object(),pts=h/29.97) for h in range(10)]
    blind = blind_window(source)
    assert all(item['coordinates'] is coordinates and item['features'] is features for item in blind)
    assert [item['pts'] for item in blind] == [item['pts'] for item in source]
    assert set(blind[0]) == {'coordinates','features','pts'}
    assert source[1]['features'] is not features
    def values(error):
        return dict(mae=error,hand_activity_mae=error,motion_mae=error,psnr=30.,hand_activity_psnr=29.)
    rows = [dict(start=0,end=1,horizon=1,correct=values(0),blinded=values(1)),
            dict(start=1,end=1,horizon=0,correct=values(0),blinded=values(3)),
            dict(start=1,end=2,horizon=1,correct=values(0),blinded=values(10))]
    aggregate = paired_aggregate(rows)
    assert aggregate['frame_uniform']['blinded_minus_correct_mae']['mae'] == 6.
    assert aggregate['physical_frames'] == 2 and aggregate['observations'] == 3
    original = [[0]*9 for _ in range(168)]
    altered = [row.copy() for row in original]; altered[0][2] = 1
    changed = id_statistics(original,altered)
    assert changed['changed_tokens'] == 1 and changed['changed_bundles'] == 1
    assert changed['per_horizon']['3']['change_rate'] == 1/168
    assert all(original[start] == [0]*9 for start in range(168))
    # Arithmetic examples check interpretation, not a second implementation of
    # the tensor/render pipeline: a static offset cancels, static prediction
    # does not reproduce a nonzero true temporal change.
    true_initial, true_future = [.1,.2,.3], [.2,.4,.1]
    predicted_initial = [value + .05 for value in true_initial]
    predicted_future = [value + .05 for value in true_future]
    true_delta = [future - initial for future, initial in zip(true_future,true_initial)]
    predicted_delta = [future - initial for future, initial in zip(predicted_future,predicted_initial)]
    assert math.isclose(statistics.fmean(abs(a-b) for a,b in zip(true_delta,predicted_delta)),0.,abs_tol=1e-12)
    static_predicted_delta = [value-value for value in predicted_initial]
    assert statistics.fmean(abs(a-b) for a,b in zip(true_delta,static_predicted_delta)) > 0
    delta_rows = [dict(start=0,end=1,horizon=1,
                       correct={key:.1 for key in DELTA_METRICS},
                       blinded={key:.3 for key in DELTA_METRICS})]
    delta_summary = paired_aggregate(delta_rows,DELTA_METRICS,DELTA_METRICS)
    assert all(math.isclose(gain,.2) for gain in
               delta_summary['frame_uniform']['blinded_minus_correct_mae'].values())
    reference = values(.1)
    nearby = dict(reference,mae=.10005,psnr=30.01)
    check_rerender(nearby,reference)
    try:
        check_rerender(dict(reference,mae=.1002),reference)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Reference parity tolerance was not enforced')
    assert 'torch' not in sys.modules
    print('CPU_SELF_TEST_PASSED: static-window identity, timestamps, frame aggregation, ID rates, '
          'static-bias cancellation, nonzero change vs static prediction, delta direction, rerender parity; no torch/GPU/files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--arm',choices=('partial','frozen','constant'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=float,default=1200.)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test();return
    if args.root is None or args.arm is None or args.output is None:
        parser.error('--root, --arm and --output are required for a diagnostic run')
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('--max-seconds must be finite and positive')
    root, output = args.root.resolve(), args.output.resolve()
    allowed = (root/'diagnostics'/'future_information').resolve()
    if not root.is_dir() or not output.is_relative_to(allowed) or output == allowed:
        parser.error('Output must be a new directory beneath ROOT/diagnostics/future_information')
    if output.exists():
        parser.error('Output already exists; refusing to overwrite or resume it')
    run(root,args.arm,output,args.max_seconds)


if __name__ == '__main__':
    main()
