"""Read-only evidence comparison; writes only root/SUMMARY.json and SUMMARY.md.

No torch, checkpoint loading, GPU, training, or automatic recovery. The paired
bootstrap resamples contiguous blocks of PHYSICAL target frames after reducing
overlapping (start,horizon) observations, never 1512 independent observations.
Its intervals describe this TRAIN clip and are not population/generalization
confidence claims. A passed screen never marks the research goal complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ('frozen', 'partial', 'constant')
CONTROLS = ('correct', 'shuffle42', 'shuffle84', 'shuffle126',
            'adapted_held', 'held_initial', 'frozen_target')
METRICS = ('mae', 'hand_activity_mae', 'motion_mae', 'psnr', 'hand_activity_psnr')
REQUIRED_KEYS = {(start, h) for start in range(168) for h in range(1, 10)}
BOOTSTRAP_SEED = 20261031
BOOTSTRAP_REPLICATES = 2000
BLOCK_LENGTH = 16


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_document(path, issues, *, required=True):
    if not path.exists():
        if required:
            issues.append(f'缺少 {path.name}')
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('顶层不是对象')
        return value
    except (OSError, ValueError) as error:
        issues.append(f'{path.name} 不能读取：{error}')
        return None


def read_rows(path, issues):
    rows, seen = [], set()
    if not path.exists():
        issues.append('缺少 targets.jsonl')
        return rows
    try:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    start, horizon, end = row['start'], row['horizon'], row['end']
                    if any(type(value) is not int for value in (start, horizon, end)):
                        raise ValueError('start/horizon/end 必须为整数')
                    key = (start, horizon)
                    if key not in REQUIRED_KEYS or end != start + horizon:
                        raise ValueError('不是完整十帧窗中的有效未来目标')
                    if key in seen:
                        raise ValueError('重复 (start,horizon)')
                    seen.add(key)
                    for control in CONTROLS:
                        values = row.get(control)
                        if not isinstance(values, dict):
                            raise ValueError(f'缺少 {control}')
                        required = ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr')
                        if control not in ('held_initial', 'frozen_target'):
                            required += ('motion_mae',)
                        if any(not numeric(values.get(metric)) for metric in required):
                            raise ValueError(f'{control} 缺指标或含非有限数值')
                    rows.append(row)
                except (KeyError, TypeError, ValueError) as error:
                    if len(issues) < 25:
                        issues.append(f'targets.jsonl 第 {line_number} 行无效：{error}')
    except OSError as error:
        issues.append(f'targets.jsonl 不能读取：{error}')
    return rows


def reduce_rows(rows, control):
    """Average errors within physical frame first, then give frames equal weight."""
    selected = [row for row in rows if isinstance(row.get(control), dict)]
    if not selected:
        return None
    metrics = [metric for metric in METRICS
               if all(numeric(row[control].get(metric)) for row in selected)]
    frames, horizons = {}, {}
    for row in selected:
        frames.setdefault(row['end'], []).append(row)
        horizons.setdefault(row['horizon'], []).append(row)

    def means(group):
        return {metric: float(np.mean([row[control][metric] for row in group]))
                for metric in metrics}

    per_frame = {frame: means(group) for frame, group in frames.items()}
    return dict(exposure_mean=means(selected),
                frame_uniform={metric: float(np.mean([value[metric] for value in per_frame.values()]))
                               for metric in metrics},
                per_horizon={str(h): means(group) for h, group in sorted(horizons.items())},
                valid_rows=len(selected), unique_target_frames=len(frames))


def load_arm(root, name):
    path, issues = root / name, []
    train = read_document(path / 'TRAIN_DONE.json', issues)
    evaluation = read_document(path / 'EVALUATION.json', issues)
    plan = read_document(path / 'PLAN.json', issues)
    failed = read_document(path / 'FAILED.json', issues, required=False)
    initial = read_document(path / 'initial_reconstruction.json', issues,
                            required=bool(evaluation and evaluation.get('initial_reconstruction')))
    initial_summary = None
    if initial is not None:
        initial_rows = initial.get('rows')
        valid_initial = (isinstance(initial_rows, list) and len(initial_rows) == 168
                         and all(isinstance(row, dict) for row in initial_rows)
                         and all(type(row.get('start')) is int for row in initial_rows)
                         and sorted(row.get('start', -1) for row in initial_rows) == list(range(168)))
        if valid_initial:
            valid_initial = all(row.get('end') == row['start'] and row.get('horizon') == 0
                                and all(isinstance(row.get(control), dict)
                                        and all(numeric(row[control].get(metric)) for metric in
                                                ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr'))
                                        for control in ('correct', 'adapted_held'))
                                for row in initial_rows)
        if not valid_initial:
            issues.append('initial_reconstruction 的168个初始重建记录无效或不完整')
        else:
            initial_summary = dict(complete=True, frames=list(range(168)),
                                   metrics={control: reduce_rows(initial_rows, control)
                                            for control in ('correct', 'adapted_held')},
                                   frame0=next(row for row in initial_rows if row['start'] == 0))
    rows = read_rows(path / 'targets.jsonl', issues)
    keys = {(row['start'], row['horizon']) for row in rows}
    coverage = keys == REQUIRED_KEYS and len(rows) == 1512
    if not coverage:
        issues.append(f'有效评测目标 {len(rows)}/1512；缺失键 {len(REQUIRED_KEYS - keys)}')
    if train and train.get('status') != 'complete':
        issues.append('TRAIN_DONE 未声明 complete')
    if evaluation and evaluation.get('status') != 'complete':
        issues.append('EVALUATION 未声明 complete')
    if failed is not None:
        issues.append('存在 FAILED.json；保留失败状态，不由其他文件覆盖')
    steps = train.get('steps') if train else None
    valid_steps = type(steps) is int and steps > 0
    if train and not valid_steps:
        issues.append('训练完成步数缺失、非正整数或为零')
    training_coverage = bool(train and train.get('full_window_coverage') is True
                             and train.get('windows_visited') == 168
                             and train.get('target_frames_seen') == list(range(177)))
    if train and not training_coverage:
        issues.append('训练未确认完整 168 窗／177 帧覆盖')
    plan_prefix = None
    if plan and valid_steps:
        batches = plan.get('starts')
        if (not isinstance(batches, list) or len(batches) < steps
                or any(not isinstance(batch, list) or len(batch) != 4
                       or any(type(start) is not int or not 0 <= start < 168 for start in batch)
                       for batch in batches[:steps])):
            issues.append('PLAN 的已完成 B4 采样序列无效')
        else:
            plan_prefix = batches[:steps]
            if {start for batch in plan_prefix for start in batch} != set(range(168)):
                issues.append('PLAN 已完成部分未实际覆盖全部窗口')
                training_coverage = False
    if train and evaluation:
        if (not train.get('final_sha256')
                or train.get('final_sha256') != evaluation.get('checkpoint_sha256')):
            issues.append('训练最终 checkpoint 与评测 checkpoint 的记录 hash 不一致或缺失')
        if (evaluation.get('windows') != 168 or evaluation.get('real_future_exposures') != 1512
                or evaluation.get('unique_future_frames') != 176):
            issues.append('EVALUATION 覆盖计数不是 168 窗／1512 曝光／176 未来帧')
    metrics = {control: reduce_rows(rows, control) for control in CONTROLS}
    if evaluation and coverage:
        stored = evaluation.get('metrics', {})
        if not isinstance(stored, dict):
            stored = {}
            issues.append('EVALUATION.metrics 不是对象')
        for control, values in metrics.items():
            stored_control = stored.get(control, {})
            expected = stored_control.get('frame_uniform', {}) if isinstance(stored_control, dict) else {}
            if not isinstance(expected, dict):
                expected = {}
            if values is None:
                continue
            for metric, value in values['frame_uniform'].items():
                if not numeric(expected.get(metric)) or not math.isclose(
                        value, expected[metric], rel_tol=1e-7, abs_tol=1e-9):
                    issues.append(f'EVALUATION 的 {control}/{metric} 与逐目标重算不符或缺失')
    files = {name: dict(path=str(path / name), sha256=digest(path / name))
             for name in ('TRAIN_DONE.json', 'EVALUATION.json', 'PLAN.json', 'targets.jsonl',
                          'initial_reconstruction.json', 'FAILED.json')
             if (path / name).is_file()}
    return dict(name=name, engineering_status='complete' if not issues else ('failed' if failed else 'incomplete_or_invalid'),
                issues=issues, steps=steps, planned_steps=train.get('planned_steps') if train else None,
                stop_reason=train.get('stop_reason') if train else None,
                training_full_coverage=training_coverage, evaluation_full_coverage=coverage,
                metrics=metrics, train_done=train, evaluation=evaluation, plan=plan,
                initial_reconstruction=initial_summary,
                source_files=files, _rows=rows, _plan_prefix=plan_prefix)


def comparison_eligibility(arms):
    reasons = []
    if any(arm['engineering_status'] != 'complete' for arm in arms.values()):
        reasons.append('至少一臂缺结果、失败、证据无效或覆盖不全')
    steps = [arm['steps'] for arm in arms.values()]
    if any(type(value) is not int or value <= 0 for value in steps) or len(set(steps)) != 1:
        reasons.append(f'完成步数不相同或无效：{dict(zip(ARMS, steps))}')
    prefixes = [arm['_plan_prefix'] for arm in arms.values()]
    if any(prefix is None for prefix in prefixes) or any(prefix != prefixes[0] for prefix in prefixes[1:]):
        reasons.append('无法确认三臂已完成的 B4 采样计划相同')
    for key in ('initial_model_hash', 'protocol_sha256', 'code_manifest_sha256'):
        values = [(arm['plan'] or {}).get(key) for arm in arms.values()]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != 1:
            reasons.append(f'三臂 {key} 不一致或缺失')
    references = [(arm['evaluation'] or {}).get('frozen_reference_sha256') for arm in arms.values()]
    if any(not isinstance(value, str) or not value for value in references) or len(set(references)) != 1:
        reasons.append('三臂冻结基线文件 hash 不一致或缺失')
    # Equal hashes do not silently replace a check that per-target baselines
    # were actually copied consistently into every result row.
    indexed = {name: {(row['start'], row['horizon']): row for row in arm['_rows']}
               for name, arm in arms.items()}
    if all(set(values) == REQUIRED_KEYS for values in indexed.values()):
        for name in ('partial', 'constant'):
            mismatch = False
            for key in REQUIRED_KEYS:
                for control in ('held_initial', 'frozen_target'):
                    for metric in ('mae', 'hand_activity_mae', 'psnr', 'hand_activity_psnr'):
                        if not math.isclose(indexed[name][key][control][metric],
                                            indexed['frozen'][key][control][metric],
                                            rel_tol=0, abs_tol=1e-12):
                            mismatch = True
            if mismatch:
                reasons.append(f'{name} 的逐目标冻结基线与 frozen 臂不一致')
    return dict(eligible=not reasons, reasons=reasons,
                matched_steps=steps[0] if not reasons else None,
                requirement='三臂相同步数、相同已完成采样计划/初始化/协议/代码来源、完整覆盖及一致冻结基线')


def moving_block_bootstrap(frame_differences):
    """Percentile interval on 176 paired physical-frame mean improvements."""
    frames = sorted(frame_differences)
    if frames != list(range(1, 177)):
        raise ValueError('Bootstrap requires all physical future frames 1..176 exactly once')
    values = np.asarray([frame_differences[frame] for frame in frames], dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError('Nonfinite frame differences')
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    block_count = math.ceil(len(values) / BLOCK_LENGTH)
    starts = generator.integers(0, len(values) - BLOCK_LENGTH + 1,
                               size=(BOOTSTRAP_REPLICATES, block_count))
    indices = (starts[:, :, None] + np.arange(BLOCK_LENGTH)).reshape(BOOTSTRAP_REPLICATES, -1)
    bootstrap_means = values[indices[:, :len(values)]].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [.025, .975])
    return dict(mean_improvement=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                positive_interval=bool(low > 0), physical_frames=len(values),
                block_length=BLOCK_LENGTH, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                interval='paired moving-block percentile, non-circular; descriptive single TRAIN clip')


def paired_comparison(partial_rows, other_rows, other_control, metrics):
    left = {(row['start'], row['horizon']): row for row in partial_rows}
    right = {(row['start'], row['horizon']): row for row in other_rows}
    if set(left) != REQUIRED_KEYS or set(right) != REQUIRED_KEYS:
        raise ValueError('Paired comparison requires complete matching target keys')
    result = {}
    for metric in metrics:
        by_frame, by_horizon = {}, {}
        for key in sorted(REQUIRED_KEYS):
            row, other = left[key], right[key]
            difference = other[other_control][metric] - row['correct'][metric]
            by_frame.setdefault(row['end'], []).append(difference)
            by_horizon.setdefault(row['horizon'], []).append(difference)
        frame_means = {frame: float(np.mean(values)) for frame, values in by_frame.items()}
        result[metric] = moving_block_bootstrap(frame_means)
        result[metric]['per_horizon_mean_improvement'] = {
            str(h): float(np.mean(values)) for h, values in sorted(by_horizon.items())}
    return result


def build_summary(root):
    arms = {name: load_arm(root, name) for name in ARMS}
    eligibility = comparison_eligibility(arms)
    comparisons, requirements, horizon_degradations = {}, [], []
    if eligibility['eligible']:
        p = arms['partial']['_rows']
        specs = [(f'partial_vs_{control}', p, control,
                  ('mae', 'hand_activity_mae', 'motion_mae'), True)
                 for control in ('shuffle42', 'shuffle84', 'shuffle126')]
        specs += [('partial_vs_constant', arms['constant']['_rows'], 'correct',
                   ('mae', 'hand_activity_mae', 'motion_mae'), True),
                  ('partial_vs_frozen_held_initial', arms['frozen']['_rows'], 'held_initial',
                   ('mae', 'hand_activity_mae'), True),
                  ('partial_vs_frozen_correct', arms['frozen']['_rows'], 'correct',
                   ('mae', 'hand_activity_mae', 'motion_mae'), False),
                  ('partial_vs_adapted_held', p, 'adapted_held',
                   ('mae', 'hand_activity_mae', 'motion_mae'), False)]
        for name, rows, control, metrics, required in specs:
            comparisons[name] = paired_comparison(p, rows, control, metrics)
            if required:
                for metric in metrics:
                    item = comparisons[name][metric]
                    requirements.append(dict(comparison=name, metric=metric,
                                             passed=item['positive_interval'],
                                             mean_improvement=item['mean_improvement'],
                                             ci95_low=item['ci95_low']))
                # This is a conservative mean-direction screen, not nine
                # significance tests and not a multiple-testing claim.
                for horizon, gain in comparisons[name]['mae']['per_horizon_mean_improvement'].items():
                    if gain < -1e-12:
                        horizon_degradations.append(dict(comparison=name, horizon=int(horizon),
                                                         full_mae_improvement=gain))
        passed = all(item['passed'] for item in requirements) and not horizon_degradations
        scientific_status = 'initial_screen_passed' if passed else 'initial_screen_not_passed'
    else:
        passed, scientific_status = False, 'inconclusive'
    public_arms = {name: {key: value for key, value in arm.items() if not key.startswith('_')}
                   for name, arm in arms.items()}
    return dict(schema='native_z_three_arm_comparison_v1',
                engineering_status='complete' if all(arm['engineering_status'] == 'complete'
                                                       for arm in arms.values()) else 'incomplete_or_failed',
                scientific_status=scientific_status, z_initial_screen_passed=passed,
                goal_complete=False, comparison_eligibility=eligibility,
                scope='单一训练片段的后验重建；无泛化、语义动作迁移或 AR rollout 结论',
                metric_scope=dict(training_source_frames=177, evaluated_future_frames=list(range(1,177)),
                                  complete_windows=168, future_exposures=1512,
                                  aggregation='先对同一真实目标帧的重复窗口预测误差取均值，再对176帧等权',
                                  initial_output='主比较只用未来帧1..176；各臂initial_reconstruction单独列h0及frame0，不混入Z初筛',
                                  motion_metric='target-dependent loss weight; camera changes can contribute; not semantic segmentation'),
                bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                               block_length=BLOCK_LENGTH, unit='physical_target_frame_after_overlap_reduction',
                               warning='单clip描述性区间，不将1512次重叠曝光当独立样本，不作总体泛化推断'),
                arms=public_arms, comparisons=comparisons,
                screen_requirements=requirements,
                horizon_rule='对3种shuffle、constant、冻结held基线逐h的full MAE均值均不得恶化；数值容差1e-12；不是显著性检验',
                horizon_degradations=horizon_degradations,
                interpretation=('满足预注册的Z贡献初筛，仍需后续独立验证；不代表最终目标完成。' if passed else
                                '未满足初筛；分别查看公平性阻断、区间及逐h方向，不能用码熵增加替代重建贡献。'))


def fmt(value, places=6):
    return f'{value:.{places}f}' if numeric(value) else '—'


def markdown(summary):
    status = dict(initial_screen_passed='Z 有效性初筛通过',
                  initial_screen_not_passed='Z 有效性初筛未通过', inconclusive='尚不能公平比较')
    lines = ['# 三臂原生 GS / Z 比较', '',
             f"**{status[summary['scientific_status']]}。** 工程状态：`{summary['engineering_status']}`。",
             '这是单片段 TRAIN 后验重建初筛，不代表泛化、AR 预测或最终目标完成。', '',
             '| 臂 | 工程状态 | 完成／计划步数 | 训练全覆盖 | 评测全覆盖 |',
             '|---|---|---:|---|---|']
    for name, arm in summary['arms'].items():
        lines.append(f"| {name} | {arm['engineering_status']} | {arm['steps']} / {arm['planned_steps']} | "
                     f"{arm['training_full_coverage']} | {arm['evaluation_full_coverage']} |")
    lines += ['', '只有三臂步数相同、初始化和已完成采样计划一致、完整覆盖且冻结基线一致，才计算强比较。']
    for reason in summary['comparison_eligibility']['reasons']:
        lines.append(f'- {reason}')
    for name, arm in summary['arms'].items():
        if arm['issues']:
            lines += ['', f"{name} 的缺失／失败证据："]
            lines += [f'- {issue}' for issue in arm['issues']]
    lines += ['', '## 按真实目标帧等权的指标', '',
              '先汇总同一目标帧的重叠窗口误差，再对物理帧 1–176 等权。主表不混入 h0／frame0，'
              '初始重建另列；不把 1512 次曝光当成独立帧。未完整臂的数字仅为已记录部分。', '',
              '| 臂／控制 | 全图 MAE ↓ | ROI MAE ↓ | 运动加权 MAE ↓ | 全图 PSNR ↑ | ROI PSNR ↑ |',
              '|---|---:|---:|---:|---:|---:|']
    for name, arm in summary['arms'].items():
        for control in ('correct', 'adapted_held', 'held_initial', 'frozen_target'):
            data = arm['metrics'].get(control)
            if data:
                values = data['frame_uniform']
                lines.append('| ' + name + '/' + control + ' | ' + ' | '.join(
                    fmt(values.get(key)) for key in METRICS) + ' |')
    lines += ['', '单独的初始重建：']
    for name, arm in summary['arms'].items():
        initial = arm['initial_reconstruction']
        if initial:
            value = initial['metrics']['correct']['frame_uniform']
            frame0 = initial['frame0']['correct']
            lines.append(f"- {name}：h0 的168帧平均全图/ROI MAE "
                         f"{fmt(value['mae'])}/{fmt(value['hand_activity_mae'])}；"
                         f"物理frame0为 {fmt(frame0['mae'])}/{fmt(frame0['hand_activity_mae'])}。")
        else:
            lines.append(f'- {name}：初始重建完整证据缺失，不能补出 frame0 质量。')
    lines += ['', '## Z 贡献的配对比较', '',
              '改善量 = 对照 MAE − partial 正确码 MAE，正值更好。连续时间移动块长度 16，'
              '种子 20261031，2000 次重采样；先按物理目标帧聚合，再重采样。区间只描述本 TRAIN clip。', '',
              '| 对照 | 指标 | MAE 改善 | 95% 区间 | 下界 > 0 |',
              '|---|---|---:|---|---|']
    if not summary['comparisons']:
        lines += ['| 未执行 | 公平比较前提不满足 | — | — | — |']
    for comparison, metrics in summary['comparisons'].items():
        for metric, values in metrics.items():
            lines.append(f"| {comparison} | {metric} | {fmt(values['mean_improvement'])} | "
                         f"[{fmt(values['ci95_low'])}, {fmt(values['ci95_high'])}] | {values['positive_interval']} |")
    lines += ['', '## 正确码逐 horizon 指标', '',
              '| 臂 | h | 全图 MAE ↓ | ROI MAE ↓ | 运动加权 MAE ↓ | 全图 PSNR ↑ | ROI PSNR ↑ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, arm in summary['arms'].items():
        data = arm['metrics'].get('correct')
        if data:
            for horizon, values in data['per_horizon'].items():
                lines.append(f'| {name} | {horizon} | ' + ' | '.join(fmt(values.get(key)) for key in METRICS) + ' |')
    lines += ['', summary['horizon_rule'] + '。']
    for item in summary['horizon_degradations']:
        lines.append(f"- {item['comparison']}，h{item['horizon']}：全图 MAE 改善 "
                     f"{fmt(item['full_mae_improvement'])}（负值表示恶化）。")
    lines += ['', summary['interpretation'], '',
              '若 partial 只优于 frozen，却不优于 constant 或同 h 置换，收益不能归因于 Z 的未来信息。'
              '条件熵、不同 ID 数量和 bundle 数量是辅助诊断，不替代上述图像证据。', '',
              '来源文件、hash、各臂原始完成状态、全部控制汇总及检查细节见同目录 `SUMMARY.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('实验根目录不存在；不会创建新的实验目录')
    summary = build_summary(root)
    for name, content in [('SUMMARY.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)),
                          ('SUMMARY.md', markdown(summary))]:
        destination = root / name
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(destination)
    print(json.dumps(dict(engineering_status=summary['engineering_status'],
                          scientific_status=summary['scientific_status'],
                          goal_complete=False, summary=str(root / 'SUMMARY.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
