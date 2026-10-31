"""Physical-frame comparisons for the predeclared four-arm continuation."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from pipeline import read, sha, verify, write


METRICS = ('mae', 'hand_activity_mae', 'motion_mae')
DELTA = dict(mae='delta_mae', hand_activity_mae='hand_activity_delta_mae',
             motion_mae='motion_delta_mae')


def delta_rows(rows):
    result = []
    for row in rows:
        item = {k:row[k] for k in ('start', 'end', 'horizon')}
        for control in ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'adapted_held', 'blinded'):
            values = row['delta'][control]
            item[control] = {key:values[field] for key,field in DELTA.items()}
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in item[control].values()):
                raise RuntimeError('Nonfinite or missing delta metrics')
        result.append(item)
    return result


def main(root):
    manifest = verify(root)
    cfg, selection = read(root/'protocol.json'), read(root/'SELECTION.json')
    if selection['code_manifest_sha256'] != manifest or selection['protocol_sha256'] != sha(root/'protocol.json'):
        raise RuntimeError('Selection belongs to different sources')
    source = Path(cfg['source_root'])
    for rel, digest in read(source/'CODE_MANIFEST.json').items():
        if sha(source/rel) != digest:
            raise RuntimeError('Original experiment sources changed')
    if sha(source/'CODE_MANIFEST.json') != cfg['source_manifest_sha256']:
        raise RuntimeError('Original source manifest changed')
    spec = importlib.util.spec_from_file_location('original_comparison', source/'code/compare_arms.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    selected = selection['selected_step']
    if selected not in cfg['train']['save_steps']:
        raise RuntimeError('Unplanned evaluation step')
    arms, delta, initials, public = {}, {}, {}, {}
    plans = []
    references = {}
    for arm in cfg['arms']:
        path = root/'evaluations'/f'step_{selected:04d}'/arm
        evaluation = read(path/'EVALUATION.json')
        expected_bindings = {
            'targets_sha256': path/'targets.jsonl',
            'initial_reconstruction_sha256': path/'initial_reconstruction.json',
            'actions_sha256': path/'actions.pt',
            'protocol_sha256': root/'protocol.json',
            'source_record_sha256': root/arm/'SOURCE.json',
            'plan_sha256': root/arm/'PLAN.json',
            'train_done_sha256': root/arm/'TRAIN_DONE.json',
        }
        if any(evaluation.get(key) != sha(value) for key,value in expected_bindings.items()):
            raise RuntimeError('Evaluation artifact/source hash mismatch: '+arm)
        source_arm = cfg['arms'][arm]['source_arm']
        if evaluation.get('source_checkpoint_sha256') != cfg['source_checkpoints'][source_arm]['sha256']:
            raise RuntimeError('Continuation source checkpoint mismatch: '+arm)
        # The runner binds the exact saved checkpoint, independently of final.
        checkpoint = selection['checkpoints'][arm]
        if (evaluation.get('status') != 'complete'
                or evaluation.get('checkpoint_sha256') != checkpoint['sha256']
                or sha(Path(checkpoint['path'])) != checkpoint['sha256']
                or evaluation.get('windows') != 168
                or evaluation.get('real_future_exposures') != 1512
                or evaluation.get('unique_future_frames') != 176):
            raise RuntimeError('Evaluation checkpoint/coverage mismatch: '+arm)
        step = evaluation.get('selected_step', evaluation.get('checkpoint_step'))
        if step != selected or evaluation.get('code_manifest_sha256') != manifest:
            raise RuntimeError('Evaluation step/source mismatch: '+arm)
        plan = read(root/arm/'PLAN.json')['starts'][:selected]
        plans.append(plan)
        if hashlib.sha256(json.dumps(plan,separators=(',', ':')).encode()).hexdigest() != selection['plan_prefix_sha256']:
            raise RuntimeError('Selected plan mismatch: '+arm)
        issues = []
        rows = comparison.read_rows(path/'targets.jsonl', issues)
        if issues or len(rows) != 1512 or {(r['start'],r['horizon']) for r in rows} != comparison.REQUIRED_KEYS:
            raise RuntimeError('Incomplete or invalid target rows: '+arm+' '+str(issues[:3]))
        if any(not isinstance(r.get('blinded'),dict) or
               any(not comparison.numeric(r['blinded'].get(m)) for m in comparison.METRICS)
               for r in rows):
            raise RuntimeError('Missing/nonfinite blind-future control: '+arm)
        arms[arm], delta[arm] = rows, delta_rows(rows)
        initial = read(path/'initial_reconstruction.json')['rows']
        if len(initial) != 168 or sorted(r['start'] for r in initial) != list(range(168)):
            raise RuntimeError('Missing initial outputs: '+arm)
        if any(r.get('end') != r['start'] or r.get('horizon') != 0 for r in initial):
            raise RuntimeError('Invalid initial indexing: '+arm)
        initials[arm] = {metric:sum(r['correct'][metric] for r in initial)/168
                         for metric in ('mae','hand_activity_mae','psnr')}
        public[arm] = dict(metrics={c:comparison.reduce_rows(rows,c) for c in (*comparison.CONTROLS,'blinded')},
                           delta={c:comparison.reduce_rows(delta[arm],c) for c in delta[arm][0]
                                  if c not in ('start','end','horizon')},
                           initial=initials[arm],
                           initial_blinded={metric:sum(r['blinded'][metric] for r in initial)/168
                                            for metric in ('mae','hand_activity_mae','psnr')},
                           blinded_id_statistics=evaluation['blinded_id_statistics'], checkpoint=checkpoint,
                           targets_sha256=sha(path/'targets.jsonl'),
                           evaluation_sha256=sha(path/'EVALUATION.json'))
        references[arm] = {(r['start'],r['horizon']):{c:r[c] for c in ('held_initial','frozen_target')} for r in rows}
    if any(p != plans[0] for p in plans):
        raise RuntimeError('Completed sample plans differ')
    if any(r != next(iter(references.values())) for r in references.values()):
        raise RuntimeError('Frozen per-target reference metrics differ')

    screens, comparisons = {}, {}
    for mode in ('rgb','delta'):
        z, constant = 'z_'+mode, 'constant_'+mode
        requirements, deteriorations = [], []
        specs = [(c, arms[z], c, METRICS) for c in ('shuffle42','shuffle84','shuffle126')]
        specs += [('constant',arms[constant],'correct',METRICS),
                  ('frozen_held',arms[z],'held_initial',METRICS[:2])]
        for name, other, control, metrics in specs:
            key = z+'_vs_'+name
            values = comparison.paired_comparison(arms[z], other, control, metrics)
            comparisons[key] = values
            for metric, item in values.items():
                requirements.append(dict(comparison=key, metric=metric,
                                         passed=item['positive_interval'], **item))
            deteriorations += [dict(comparison=key,horizon=int(h),full_mae_improvement=gain)
                               for h,gain in values['mae']['per_horizon_mean_improvement'].items()
                               if gain < -1e-12]
        screens[mode] = dict(passed=all(x['passed'] for x in requirements) and not deteriorations,
                            requirements=requirements,horizon_degradations=deteriorations)
        comparisons[z+'_delta_vs_constant'] = comparison.paired_comparison(delta[z],delta[constant],'correct',METRICS)
        comparisons[z+'_vs_adapted_held'] = comparison.paired_comparison(arms[z],arms[z],'adapted_held',METRICS)
        comparisons[z+'_vs_blinded'] = comparison.paired_comparison(arms[z],arms[z],'blinded',METRICS)
        comparisons[z+'_changes_vs_blinded'] = comparison.paired_comparison(delta[z],delta[z],'blinded',METRICS)
    comparisons['z_delta_vs_z_rgb'] = comparison.paired_comparison(arms['z_delta'],arms['z_rgb'],'correct',METRICS)
    comparisons['z_delta_vs_z_rgb_changes'] = comparison.paired_comparison(delta['z_delta'],delta['z_rgb'],'correct',METRICS)
    comparisons['constant_delta_vs_constant_rgb_changes'] = comparison.paired_comparison(delta['constant_delta'],delta['constant_rgb'],'correct',METRICS)
    ordinary = comparisons['z_delta_vs_z_rgb']
    changes = comparisons['z_delta_vs_z_rgb_changes']
    checks = dict(delta_roi_positive_interval=changes['hand_activity_mae']['positive_interval'],
                  delta_motion_positive_interval=changes['motion_mae']['positive_interval'],
                  full_mae_nonworse=ordinary['mae']['mean_improvement'] >= 0,
                  roi_mae_nonworse=ordinary['hand_activity_mae']['mean_improvement'] >= 0,
                  initial_mae_nonworse=initials['z_delta']['mae'] <= initials['z_rgb']['mae'],
                  matched_constant_and_shuffle_screen=screens['delta']['passed'])
    summary = dict(schema='gs_temporal_four_arm_comparison_v7', engineering_status='complete',
                   comparison_eligible=True,selected_step=selected,arms=public,
                   z_contribution_screens=screens,comparisons=comparisons,
                   delta_hypothesis_checks=checks,delta_hypothesis_supported=all(checks.values()),
                   goal_complete=False,scope='Single TRAIN clip posterior GS reconstruction; no action transfer or AR proof',
                   checkpoint_selection=selection,code_manifest_sha256=manifest,
                   bootstrap='Original physical-frame moving block method;16 frames,2000 draws,seed20261031',
                   caution='A screen pass is not the active research objective. Inspect motion videos and validate independent data before success claims.')
    write(root/'SUMMARY.json',summary)
    lines=['# 四臂继续训练与时间差分实验','',f'统一评测步数：{selected}。仅单个 TRAIN 片段；目标尚未完成。','',
           '| 分支 | PSNR | 全图 MAE | ROI MAE | 运动加权 MAE | 全图差分 MAE |','|---|---:|---:|---:|---:|---:|']
    for arm,x in public.items():
        m=x['metrics']['correct']['frame_uniform'];d=x['delta']['correct']['frame_uniform']
        lines.append(f"| {arm} | {m['psnr']:.4f} | {m['mae']:.6f} | {m['hand_activity_mae']:.6f} | {m['motion_mae']:.6f} | {d['mae']:.6f} |")
    lines += ['',f"原损失 Z 初筛：{screens['rgb']['passed']}；差分损失 Z 初筛：{screens['delta']['passed']}。",
              '',f'差分机制全部预注册条件通过：{all(checks.values())}。', '',
              '统计先合并同一物理目标帧的重复窗口预测，再对176帧等权；区间仅描述当前片段。',
              '完整效应、区间、逐h和源文件hash见 SUMMARY.json。']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root.resolve())
