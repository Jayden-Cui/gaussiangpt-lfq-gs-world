"""CPU-only, source-bound comparison on one cross-source posterior clip.

No Torch, training, checkpoint selection, GPU, or automatic experiment retry.
The old bootstrap algorithm is reused unchanged; its new scope is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ARMS = ('z_rgb', 'constant_rgb', 'z_delta', 'constant_delta')
CONTROLS = ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'blinded',
            'adapted_held', 'teacher_state_reference', 'frozen_teacher_reference')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr', 'motion_mae')
ERRORS = ('mae', 'workspace_roi_mae', 'motion_mae')
KEYS = {(s, h) for s in range(168) for h in range(1, 10)}
CLIP = 'buildingblock_tower_14_0'
SCOPE = ('Single project cross-source clip posterior reconstruction; historical exposure unknown; '
         'not official holdout, population generalization, action semantics or AR proof')
MAIN_MANIFEST = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
POSTFIT_MANIFEST = '1669fe20188831af3cb6cae31dd1f8e8a2419cf8ed72010ef1130f23bc492f10'
PREPARATION_MANIFEST = '8c9381593b2a7bce009e38d73c07f9de1b0ab1a411cfa7feb9ce6729d268ef57'
PREPARATION_BINDING = 'a422de97bdfeb7b9dcae47a00478f43ddb63185b9b352b85e221cb914c7d1e8d'


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SourceCheck:
    """Deduplicate large read-only inputs, rejecting changes during comparison."""
    def __init__(self):
        self.checked = {}

    def verify(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if path in self.checked:
            old, digest = self.checked[path]
            require(identity == old, 'Input changed during comparison: ' + str(path))
        else:
            digest = sha(path)
            after = path.stat()
            require(identity == (after.st_size, after.st_mtime_ns), 'Input changed while hashing')
            self.checked[path] = (identity, digest)
        require(expected is None or (is_digest(expected) and digest == expected),
                'Source hash mismatch: ' + str(path))
        return digest

    def unchanged(self):
        for path, (identity, _) in self.checked.items():
            stat = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == identity, 'Input changed: ' + str(path))


def validate_metrics(values, delta=False, initial=False):
    keys = ERRORS if delta else METRICS
    require(isinstance(values, dict) and set(values) == set(keys), 'Wrong metric keys')
    for key in keys:
        value = values[key]
        if key == 'motion_mae' and value is None:
            continue
        require(finite(value), 'Nonfinite or invalid metric: ' + key)
        require('psnr' in key or value >= 0, 'Negative error metric')
    if initial:
        require(values['motion_mae'] is None, 'h0 has zero motion weight; it must be null')


def validate_ids(ids):
    require(isinstance(ids, list) and len(ids) == 9
            and all(type(v) is int and 0 <= v < 65536 for v in ids),
            'Expected nine integer 16-bit LFQ IDs')


def validate_rows(rows, initial=False):
    expected = {(s, 0) for s in range(168)} if initial else KEYS
    seen, bundles = set(), {}
    for row in rows:
        s, h, end = row.get('start'), row.get('horizon'), row.get('end')
        require(all(type(v) is int for v in (s, h, end)) and (s, h) in expected
                and end == s + h and (s, h) not in seen, 'Invalid/duplicate target identity')
        require(row.get('clip', CLIP) == CLIP, 'Wrong target clip')
        seen.add((s, h))
        for control in CONTROLS:
            validate_metrics(row.get(control), initial=initial)
        if initial:
            continue
        require(isinstance(row.get('delta'), dict) and set(row['delta']) == set(CONTROLS), 'Missing delta controls')
        for control in CONTROLS:
            validate_metrics(row['delta'][control], delta=True)
        nulls = {row[c]['motion_mae'] is None for c in CONTROLS}
        nulls.update(row['delta'][c]['motion_mae'] is None for c in CONTROLS)
        require(len(nulls) == 1, 'Motion-weight validity depends on target, never control')
        ids = (row.get('motion_ids'), row.get('blinded_motion_ids'))
        for values in ids:
            validate_ids(values)
        require(s not in bundles or bundles[s] == ids, 'IDs changed within the same window')
        bundles[s] = ids
    require(seen == expected, 'Incomplete real-frame coverage')
    return bundles


def reduce_rows(rows, control, delta=False):
    """Report valid-count-aware means, first reducing overlapping physical targets."""
    keys = ERRORS if delta else METRICS
    values = lambda row: row['delta'][control] if delta else row[control]
    frames, horizons = {}, {}
    for row in rows:
        frames.setdefault(row['end'], []).append(row)
        horizons.setdefault(row['horizon'], []).append(row)

    def reduce_group(group):
        result, counts = {}, {}
        for key in keys:
            selected = [values(row)[key] for row in group if values(row)[key] is not None]
            result[key] = statistics.fmean(selected) if selected else None
            counts[key] = len(selected)
        return result, counts

    per_frame = {frame: reduce_group(group)[0] for frame, group in frames.items()}
    frame_mean, valid_frames = {}, {}
    for key in keys:
        selected = [value[key] for value in per_frame.values() if value[key] is not None]
        frame_mean[key] = statistics.fmean(selected) if selected else None
        valid_frames[key] = len(selected)
    exposure, valid_exposures = reduce_group(rows)
    return dict(frame_uniform=frame_mean, exposure_mean=exposure,
                per_horizon={str(h): reduce_group(group)[0] for h, group in sorted(horizons.items())},
                valid_physical_frames=valid_frames, valid_exposures=valid_exposures,
                observations=len(rows), unique_physical_frames=len(frames))


def paired(left_rows, right_rows, control, bootstrap, delta=False, metrics=ERRORS):
    left = {(r['start'], r['horizon']): r for r in left_rows}
    right = {(r['start'], r['horizon']): r for r in right_rows}
    require(set(left) == set(right) == KEYS, 'Paired comparisons require all matching windows')
    results = {}
    for metric in metrics:
        frames, horizons = {}, {}
        valid_pairs = 0
        for key in sorted(KEYS):
            a = left[key]['delta']['correct'] if delta else left[key]['correct']
            b = right[key]['delta'][control] if delta else right[key][control]
            av, bv = a[metric], b[metric]
            require((av is None) == (bv is None), 'Different valid-target masks across paired arms')
            if av is None:
                continue
            require(finite(av) and finite(bv), 'Nonfinite paired values')
            difference = bv - av
            frames.setdefault(left[key]['end'], []).append(difference)
            horizons.setdefault(key[1], []).append(difference)
            valid_pairs += 1
        frame_means = {frame: statistics.fmean(values) for frame, values in frames.items()}
        if sorted(frame_means) == list(range(1, 177)):
            value = dict(bootstrap(frame_means))
            value.update(available=True, interval='paired non-circular moving-block percentile; descriptive single cross-source posterior clip')
        else:
            value = dict(available=False, positive_interval=False, ci95_low=None, ci95_high=None,
                         mean_improvement=statistics.fmean(frame_means.values()) if frame_means else None,
                         physical_frames=len(frame_means), reason='Missing physical frames; no zero fill or compressed-time bootstrap',
                         missing_physical_frames=sorted(set(range(1,177)).difference(frame_means)))
        value.update(valid_exposures=valid_pairs, per_horizon_mean_improvement={
            str(h): statistics.fmean(horizons[h]) if h in horizons else None for h in range(1,10)})
        results[metric] = value
    return results


def evaluation_contract(evaluation, arm, selection, expected, target_context=None):
    """Pure metadata gate, shared with CPU rejection tests."""
    require(evaluation.get('schema') == 'gs_tower14_transfer_evaluation_v1'
            and evaluation.get('status') == 'complete' and evaluation.get('arm') == arm, 'Invalid EVALUATION identity')
    selected = selection['checkpoints'][arm]
    require(evaluation.get('selected_step') == selection['selected_step']
            and evaluation.get('checkpoint_path') == selected['path']
            and evaluation.get('checkpoint_sha256') == selected['sha256'], 'Different selected checkpoint/step')
    for key, value in expected.items():
        require(evaluation.get(key) == value, 'Evaluation source mismatch: ' + key)
    for key, value in dict(windows=168, future_exposures=1512, unique_future_frames=176, initial_frames=168,
                           optimizer_updates=0, all_parameters_frozen=True, goal_complete=False).items():
        require(type(evaluation.get(key)) is type(value) and evaluation[key] == value, 'Invalid completed scope: ' + key)
    require('scientific_transfer_passed' in evaluation and evaluation['scientific_transfer_passed'] is None,
            'Evaluation must explicitly leave scientific transfer success unproven')
    require(evaluation.get('clip') == CLIP and evaluation.get('resolution_hw') == [252,448]
            and evaluation.get('workspace_roi_xyxy') == [67,38,380,227], 'Wrong target clip or ROI/resolution')
    require(evaluation.get('all_initial_latent_points_retained') is True
            and evaluation.get('native_internal_hard_pruning_retained') is True,
            'Original GS/support guards were not retained')
    require(finite(evaluation.get('gpu_peak_gib')) and 0 <= evaluation['gpu_peak_gib'] <= 48,
            'Invalid native memory guard result')
    require(evaluation.get('code_mode') == ('constant' if arm.startswith('constant_') else 'posterior'),
            'Wrong posterior/constant model mode')
    for family in ('model_state', 'codec_state'):
        before, after = evaluation.get(family+'_sha256_before'), evaluation.get(family+'_sha256_after')
        require(is_digest(before) and before == after, 'Model parameters or buffers changed: ' + family)
    values = [evaluation.get('output_std_sha256_'+part) for part in ('source','before','after')]
    require(is_digest(values[0]) and len(set(values)) == 1, 'TRAIN output units changed or were refit')
    inputs = evaluation.get('target_input_sha256')
    require(isinstance(inputs, dict) and inputs, 'Missing target input binding')
    digest = hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(evaluation.get('target_context_sha256') == digest
            and (target_context is None or target_context == digest), 'Different target data context')
    return digest


def load_evidence(root):
    check = SourceCheck()
    manifest = read(root/'CODE_MANIFEST.json')
    cfg = read(root/'protocol.json')
    require(cfg.get('schema') == 'gs_neoverse_cross_source_eval_v1', 'Wrong comparison protocol')
    main, preparation = Path(cfg['main_root']).resolve(), Path(cfg['preparation_root']).resolve()
    allowed = (root, main, preparation, main.parent/'gaussiangpt_neoverse_native_z_20261031_v6',
               main.parent/'gaussiangpt_neoverse_step_acceleration_20261031_v2')
    for relative, digest in manifest.items():
        path = (root/relative).resolve()
        require(not Path(relative).is_absolute() and any(path.is_relative_to(base) for base in allowed), 'Unadmitted source path')
        check.verify(path,digest)
    for name in ('compare_transfer.py','protocol.json'):
        require(name in manifest, 'Comparison source is not bound')
    require(cfg.get('arms') == list(ARMS) and cfg.get('controls') == list(CONTROLS), 'Wrong fixed arms/controls')
    for key,value in dict(frames=177,windows=168,horizons=9,frame_gap=1,resolution_hw=[252,448],
                           workspace_roi_xyxy=[67,38,380,227],max_seconds_per_arm=2400).items():
        require(cfg.get(key)==value,'Fixed comparison scope changed: '+key)
    fixed = dict(main_manifest_sha256=MAIN_MANIFEST, postfit_manifest_sha256=POSTFIT_MANIFEST,
                 preparation_manifest_sha256=PREPARATION_MANIFEST, preparation_binding_sha256=PREPARATION_BINDING)
    for key, value in fixed.items():
        require(cfg.get(key) == value, 'Wrong pinned input manifest: ' + key)
    expected = dict(**fixed, code_manifest_sha256=check.verify(root/'CODE_MANIFEST.json'),
                    protocol_sha256=check.verify(root/'protocol.json'))
    check.verify(main/'CODE_MANIFEST.json', MAIN_MANIFEST)
    check.verify(main/'POSTFIT_MANIFEST.json', POSTFIT_MANIFEST)
    check.verify(preparation/'CODE_MANIFEST.json', PREPARATION_MANIFEST)
    check.verify(preparation/'input_binding.json', PREPARATION_BINDING)
    require(not (main/'queue/FAILED.json').exists() and not (preparation/'queue/FAILED.json').exists(), 'Upstream engineering failure')
    main_done, preparation_done = read(main/'queue/DONE.json'), read(preparation/'queue/DONE.json')
    require(main_done.get('status') == preparation_done.get('status') == 'complete', 'Upstream not complete')
    selection = read(main/'SELECTION.json')
    require(selection.get('status') == 'selected' and selection.get('selected_step') in (200,300,400)
            and selection.get('selected_step') == main_done.get('selected_step')
            and selection.get('code_manifest_sha256') == MAIN_MANIFEST
            and set(selection.get('checkpoints',{})) == set(ARMS), 'Invalid exact common selection')
    for path in (main/'SELECTION.json',main/'queue/DONE.json',preparation/'queue/DONE.json'):
        check.verify(path)
    check.verify(preparation/'SUMMARY.json',preparation_done['summary_sha256'])
    bootstrap_path = main.parent/'gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py'
    names = {(root/k).resolve(): v for k,v in manifest.items()}
    require(bootstrap_path in names, 'Bootstrap helper must be explicitly pinned')
    check.verify(bootstrap_path,names[bootstrap_path])
    spec = importlib.util.spec_from_file_location('_bound_old_bootstrap',bootstrap_path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    require((helper.BLOCK_LENGTH,helper.BOOTSTRAP_REPLICATES,helper.BOOTSTRAP_SEED)==(16,2000,20261031), 'Bootstrap protocol changed')
    arms, target_context, baseline = {}, None, None
    for arm in ARMS:
        directory=root/'evaluations'/arm
        require(not (directory/'FAILED.json').exists(), 'Arm failed: '+arm)
        evaluation=read(directory/'EVALUATION.json')
        digest=evaluation_contract(evaluation,arm,selection,expected,target_context)
        target_context=digest
        for key in ('main_input_sha256','target_input_sha256'):
            hashes=evaluation.get(key)
            require(isinstance(hashes,dict) and hashes,'Missing source hashes: '+key)
            for path, value in hashes.items():
                require(Path(path).is_absolute(),'Input source must be absolute')
                check.verify(path,value)
        require(evaluation['main_input_sha256'].get(str(main/'SELECTION.json')) == check.verify(main/'SELECTION.json'), 'Missing selection binding')
        require(evaluation['target_input_sha256'].get(str(preparation/'SUMMARY.json')) == check.verify(preparation/'SUMMARY.json'), 'Missing prepared summary binding')
        selected=selection['checkpoints'][arm]
        require(Path(selected['path']).resolve()==main/arm/f"step_{selection['selected_step']:04d}.pt",'Wrong checkpoint location')
        check.verify(selected['path'],selected['sha256'])
        artifacts=evaluation.get('artifact_sha256')
        require(isinstance(artifacts,dict) and {'targets.jsonl','initial_reconstruction.json','actions.pt'}.issubset(artifacts),'Incomplete artifacts')
        for relative,value in artifacts.items():
            path=(directory/relative).resolve()
            require(not Path(relative).is_absolute() and path.is_relative_to(directory),'Artifact outside arm directory')
            check.verify(path,value)
        for name,key in (('targets.jsonl','targets_sha256'),('initial_reconstruction.json','initial_reconstruction_sha256'),('actions.pt','actions_sha256')):
            require(evaluation.get(key)==artifacts[name],'Duplicate artifact hash mismatch')
        rows=[json.loads(line) for line in (directory/'targets.jsonl').read_text().splitlines() if line.strip()]
        bundles=validate_rows(rows)
        initial=read(directory/'initial_reconstruction.json')
        require(initial.get('unique_initial_frames')==168,'Wrong initial coverage')
        validate_rows(initial.get('rows',[]),initial=True)
        for family, is_delta in (('metrics',False),('delta_metrics',True)):
            require(set(evaluation.get(family,{})) == set(CONTROLS),'Evaluation aggregate controls missing')
            for control in CONTROLS:
                calculated=reduce_rows(rows,control,delta=is_delta)
                reported=evaluation[family][control]
                require(reported.get('valid_physical_frames')==calculated['valid_physical_frames'],'Aggregate valid counts differ')
                require(set(reported.get('frame_uniform',{}))==set(calculated['frame_uniform']),'Aggregate metric keys differ')
                for key,value in calculated['frame_uniform'].items():
                    observed=reported['frame_uniform'][key]
                    require((observed is None and value is None) or
                            (finite(observed) and value is not None and math.isclose(observed,value,rel_tol=1e-10,abs_tol=1e-12)),
                            'Evaluation aggregate differs from bound rows')
        if arm.startswith('constant_'):
            code=evaluation.get('constant_id')
            require(type(code) is int and 0<=code<65536,'Constant ID missing')
            require(all(a==b==[code]*9 for a,b in bundles.values()),'Constant control IDs are not constant')
            for row in rows:
                require(all(row[c]==row['correct'] and row['delta'][c]==row['delta']['correct']
                            for c in ('shuffle42','shuffle84','shuffle126','blinded')),'Constant render reuse differs')
        references={(r['start'],r['horizon']):(r['frozen_teacher_reference'],r['delta']['frozen_teacher_reference']) for r in rows}
        require(baseline is None or baseline==references,'Frozen cached references differ between arms')
        baseline=references
        arms[arm]=dict(evaluation=evaluation,evaluation_sha256=check.verify(directory/'EVALUATION.json'),
                       rows=rows,initial_rows=initial['rows'])
    return cfg,selection,arms,helper.moving_block_bootstrap,check,target_context


def summarize(arms, bootstrap):
    comparisons, screens = {}, {}
    for mode in ('rgb','delta'):
        name,constant='z_'+mode,'constant_'+mode
        rowset=arms[name]['rows'];requirements=[];degradations=[];delta_requirements=[]
        for control in ('shuffle42','shuffle84','shuffle126','blinded','adapted_held','constant'):
            other=arms[constant]['rows'] if control=='constant' else rowset
            other_control='correct' if control=='constant' else control
            key=name+'_vs_'+control
            comparisons[key]=paired(rowset,other,other_control,bootstrap)
            comparisons[key+'_changes']=paired(rowset,other,other_control,bootstrap,delta=True)
            required=ERRORS if control in ('shuffle42','shuffle84','shuffle126','constant') else ('mae','workspace_roi_mae') if control=='adapted_held' else ()
            for metric in required:
                result=comparisons[key][metric]
                requirements.append(dict(comparison=key,metric=metric,passed=result['available'] and result['positive_interval'],
                                         ci95_low=result['ci95_low'],mean_improvement=result['mean_improvement']))
            if required:
                for h,gain in comparisons[key]['mae']['per_horizon_mean_improvement'].items():
                    if gain is None or gain < -1e-12:
                        degradations.append(dict(comparison=key,horizon=int(h),full_mae_improvement=gain))
            if control in ('shuffle42','shuffle84','shuffle126','constant'):
                for metric in ('workspace_roi_mae','motion_mae'):
                    result=comparisons[key+'_changes'][metric]
                    delta_requirements.append(dict(comparison=key+'_changes',metric=metric,
                        passed=result['available'] and result['positive_interval'],ci95_low=result['ci95_low']))
        require(len(requirements)==14 and len(delta_requirements)==8,'Screen definition changed')
        screens[mode]=dict(ordinary_passed=all(r['passed'] for r in requirements) and not degradations,
                          ordinary_requirements=requirements,ordinary_horizon_degradations=degradations,
                          delta_information_passed=all(r['passed'] for r in delta_requirements),delta_requirements=delta_requirements)
    comparisons['z_delta_vs_z_rgb']=paired(arms['z_delta']['rows'],arms['z_rgb']['rows'],'correct',bootstrap)
    comparisons['z_delta_vs_z_rgb_changes']=paired(arms['z_delta']['rows'],arms['z_rgb']['rows'],'correct',bootstrap,delta=True)
    return comparisons,screens


def main(root):
    root=root.resolve()
    require(not (root/'SUMMARY.json').exists() and not (root/'SUMMARY.md').exists(),'Preserve existing summary; no overwrite')
    cfg,selection,arms,bootstrap,check,target_context=load_evidence(root)
    comparisons,screens=summarize(arms,bootstrap)
    public={arm:dict(evaluation_sha256=value['evaluation_sha256'],checkpoint=selection['checkpoints'][arm],
        metrics={c:reduce_rows(value['rows'],c) for c in CONTROLS},
        delta_metrics={c:reduce_rows(value['rows'],c,delta=True) for c in CONTROLS},
        initial_metrics={c:reduce_rows(value['initial_rows'],c) for c in CONTROLS}) for arm,value in arms.items()}
    summary=dict(schema='gs_tower14_transfer_comparison_v1',engineering_status='complete',comparison_eligible=True,
        selected_step=selection['selected_step'],checkpoint_selection=selection,clip=CLIP,scope=SCOPE,
        target_context_sha256=target_context,arms=public,comparisons=comparisons,z_information_screens=screens,
        goal_complete=False,scientific_transfer_passed=None,
        code_manifest_sha256=check.verify(root/'CODE_MANIFEST.json'),protocol_sha256=check.verify(root/'protocol.json'),
        source_sha256={str(p):v[1] for p,v in check.checked.items()},
        roi=dict(name='fixed central workspace crop, not hand segmentation',xyxy=[67,38,380,227]),
        aggregation='Mean valid overlapping observations per physical frame, then equally weight physical frames1..176; h0 separate',
        bootstrap=dict(block_length=16,replicates=2000,seed=20261031,scope='Descriptive single cross-source posterior clip; incomplete physical series receives no CI'),
        references='Teacher-state references show codec quality with true future states; neither is a theoretical upper bound',
        action_semantics='Nine-code whole-window posterior bundles, periodic real-initial reset; not independent AR actions')
    check.unchanged()
    lines=['# 单一跨来源片段的冻结模型评测','',f"共同 checkpoint：{selection['selected_step']} 步；片段 `{CLIP}`。",'',
           '这是后验窗口重建，初态来自真实状态；没有微调，不是 AR rollout。历史接触情况未知，不称官方独立测试集。','',
           '| 分支 | 全图 MAE ↓ | 工作区 MAE ↓ | 运动加权 MAE ↓ | PSNR ↑ | 全图差分 MAE ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(value):return '—' if value is None else f'{value:.6f}'
    for arm in ARMS:
        m=public[arm]['metrics']['correct']['frame_uniform'];d=public[arm]['delta_metrics']['correct']['frame_uniform']
        lines.append('| '+arm+' | '+' | '.join(fmt(v) for v in (m['mae'],m['workspace_roi_mae'],m['motion_mae'],m['psnr'],d['mae']))+' |')
    for mode in ('rgb','delta'):
        screen=screens[mode]
        lines += ['',f"{mode}：普通信息贡献初筛 {screen['ordinary_passed']}；差分信息检查 {screen['delta_information_passed']}。",
                  f"普通置信区间通过 {sum(r['passed'] for r in screen['ordinary_requirements'])}/14；逐h全图恶化或缺失 {len(screen['ordinary_horizon_degradations'])} 项。"]
    lines += ['','全部指标先按176个物理未来帧归并；1512次重叠曝光不是1512个独立样本。零运动权重记null，不补0；缺物理帧的运动指标不计算连续块置信区间。',
              '工作区是固定中央crop，不能解释成新片段的手部分割。真实未来状态的两种codec重建只作质量参考，不是理论上限。',
              '完整正确/打乱/盲化/保持初态/固定码配对效应、区间和逐h结果见 SUMMARY.json。通过本初筛也不证明动作语义、总体泛化或 AR；目标仍未完成。']
    with (root/'SUMMARY.json').open('x',encoding='utf-8') as stream:json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
    with (root/'SUMMARY.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(engineering_status='complete',selected_step=selection['selected_step'],goal_complete=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
