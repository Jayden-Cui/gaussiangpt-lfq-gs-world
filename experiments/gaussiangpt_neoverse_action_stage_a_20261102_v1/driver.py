"""Run the fixed-decoder free-code diagnostic, evaluation, and cold replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from stage_context import ROOT,setup,setup_gpu,load_codes,load_decoder,dataset,source_record,assert_frozen,read,write,sha,explicit
from sampling import audit


def merge_sweep(sweep,c,helper,orig,p,paths,receipt):
    import torch
    from merge import merge_parts
    parts=[];hashes={}
    for shard in (0,1):
        folder=ROOT/f'shard_{shard}';checkpoint=folder/f'sweep_{sweep:02d}.pt'
        meta=read(folder/f'sweep_{sweep:02d}.json')
        if meta['status']!='complete' or sha(checkpoint)!=meta['checkpoint_sha256']:
            raise RuntimeError('Incomplete or changed free-code checkpoint')
        item=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (item['schema']!='fixed_decoder_free_codes_shard_v1' or item['shard']!=shard
                or item['sweep']!=sweep or item['starts']!=c['shards'][str(shard)]
                or item['source']['decoder_digest']!=c['decoder_tensor_digest']
                or item['source']['baseline_codes_sha256']!=c['bindings']['codes']):
            raise RuntimeError('Shard checkpoint provenance differs')
        parts.append((item['starts'],item['codes']));hashes[str(shard)]=meta['checkpoint_sha256']
    merged=merge_parts(parts)
    folder=ROOT/f'sweep_{sweep:02d}';folder.mkdir()
    path=folder/'merged_codes.pt'
    with path.open('xb') as stream:
        torch.save(dict(schema='fixed_decoder_free_codes_merged_v1',sweep=sweep,starts=c['starts'],
            codes=merged,decoder_digest=c['decoder_tensor_digest'],baseline_codes_sha256=c['bindings']['codes'],
            shard_checkpoint_sha256=hashes),stream)
    replay=torch.load(path,map_location='cpu',weights_only=True)
    if not torch.equal(replay['codes'],merged):raise RuntimeError('Merged codes did not round-trip')
    write(folder/'MERGED.json',dict(status='complete',sweep=sweep,shape=list(merged.shape),
          sha256=sha(path),shard_checkpoint_sha256=hashes,decoder_digest=c['decoder_tensor_digest']))


def dynamics(rows,store,p):
    import numpy as np
    from core import color_mask
    result={}
    for start in [35,123]:
        sample_rows=sorted((r for r in rows if r['start']==start and r['horizon']>0),key=lambda r:r['horizon'])
        if len(sample_rows)!=9:raise RuntimeError('Anchor window incomplete')
        entries=[]
        for row in sample_rows:
            proxy=row['metrics']['correct']['object_proxy']
            image=store.frames['samples'][row['frame']]['record']['real'].cpu().numpy()
            yy,xx=np.nonzero(color_mask(image,p['object_proxies'][str(start)]))
            diagonal=float(np.hypot(xx.max()-xx.min()+1,yy.max()-yy.min()+1)) if len(xx)>=10 else None
            normalized=(proxy['centroid_error_px']/diagonal if diagonal and proxy['centroid_error_px'] is not None else None)
            entries.append(dict(frame=row['frame'],horizon=row['horizon'],target_visible=proxy['target']['valid'],
                prediction_visible=proxy['prediction']['valid'],centroid_error_over_target_bbox_diagonal=normalized,
                axis_error_degrees=proxy['axis_error_degrees'],iou=proxy['iou'],
                target_centroid=proxy['target']['centroid_xy'],prediction_centroid=proxy['prediction']['centroid_xy']))
        visible=[e for e in entries if e['target_visible']]
        retained=sum(e['prediction_visible'] for e in visible)
        center_pass=sum(e['centroid_error_over_target_bbox_diagonal'] is not None and
                        e['centroid_error_over_target_bbox_diagonal']<=.1 for e in visible)
        angles=[e['axis_error_degrees'] for e in visible if e['axis_error_degrees'] is not None]
        longest=run=0
        for e in entries:
            run=run+1 if e['target_visible'] and not e['prediction_visible'] else 0
            longest=max(longest,run)
        matched=[e for e in visible if e['prediction_visible']]
        direction=None
        if len(matched)>=2:
            ta=np.array(matched[-1]['target_centroid'])-np.array(matched[0]['target_centroid'])
            pa=np.array(matched[-1]['prediction_centroid'])-np.array(matched[0]['prediction_centroid'])
            if np.linalg.norm(ta)>1 and np.linalg.norm(pa)>1:
                direction=float(np.dot(ta,pa)/(np.linalg.norm(ta)*np.linalg.norm(pa)))
        result[str(start)]=dict(name=p['object_proxies'][str(start)]['name'],visible_target_frames=len(visible),
            retention_fraction=retained/max(len(visible),1),center_within_10pct_fraction=center_pass/max(len(visible),1),
            median_axis_error_degrees=statistics.median(angles) if angles else None,
            maximum_consecutive_missing_frames=longest,trajectory_direction_cosine=direction,frames=entries,
            scope='HSV/ROI color proxy on one known window; bbox diagonal is used as a scale proxy, not a tracked 3D object')
    return result


def evaluate_sweep(sweep):
    import torch
    c,helper,orig,p,paths,receipt,baseline_evaluation=setup()
    if sweep not in c['checkpoint_sweeps']:raise ValueError('Unexpected sweep')
    folder=ROOT/f'sweep_{sweep:02d}';meta=read(folder/'MERGED.json')
    code_file=folder/'merged_codes.pt'
    if sha(code_file)!=meta['sha256']:raise RuntimeError('Merged code checksum differs')
    out=folder/'free_z';out.mkdir()
    write(out/'STARTED.json',dict(pid=os.getpid(),sweep=sweep,mode='free_z'))
    write(out/'STATUS.json',dict(status='running',phase='initializing'))
    try:
        setup_gpu(p)
        decoder,modules=load_decoder(c,helper,orig,p,paths,receipt,'cuda')
        fm,data,runtime=modules
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        store=dataset(modules,p,c);baseline_codes=load_codes(paths,baseline_evaluation)
        merged=torch.load(code_file,map_location='cpu',weights_only=True)
        if (merged['schema']!='fixed_decoder_free_codes_merged_v1' or merged['sweep']!=sweep
                or merged['starts']!=c['starts'] or merged['decoder_digest']!=c['decoder_tensor_digest']
                or tuple(merged['codes'].shape)!=(168,9,16)):
            raise RuntimeError('Merged code provenance differs')
        baseline_rows={(row['start'],row['horizon']):row for row in baseline_evaluation['rows'] if row['horizon']>0}
        if set(baseline_rows)!={(s,h) for s in c['starts'] for h in range(1,10)}:
            raise RuntimeError('Baseline row coverage differs')
        local_p=dict(p,eval_controls=p['eval_controls']+['baseline_qformer'],
                     evaluation_max_seconds=c['evaluation_max_seconds'])
        source=dict(source_record(c,paths,receipt,audit(c)),sweep=sweep,
                    merged_codes_sha256=meta['sha256'],baseline_control_from_previous_frozen_result=True)
        write(out/'STATUS.json',dict(status='running',phase='evaluation',sweep=sweep),replace=True)
        evaluator=explicit('_stage_a_full_evaluator',ROOT/'evaluate.py')
        evaluator.evaluate(decoder,bridge,store,local_p,out,source,'free_z',merged['codes'],baseline_codes,baseline_rows)
        assert_frozen(decoder,bridge,runtime,c['decoder_tensor_digest'])
        result=read(out/'evaluation/EVALUATION.json')
        if result['window_predictions']!=1512 or result['unique_future_frames']!=176:
            raise RuntimeError('Incomplete full evaluation')
        future=[r for r in result['rows'] if r['horizon']>0]
        by_horizon={str(h):fm.aggregate([r for r in future if r['horizon']==h],local_p['eval_controls'])['window_prediction_equal']
                    for h in range(1,10)}
        anchors=dynamics(future,store,p)
        controls=result['metrics']['unique_frame_equal']
        free=controls['correct']['motion_mae'];baseline=controls['baseline_qformer']['motion_mae']
        anchor_mae=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in future if r['start'] in [35,123])/18
        baseline_anchor=sum(r['metrics']['baseline_qformer']['metrics']['motion_mae'] for r in future if r['start'] in [35,123])/18
        thresholds=c['anchor_proxy_thresholds']
        def pass_anchor(value):
            return (value['visible_target_frames']>0 and value['retention_fraction']>=thresholds['retention_min']
                and value['center_within_10pct_fraction']>=thresholds['center_pass_fraction_min']
                and value['median_axis_error_degrees'] is not None
                and value['median_axis_error_degrees']<=thresholds['median_axis_error_degrees_max']
                and value['maximum_consecutive_missing_frames']<=thresholds['maximum_missing_run'])
        write(out/'STAGE_METRICS.json',dict(status='complete',sweep=sweep,metrics=result['metrics'],
            support=result['support'],by_horizon=by_horizon,dynamic_anchors=anchors,
            anchor_motion_mae=anchor_mae,baseline_anchor_motion_mae=baseline_anchor,
            global_motion_mae_gain_vs_baseline=1-free/max(baseline,1e-12),
            preliminary_known_anchor_proxy_pass=all(pass_anchor(v) for v in anchors.values()),
            provisional_fit_gain_10pct=(1-free/max(baseline,1e-12))>=.1,
            manual_visual_review_required=True,unannotated_other_windows_not_claimed_success=True,
            original_reference='Baseline Q-Former metrics are copied from the hash-bound 9560-step evaluation for exactly the same start/horizon.',
            controls='correct=optimized free code; baseline_qformer=original correct posterior; initial_only=original posterior with repeated S0; swapped=free code from adjacent window.'))
        write(out/'STATUS.json',dict(status='complete',phase='evaluation',native_replay='pending',sweep=sweep),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase='evaluation',sweep=sweep,error=repr(error),traceback=traceback.format_exc())
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


def replay_sweep(sweep):
    c,helper,orig,p,paths,receipt,evaluation=setup()
    if sweep not in c['checkpoint_sweeps']:raise ValueError('Unexpected sweep')
    replay=explicit('_stage_a_native_replay',orig/'replay.py')
    replay.ROOT=ROOT/f'sweep_{sweep:02d}'
    replay.main('free_z')


def summarize(c):
    diagnostics=read(ROOT/'diagnostics/DIAGNOSTIC.json')
    if diagnostics['status']!='complete' or diagnostics['window_predictions']!=1512:
        raise RuntimeError('Missing target-support or residual diagnostic')
    stages={};lines=['# 阶段A：冻结9560步解码器的全片段自由连续码','',
        '168个训练窗口的码以9560步Q-Former输出初始化；只优化连续码，decoder与codec全程冻结。',
        '主要指标对同一物理帧的不同上下文求平均，再对176个物理未来帧等权。','',
        '| 追加轮数 | 自由码运动MAE | 原Q-Former运动MAE | 自由码相对改善 | 两关键窗口运动MAE |',
        '|---|---:|---:|---:|---:|']
    for sweep in c['checkpoint_sweeps']:
        out=ROOT/f'sweep_{sweep:02d}/free_z'
        m=read(out/'STAGE_METRICS.json');replay=read(out/'evaluation/REPLAY.json')
        if (m['status']!='complete' or replay['status']!='passed'
                or replay['packets']!=104 or replay['views']!=208):
            raise RuntimeError('Stage incomplete')
        free=m['metrics']['unique_frame_equal']['correct']['motion_mae']
        original=m['metrics']['unique_frame_equal']['baseline_qformer']['motion_mae']
        stages[str(sweep)]=dict(metrics=m,replay={k:v for k,v in replay.items() if k!='rows'})
        lines.append(f"| {sweep} | {free:.6f} | {original:.6f} | {m['global_motion_mae_gain_vs_baseline']*100:.2f}% | {m['anchor_motion_mae']:.6f} |")
    final=stages[str(c['sweeps'])]['metrics']
    lines+=['','颜色与ROI结果只是两段已知训练样例的近似代理，必须复核完整运动视频和其他动态窗口。',
            '真实占据替换是只用于诊断的oracle，不进入动作编码器或decoder预测输入。',
            '自动分支建议仅是线索，不代替完整动态内容审查；未完成动作提取器和运动/静止控制测试。','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,stages=stages,
        diagnostic=dict(metrics=diagnostics['metrics'],by_horizon=diagnostics['by_horizon'],
                        reachability_ratios=diagnostics['reachability_ratios']),
        final_sweep=c['sweeps'],primary_checkpoint='final',
        stage_a_anchor_proxy_pass=final['preliminary_known_anchor_proxy_pass'],
        stage_a_fit_gain_10pct=final['provisional_fit_gain_10pct'],
        next_step_hint='Review dynamic windows; if free code still fails use stage B, otherwise stage C with the same decoder',
        action_codec_complete=False,manual_visual_review_required=True,
        no_unseen_action_or_ar_claim=True,code_sha256=sha(ROOT/'CODE.sha256')))


def child(name,script,gpu,timeout,args=()):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
             OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-B',str(ROOT/script)]+list(args)
    with (ROOT/f'{name}.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(ROOT/f'{name}_PROCESS.json',dict(pid=proc.pid,phase=script,gpu=gpu,args=list(args)),replace=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            return dict(status='failed',name=name,error='deadline')
    return dict(status='complete' if code==0 else 'failed',name=name,exit_code=code)


def coordinate():
    began=time.monotonic()
    try:
        c,helper,orig,p,paths,receipt,evaluation=setup()
        witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256') or witness['sampling']!=audit(c):
            raise RuntimeError('CPU witness or sampling changed')
        available=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
        manifest=read(paths['cache_manifest']);cache_bytes=sum(v['bytes'] for v in manifest['windows'].values())+manifest['frames']['bytes']
        if available<max(p['minimum_host_available_gib']*2**30,2*cache_bytes+8*2**30):raise RuntimeError('Insufficient RAM')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:raise RuntimeError('Insufficient disk')
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),stage='A',code_sha256=sha(ROOT/'CODE.sha256'),
              decoder_digest=c['decoder_tensor_digest'],sweeps=c['sweeps']))
        write(ROOT/'STATUS.json',dict(status='running',phase='parallel_free_code_fit_and_diagnostics'))
        def shard0():return child('shard_0','optimize.py',0,c['training_max_seconds_per_shard']+60,['--shard','0'])
        def shard1():
            result=child('diagnostics','diagnose.py',1,c['diagnostic_max_seconds']+60)
            if result['status']!='complete':return result
            return child('shard_1','optimize.py',1,c['training_max_seconds_per_shard']+60,['--shard','1'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(shard0),pool.submit(shard1)];results=[f.result() for f in futures]
        if any(x['status']!='complete' for x in results):raise RuntimeError('Fitting/diagnostic branch failed: '+str(results))
        for sweep in c['checkpoint_sweeps']:merge_sweep(sweep,c,helper,orig,p,paths,receipt)
        write(ROOT/'STATUS.json',dict(status='running',phase='full_evaluation_and_native_replay',sweeps=c['checkpoint_sweeps']),replace=True)
        def evaluate_and_replay(sweep,gpu):
            result=child(f'eval_{sweep:02d}','driver.py',gpu,c['evaluation_max_seconds']+90,['evaluate','--sweep',str(sweep)])
            if result['status']!='complete':return result
            result=child(f'replay_{sweep:02d}','driver.py',gpu,p['replay_max_seconds']+30,['replay','--sweep',str(sweep)])
            if result['status']=='complete':
                write(ROOT/f'sweep_{sweep:02d}/free_z/STATUS.json',dict(status='complete',native_replay='passed',sweep=sweep),replace=True)
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluate_and_replay,5,0),pool.submit(evaluate_and_replay,10,1)]
            results=[f.result() for f in futures]
        if any(x['status']!='complete' for x in results):raise RuntimeError('Intermediate evaluation failed: '+str(results))
        final=evaluate_and_replay(c['sweeps'],0)
        if final['status']!='complete':raise RuntimeError('Final evaluation failed: '+str(final))
        summarize(c)
        write(ROOT/'STATUS.json',dict(status='complete',stage='A',action_codec_complete=False,
            manual_visual_review_required=True,elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc())
        write(ROOT/'FAILED.json',failure);write(ROOT/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['coordinate','evaluate','replay'])
    parser.add_argument('--sweep',type=int)
    args=parser.parse_args()
    if args.phase in ['evaluate','replay'] and args.sweep is None:parser.error('--sweep required')
    if args.phase=='coordinate':coordinate()
    elif args.phase=='evaluate':evaluate_sweep(args.sweep)
    else:replay_sweep(args.sweep)
