"""Paired physical-frame statistics; visual/action success is never automatic."""
import argparse
import importlib.util
import json
from pathlib import Path
from contract import read,sha,require,verify,write_new,ARMS

METRICS=('mae','hand_activity_mae','motion_mae')
CONTROLS=('correct','shuffle42','shuffle84','shuffle126','blinded','adapted_held')
STAT_PATH='gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py'
STAT_SHA='14fa085be07f8ab3f6ef79ec671362f062d6b2726f3968b02f8c4d95ff91c1b8'

def main(root):
    root=Path(root).resolve();p,proof=verify(root)
    path=root.parent/STAT_PATH;require(sha(path)==STAT_SHA,'Statistical method changed')
    spec=importlib.util.spec_from_file_location('_v10_statistics',path);stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    arms={};rows={};h0={};deltas={}
    for arm in ARMS:
        d=root/'evaluations'/arm;t=root/'arms'/arm
        e=read(d/'EVALUATION.json');train=read(t/'TRAIN_DONE.json')
        require(e['status']==train['status']=='complete' and e['steps']==train['steps']==800,'Incomplete training/evaluation')
        require(e['checkpoint_sha256']==train['checkpoint_sha256']['800']==sha(t/'step_0800.pt'),'Checkpoint changed')
        require(e['train_done_sha256']==sha(t/'TRAIN_DONE.json'),'Train receipt changed')
        for key,filename in [('targets','targets.jsonl'),('actions','actions.pt'),('initial_reconstruction','initial_reconstruction.json'),
                             ('snapshots','snapshots.json'),('native_packets','native_packets.json')]:
            require(e[key+'_sha256']==sha(d/filename),'Evaluation artifact changed: '+filename)
        for name,digest in read(d/'snapshots.json')['all_pngs'].items():require(sha(d/'snapshots'/name)==digest,'Snapshot changed')
        for item in read(d/'native_packets.json')['rows']:require(sha(item['path'])==item['sha256'],'Native packet changed')
        rows[arm]=[json.loads(line) for line in (d/'targets.jsonl').read_text().splitlines()]
        require(len(rows[arm])==1512 and {(r['start'],r['horizon']) for r in rows[arm]}=={(s,h) for s in range(168) for h in range(1,10)},'Incomplete future coverage')
        h0[arm]=read(d/'initial_reconstruction.json')['rows'];require(len(h0[arm])==168,'Incomplete h0')
        deltas[arm]=[dict(start=r['start'],end=r['end'],horizon=r['horizon'],**{c:{m:r['delta'][c][f] for m,f in zip(METRICS,('delta_mae','hand_activity_delta_mae','motion_delta_mae'))} for c in CONTROLS}) for r in rows[arm]]
        # Retain object rectangle as a separate descriptive metric, not a segmentation score.
        obj={}
        for r in rows[arm]:obj.setdefault(r['end'],[]).append(r['correct']['object_box_mae'])
        arms[arm]=dict(metrics=stats.reduce_rows(rows[arm],'correct'),initial=stats.reduce_rows(h0[arm],'correct'),
             object_box_mae=sum(sum(v)/len(v) for v in obj.values())/len(obj),evaluation_sha256=sha(d/'EVALUATION.json'),
             source_checkpoint_sha256=e['source_checkpoint_sha256'],per_h_distinct_ids=e['per_h_distinct_ids'],distinct_bundles=e['distinct_bundles'])
    comparisons={};screens={}
    for prefix in ('rgb','gs'):
        z=prefix+'_z';constant=prefix+'_constant';requirements=[];degraded=[]
        for control in ('shuffle42','shuffle84','shuffle126','matched_constant','adapted_held'):
            other=rows[constant] if control=='matched_constant' else rows[z]
            actual='correct' if control=='matched_constant' else control
            metrics=METRICS[:2] if control=='adapted_held' else METRICS
            c=stats.paired_comparison(rows[z],other,actual,metrics);comparisons[z+'_vs_'+control]=c
            requirements += [dict(control=control,metric=m,passed=v['positive_interval']) for m,v in c.items()]
            degraded += [dict(control=control,horizon=h,improvement=v) for h,v in c['mae']['per_horizon_mean_improvement'].items() if v < -1e-12]
        require(len(requirements)==14,'Wrong action-screen size')
        screens[prefix]=dict(passed=all(x['passed'] for x in requirements) and not degraded,requirements=requirements,horizon_degradations=degraded)
        comparisons[z+'_vs_blinded']=stats.paired_comparison(rows[z],rows[z],'blinded',METRICS)
        comparisons[z+'_delta_vs_constant']=stats.paired_comparison(deltas[z],deltas[constant],'correct',METRICS)
    for suffix in ('z','constant'):
        comparisons['gs_'+suffix+'_vs_rgb_'+suffix]=stats.paired_comparison(rows['gs_'+suffix],rows['rgb_'+suffix],'correct',METRICS+('object_box_mae',))
    effect=comparisons['gs_z_vs_rgb_z']
    result=dict(status='complete',**proof,arms=arms,comparisons=comparisons,z_contribution_screens=screens,
        supervision_checks=dict(motion_positive_interval=effect['motion_mae']['positive_interval'],
            full_nonworse=effect['mae']['mean_improvement']>=0,roi_nonworse=effect['hand_activity_mae']['mean_improvement']>=0,
            object_rectangle_nonworse=effect['object_box_mae']['mean_improvement']>=0,
            h0_full_nonworse=arms['gs_z']['initial']['frame_uniform']['mae']<=arms['rgb_z']['initial']['frame_uniform']['mae']),
        formal_updates=800,source_frames=177,future_windows=168,physical_future_frames=176,
        bootstrap=dict(block_length=16,replicates=2000,seed=20261031),visual_acceptance='pending',
        teacher_mask_is_geometry_truth=False,learned_reliable_action=None,goal_complete=False,
        scope='One already used TRAIN clip; future targets supervise loss only; no AR/transfer claim')
    write_new(root/'SUMMARY.json',result)
    lines=['# v10：固定架构的伪视角监督对照','', '全部数值须结合44/132物体完整性验收；此文件不自动宣称Z成功。','',
           '| 方法 | PSNR | 运动MAE | 物体矩形MAE |','|---|---:|---:|---:|']
    for a,v in arms.items():
        m=v['metrics']['frame_uniform'];lines.append(f"| {a} | {m['psnr']:.4f} | {m['motion_mae']:.6f} | {v['object_box_mae']:.6f} |")
    with (root/'SUMMARY.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
