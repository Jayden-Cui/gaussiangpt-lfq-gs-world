"""CPU-only complete-frame codec comparison; no Torch or checkpoint deserialization."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import numpy as np
import pool_index as pool

ARMS=('decoder_only','full_codec')
LABELS=('baseline',*ARMS)
EVAL_CLIP='buildingblock_tower_14_0'
CLIPS=(*sorted(pool.EXPECTED),EVAL_CLIP)
KEYS={(c,f) for c in CLIPS for f in range(177)}
GROUPS=('codec_real','teacher_real','codec_teacher','pseudo_view_teacher')
METRICS=('mae','psnr','workspace_roi_mae','workspace_roi_psnr')
PICTURES=('real','teacher','codec','nearby','nearby_teacher')
SNAPSHOTS=(0,44,88,132,176)
INITIAL_SHA='a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
INITIAL_ENCODER_SHA='c2ddb2b1dd5a43ee4a86286ea09ef141f61661963fa1d5979edf3d4f5917737b'
LEGACY_BINDING_SHA='695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715'
OLD_COMPARE_SHA='0d0750f6655c41fd68d471c6af8fe77ee8407572e457c32545b9e153a9e32047'
OLD_WORKER_SHA='f210f2f0961d87ed0f4e9d448b54b34118da00e9c958ae32ac1fb70a6335e3f7'

def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value,dict),'Expected JSON object: '+str(path))
    return value

def rows(path):
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()),'Empty JSONL row')
            value=json.loads(line);require(isinstance(value,dict),'JSONL row is not object');result.append(value)
    return result

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def check_hash(path,digest,size=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and pool.valid_sha(digest),'Missing file or invalid digest')
    before=pool.signature(path)
    require(size is None or type(size) is int and size>0 and before[2]==size,'Artifact size mismatch: '+str(path))
    require(sha(path)==digest and pool.signature(path)==before,'Artifact hash/change mismatch: '+str(path))
    return digest

def fields(actual,expected,context):
    for k,v in expected.items():
        require(type(actual.get(k)) is type(v) and actual[k]==v,context+'.'+k+' differs')

def metric_values(m):
    require(isinstance(m,dict) and all(type(m.get(k)) in (int,float) and math.isfinite(m[k]) for k in METRICS),
            'Missing or nonfinite image metrics')
    require(0<=m['mae']<=1 and 0<=m['workspace_roi_mae']<=1,'MAE outside normalized RGB range')

def check_replay(value):
    require(isinstance(value,dict) and set(value)=={'real_view','nearby_view'},'Both replay views required')
    for metrics in value.values():
        metric_values(metrics)
        require(metrics['mae']<=1e-5 and metrics['psnr']>=60.,'Latent packet replay failed')

def index_frames(values,expected=KEYS):
    indexed={}
    for r in values:
        key=(r.get('clip'),r.get('frame'))
        require(type(r.get('frame')) is int and key in expected and key not in indexed,'Duplicate/foreign frame')
        indexed[key]=r
    require(set(indexed)==expected,'Missing complete-frame coverage')
    return indexed

def mean_metrics(selected):
    require(bool(selected),'Cannot average empty group')
    return {g:{m:float(np.mean([r[g][m] for r in selected])) for m in METRICS} for g in GROUPS}

def compare_means(actual,stored,context):
    require(isinstance(stored,dict),context+': missing summary')
    for g in GROUPS:
        metric_values(stored.get(g))
        for m in METRICS:
            require(math.isclose(actual[g][m],stored[g][m],rel_tol=1e-7,abs_tol=1e-9),context+': summary/frame mismatch')

def aggregates(by_clip,train_groups=pool.EXPECTED,eval_clip=EVAL_CLIP):
    require(set(train_groups)<=set(by_clip) and eval_clip not in train_groups,'Aggregate membership mismatch')
    groups={group:mean_metrics([by_clip[c] for c,g in train_groups.items() if g==group]) for group in sorted(set(train_groups.values()))}
    return dict(train_clip_equal=mean_metrics([by_clip[c] for c in train_groups]),
        train_source_group_equal=mean_metrics(list(groups.values())),by_train_source_group=groups,
        exposed_eval_only=by_clip[eval_clip],group_clip_counts=dict(sorted(Counter(train_groups.values()).items())))

def teacher_alignment(a,b):
    for g in ('teacher_real',):
        metric_values(a[g]);metric_values(b[g])
        for metric in METRICS:
            absolute,relative=(2e-6,1e-5) if 'mae' in metric else (.003,1e-6)
            require(math.isclose(a[g][metric],b[g][metric],abs_tol=absolute,rel_tol=relative),
                    'Teacher/real alignment differs beyond CPU/GPU numerical tolerance')

def bootstrap_function(root):
    path=root.parent/pool.LEGACY_RUN/'compare.py';check_hash(path,OLD_COMPARE_SHA)
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='moving_block_bootstrap']
    require(len(functions)==1,'Pinned bootstrap function missing')
    namespace={'np':np,'math':math,'require':require}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['moving_block_bootstrap']

def load_contract(root):
    manifest=read(root/'CODE_MANIFEST.json')
    require({'worker.py','compare.py','pool_index.py','baseline_adapter.py','protocol.json'}<=set(manifest),
            'Manifest omits implementation/protocol')
    require(not {'SUMMARY.json','SUMMARY.md'}&set(manifest),'Generated summaries must not enter source manifest')
    for name,digest in manifest.items():
        require(Path(name).name==name,'Unsafe manifest path');check_hash(root/name,digest)
    proof=read(root/'CPU_VALIDATION.json')
    fields(proof,dict(status='passed',gpu_executed=False,code_manifest_sha256=sha(root/'CODE_MANIFEST.json')),'CPU proof')
    p=read(root/'protocol.json');fields(p,dict(seed=20261031),'Protocol')
    fields(p['train'],dict(steps=4426,batch_size=4,epochs=2,epoch_seeds=[20261031,20261032],force_target_epochs=[True,False]),'Train')
    fields(p['data'],dict(train_clips=sorted(pool.EXPECTED),eval_only_clips=[EVAL_CLIP],frames_per_clip=177),'Data')
    fields(p['evaluation'],dict(records_per_model=9027,snapshot_frames=list(SNAPSHOTS),target_forcing=False),'Evaluation')
    index=tuple(SimpleNamespace(clip=c,frame=f) for c in sorted(pool.EXPECTED) for f in range(177))
    plans=[pool.balanced_plan(index,seed,4) for seed in (20261031,20261032)]
    identities=[[[r.clip,r.frame] for b in plan for r in b] for plan in plans]
    plan_shas=[hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest() for x in identities]
    source=dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),protocol_sha256=sha(root/'protocol.json'),
        cpu_validation_sha256=sha(root/'CPU_VALIDATION.json'),pool_complete_sha256=pool.COMPLETE_SHA,
        pool_done_sha256=pool.DONE_SHA,legacy_worker_sha256=OLD_WORKER_SHA,legacy_sources_sha256=pool.LEGACY_SHA,
        legacy_binding_sha256=LEGACY_BINDING_SHA,initial_checkpoint_sha256=INITIAL_SHA,epoch_plan_sha256=plan_shas)
    preflight=read(root/'PREFLIGHT.json')
    fields(preflight,dict(status='passed',source=source,initial_arms_identical=True),'Preflight')
    require(set(preflight['arms'])==set(ARMS),'Incomplete preflight arms')
    initials=[]
    for a in ARMS:
        row=preflight['arms'][a]
        fields(row,dict(passed=True,discarded_updates=2,production_updates=0,actual_batch_sizes=[4,4]),'Preflight/'+a)
        require(pool.valid_sha(row.get('initial_model_sha256')),'Initial model digest missing');initials.append(row['initial_model_sha256'])
    require(len(set(initials))==1,'Preflight initialization differs')
    pool_root=Path(p['data']['pool_root'])
    check_hash(pool_root/'POOL_COMPLETE.json',pool.COMPLETE_SHA);complete=read(pool_root/'POOL_COMPLETE.json')
    check_hash(pool_root/'queue/DONE.json',pool.DONE_SHA)
    records={r['path']:r for r in complete['files']}
    old_root=root.parent/pool.LEGACY_RUN
    check_hash(old_root/'DATA_BINDING.json',LEGACY_BINDING_SHA)
    old_binding=read(old_root/'DATA_BINDING.json')
    cache={}
    for r in [*records.values(),*old_binding['files']]:
        path=Path(r['path'])
        if path.parent.parent.name=='cache' and path.parent.name in CLIPS and path.suffix=='.pt':
            key=(path.parent.name,int(path.stem))
            require(key not in cache or cache[key]==r['sha256'],'Conflicting cache provenance')
            cache[key]=r['sha256']
    require(set(cache)==KEYS,'Expected 9027 source caches in admitted receipts')
    normals={}
    old_records={r['path']:r for r in old_binding['files']}
    for clip in complete['clips']:
        path=Path(clip['paths']['cache']).parent.parent/'normalization.json'
        check_hash(path,records[str(path)]['sha256'],records[str(path)]['bytes']);normals[clip['clip_id']]=read(path)
    path=old_root.parent/'gaussiangpt_neoverse_tower14_transfer_20261031_v1'/'normalization.json'
    check_hash(path,old_records[str(path)]['sha256'],old_records[str(path)]['bytes']);normals[EVAL_CLIP]=read(path)
    return dict(source=source,plans=plans,cache=cache,normals=normals,initial_model_sha256=initials[0])

def load_training(root,arm,contract):
    folder=root/'arms'/arm;require(not (folder/'FAILURE.json').exists(),'Failed training arm')
    done=read(folder/'TRAIN_DONE.json')
    fields(done,dict(status='complete',arm=arm,steps=4426,source=contract['source'],epoch_frames=[8850,8850],
        total_frame_exposures=17700,forcing_updates=2213,unforced_updates=2213,logical_batch_size=4,
        tail_batch_sizes=[2,2],fresh_optimizer_verified=True,initial_model_sha256=contract['initial_model_sha256'],
        initial_encoder_sha256=INITIAL_ENCODER_SHA,frozen_encoder_verified=arm=='decoder_only',goal_complete=False),'Training')
    require(pool.valid_sha(done.get('final_encoder_sha256')),'Missing final encoder digest')
    require((done['final_encoder_sha256']==INITIAL_ENCODER_SHA)==(arm=='decoder_only'),'Encoder freeze/update contradicted')
    check_hash(folder/'step_4426.pt',done['checkpoint_sha256']);check_hash(folder/'step_2213.pt',done['first_epoch_checkpoint_sha256'])
    check_hash(folder/'trace.jsonl',done['trace_sha256']);check_hash(folder/'PLAN.json',done['plan_sha256'])
    plan_receipt=read(folder/'PLAN.json')
    fields(plan_receipt,dict(epoch_seeds=[20261031,20261032],forcing=[True,False]),'Plan')
    require(len(plan_receipt['epochs'])==2,'Plan epoch count differs')
    for epoch,record in enumerate(plan_receipt['epochs']):
        fields(record,dict(frames=8850,updates=2213,actual_batch_sizes=[4]*2212+[2],
            clip_exposure={c:177 for c in pool.EXPECTED},
            source_group_exposure=dict(Counter(g for g in pool.EXPECTED.values() for _ in range(177))),
            plan_sha256=contract['source']['epoch_plan_sha256'][epoch]),'Plan epoch')
    fields(read(folder/'SOURCE.json'),dict(source=contract['source'],arm=arm,preflight_sha256=sha(root/'PREFLIGHT.json')),'Train source')
    trace=rows(folder/'trace.jsonl');require(len(trace)==4426,'Incomplete training trace')
    for n,(row,batch) in enumerate(zip(trace,[*contract['plans'][0],*contract['plans'][1]]),1):
        fields(row,dict(step=n,arm=arm,epoch=1 if n<=2213 else 2,force_target=n<=2213,actual_batch_size=len(batch)),'Trace')
        require([(r['clip'],r['frame']) for r in row['frames']]==[(r.clip,r.frame) for r in batch],'Trace differs from full-coverage plan')
    return done

def check_artifacts(folder,evaluation,targets,contract,label):
    packets=index_frames(rows(folder/'state_manifest.jsonl'))
    for key,packet in packets.items():
        row=targets[key]
        fields(packet,dict(arm=label,split=row['split'],checkpoint_sha256=evaluation['checkpoint_sha256'],
            encoder_sha256=evaluation['encoder_sha256'],source=contract['source'],inherited_codec_steps=3000,
            additional_codec_updates=4426,decoder_target_key=None,fresh_encode=True,normalization=contract['normals'][key[0]],
            source_cache_sha256=contract['cache'][key],shape=[row['latent_positions'],64],decoded_gaussians=row['decoded_gaussians'],
            state_path=f'states/{key[0]}/{key[1]:03d}.pt'),'State packet')
        require(packet['pts']==row['pts'],'Packet time differs')
        check_replay(packet.get('saved_packet_replay'));require(packet['saved_packet_replay']==row['replay'],'Replay receipts disagree')
        check_hash(folder/packet['state_path'],packet['state_sha256'],packet['state_bytes'])
    expected={(c,f,k) for c in CLIPS for f in SNAPSHOTS for k in PICTURES};seen=set()
    for item in rows(folder/'screenshot_manifest.jsonl'):
        key=(item.get('clip'),item.get('frame'),item.get('kind'))
        require(key in expected and key not in seen,'Duplicate/foreign screenshot');seen.add(key)
        relative=f'screenshots/{key[0]}/{key[1]:03d}_{key[2]}.png'
        require(item.get('path')==relative,'Screenshot identity/path differs');path=folder/relative
        check_hash(path,item.get('sha256'),item.get('bytes'))
        with path.open('rb') as f:head=f.read(24)
        require(head[:8]==b'\x89PNG\r\n\x1a\n' and head[12:16]==b'IHDR' and struct.unpack('>II',head[16:24])==(448,252),
                'Screenshot dimensions differ')
    require(seen==expected and len(seen)==1275,'Missing fixed screenshots')
    return dict(state_files_checked=len(packets),screenshot_files_checked=len(seen),tensor_deserialization=False)

def load_evaluation(root,label,contract,training):
    folder=root/'evaluations'/label;require(not (folder/'FAILURE.json').exists(),'Failed evaluation')
    e=read(folder/'EVALUATION.json');baseline=label=='baseline'
    fields(e,dict(status='evaluated',arm=label,frames=9027,train_frames=8850,exposed_eval_only_frames=177,
        additional_codec_updates=0 if baseline else 4426,source=contract['source'],goal_complete=False),'Evaluation')
    require(e['checkpoint_sha256']==(INITIAL_SHA if baseline else training[label]['checkpoint_sha256']),'Evaluation checkpoint differs')
    require(e['encoder_sha256']==(INITIAL_ENCODER_SHA if baseline else training[label]['final_encoder_sha256']),'Evaluation encoder differs')
    for name in ('targets','state_manifest','screenshot_manifest'):check_hash(folder/(name+'.jsonl'),e[name+'_sha256'])
    indexed=index_frames(rows(folder/'targets.jsonl'))
    for key,row in indexed.items():
        fields(row,dict(checkpoint_sha256=e['checkpoint_sha256'],encoder_sha256=e['encoder_sha256'],
            additional_codec_updates=0 if baseline else 4426,split='TRAIN' if key[0] in pool.EXPECTED else 'EVAL-only'),'Target row')
        require(type(row.get('pts')) in (int,float) and math.isfinite(row['pts']),'Missing finite time')
        require(math.isclose(row['pts'],key[1]/29.97,abs_tol=1e-9,rel_tol=0),'Frame time mapping differs')
        for g in GROUPS:metric_values(row.get(g))
        if not baseline:
            fields(row,dict(latent_channels=64,input_gaussians=112896),'Target shape')
            for name,maximum in [('input_voxels',80000),('latent_positions',80000),('decoded_gaussians',2400000)]:
                require(type(row.get(name)) is int and 0<row[name]<=maximum,'Invalid frame count '+name)
            check_replay(row.get('replay'))
    means={c:mean_metrics([indexed[c,f] for f in range(177)]) for c in CLIPS}
    require(set(e['summary_by_clip'])==set(CLIPS),'Summary clip membership differs')
    for c in CLIPS:compare_means(means[c],e['summary_by_clip'][c],label+'/'+c)
    aggregate=aggregates(means)
    compare_means(aggregate['train_clip_equal'],e['summary_train_frames'],label+'/train')
    compare_means(aggregate['exposed_eval_only'],e['summary_exposed_eval'],label+'/exposed_eval')
    if baseline:
        fields(e,dict(baseline_reused=True,saved_latent_state_replay=False,every_packet_saved_replay_passed=False,
            historical_saved_latent_state_replay_frames=354,saved_latent_state_replay_unavailable_frames=8673,
            reconstruction_cpu_metric_frames=8673,exact_historical_metric_frames=354,
            actual_cache_bytes_sha_verified_frames=9027,actual_state_bytes_sha_verified_frames=9027,
            screenshots_generated=False,all_frames_freshly_encoded=False,
            baseline_adapter_sha256=sha(root/'baseline_adapter.py')),'Reused baseline scope')
        historical=index_frames(rows(folder/'state_manifest.jsonl'))
        require(not rows(folder/'screenshot_manifest.jsonl'),'Reused baseline unexpectedly claims new screenshots')
        for key,packet in historical.items():
            fields(packet,dict(fresh_encode=False,state_path_is_external=True,checkpoint_sha256=INITIAL_SHA,
                encoder_sha256=INITIAL_ENCODER_SHA,additional_codec_updates=0,
                normalization=contract['normals'][key[0]],source_cache_sha256=contract['cache'][key]),'Historical packet metadata')
            require(pool.valid_sha(packet.get('state_sha256')) and type(packet.get('state_bytes')) is int
                    and packet['state_bytes']>0,'Historical packet lacks byte binding')
            if key[0] in ('buildingblock_tower_17_176',EVAL_CLIP):
                check_replay(packet.get('saved_packet_replay'))
            else:
                require(packet.get('saved_packet_replay') is None,'Unavailable historical replay was fabricated')
        artifact_scope=dict(new_state_files_checked=0,new_screenshot_files_checked=0,
            historical_state_manifest_entries_checked=len(historical),
            all_frame_saved_replay=False,historical_saved_replay_frames=354)
    else:
        fields(e,dict(every_packet_saved_replay_passed=True,all_frames_freshly_encoded=True,
            saved_latent_state_replay=True,independent_native_gs_replay=False,decoder_target_key=None),'New replay scope')
        artifact_scope=check_artifacts(folder,e,indexed,contract,label)
    return dict(receipt_sha256=sha(folder/'EVALUATION.json'),checkpoint_sha256=e['checkpoint_sha256'],
        encoder_sha256=e['encoder_sha256'],per_clip=means,aggregates=aggregate,artifact_validation=artifact_scope,_rows=indexed)

def scientific_report(evaluations,bootstrap):
    result={}
    for candidate,reference in [('decoder_only','baseline'),('full_codec','baseline'),('full_codec','decoder_only')]:
        a,b=evaluations[candidate],evaluations[reference]
        per_clip={c:{metric:bootstrap({f:b['_rows'][c,f]['codec_real'][metric]-a['_rows'][c,f]['codec_real'][metric]
                      for f in range(177)}) for metric in ('mae','workspace_roi_mae')} for c in CLIPS}
        aggregate={scope:{g:{m:b['aggregates'][scope][g][m]-a['aggregates'][scope][g][m] for m in METRICS} for g in GROUPS}
                   for scope in ('train_clip_equal','train_source_group_equal','exposed_eval_only')}
        result[candidate+'_vs_'+reference]=dict(per_clip_paired_intervals=per_clip,
            aggregate_reference_minus_candidate=aggregate,mae_positive_is_improvement=True,
            psnr_negative_is_improvement=True,aggregate_confidence_intervals_computed=False,
            scope='Each interval describes one clip; overlapping clips are not independent population samples')
    return result

def main(root):
    root=Path(root).resolve()
    require(all(not (root/n).exists() and not (root/n).is_symlink() for n in ('SUMMARY.json','SUMMARY.md')),
            'Summary already exists; no overwrite/retry')
    contract=load_contract(root)
    training={a:load_training(root,a,contract) for a in ARMS}
    evaluations={label:load_evaluation(root,label,contract,training) for label in LABELS}
    for key in KEYS:
        base=evaluations['baseline']['_rows'][key]
        for arm in ARMS:
            row=evaluations[arm]['_rows'][key]
            require(math.isclose(row['pts'],base['pts'],abs_tol=1e-9,rel_tol=0),'Source time differs between arms')
            teacher_alignment(row,base)
    comparisons=scientific_report(evaluations,bootstrap_function(root))
    result=dict(status='complete',source=contract['source'],models={k:{a:b for a,b in v.items() if a!='_rows'} for k,v in evaluations.items()},
        comparisons=comparisons,train_clips=50,train_source_groups=19,clip_internal_train_frames=8850,
        exposed_eval_only_frames=177,cross_clip_physical_frame_unique_count=None,
        bootstrap=dict(block_length=16,replicates=2000,seed=20261031,implementation_sha256=OLD_COMPARE_SHA),
        teacher_numerical_tolerance=dict(mae_absolute=2e-6,mae_relative=1e-5,psnr_absolute=.003,psnr_relative=1e-6),
        action_Z_not_trained=True,goal_complete=False,visual_acceptance='pending',automatic_ldm_codec_replacement=False,
        checkpoint_selection='Fixed additional step4426; exposed tower14 never used for selection',
        baseline_scope='Reused exact frozen-source metrics; all-frame saved latent replay unavailable',
        independent_native_gs_replay=False)
    with (root/'SUMMARY.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    text=['# 完整池 codec 对照','', '这是状态重建结果；动作 Z 未训练，不能据此认定 LDM 学好。','',
          '|模型|TRAIN clip等权 PSNR|TRAIN来源组等权 PSNR|来源组等权 MAE|已暴露14 PSNR|', '|---|---:|---:|---:|---:|']
    for name,item in evaluations.items():
        a=item['aggregates'];text.append(f"|{name}|{a['train_clip_equal']['codec_real']['psnr']:.4f}|{a['train_source_group_equal']['codec_real']['psnr']:.4f}|{a['train_source_group_equal']['codec_real']['mae']:.6f}|{a['exposed_eval_only']['codec_real']['psnr']:.4f}|")
    text+=['','每段配对区间使用移动块16、2000次bootstrap。50段含重叠，不能视为50个独立来源；19来源组等权均值与clip等权均值分开报告，未计算跨来源总体置信区间。',
        '新两臂每个9027帧连续状态包和1275张固定截图均核bytes/SHA；重放是latent经过模型的双视角重放，不是独立native GS重放。',
        'baseline复用历史指标，未伪造全量落盘重放或新截图。仍需人工检查固定帧；不按tower14挑checkpoint，不自动替换LDM底座。']
    with (root/'SUMMARY.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    print(json.dumps(dict(status='complete',summary_sha256=sha(root/'SUMMARY.json'),goal_complete=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
