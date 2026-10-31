"""Fixed v9 600 -> 800 loss-only LDM continuation; no network/data reduction."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import torch
from contract import read, sha, require, verify, verify_sources
from loss_step import step_native, render_views

sys.dont_write_bytecode=True
ARMS=('rgb_z','gs_z','rgb_constant','gs_constant')
CONTROLS=('correct','shuffle42','shuffle84','shuffle126','blinded','adapted_held')


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def save(path,value):
    with Path(path).open('xb') as stream:torch.save(value,stream)


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def memory_admission(p):
    torch.set_num_threads(p['resources']['cpu_threads']);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info();cap=p['resources']['peak_limit_gib']
    require(free>=p['resources']['minimum_free_gib']*2**30,'Insufficient actual free GPU memory')
    require(cap==16 and p['resources']['minimum_free_gib']==20,'Resource policy changed')
    torch.cuda.set_per_process_memory_fraction(cap*2**30/total)
    torch.cuda.reset_peak_memory_stats()
    return dict(free_gib=free/2**30,total_gib=total/2**30,allocator_cap_gib=cap,
                measurement_scope='PyTorch allocated/reserved, not NVML process peak')


def guard(deadline):
    require(time.monotonic()<deadline,'Fixed worker wall budget expired')
    require(torch.cuda.max_memory_allocated()/2**30<=16 and torch.cuda.max_memory_reserved()/2**30<=16,
            'PyTorch allocation/reservation exceeded16GiB')


def runtime(root,p,binding):
    """Explicit new resource adapter; immutable core.setup and cfg remain unchanged."""
    verify_sources(p)
    source=Path(p['source_root']);sys.path.insert(0,str(source))
    v9=module_at('_v10_bound_v9',source/'worker.py')
    e=v9.load_verified(source);evidence=e['context']['evidence']
    for path in (evidence['source']/'code',evidence['acceleration'],evidence['root']):
        sys.path.insert(0,str(path))
    import native_experiment as core
    import temporal_step
    require(Path(core.__file__).resolve()==(evidence['source']/'code/native_experiment.py').resolve(),
            'Native module imported from wrong location')
    require(Path(temporal_step.__file__).resolve()==(evidence['root']/'temporal_step.py').resolve(),
            'Temporal module imported from wrong location')
    cfg=read(evidence['source']/'protocol.json')
    require(cfg==evidence['source_cfg'],'Historical source config changed')
    require(cfg['train']['batch_size']==4 and cfg['data']['frames_per_window']==10,'Full B4T10 required')
    for name,digest in read(evidence['source']/'CODE_MANIFEST.json').items():
        require(sha(evidence['source']/name)==digest,'Bound original source changed')
    torch.manual_seed(cfg['seed']);core.np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    data=core.Data()
    require(len(data.states)==177 and len(data.sources)==354,'Incomplete177-frame source')
    plan=e['context']['runner'].make_plan(cfg['seed'],800)
    require(p['train_plan']==plan and plan[:600]==e['plan'],'Seeded continuation plan changed')
    assets=read(root/'TEACHER_ASSETS.json')
    require(assets['status']=='complete' and assets['teachers_sha256']==sha(root/'teachers.pt'),
            'Teacher alpha assets not complete/bound')
    require(all(assets.get(k)==v for k,v in binding.items()) and assets['frames']==177
            and assets['alpha_threshold']==.90 and assets['erosion_kernel']==3
            and assets['prediction_dependent_mask'] is False,'Teacher asset method/provenance changed')
    masks=torch.load(root/'teachers.pt',map_location='cpu',weights_only=True)['masks']
    require(masks.shape==(177,252,448) and masks.dtype==torch.bool and bool(masks.flatten(1).any(1).all()),
            'Teacher asset masks incomplete/empty')
    data.teacher_masks=masks.cuda();data.teacher_targets=[]
    for frame in range(177):
        path=core.STATE_ROOT/'cache'/core.CLIP/f'{frame:03d}.pt'
        raw=torch.load(path,map_location='cpu',weights_only=True)
        target=raw['targets']
        require(target.shape==(2,252,448,3) and not target.requires_grad and bool(torch.isfinite(target).all()),
                'Invalid frozen native teacher targets')
        data.teacher_targets.append(target.cuda())
        data.normalization=raw['normalization']
    r=dict(torch=torch,core=core,temporal=temporal_step,cfg=cfg,data=data,
           data_hash=v9.digest(data.sources),runner=e['context']['runner'])
    return dict(root=root,p=p,binding=binding,v9=v9,e=e,r=r,assets_hash=sha(root/'TEACHER_ASSETS.json'),
                teachers_hash=sha(root/'teachers.pt'))


def unchanged(x):
    _,proof=verify(x['root'],cpu_witness=True)
    require(proof==x['binding'],'Own bound sources changed')
    verify_sources(x['p'])
    x['v9'].verify_unchanged(x['e'])
    require(sha(x['root']/'TEACHER_ASSETS.json')==x['assets_hash'] and
            sha(x['root']/'teachers.pt')==x['teachers_hash'],'Fixed alpha assets changed')


def restore600(x,arm):
    source_arm=x['p']['arms'][arm]['source_arm']
    state,done,checkpoint=x['v9'].restore600(x['e'],x['r'],source_arm)
    return state,dict(source_arm=source_arm,source_step=600,source_checkpoint_sha256=sha(checkpoint),
                      source_train_done_sha256=sha(Path(checkpoint).parent/'TRAIN_DONE.json'))


def evidence(x,arm,source):
    return dict(**x['binding'],**source,arm=arm,teacher_weights=x['p']['arms'][arm]['teacher_weights'],
                teacher_assets_sha256=x['assets_hash'],teachers_sha256=x['teachers_hash'],
                full_plan_sha256=x['v9'].digest(x['p']['train_plan']),source_data_hash=x['r']['data_hash'],
                source_frames=177,frame_gap=1,batch_size=4,frames_per_window=10,
                full_native_resolution=[252,448],all_initial_latent_points_retained=True,goal_complete=False)


def metrics(data,image,start,h):
    result=data.metrics(image,start,h)
    result['object_box_mae']=float((image[25:235,140:320]-data.records[start+h]['real'][25:235,140:320]).abs().mean())
    return result


def preflight(x,out,deadline):
    results={};v9=x['v9'];r=x['r'];data=r['data']
    for arm in ARMS:
        guard(deadline);state,source=restore600(x,arm)
        baseline=read(Path(x['p']['source_root'])/'evaluations'/source['source_arm']/'initial_reconstruction.json')
        initial={row['start']:row for row in baseline['rows']}
        target={(row['start'],row['horizon']):row for row in
                map(json.loads,(Path(x['p']['source_root'])/'evaluations'/source['source_arm']/'targets.jsonl').read_text().splitlines())}
        actions=torch.load(Path(x['p']['source_root'])/'evaluations'/source['source_arm']/'actions.pt',
                           map_location='cpu',weights_only=True)['ids'].cuda()
        parity=[]
        with torch.no_grad():
            for start in x['p']['preflight']['parity_starts']:
                outputs=state['model'].decode_ids(actions[start:start+1],[r['core'].make_condition(data.states[start])])[0]
                for h,pred in enumerate(outputs):
                    image=state['bridge'].render(pred,data.records[start+h]);actual=data.metrics(image,start,h)
                    expected=initial[start]['correct'] if h==0 else target[start,h]['correct']
                    for key,value in expected.items():
                        require(abs(actual[key]-value)<= (.02 if 'psnr' in key else 1e-4),'Source600 parity failed')
                    parity.append(dict(start=start,horizon=h,metrics=actual))
        # Parity is no-grad and model eval. Reset original RNG explicitly before one diagnostic update.
        saved=torch.load(Path(x['p']['source_root'])/'arms'/source['source_arm']/'step_0600.pt',
                         map_location='cpu',weights_only=False)
        v9.restore_rng(r,saved);del saved
        state['model'].train();state['bridge'].train()
        before_model=r['core'].tensor_hash(state['model']);before_codec=r['core'].tensor_hash(state['bridge'].codec)
        measured=step_native(state['model'],state['bridge'],state['optimizer'],data,x['p']['preflight']['starts'],
                             r['cfg'],r['temporal'],x['p']['arms'][arm]['teacher_weights'])
        v9.invariants(r,state);guard(deadline)
        v9.check_gradients(measured,posterior=arm.endswith('_z'))
        expected_groups=['residual_head','motion_decoder','native_out','native_heads','native_multiplier']
        if arm.endswith('_z'):expected_groups.append('motion_encoder')
        require(all(math.isfinite(measured['gradient_norms'][key]) and measured['gradient_norms'][key]>0
                    for key in expected_groups),'A required native/LDM group has no finite reconstruction gradient')
        require(r['core'].tensor_hash(state['model'])!=before_model,'LDM model did not update')
        require(r['core'].tensor_hash(state['bridge'].codec)!=before_codec,'Native trainable codec did not update')
        if arm.endswith('_z'):
            require(all(value>0 for value in measured['rgb_quantized_grad_per_h']),
                    'A motion token has no reconstruction gradient')
        results[arm]=dict(**evidence(x,arm,source),optimizer_updates=1,parity=parity,measured=measured,
                         model_updated=True,codec_updated=True,frozen_encoder_preserved=True)
        del state,actions;gc.collect();torch.cuda.empty_cache()
    unchanged(x);guard(deadline)
    write(x['root']/'PREFLIGHT.json',dict(status='passed',gpu_witness=True,**x['binding'],arms=results,
        model_updates_per_arm=1,training_checkpoints_written=False,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,goal_complete=False))


def save_checkpoint(path,x,state,arm,source,step,partial=False):
    payload=dict(model=state['model'].state_dict(),codec=state['bridge'].codec.state_dict(),
        optimizer=state['optimizer'].state_dict(),config=x['r']['cfg'],plan=x['p']['train_plan'],completed_steps=step,
        continuation=dict(schema='gs_supervision_checkpoint_v10',**evidence(x,arm,source),partial=partial),
        **x['v9'].rng_state(x['r']))
    save(path,payload)


def require_preflight(x):
    pre=read(x['root']/'PREFLIGHT.json')
    require(pre['status']=='passed' and all(pre[k]==v for k,v in x['binding'].items()),'Passed bound preflight required')
    require(set(pre['arms'])==set(ARMS),'Preflight omitted arm')
    return sha(x['root']/'PREFLIGHT.json')


def train(x,out,arm,deadline):
    prehash=require_preflight(x);state,source=restore600(x,arm);r=x['r'];started=time.monotonic();completed=600
    state['model'].train();state['bridge'].train()
    write(out/'SOURCE.json',evidence(x,arm,source))
    write(out/'PLAN.json',dict(starts=x['p']['train_plan'],source_steps=600,total_steps=800,**x['binding']))
    hashes={}
    try:
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for index in range(600,800):
                guard(deadline);r['runner'].ensure_eval_codec(state['bridge'])
                measured=step_native(state['model'],state['bridge'],state['optimizer'],r['data'],x['p']['train_plan'][index],
                                     r['cfg'],r['temporal'],x['p']['arms'][arm]['teacher_weights'])
                completed=index+1;x['v9'].invariants(r,state);guard(deadline)
                measured.update(step=completed,added_steps=completed-600,starts=x['p']['train_plan'][index],
                                elapsed_seconds=time.monotonic()-started,
                                spatial_action_grad_norm_after_clip=state['spatial_grad']())
                log.write(json.dumps(measured,allow_nan=False)+'\n')
                if completed%10==0:print(arm,'STEP',completed,'L1',measured['full_l1'],flush=True)
                if completed in x['p']['train']['save_steps']:
                    path=out/f'step_{completed:04d}.pt';save_checkpoint(path,x,state,arm,source,completed);hashes[str(completed)]=sha(path)
        unchanged(x);guard(deadline)
        write(out/'TRAIN_DONE.json',dict(status='complete',**evidence(x,arm,source),completed_steps=800,steps=800,
            additional_steps=200,checkpoint_sha256=hashes,preflight_sha256=prehash,source_sha256=sha(out/'SOURCE.json'),
            plan_sha256=sha(out/'PLAN.json'),train_jsonl_sha256=sha(out/'train.jsonl'),last=measured,
            training_seconds=time.monotonic()-started,encoder_unchanged=True,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30))
    except BaseException:
        try:save_checkpoint(out/'partial.pt',x,state,arm,source,completed,partial=True)
        except BaseException:pass
        raise


def restore800(x,arm):
    out=x['root']/'arms'/arm;done=read(out/'TRAIN_DONE.json');path=out/'step_0800.pt'
    require(done['status']=='complete' and done['steps']==800 and done['checkpoint_sha256']['800']==sha(path),
            'Complete bound step800 required')
    state,source=restore600(x,arm)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    require(saved['completed_steps']==800 and saved['plan']==x['p']['train_plan'] and saved['config']==x['r']['cfg'],
            'Stored checkpoint scope changed')
    require(saved['continuation']==dict(schema='gs_supervision_checkpoint_v10',**evidence(x,arm,source),partial=False),
            'Checkpoint provenance changed')
    state['model'].load_state_dict(saved['model'],strict=True);state['bridge'].codec.load_state_dict(saved['codec'],strict=True)
    state['optimizer'].load_state_dict(saved['optimizer']);x['v9'].restore_rng(x['r'],saved)
    x['v9'].invariants(x['r'],state);state['model'].eval();state['bridge'].eval()
    return state,source,done,path


def native_packet(bridge,prediction,record,start,h,pts,label,normalization):
    points=bridge.codec.decode_state(prediction['coordinates'],prediction['features']);gs,sh=bridge.prediction_gs(points)
    require(gs.shape[1:]==(14,) and sh.shape==(len(gs),4,3) and len(gs)>0,'Invalid complete native packet')
    return dict(schema='gs_supervision_native_packet_v10',gs=gs.cpu(),sh=sh.cpu(),
        cameras=record['cameras'].cpu(),intrinsics=record['intrinsics'].cpu(),near=float(record['near']),far=float(record['far']),
        resolution_hw=[252,448],start=start,horizon=h,frame=start+h,pts=float(pts),label=label,normalization=normalization,
        target_state_reference=(label=='decoded_actual_target_state'),posterior_reconstruction=True,goal_complete=False)


def evaluate(x,out,arm,deadline):
    require_preflight(x);state,source,done,checkpoint=restore800(x,arm)
    r=x['r'];data=r['data'];core=r['core'];v7=r['runner'];model,bridge=state['model'],state['bridge']
    rows=[];initial_rows=[];snapshot_rows=[];packet_rows=[]
    mapping=x['v9'].snapshot_mapping(x['e']['context']);(out/'snapshots').mkdir();(out/'native_packets').mkdir()
    def capture(start,h,images):
        for frame,pair in mapping.items():
            if pair!=(start,h):continue
            files={}
            for name,image in images.items():
                path=out/'snapshots'/f'frame_{frame:03d}_{name}.png';x['v9'].save_image(path,image)
                files[name]=dict(path=str(path),sha256=sha(path))
            snapshot_rows.append(dict(source_frame=frame,start=start,horizon=h,files=files))
    with torch.no_grad():
        ids=[];blind=[];constant=model.code_mode=='constant'
        for start in range(0,168,4):
            guard(deadline);windows=[data.window(s) for s in range(start,start+4)]
            ids.append(model.encode_windows(windows)['indices'])
            if not constant:blind.append(model.encode_windows([x['e']['context']['probe'].blind_window(w) for w in windows])['indices'])
        ids=torch.cat(ids);blind_ids=ids if constant else torch.cat(blind)
        require(ids.shape==(168,9) and blind_ids.shape==(168,9) and bool(((ids>=0)&(ids<65536)).all()),'Invalid whole-window IDs')
        require(not constant or bool((ids==model.constant_id).all()),'Constant control not constant')
        with (out/'targets.jsonl').open('x',encoding='utf-8') as log:
            for start in range(168):
                guard(deadline);condition=[core.make_condition(data.states[start])]
                predictions={'correct':model.decode_ids(ids[start:start+1],condition)[0]}
                if not constant:
                    for offset in (42,84,126):
                        index=(start+offset)%168;predictions[f'shuffle{offset}']=model.decode_ids(ids[index:index+1],condition)[0]
                    predictions['blinded']=model.decode_ids(blind_ids[start:start+1],condition)[0]
                require(all(len(frames)==10 and all(torch.equal(f['coordinates'],data.states[start]['coordinates']) for f in frames)
                            for frames in predictions.values()),'Full initial support/ten frames changed')
                initial={name:bridge.render(frames[0],data.records[start]) for name,frames in predictions.items()}
                if constant:
                    for name in ('shuffle42','shuffle84','shuffle126','blinded'):initial[name]=initial['correct']
                initial['adapted_held']=bridge.render(data.states[start],data.records[start])
                initial_row=dict(start=start,end=start,horizon=0,held_initial=None,frozen_target=None)
                initial_row.update({name:metrics(data,image,start,0) for name,image in initial.items()});initial_rows.append(initial_row)
                capture(start,0,dict(initial,real=data.records[start]['real']))
                for h in range(1,10):
                    row=dict(start=start,end=start+h,horizon=h,motion_ids=ids[start].cpu().tolist(),
                        blinded_motion_ids=blind_ids[start].cpu().tolist(),blinded_horizon_id_changed=bool(ids[start,h-1]!=blind_ids[start,h-1]),
                        **data.reference[start,h],delta={'held_initial':None,'frozen_target':None})
                    snapshots={};wanted=(start,h) in mapping.values()
                    for name,frames in predictions.items():
                        image=bridge.render(frames[h],data.records[start+h]);row[name]=metrics(data,image,start,h)
                        row['delta'][name]=v7.delta_metrics(image,initial[name],data,start,h)
                        if wanted and name in ('correct','shuffle84','blinded'):snapshots[name]=image
                    if constant:
                        for name in ('shuffle42','shuffle84','shuffle126','blinded'):
                            row[name]=dict(row['correct']);row['delta'][name]=dict(row['delta']['correct'])
                            if wanted and name in ('shuffle84','blinded'):snapshots[name]=snapshots['correct']
                    held=bridge.render(data.states[start],data.records[start+h]);row['adapted_held']=metrics(data,held,start,h)
                    row['delta']['adapted_held']=v7.delta_metrics(held,initial['adapted_held'],data,start,h)
                    if wanted:capture(start,h,dict(snapshots,adapted_held=held,real=data.records[start+h]['real']))
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');guard(deadline)
                if start in (35,123):
                    # Save independent native packets for both complete failure windows.
                    for label in ('correct','shuffle84','decoded_actual_target_state'):
                        frames=(data.states[start:start+10] if label=='decoded_actual_target_state' else
                                predictions['correct' if constant and label=='shuffle84' else label])
                        for h,pred in enumerate(frames):
                            packet=native_packet(bridge,pred,data.records[start+h],start,h,data.states[start+h]['pts'],label,data.normalization)
                            path=out/'native_packets'/f's{start:03d}_h{h:02d}_{label}.pt';save(path,packet)
                            packet_rows.append(dict(start=start,horizon=h,label=label,path=str(path),sha256=sha(path),gs_count=len(packet['gs'])))
                            if h==9:
                                views=render_views(bridge,pred,data.records[start+h])
                                for view in range(2):
                                    png=out/'snapshots'/f'frame_{start+h:03d}_{label}_view{view}.png'
                                    x['v9'].save_image(png,views[view])
                    for view in range(2):
                        x['v9'].save_image(out/'snapshots'/f'frame_{start+9:03d}_raw_teacher_view{view}.png',data.teacher_targets[start+9][view])
                log.flush();del predictions,initial
                if start%21==0:print(arm,'EVAL',start+1,'/168',flush=True)
        v7.validate_targets(rows);require(len(initial_rows)==168 and len(snapshot_rows)==5 and len(packet_rows)==60,'Incomplete evaluation artifacts')
        save(out/'actions.pt',dict(schema='gs_supervision_actions_v10',ids=ids.cpu(),blinded_ids=blind_ids.cpu(),arm=arm,steps=800,
            starts=list(range(168)),horizons=list(range(1,10)),source_frames=177,frame_gap=1,checkpoint_sha256=sha(checkpoint),
            **x['binding'],code_mode=model.code_mode,constant_id=model.constant_id if constant else None,
            semantics='Whole nine-code posterior bundles, not independent adjacent AR actions'))
        write(out/'initial_reconstruction.json',dict(rows=initial_rows,unique_initial_frames=168,uncomputed_controls=['held_initial','frozen_target']))
        write(out/'snapshots.json',dict(rows=snapshot_rows,all_pngs={p.name:sha(p) for p in (out/'snapshots').glob('*.png')}))
        write(out/'native_packets.json',dict(rows=packet_rows,model_free_native_format=True,independent_replay_performed=False))
        entropy=[];distinct=[]
        for h in range(9):
            _,counts=ids[:,h].unique(return_counts=True);prob=counts.float()/168
            entropy.append(float(-(prob*prob.log()).sum()));distinct.append(len(counts))
        x['v9'].invariants(r,state);unchanged(x);guard(deadline)
        write(out/'EVALUATION.json',dict(status='complete',**evidence(x,arm,source),steps=800,completed_steps=800,selected_step=800,
            checkpoint_sha256=sha(checkpoint),train_done_sha256=sha(x['root']/'arms'/arm/'TRAIN_DONE.json'),optimizer_updates=0,
            windows=168,real_future_exposures=1512,unique_future_frames=176,physical_frames_with_some_reconstruction=177,
            metrics={name:core.aggregate(rows,name) for name in v7.ALL_CONTROLS},
            delta_metrics={name:v7.aggregate_delta(rows,name) for name in v7.ALL_CONTROLS},
            conditional_entropy_nats=sum(entropy)/9,per_h_entropy_nats=entropy,per_h_distinct_ids=distinct,
            distinct_bundles=len(torch.unique(ids,dim=0)),
            blinded_id_statistics=x['e']['context']['probe'].id_statistics(ids.cpu().tolist(),blind_ids.cpu().tolist()),
            targets_sha256=sha(out/'targets.jsonl'),initial_reconstruction_sha256=sha(out/'initial_reconstruction.json'),
            actions_sha256=sha(out/'actions.pt'),snapshots_sha256=sha(out/'snapshots.json'),native_packets_sha256=sha(out/'native_packets.json'),
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,
            scope='single TRAIN clip; posterior reconstruction, not AR or cross-source',
            object_box=[140,25,320,235],object_box_semantics='fixed rectangle, not object segmentation',visual_acceptance='pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=('preflight','train','evaluate'));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();root=args.root.resolve()
    if (args.mode=='preflight')!=(args.arm is None):parser.error('--arm required exactly for train/evaluate')
    p,binding=verify(root,cpu_witness=True);section={'preflight':'preflight','train':'train','evaluate':'evaluation'}[args.mode]
    deadline=time.monotonic()+p[section]['max_seconds']
    def expired(*unused):raise TimeoutError('Fixed worker wall budget expired')
    if hasattr(signal,'SIGALRM'):
        signal.signal(signal.SIGALRM,expired);signal.alarm(p[section]['max_seconds'])
    signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGINT,expired)
    out=root/'preflight' if args.mode=='preflight' else root/('arms' if args.mode=='train' else 'evaluations')/args.arm
    require(not out.exists(),'Existing attempt cannot be overwritten/retried');out.mkdir(parents=True)
    try:
        admission=memory_admission(p);x=runtime(root,p,binding);guard(deadline)
        write(out/'MEMORY_ADMISSION.json',admission)
        if args.mode=='preflight':preflight(x,out,deadline)
        elif args.mode=='train':train(x,out,args.arm,deadline)
        else:evaluate(x,out,args.arm,deadline)
    except BaseException as error:
        write(out/'FAILED.json',dict(status='failed',mode=args.mode,arm=args.arm,error=repr(error),
            traceback=traceback.format_exc(),goal_complete=False));raise
    finally:
        if hasattr(signal,'SIGALRM'):signal.alarm(0)


if __name__=='__main__':main()
