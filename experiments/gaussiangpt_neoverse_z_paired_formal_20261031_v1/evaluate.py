"""Evaluate all windows with saved actual action codes and native GS packets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
from codec_binding import require,sha
from formal_common import verify_sources,gpu_setup,guard,build,write_new,model_digest
from native_packet import save_new
from runner_core import (CODE_CONTROLS,blind_window,make_action_packet,write_action_packet,
                         read_action_packet,select_control_codes,evaluation_coverage)
from runtime_bridge import _bound_module

BASE=Path('/data/CJH/AGS-V6Direct/outputs')
INFERENCE=BASE/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/inference_profile.py'
INFERENCE_SHA='fa6695b2ec3c979d11e2e6c7b2a50f34965cf41204e2b551d778349f151ec7f0'


def save_png(path,image):
    array=(image.detach().cpu().numpy().clip(0,1)*255).round().astype('uint8')
    Image.fromarray(array).save(path)


def render_and_packet(bridge,pred,record,packet_path=None,reference_path=None,metadata=None):
    points=bridge.decode_points(pred['coordinates'],pred['features'])
    gs,sh=bridge.prediction_gs(points)
    require(0<len(gs)<=2400000 and tuple(sh.shape)==(len(gs),4,3),'Invalid complete decoded native GS')
    images=torch.stack([bridge.render_fn(gs,cam,record['intrinsics'],record['near'],record['far'],sh)[0].clamp(0,1)
                        for cam in record['cameras']])
    require(tuple(images.shape)==(2,252,448,3) and bool(torch.isfinite(images).all()),'Invalid rendered views')
    if packet_path is None: return images,None
    packet=dict(schema='z_action_native_packet_v1',gs=gs.detach().cpu().contiguous(),sh=sh.detach().cpu().contiguous(),
        cameras=record['cameras'].detach().cpu().contiguous(),intrinsics=record['intrinsics'].detach().cpu().contiguous(),
        near=float(record['near']),far=float(record['far']),resolution_hw=[252,448],**metadata)
    info=save_new(packet_path,packet)
    reference=save_new(reference_path,images.detach().cpu().contiguous())
    return images,dict(**metadata,gs_count=len(gs),packet=info,reference=reference,
                       packet_file=packet_path.name,reference_file=reference_path.name)


def delta_metrics(image,initial,data,start,h):
    actual=data.records[start+h]['real']-data.records[start]['real']
    error=((image-initial)-actual).abs().mean(-1)
    weight=data.motion_weight(start,h)
    return dict(delta_mae=float(error.mean()),hand_activity_delta_mae=float(error[data.roi].mean()),
                motion_delta_mae=float((error*weight).sum()/weight.sum().clamp_min(1e-6)))


def metrics(data,image,start,h):
    result=data.metrics(image,start,h)
    result['object_box_mae']=float((image[25:235,140:320]-data.records[start+h]['real'][25:235,140:320]).abs().mean())
    return result


def aggregate(rows,controls):
    require(len(rows)==1512 and {(r['start'],r['horizon']) for r in rows}=={(s,h) for s in range(168) for h in range(1,10)},
            'Incomplete/duplicate future coverage')
    result={}
    for name in controls:
        keys=list(rows[0]['metrics'][name])
        grouped={frame:[r for r in rows if r['frame']==frame] for frame in range(1,177)}
        require(all(grouped.values()),'Missing physical frame')
        by_frame={str(f):{k:float(np.mean([r['metrics'][name][k] for r in group])) for k in keys}
                  for f,group in grouped.items()}
        delta_keys=list(rows[0]['delta'][name])
        delta_by_frame={str(f):{k:float(np.mean([r['delta'][name][k] for r in group])) for k in delta_keys}
                        for f,group in grouped.items()}
        result[name]=dict(frame_uniform={k:float(np.mean([v[k] for v in by_frame.values()])) for k in keys},
            exposure_mean={k:float(np.mean([r['metrics'][name][k] for r in rows])) for k in keys},
            per_horizon={str(h):{k:float(np.mean([r['metrics'][name][k] for r in rows if r['horizon']==h])) for k in keys}
                         for h in range(1,10)},by_physical_frame=by_frame,
            delta_frame_uniform={k:float(np.mean([v[k] for v in delta_by_frame.values()])) for k in delta_keys},
            delta_per_horizon={str(h):{k:float(np.mean([r['delta'][name][k] for r in rows if r['horizon']==h]))
                               for k in delta_keys} for h in range(1,10)},delta_by_physical_frame=delta_by_frame)
    return result


def verify_final_latent_replay(rt,p,out,frames=None):
    inference=_bound_module('_z_final_inference_profile',INFERENCE,INFERENCE_SHA)
    bridge,data=rt['bridge'],rt['data']
    frames=list(range(177)) if frames is None else list(frames)
    require(frames and len(set(frames))==len(frames) and all(type(f) is int and 0<=f<177 for f in frames),
            'Invalid latent replay frame selection')
    records=[]
    def decode_sparse(z,record):
        points,_=bridge.codec.ae.decode(bridge.precision.prepare_sparse(z,bridge.ME),target_key=None)
        points=bridge.precision.to_render_points(bridge.codec._world(points))
        gs,sh=bridge.prediction_gs(points)
        return torch.stack([bridge.render_fn(gs,c,record['intrinsics'],record['near'],record['far'],sh)[0].clamp(0,1)
                            for c in record['cameras']]).detach().cpu()
    with (out/'latent_replay.jsonl').open('x',encoding='utf8',buffering=1) as log:
        for frame in frames:
            scope={};state=data.states[frame];record=data.records[frame]
            def direct(register):
                z=bridge.codec._sparse(state['coordinates'],state['features'],4)
                register(z.coordinate_manager)
                return decode_sparse(z,record)
            def load_from_disk(_): return data.binding.load(frame,device='cuda')
            def replay(loaded,release):
                z=bridge.codec._sparse(loaded['coordinates'],loaded['features'],4)
                release(z.coordinate_manager)
                return decode_sparse(z,record)
            first,second=inference.independent_replay(torch,bridge.ME,direct,replay,load_from_disk,scope)
            difference=first-second
            views=[]
            for view in range(2):
                mae=float(difference[view].abs().mean())
                psnr=float(-10*torch.log10(difference[view].square().mean().clamp_min(1e-12)))
                views.append(dict(mae=mae,psnr=psnr,passed=mae<=p['replay_mae_max'] and psnr>=p['replay_psnr_min']))
            row=dict(frame=frame,views=views,passed=all(v['passed'] for v in views),independent_manager=True)
            records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
    require(all(r['passed'] for r in records),'Final adapted codec latent replay failed')
    return dict(frames=len(frames),views=2*len(frames),failed_frames=0,max_mae=max(v['mae'] for r in records for v in r['views']),
                min_psnr=min(v['psnr'] for r in records for v in r['views']),
                rows_sha256=sha(out/'latent_replay.jsonl'),independent_manager=True)


def main(mode):
    root=Path(__file__).resolve().parent
    p,source=verify_sources(root)
    arm=root/'arms'/mode
    require((arm/'TRAIN_DONE.json').is_file() and not (arm/'TRAIN_FAILED.json').exists(),'Complete successful training required')
    done=json.loads((arm/'TRAIN_DONE.json').read_text())
    path=arm/'step_0800.pt'
    require(done['status']=='complete' and done['mode']==mode and done['steps']==800 and done['source']==source
            and sha(path)==done['checkpoint_sha256'],'Final training receipt mismatch')
    out=arm/'evaluation'
    out.mkdir(exist_ok=False)
    for folder in ('native_packets','native_refs','snapshots'): (out/folder).mkdir()
    started=time.monotonic();context={'phase':'initializing'}
    write_new(out/'STARTED.json',dict(pid=os.getpid(),mode=mode,checkpoint_sha256=done['checkpoint_sha256']))
    try:
        gpu_setup(p)
        rt=build(p,source,mode,training=False)
        model,bridge,data=rt['model'],rt['bridge'],rt['data']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['schema']=='paired_action_checkpoint_v1' and saved['mode']==mode and saved['completed_steps']==800
                and saved['source']==source and saved['plan_sha256']==p['plan_sha256'],'Checkpoint metadata mismatch')
        model.load_state_dict(saved['model'],strict=True)
        bridge.codec.load_state_dict(saved['codec'],strict=True)
        del saved
        model.eval();bridge.eval();bridge.assert_frozen()
        require(model_digest(model)==done['final_model_digest'] and model_digest(bridge.codec)==done['final_codec_digest'],
                'Final model/codec digest differs from training')
        context['phase']='final_codec_latent_replay'
        with torch.no_grad():
            latent=verify_final_latent_replay(rt,p,out)
            require(latent['frames']==177 and latent['views']==354,'Full final codec replay required')
            context['phase']='encode_actual_actions'
            action_batches=[];blind_batches=[];id_batches=[];diagnostics=[]
            for start in range(0,168,4):
                guard(p,started,'evaluation')
                windows=[data.window(s) for s in range(start,start+4)]
                encoded=model.encode_windows(windows)
                action_batches.append(encoded['action_codes'].cpu())
                diagnostics.append(encoded['diagnostic_ids'].cpu())
                if mode=='lfq16': id_batches.append(encoded['indices'].cpu())
                else: require(encoded['indices'] is None,'Continuous actions unexpectedly expose IDs')
                blinded=[]
                for s in range(start,start+4):
                    c=blind_window([data.binding.load(f) for f in range(s,s+10)])
                    blinded.append([{k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in state.items()} for state in c])
                blind_batches.append(model.encode_windows(blinded)['action_codes'].cpu())
            actions=torch.cat(action_batches);blinded=torch.cat(blind_batches);diag=torch.cat(diagnostics)
            require(tuple(actions.shape)==tuple(blinded.shape)==(168,9,16) and tuple(diag.shape)==(168,9),'Action coverage')
            if mode=='lfq16':
                ids=torch.cat(id_batches)
                require(torch.equal(model.quantizer.indices_to_codes(ids.cuda(),project_out=False).cpu(),actions),'LFQ IDs/signs differ')
            else: ids=None
            packet_source=dict(model_checkpoint_sha256=done['checkpoint_sha256'],codec_binding_sha256=sha(root/'codec_binding.py'),
                state_manifest_sha256=source['state_manifest_sha256'],adapter_sha256=sha(root/'action_bottleneck.py'),
                final_codec_digest=done['final_codec_digest'],kind='actual_actions')
            packet=make_action_packet(actions,mode=mode,starts=list(range(168)),s_ref=p['s_ref'],source=packet_source)
            packet_info=write_action_packet(out/'actions.pt',packet)
            packet=read_action_packet(out/'actions.pt',expected_file_sha256=packet_info['file_sha256'],expected_mode=mode,
                expected_source=packet_source,expected_s_ref=p['s_ref'],expected_starts=list(range(168)))
            blind_source=dict(packet_source,kind='blinded_posterior')
            blind_packet=make_action_packet(blinded,mode=mode,starts=list(range(168)),s_ref=p['s_ref'],source=blind_source)
            blind_info=write_action_packet(out/'blinded_actions.pt',blind_packet)
            blind_packet=read_action_packet(out/'blinded_actions.pt',expected_file_sha256=blind_info['file_sha256'],
                expected_mode=mode,expected_source=blind_source,expected_s_ref=p['s_ref'],expected_starts=list(range(168)))
            ids_info=save_new(out/'action_ids.pt',dict(mode=mode,actual_ids=ids,diagnostic_ids=diag,
                                                      continuous_ids_are_diagnostics_only=(mode=='continuous16')))
            control_codes={name:select_control_codes(packet,name,list(range(168)),expected_mode=mode,
                expected_source=packet_source,expected_s_ref=p['s_ref'])['codes'] for name in CODE_CONTROLS}
            control_codes['blinded_posterior']=blind_packet['codes']
            labels=list(p['evaluation_controls']);rows=[];initial_rows=[];native_rows=[]
            context['phase']='render_full_controls'
            with (out/'targets.jsonl').open('x',encoding='utf8',buffering=1) as log:
                for start in range(168):
                    context['start']=start
                    guard(p,started,'evaluation')
                    condition=[rt['api'].make_condition(data.states[start])]
                    predicted={name:model.decode_action_codes(codes[start:start+1].cuda(),condition)[0]
                               for name,codes in control_codes.items()}
                    predicted['adapted_held']=[data.states[start]]*10
                    predicted['decoded_actual_target_state']=data.states[start:start+10]
                    require(set(predicted)==set(labels) and all(len(frames)==10 for frames in predicted.values()),
                            'Control coverage mismatch')
                    initial={}
                    for h in range(10):
                        row=dict(start=start,horizon=h,frame=start+h,metrics={},delta={})
                        for name in labels:
                            need=start in p['native_packet_windows'] and name in p['native_packet_controls']
                            name_base=f's{start:03d}_h{h:02d}_{name}'
                            meta=dict(start=start,horizon=h,frame=start+h,control=name,mode=mode,
                                pts=float(data.states[start+h]['pts']),normalization=data.normalization,
                                model_checkpoint_sha256=done['checkpoint_sha256'],final_codec_digest=done['final_codec_digest'],
                                target_state_reference=name=='decoded_actual_target_state',posterior_reconstruction=True)
                            images,native_row=render_and_packet(bridge,predicted[name][h],data.records[start+h],
                                out/'native_packets'/f'{name_base}.pt' if need else None,
                                out/'native_refs'/f'{name_base}.pt' if need else None,meta if need else None)
                            if native_row:
                                native_rows.append(native_row)
                                for view in (0,1): save_png(out/'snapshots'/f'{name_base}_v{view}.png',images[view])
                            if h==0: initial[name]=images[0].clone()
                            row['metrics'][name]=metrics(data,images[0],start,h)
                            if h: row['delta'][name]=delta_metrics(images[0],initial[name],data,start,h)
                        if start in p['native_packet_windows']:
                            save_png(out/'snapshots'/f's{start:03d}_h{h:02d}_real_v0.png',data.records[start+h]['real'])
                            save_png(out/'snapshots'/f's{start:03d}_h{h:02d}_teacher_v1.png',data.teacher_targets[start+h][1])
                        if h==0: initial_rows.append(row)
                        else: rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                    if start%21==0: print('EVALUATE',mode,start+1,'/168',flush=True)
            require(len(initial_rows)==168 and len(native_rows)==80,'Initial/native packet coverage')
            aggregate_result=aggregate(rows,labels)
            write_new(out/'initial.json',dict(rows=initial_rows))
            write_new(out/'native_manifest.json',dict(schema='z_native_packet_manifest_v1',mode=mode,
                checkpoint_sha256=done['checkpoint_sha256'],final_codec_digest=done['final_codec_digest'],
                packets=native_rows,independent_replay_performed=False))
            stats={}
            for h in range(9):
                if mode=='lfq16':
                    _,counts=ids[:,h].unique(return_counts=True);probs=counts.float()/168
                    stats[str(h+1)]=dict(distinct_ids=len(counts),entropy_nats=float(-(probs*probs.log()).sum()),
                                         bit_occupancy=(actions[:,h,:]>0).float().mean(0).tolist())
                else:
                    stats[str(h+1)]=dict(variance=actions[:,h,:].var(0,unbiased=False).tolist(),
                                         rms=float(actions[:,h,:].square().mean().sqrt()))
            write_new(out/'EVALUATION.json',dict(status='complete',mode=mode,source=source,steps=800,
                checkpoint_sha256=done['checkpoint_sha256'],final_codec_digest=done['final_codec_digest'],
                windows=168,future_exposures=1512,physical_future_frames=176,initial_exposures=168,
                actual_codes_shape=[168,9,16],actions=packet_info,blinded_actions=blind_info,ids=ids_info,
                latent_replay=latent,controls=labels,metrics=aggregate_result,action_statistics=stats,
                targets_sha256=sha(out/'targets.jsonl'),native_manifest_sha256=sha(out/'native_manifest.json'),
                independent_native_replay_performed=False,visual_acceptance='pending',scope=p['scope'],
                elapsed_seconds=time.monotonic()-started))
        bridge.assert_frozen()
        require(verify_sources(root)[1]==source,'Source drift during evaluation')
    except BaseException as error:
        write_new(out/'EVALUATE_FAILED.json',dict(status='failed',mode=mode,context=context,error=repr(error),
                  traceback=traceback.format_exc(),source=source))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['continuous16','lfq16'])
    main(parser.parse_args().mode)
