"""Fit a causal Q-Former to a fixed two-window motion/static free-code teacher."""
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from q_utils import ROOT, expected_packet_names, read, schedule, sha, validate_protocol, verify_self, write

OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
BROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PREVIEW=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
STATIC=OUTPUTS/'gaussiangpt_neoverse_qformer_static_control_20261102_v1'
PAIRED=OUTPUTS/'gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1'
VISUAL=OUTPUTS/'gaussiangpt_neoverse_two_future_visual_20261102_v2'
REPLAY=OUTPUTS/'gaussiangpt_neoverse_two_future_native_replay_20261102_v2'


def verify_teacher_sources(cfg):
    paths=[(PAIRED/'CODE.sha256','paired_source_code_sha256'),
           (PAIRED/'protocol.json','paired_protocol_sha256'),
           (PAIRED/'result/SUMMARY.json','paired_summary_sha256'),
           (PAIRED/'result/step_00360/EVALUATION.json','paired_evaluation_sha256'),
           (PAIRED/'result/checkpoint_00360.pt','teacher_checkpoint_sha256'),
           (VISUAL/'PILOT_REVIEW.json','visual_review_sha256'),
           (REPLAY/'result/REPLAY.json','native_replay_sha256'),
           (REPLAY/'result/PACKETS.json','native_packets_sha256')]
    for path,key in paths:
        if sha(path)!=cfg[key]:raise RuntimeError('Bound teacher source changed: '+str(path))
    summary=read(PAIRED/'result/SUMMARY.json')
    evaluation=read(PAIRED/'result/step_00360/EVALUATION.json')
    review=read(VISUAL/'PILOT_REVIEW.json')
    replay=read(REPLAY/'result/REPLAY.json')
    manifest=read(REPLAY/'result/PACKETS.json')
    done=read(REPLAY/'DONE.json')
    if (summary['status']!='complete' or summary['updates']!=360
            or summary['checkpoints']['360']!=cfg['teacher_checkpoint_sha256']
            or evaluation['status']!='complete' or evaluation['summary']!=summary['summaries']['360']
            or review['status']!='reviewed' or review['scope']!='two_known_training_windows_only'
            or review['suitable_for_limited_qformer_diagnostic'] is not True
            or review['full_168_teacher_accepted'] is not False
            or review['strict_proxy_gate_passed'] is not False
            or review['evaluation_sha256']!=cfg['paired_evaluation_sha256']):
        raise RuntimeError('Paired pilot review, scope or evaluation differs')
    if (sha(VISUAL/'result/MANIFEST.json')!=review['visual_manifest_sha256']
            or sha(VISUAL/'result/OBJECT_PROXY.json')!=review['object_proxy_sha256']):
        raise RuntimeError('Paired visual review evidence changed')
    if (done['status']!='complete' or done['replay_sha256']!=cfg['native_replay_sha256']
            or replay['status']!='passed' or replay['packets']!=36 or replay['views']!=72
            or replay['packets_sha256']!=cfg['native_packets_sha256']
            or replay['encoder_or_codec_loaded'] is not False or replay['producer_pid']==replay['replay_pid']
            or len(replay['rows'])!=36 or {r['name'] for r in replay['rows']}!=expected_packet_names(cfg)
            or not all(r['passed'] and len(r['views'])==2 and all(v['passed'] for v in r['views']) for r in replay['rows'])
            or manifest['source']['checkpoint_sha256']!=cfg['teacher_checkpoint_sha256']
            or manifest['source']['decoder_digest']!=cfg['teacher_decoder_digest']):
        raise RuntimeError('Independent teacher GS replay failed or differs')
    return evaluation


def camera_only(tree,record):
    return {k:tree(record[k],'cuda') for k in ('cameras','intrinsics','near','far')}


def train_step(posterior,decoder,bridge,store,items,teacher_codes,scale,optimizer,std,p,phase,horizon,*,probe=False):
    import torch
    from core import image_losses
    from support_decoder import select_state,state_loss
    from runtime import render
    if phase not in ('distill','joint') or (phase=='joint' and not 1<=horizon<=9):
        raise ValueError('Invalid paired Q training phase')
    expected=[v for v in posterior.parameters() if v.requires_grad]
    actual=[v for group in optimizer.param_groups for v in group['params']]
    if (len(expected)!=len(actual) or {id(v) for v in expected}!={id(v) for v in actual}
            or any(v.requires_grad for v in decoder.parameters()) or any(v.requires_grad for v in bridge.parameters())):
        raise RuntimeError('Only the Q-Former may be updated')
    optimizer.zero_grad(set_to_none=True)
    rows=[]
    for i,start in enumerate((35,123)):
        item=items[start];w=item['window']
        for branch in ('motion','static'):
            memories=w['memories'] if branch=='motion' else w['initial_memories']
            codes=posterior([memories])['codes'][0]
            target=teacher_codes[branch][i]
            distill=(codes-target).square().mean()/scale
            loss=distill
            row={'start':start,'branch':branch,'code_loss':float(distill.detach())}
            if phase=='joint':
                sample=store.sample(start,horizon)
                if branch=='motion':
                    labels=w['targets'];image_target=sample['record']['real'];weight=sample['motion']
                else:
                    labels=item['static_labels'];image_target=item['static_images'][horizon];weight=item['weights'][horizon]
                prediction=decoder(item['condition'],codes)
                latent,latent_terms=state_loss(prediction,labels,std)
                state=select_state(item['condition'].coordinates,prediction['features'][horizon-1],
                                   prediction['logits'][horizon-1],p['max_predicted_latent_nodes'])
                images,_,_=render(bridge,state,item['records'][horizon],views=(0,))
                rgb,rgb_terms=image_losses(images[0],image_target,item['roi'],weight)
                grads=None
                if probe:
                    grads=[float(g.norm()) for g in torch.autograd.grad(
                        rgb,(posterior.queries.weight,posterior.project_in.weight),retain_graph=True)]
                    if any(not math.isfinite(v) or v<=0 for v in grads):
                        raise RuntimeError('RGB gradient failed to reach '+branch+' Q-Former')
                loss=rgb+.1*latent+.1*distill
                row.update(horizon=horizon,rgb=float(rgb.detach()),latent=float(latent.detach()),
                    motion_weighted_mae=float(rgb_terms['motion_mae'].detach()),
                    occupancy=float(latent_terms['occupancy'].detach()),feature=float(latent_terms['feature'].detach()),
                    rgb_to_query_projection_gradients=grads)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite Q-Former loss')
            row['loss']=float(loss.detach())
            (loss/4).backward()
            rows.append(row)
    if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in expected):
        raise RuntimeError('Nonfinite Q-Former gradient')
    if any(v.grad is not None for v in decoder.parameters()) or any(v.grad is not None for v in bridge.parameters()):
        raise RuntimeError('Frozen decoder or codec acquired gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(expected,p['grad_clip']))
    optimizer.step()
    if any(not bool(torch.isfinite(v).all()) for v in expected):raise RuntimeError('Nonfinite posterior parameter')
    return {'examples':rows,'preclip_gradient_norm':norm}


def causal_check(posterior,decoder,items):
    """Change true raw states after h, then rebuild all memories independently."""
    import torch
    from posterior import make_memories
    rows=[];posterior.eval();decoder.eval()
    with torch.no_grad():
        for start in (35,123):
            states=items[start]['window']['states']
            for horizon in (3,7):
                changed=[]
                for t,state in enumerate(states):
                    if t<=horizon:changed.append(state)
                    else:changed.append(dict(state,features=state['features']+1.,coordinates=state['coordinates']+100))
                before=posterior([make_memories(decoder.base,states)])['codes'][0,:horizon]
                after=posterior([make_memories(decoder.base,changed)])['codes'][0,:horizon]
                error=float((before-after).abs().max())
                if not math.isfinite(error) or error>2e-5:
                    raise RuntimeError('Raw future-prefix causality failed: '+str((start,horizon,error)))
                rows.append({'start':start,'horizon':horizon,'earlier_code_max_abs_error':error})
    return {'status':'passed','rows':rows,'tolerance':2e-5,
            'method':'Recompute memories after perturbing raw states strictly after each checked horizon',
            'scope':'Two known training windows; encoder cached/frozen, no online prediction claim'}


def evaluate(step_number,posterior,decoder,bridge,store,items,teacher_codes,p,cfg,teacher_evaluation,source):
    import numpy as np
    from PIL import Image
    import torch
    from data import evaluate_metrics
    from runtime import cpu,render,save
    from support_decoder import select_state
    import preview
    import static_probe
    out=ROOT/'result'/f'step_{step_number:05d}'
    out.mkdir()
    final=step_number==1194
    if final:
        (out/'packets').mkdir();(out/'references').mkdir()
    old={(r['start'],r['horizon']):r for r in teacher_evaluation['rows']}
    student_rows=[];teacher_rows=[];packets=[];teacher_diff=[]
    posterior.eval();decoder.eval()
    with torch.no_grad():
        for i,start in enumerate(cfg['windows']):
            item=items[start];w=item['window']
            qcodes={b:posterior([w['memories'] if b=='motion' else w['initial_memories']])['codes'][0]
                    for b in cfg['conditions']}
            codes={'q_motion':qcodes['motion'],'q_static':qcodes['static'],
                   'teacher_motion':teacher_codes['motion'][i],'teacher_static':teacher_codes['static'][i]}
            predictions={name:decoder(item['condition'],code) for name,code in codes.items()}
            folder=out/f's{start:03d}'
            if final:folder.mkdir()
            for h in cfg['horizons']:
                sample=store.sample(start,h);record=item['records'][h];rendered={}
                for name,pred in predictions.items():
                    state=select_state(item['condition'].coordinates,pred['features'][h-1],
                                       pred['logits'][h-1],p['max_predicted_latent_nodes'])
                    rendered[name]=render(bridge,state,record,views=(0,1))
                ref_motion,ref_static=item['moving_images'][h],item['static_images'][h]
                q=static_probe.weighted_controls(rendered['q_motion'][0][0],rendered['q_static'][0][0],ref_motion,ref_static)
                t=static_probe.weighted_controls(rendered['teacher_motion'][0][0],rendered['teacher_static'][0][0],ref_motion,ref_static)
                qm=evaluate_metrics(rendered['q_motion'][0],w,sample,p)['metrics']
                tm=evaluate_metrics(rendered['teacher_motion'][0],w,sample,p)['metrics']
                prior=old[(start,h)]
                teacher_diff.extend(abs(t[k]-prior['metrics'][k]) for k in t if k!='dynamic_weight_sum')
                teacher_diff.append(abs(tm['motion_mae']-prior['motion_source_metrics']['motion_mae']))
                row={'start':start,'horizon':h,'frame':start+h,'metrics':q,'motion_source_metrics':qm,
                     'teacher_metrics':t,'teacher_motion_source_metrics':tm,
                     'q_motion_static_code_rms':float((qcodes['motion'][h-1]-qcodes['static'][h-1]).square().mean().sqrt())}
                student_rows.append(row)
                teacher_rows.append({'start':start,'horizon':h,'metrics':t})
                if final:
                    panels=[sample['record']['real'],rendered['q_motion'][0][0],rendered['q_static'][0][0]]
                    image=np.concatenate([preview.rgb_array(v) for v in panels],axis=1)
                    Image.fromarray(image,mode='RGB').save(folder/f'frame_{h:02d}.png')
                    for branch in cfg['conditions']:
                        images,gs,sh=rendered['q_'+branch]
                        name=f's{start:03d}_h{h:02d}_{branch}'
                        packet={'schema':'two_future_qformer_native_packet_v1','name':name,'start':start,
                            'horizon':h,'branch':branch,'source':source,'gs':cpu(gs),'sh':cpu(sh),
                            'cameras':cpu(record['cameras']),'intrinsics':cpu(record['intrinsics']),
                            'near':record['near'],'far':record['far'],'resolution_hw':[252,448]}
                        packets.append({'name':name,'start':start,'horizon':h,'branch':branch,'gs_count':len(gs),
                            'packet':save(out/'packets'/(name+'.pt'),packet),
                            'reference':save(out/'references'/(name+'.pt'),cpu(images))})
            if final:
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1',
                    '-i',str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                    str(out/f's{start:03d}_source_q_motion_q_static_5fps.mp4')],check=True,timeout=90)
    if len(teacher_diff)!=126 or max(teacher_diff)>5e-5:
        raise RuntimeError('Bound teacher outputs did not reproduce')
    a=static_probe.aggregate(student_rows);b=static_probe.aggregate(teacher_rows)
    q_motion=sum(r['motion_source_metrics']['motion_mae'] for r in student_rows)/18
    t_motion=sum(r['teacher_motion_source_metrics']['motion_mae'] for r in student_rows)/18
    q_h9=sum(r['motion_source_metrics']['motion_mae'] for r in student_rows if r['horizon']==9)/2
    t_h9=sum(r['teacher_motion_source_metrics']['motion_mae'] for r in student_rows if r['horizon']==9)/2
    q_static=a['overall']['static_code_to_static_reference'];t_static=b['overall']['static_code_to_static_reference']
    if min(t_motion,t_h9,t_static)<=0:raise RuntimeError('Invalid teacher denominator')
    static_closer=sum(r['metrics']['static_code_to_static_reference']<r['metrics']['static_code_to_motion_reference'] for r in student_rows)
    motion_closer=sum(r['metrics']['motion_code_to_motion_reference']<r['metrics']['motion_code_to_static_reference'] for r in student_rows)
    summary={'q':a,'teacher':b,'q_motion_source_motion_mae':q_motion,'teacher_motion_source_motion_mae':t_motion,
        'q_motion_h9_source_motion_mae':q_h9,'teacher_motion_h9_source_motion_mae':t_h9,
        'motion_overall_ratio':q_motion/t_motion,'motion_h9_ratio':q_h9/t_h9,'static_reference_ratio':q_static/t_static,
        'q_static_closer_frames':static_closer,'q_motion_closer_frames':motion_closer,
        'mean_q_motion_static_code_rms':sum(r['q_motion_static_code_rms'] for r in student_rows)/18,
        'preliminary_numeric_pass':q_motion<=1.1*t_motion and q_h9<=1.1*t_h9 and q_static<=1.1*t_static
                                   and static_closer>=16 and motion_closer>=16,
        'visual_and_native_replay_required':True}
    if final:
        if len(packets)!=36 or {r['name'] for r in packets}!=expected_packet_names(cfg):
            raise RuntimeError('Incomplete Q native GS packets')
        write(out/'PACKETS.json',{'status':'complete','source':source,'rows':packets,
              'producer_pid':os.getpid(),'decoder_future_geometry_input':False,'quantized':False})
    write(out/'EVALUATION.json',{'status':'complete','step':step_number,'summary':summary,'rows':student_rows,
        'teacher_reproduction_max_abs':max(teacher_diff),'source':source,
        'packets_sha256':sha(out/'PACKETS.json') if final else None,'visual_review':'pending'})
    return summary


def run():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    verify_self();cfg=read(ROOT/'protocol.json');validate_protocol(cfg)
    result=ROOT/'result';result.mkdir();began=time.monotonic();phase='binding'
    write(result/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
        'code_sha256':sha(ROOT/'CODE.sha256'),'main_training_signalled':False})
    try:
        teacher_evaluation=verify_teacher_sources(cfg)
        sys.path.insert(0,str(STATIC));import static_probe;static_probe.verify_self()
        sys.path.insert(0,str(PREVIEW));import preview;preview.verify_sources()
        sys.path.insert(0,str(BROOT))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle=load_bound_sources('cpu')
        runtime,store,decoder,posterior,fm,p=(bundle[k] for k in ('runtime','store','decoder','posterior','fm','p'))
        if (sha(bundle['paths']['cache_manifest'])!=cfg['cache_manifest_sha256']
                or p['codec_checkpoint_sha256']!=cfg['codec_sha256']
                or sha(p['posterior_checkpoint'])!=cfg['posterior_initial_checkpoint_sha256']
                or runtime.digest(posterior)!=cfg['posterior_initial_digest']):
            raise RuntimeError('Frozen GS state cache, codec or Q initialization differs')
        checkpoint=torch.load(PAIRED/'result/checkpoint_00360.pt',map_location='cpu',weights_only=True)
        if (checkpoint['schema']!='two_window_same_initial_motion_static_free_codes_v1'
                or checkpoint['step']!=360 or checkpoint['windows']!=cfg['windows']
                or checkpoint['decoder_digest']!=cfg['teacher_decoder_digest']
                or checkpoint['protocol_sha256']!=cfg['paired_protocol_sha256']):
            raise RuntimeError('Teacher checkpoint metadata differs')
        teacher_codes={branch:checkpoint[branch+'_codes'] for branch in cfg['conditions']}
        if any(v.dtype!=torch.float32 or tuple(v.shape)!=(2,9,16) or not bool(torch.isfinite(v).all())
               for v in teacher_codes.values()):raise RuntimeError('Invalid paired free-code teacher')
        decoder.load_state_dict(checkpoint['decoder'],strict=True)
        if runtime.digest(decoder)!=cfg['teacher_decoder_digest'] or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
            raise RuntimeError('Teacher decoder or initial-state encoder differs')
        decoder.requires_grad_(False).eval();posterior.requires_grad_(True)
        torch.manual_seed(cfg['seed'])
        phase='gpu_preflight'
        setup_gpu_once(dict(p,allocator_limit_gib=cfg['allocator_limit_gib'],minimum_free_gpu_gib=cfg['minimum_free_gpu_gib']))
        decoder=decoder.cuda();posterior=posterior.cuda()
        teacher_codes={k:v.cuda() for k,v in teacher_codes.items()}
        scale=torch.stack([v.square().mean() for v in teacher_codes.values()]).mean().clamp_min(1e-8)
        from runtime_bridge import BoundNativeBridge
        from support_decoder import target_labels
        from runtime import render
        from data import tree
        bridge=BoundNativeBridge(partial=False,device='cuda').eval();std=store.frames['std'].cuda();items={}
        with torch.no_grad():
            for start in cfg['windows']:
                w=store.get(start);s0={'coordinates':w['states'][0]['coordinates'].detach(),
                                      'features':w['states'][0]['features'].detach()}
                records={};still={};moving={};weights={}
                for h in cfg['horizons']:
                    record=camera_only(tree,store.frames['samples'][start+h]['record'])
                    x=render(bridge,s0,record,views=(0,))[0][0].detach()
                    y=render(bridge,w['states'][h],record,views=(0,))[0][0].detach()
                    weight=((y-x).abs().mean(-1)/.1).clamp(0,1)
                    if float(weight.sum())<=1e-6:raise RuntimeError('No measurable paired reference difference')
                    records[h],still[h],moving[h],weights[h]=record,x,y,weight
                items[start]={'window':w,'condition':w['condition'],'roi':w['roi'],'records':records,
                    'static_images':still,'moving_images':moving,'weights':weights,
                    'static_labels':target_labels(w['condition'],[s0]*9)}
        def frozen():
            bridge.assert_frozen()
            if (runtime.digest(decoder)!=cfg['teacher_decoder_digest']
                    or any(v.requires_grad or v.grad is not None for v in decoder.parameters())):
                raise RuntimeError('Fixed teacher decoder changed or acquired gradients')
        def optimizer():return torch.optim.Adam(posterior.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'])
        initial={k:v.detach().cpu().clone() for k,v in posterior.state_dict().items()}
        initial_digest=runtime.digest(posterior);opt=optimizer();posterior.train()
        pre_distill=train_step(posterior,decoder,bridge,store,items,teacher_codes,scale,opt,std,p,'distill',None)
        pre_joint=train_step(posterior,decoder,bridge,store,items,teacher_codes,scale,opt,std,p,'joint',9,probe=True)
        frozen();posterior.load_state_dict(initial,strict=True)
        if runtime.digest(posterior)!=initial_digest:raise RuntimeError('Q preflight reset failed')
        del initial,opt
        opt=optimizer()
        if opt.state:raise RuntimeError('Q preflight optimizer state was not reset')
        write(result/'PREFLIGHT.json',{'status':'passed','discarded_updates':2,'posterior_and_optimizer_restored':True,
            'decoder_and_codec_frozen':True,'batch_size':4,'code_scale':float(scale),
            'rgb_only_gradients':[{'start':r['start'],'branch':r['branch'],'query_projection':r['rgb_to_query_projection_gradients']}
                                  for r in pre_joint['examples']]})
        provenance={'teacher_checkpoint_sha256':cfg['teacher_checkpoint_sha256'],
            'teacher_decoder_digest':cfg['teacher_decoder_digest'],'codec_sha256':cfg['codec_sha256'],
            'cache_manifest_sha256':cfg['cache_manifest_sha256'],'teacher_replay_sha256':cfg['native_replay_sha256'],
            'posterior_initial_digest':initial_digest,'protocol_sha256':sha(ROOT/'protocol.json'),
            'code_sha256':sha(ROOT/'CODE.sha256'),'quantized':False}
        phase='baseline'
        summaries={'0':evaluate(0,posterior,decoder,bridge,store,items,teacher_codes,p,cfg,teacher_evaluation,provenance)}
        steps=schedule(cfg)
        with (result/'train.jsonl').open('x',encoding='utf8',buffering=1) as log:
            for item in steps:
                number=item['step'];phase=item['phase']
                if time.monotonic()-began>cfg['maximum_train_seconds']:raise TimeoutError('Fixed Q training budget exceeded')
                posterior.train()
                lr=cfg['final_learning_rate']+.5*(cfg['learning_rate']-cfg['final_learning_rate'])*(1+math.cos(math.pi*(number-1)/(len(steps)-1)))
                for group in opt.param_groups:group['lr']=lr
                values=train_step(posterior,decoder,bridge,store,items,teacher_codes,scale,opt,std,p,phase,item['horizon'])
                log.write(__import__('json').dumps({'step':number,'phase':phase,'horizon':item['horizon'],'lr':lr,**values},allow_nan=False)+'\n')
                if number==1 or number%20==0:
                    write(result/'STATUS.json',{'status':'running','phase':phase,'completed_updates':number,
                          'total_updates':len(steps),'elapsed_seconds':time.monotonic()-began},replace=True)
                if number in cfg['checkpoint_steps']:
                    frozen();path=result/f'checkpoint_{number:05d}.pt'
                    with path.open('xb') as stream:
                        torch.save({'schema':cfg['schema'],'step':number,'posterior':posterior.state_dict(),
                            'posterior_digest':runtime.digest(posterior),'optimizer':opt.state_dict(),
                            'teacher_checkpoint_sha256':cfg['teacher_checkpoint_sha256'],'source':provenance},stream)
                    if number==1194:
                        phase='causality'
                        write(result/'CAUSALITY.json',causal_check(posterior,decoder,items))
                    source=dict(provenance,posterior_checkpoint_sha256=sha(path),posterior_digest=runtime.digest(posterior))
                    phase=f'evaluation_{number}'
                    summaries[str(number)]=evaluate(number,posterior,decoder,bridge,store,items,teacher_codes,p,cfg,teacher_evaluation,source)
        frozen()
        write(result/'SUMMARY.json',{'status':'complete','updates':len(steps),'summaries':summaries,
            'causality_sha256':sha(result/'CAUSALITY.json'),'teacher_decoder_and_codec_frozen':True,
            'model_scope':cfg['scope'],'visual_review':'pending','native_replay':'pending',
            'elapsed_seconds':time.monotonic()-began,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'checkpoints':{str(s):sha(result/f'checkpoint_{s:05d}.pt') for s in (600,900,1194)}})
        write(result/'STATUS.json',{'status':'complete','phase':'training_and_evaluation',
              'completed_updates':len(steps),'native_replay':'pending'},replace=True)
        print('PASS: paired Q training and evaluation complete; cold replay and visual review pending',flush=True)
    except BaseException as error:
        write(result/'FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'main_training_signalled':False,'no_automatic_retry':True})
        raise


if __name__=='__main__':
    run()
