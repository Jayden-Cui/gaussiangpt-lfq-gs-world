"""Bounded motion optimization with atomic regular and exception checkpoints."""
import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from motion_common import BASE, read, write, sha, frozen_codec, load_pair, load_target, Vgg19, pair_ids
from qmotion import MotionQFormer
from losses_motion import all_losses


def parameter_sha(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path,model,optimizer,**metadata):
    temporary=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),**metadata},temporary)
    temporary.replace(path)


def train(root,arm,verification_steps=None):
    protocol=read(root/'protocol.json');settings=protocol['train']
    config=next(a for a in protocol['arms'] if a['name']==arm)
    assert read(root/'prepare_done.json')['total_pairs']==protocol['evaluation']['pairs']
    assert sha(BASE/'codec_final.pt')==protocol['source_checkpoint_sha256']
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    codec=frozen_codec(root);frozen_before=parameter_sha(codec)
    model=MotionQFormer(config['motion_tokens']).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],betas=tuple(settings['betas']),weight_decay=settings['weight_decay'])
    vgg=Vgg19(BASE/'torch_cache').cuda().eval().requires_grad_(False)
    folder=(root/'verification'/arm) if verification_steps else (root/arm)
    folder.mkdir(parents=True,exist_ok=True)
    pairs=pair_ids(protocol);rng=random.Random(protocol['seed']);order=list(range(len(pairs)));rng.shuffle(order)
    visited=set();epochs=0;no_force_steps=0;max_kept=0;max_predicted=0
    started=time.monotonic();completed=0;last={};context={};event='initialization'
    def snapshot(path,resumable=True):
        save_checkpoint(path,model,optimizer,arm=arm,motion_tokens=config['motion_tokens'],step=completed,
            protocol_sha256=sha(root/'protocol.json'),source_checkpoint_sha256=protocol['source_checkpoint_sha256'],
            resumable=resumable,elapsed_seconds=time.monotonic()-started,
            sampler={'order':order,'rng':rng.getstate(),'visited':sorted(visited),'epochs':epochs},
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
            no_force_steps=no_force_steps,verification_only=bool(verification_steps))
    def terminate(signum,frame):raise RuntimeError(f'Training interrupted by signal {signum}')
    previous_handler=signal.signal(signal.SIGTERM,terminate)
    target_steps=verification_steps or settings['max_steps']
    try:
        with (folder/'train_metrics.jsonl').open('x',buffering=1) as log:
            for step in range(1,target_steps+1):
                elapsed=time.monotonic()-started
                if elapsed>=settings['max_seconds']:break
                offset=(step-1)%len(pairs)
                if offset==0 and step>1:rng.shuffle(order);epochs+=1
                clip,start,end=pairs[order[offset]]
                event='load_pair';context={'attempted_step':step,'clip':clip,'start':start,'end':end}
                s,t,initial,condition,target_indices=load_pair(root,clip,start,end,codec)
                rendering=step>settings['warmup_steps']
                # Regression uses the same 3000-step schedule, not a shortened one.
                fraction=max((step-1)/settings['max_steps'],elapsed/settings['max_seconds'])
                force_target=fraction<.8
                images=load_target(clip,end) if rendering else None
                optimizer.zero_grad(set_to_none=True)
                event='forward';z=model.encode_motion(s,t,initial.dt)
                prediction=model.restore(z['z_quantized'],condition)
                raw_keep=prediction['occupancy_logits']>0
                union=raw_keep.clone()
                if force_target:union[target_indices]=True
                context.update(predicted_count=int(raw_keep.sum()),forced_additions=int((union&~raw_keep).sum()),
                    kept_count=int(union.sum()),target_count=len(target_indices),candidate_count=len(raw_keep))
                terms,aux=all_losses(codec,prediction,t,target_indices,images,vgg,force_target=force_target,
                    rendering=rendering,max_tokens=protocol['support']['max_predicted_tokens'])
                weights=settings['render_loss'] if rendering else settings['warmup_loss']
                loss=sum(weights[key]*value for key,value in terms.items())
                if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite loss at step {step}')
                event='backward';loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True)
                if any(p.grad is not None or p.requires_grad for p in codec.parameters()):
                    raise RuntimeError('Frozen state codec received a parameter gradient')
                event='optimizer_step';optimizer.step();completed=step;event='step_complete'
                visited.add((clip,start))
                if rendering and not force_target:no_force_steps+=1
                max_kept=max(max_kept,context['kept_count']);max_predicted=max(max_predicted,context['predicted_count'])
                last={'step':step,'max_steps':settings['max_steps'],'elapsed_seconds':time.monotonic()-started,
                    'stage':'render' if rendering else 'latent_warmup',**context,'dt':initial.dt,
                    'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
                    'decoded_gaussians':codec._motion_decode_stats.get('gaussians'),
                    'decoder_candidates':codec._motion_decode_stats.get('max_candidates'),
                    'terms':{k:float(v.detach()) for k,v in terms.items()},'motion_ids':z['indices'].detach().cpu().tolist(),
                    'target_support_forcing':force_target,'unique_pairs_seen':len(visited),
                    'no_target_forcing_render_steps':no_force_steps,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
                if step==1 or step%10==0 or step==target_steps:
                    log.write(json.dumps(last,allow_nan=False)+'\n');write(folder/'train_progress.json',last)
                    print('TRAIN',arm,step,'loss',round(last['loss'],6),'kept',context['kept_count'],flush=True)
                if step==1 or step%settings['save_every']==0:snapshot(folder/'latest.pt')
                del s,t,initial,condition,target_indices,images,z,prediction,terms,aux,loss,raw_keep,union
        assert completed>0
        frozen_after=parameter_sha(codec);assert frozen_after==frozen_before,'Frozen state codec changed'
        snapshot(folder/'motion_final.pt')
        result={'status':'verification_complete' if verification_steps else 'complete','arm':arm,'steps':completed,
            'max_steps':settings['max_steps'],'stop_reason':'verification_budget' if verification_steps else
                ('step_budget' if completed==settings['max_steps'] else 'wall_time_budget'),
            'elapsed_seconds':time.monotonic()-started,'last':last,'unique_pairs_seen':len(visited),
            'all_pairs_seen':len(visited)==len(pairs),'no_target_forcing_render_steps':no_force_steps,
            'max_kept_tokens':max_kept,'max_predicted_tokens':max_predicted,
            'frozen_parameter_sha256_before':frozen_before,'frozen_parameter_sha256_after':frozen_after,
            'checkpoint_sha256':sha(folder/'motion_final.pt'),'completed_epochs':epochs,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write(folder/'train_progress.json',last);write(folder/'train_done.json',result)
        print('TRAIN_COMPLETE',arm,completed,flush=True)
        return result
    except BaseException as error:
        checkpoint_error=None
        try:snapshot(folder/'exception.pt',resumable=event!='optimizer_step')
        except BaseException as save_error:checkpoint_error=repr(save_error)
        write(folder/'training_failure.json',{'status':'failed','error':repr(error),'last_completed_step':completed,
            'event':event,'context':context,'exception_checkpoint_saved':checkpoint_error is None,'checkpoint_error':checkpoint_error})
        raise
    finally:
        signal.signal(signal.SIGTERM,previous_handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();train(args.root,args.arm)
