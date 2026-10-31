"""Full H9 loss and batch4 LFQ, memory bounded by exact chain-rule staging.

All 36 decoded targets contribute before a single optimizer step. A detached
bottleneck leaf is solely a gradient buffer: its complete gradient is sent
back through the original quantizer/encoder graph. No temporal truncation.
"""
import argparse,hashlib,json,random,signal,time
from pathlib import Path
import numpy as np
import torch
from motion_common import read,write,sha,frozen_codec,load_target
from qmotion import build_model
from sequence_data import window_ids,load_record
from losses_motion import all_losses
from contracts import verify_sources

def parameter_sha(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def sequence_backward(model,codec,records,clips,protocol,*,rendering,aux_weight=1.):
    batch=len(records);horizon=9
    z=model.encode_windows([r['window'] for r in records])
    weights=protocol['train']['render_loss'] if rendering else protocol['train']['warmup_loss']
    code_gradient=torch.zeros_like(z['z_quantized'])
    terms_mean={};loss_value=0.;details=[];horizon_loss=[0.]*horizon
    for b,(record,clip) in enumerate(zip(records,clips)):
        leaf=z['z_quantized'][b:b+1].detach().requires_grad_(True)
        predictions=model.decode_latents(leaf,[record['condition']])[0]
        for h,prediction in enumerate(predictions):
            images=load_target(clip,record['frames'][h+1]) if rendering else None
            terms=all_losses(codec,prediction,record['window'][h+1],record['labels'][h],images,rendering=rendering)
            loss=sum(weights[k]*value for k,value in terms.items())/(batch*horizon)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite sequence reconstruction loss')
            # Each backward frees its expensive spatial renderer graph after
            # the local tensors are released; shared temporal graph is retained.
            loss.backward(retain_graph=h<horizon-1)
            for key,value in terms.items():terms_mean[key]=terms_mean.get(key,0.)+float(value.detach())/(batch*horizon)
            loss_value+=float(loss.detach());horizon_loss[h]+=float(loss.detach())*horizon
            details.append(dict(batch=b,horizon=h+1,frame=record['frames'][h+1],padding=not record['valid'][h],
                predicted_tokens=int((prediction['occupancy_logits']>0).sum()),target_tokens=len(record['labels'][h]),
                target_support_forcing=False,decoded_gaussians=codec._motion_decode_stats.get('gaussians') if rendering else None))
            del loss,terms,images
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
        code_gradient[b]=leaf.grad[0]
        del predictions,prediction,leaf
    # This vector-Jacobian product restores the exact reconstruction gradient.
    surrogate=(z['z_quantized']*code_gradient).sum()+aux_weight*z['aux_loss']
    surrogate.backward()
    return dict(loss=loss_value+aux_weight*float(z['aux_loss'].detach()),terms=terms_mean,horizon_losses=horizon_loss,
        quantizer_aux_loss=float(z['aux_loss'].detach()),per_sample_entropy=float(z['per_sample_entropy'].detach()),
        codebook_entropy=float(z['codebook_entropy'].detach()),motion_ids=z['indices'].detach().cpu().tolist(),
        bottleneck_gradient_per_horizon=code_gradient.square().sum((0,2)).sqrt().cpu().tolist(),details=details)

def train(root,arm):
    verify_sources(root);protocol=read(root/'protocol.json');settings=protocol['train'];config=protocol['arms'][0]
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    codec=frozen_codec(root);frozen_before=parameter_sha(codec)
    model=build_model(config).cuda().train();initial_sha=parameter_sha(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],betas=tuple(settings['betas']),weight_decay=settings['weight_decay'])
    folder=root/arm;folder.mkdir(exist_ok=True)
    items=window_ids();rng=random.Random(protocol['seed']);order=list(range(len(items)));rng.shuffle(order)
    cursor=0;epochs=0;visited=set();exposures=0;real_exposures=0;completed=0;last={};event='init';context={}
    started=time.monotonic()
    def save(name,resumable=True):
        path=folder/name;temp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=completed,
            protocol_sha256=sha(root/'protocol.json'),source_binding_sha256=sha(root/'source_binding.json'),
            source_checkpoint_sha256=protocol['source_checkpoint_sha256'],resumable=resumable,
            sampler=dict(order=order,cursor=cursor,rng=rng.getstate(),visited=sorted(visited),epochs=epochs),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),window_exposures=exposures),temp)
        temp.replace(path)
    def terminate(signum,frame):raise RuntimeError(f'Terminated by signal {signum}')
    old=signal.signal(signal.SIGTERM,terminate)
    try:
        with (folder/'train_metrics.jsonl').open('x',buffering=1) as log:
            for step in range(1,settings['max_steps']+1):
                if time.monotonic()-started>=settings['max_seconds']:break
                selected=[]
                for _ in range(settings['batch_size']):
                    if cursor==len(order):rng.shuffle(order);cursor=0;epochs+=1
                    selected.append(items[order[cursor]]);cursor+=1
                event='load';context=dict(attempted_step=step,windows=selected)
                records=[load_record(root,c,s,codec) for c,s in selected]
                rendering=step>settings['warmup_steps']
                optimizer.zero_grad(set_to_none=True);event='forward_backward'
                output=sequence_backward(model,codec,records,[c for c,s in selected],protocol,rendering=rendering)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True)
                assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
                event='optimizer_step';optimizer.step();completed=step;event='complete_step'
                visited.update(selected);exposures+=len(selected);real_exposures+=sum(sum(r['valid']) for r in records)
                last=dict(step=step,max_steps=settings['max_steps'],batch_size=len(records),horizon=9,
                    stage='future_render' if rendering else 'latent_warmup',elapsed_seconds=time.monotonic()-started,
                    window_exposures=exposures,future_target_exposures=exposures*9,real_future_target_exposures=real_exposures,
                    unique_windows_seen=len(visited),target_support_forcing=False,windows=selected,
                    gradient_norm_before_clip=float(norm),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,**output)
                if step==1 or step%10==0 or step==settings['max_steps']:
                    log.write(json.dumps(last,allow_nan=False)+'\n');write(folder/'train_progress.json',last)
                    print('TRAIN',step,'H9 loss',round(last['loss'],6),'future_targets',exposures*9,flush=True)
                if step==1 or step%settings['save_every']==0:save('latest.pt')
                del records,output
        assert completed>settings['warmup_steps'],'No future rendering training completed'
        frozen_after=parameter_sha(codec);assert frozen_after==frozen_before
        save('motion_final.pt')
        result=dict(status='complete',arm=arm,steps=completed,max_steps=settings['max_steps'],batch_size=settings['batch_size'],
            horizon=9,window_exposures=exposures,future_target_exposures=exposures*9,real_future_target_exposures=real_exposures,
            all_windows_seen=len(visited)==len(items),unique_windows_seen=len(visited),no_target_forcing_render_steps=completed-settings['warmup_steps'],
            elapsed_seconds=time.monotonic()-started,stop_reason='step_budget' if completed==settings['max_steps'] else 'wall_time_budget',
            checkpoint_sha256=sha(folder/'motion_final.pt'),initial_parameter_sha256=initial_sha,
            frozen_parameter_sha256_before=frozen_before,frozen_parameter_sha256_after=frozen_after,
            trainable_parameters=sum(p.numel() for p in model.parameters()),last=last)
        write(folder/'train_done.json',result);print('TRAIN_COMPLETE',completed,flush=True)
    except BaseException as error:
        save_error=None
        try:save('exception.pt',event=='complete_step')
        except BaseException as e:save_error=repr(e)
        write(folder/'training_failure.json',dict(status='failed',error=repr(error),event=event,context=context,
            last_completed_step=completed,checkpoint_error=save_error));raise
    finally:signal.signal(signal.SIGTERM,old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();train(a.root,a.arm)
