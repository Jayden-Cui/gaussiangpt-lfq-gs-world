import copy,json,math,os,shutil,time,traceback
import torch
from io_utils import ROOT,read,write,sha,verify
from runtime import load_data,digest,render,save,cpu
from posterior import build_pair,make_memories,PrefixPosterior
from support_decoder import state_loss,select_state
from core import image_losses
from evaluate import evaluate


def update(posterior,teacher,bridge,windows,std,optimizer,phase,index,p,probe_rgb=False):
    optimizer.zero_grad(set_to_none=True);rows=[];scale=teacher.codes.detach().square().mean().clamp_min(1e-8)
    h=1+index%9
    for k,w in enumerate(windows):
        codes=posterior([w['memories']])['codes'][0]
        distill=(codes-teacher.codes[k].detach()).square().mean()/scale
        loss=distill;row=dict(start=w['start'],code_mse=float(distill.detach()))
        if phase=='render':
            prediction=teacher(w['condition'],codes)
            latent,terms=state_loss(prediction,w['targets'],std)
            state=select_state(w['condition'].coordinates,prediction['features'][h-1],prediction['logits'][h-1],p['max_predicted_latent_nodes'])
            images,gs,sh=render(bridge,state,w['samples'][h]['record'])
            rgb,_=image_losses(images[0],w['samples'][h]['record']['real'],w['roi'],w['motion'][h])
            if probe_rgb:
                gradients=torch.autograd.grad(rgb,(posterior.queries.weight,posterior.project_in.weight),retain_graph=True)
                row['rgb_only_gradient']={'last_query':float(gradients[0][-1].norm()),'projection':float(gradients[1].norm())}
                if not all(math.isfinite(v) and v>0 for v in row['rgb_only_gradient'].values()):
                    raise RuntimeError('RGB does not reach the posterior')
            loss=rgb+.1*latent+.1*distill
            row.update(horizon=h,rgb_loss=float(rgb.detach()),latent_loss=float(latent.detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite posterior loss')
        row['loss']=float(loss.detach());(loss/len(windows)).backward();rows.append(row)
        if phase=='render':del prediction,latent,state,images,gs,sh,rgb,terms
        del loss,distill,codes
    if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in posterior.parameters()):
        raise RuntimeError('Nonfinite posterior gradient')
    norms=posterior.queries.weight.grad.norm(dim=-1)
    if probe_rgb and not bool((norms>0).all()):raise RuntimeError('Missing query gradient in preflight')
    gradient_norm=float(torch.nn.utils.clip_grad_norm_(posterior.parameters(),p['grad_clip']))
    optimizer.step()
    return dict(windows=rows,query_gradient_norms=norms.tolist(),preclip_gradient_norm=gradient_norm,clip_triggered=gradient_norm>p['grad_clip'])


def main():
    p=verify();mode=p['modes'][0];out=ROOT/mode;out.mkdir(exist_ok=False);began=time.monotonic();context={'phase':'initializing'}
    write(out/'STARTED.json',dict(pid=os.getpid(),mode=mode))
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU')
        if shutil.which('ffmpeg') is None:raise RuntimeError('Existing ffmpeg is required')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info()
        if free<p['minimum_free_gpu_gib']*2**30:raise RuntimeError('Insufficient free GPU memory')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
        teacher,posterior=build_pair(p,'cuda');initial_digest=digest(posterior)
        if initial_digest!=read(ROOT/'CPU_TESTS.json')['initial_posterior_digest']:raise RuntimeError('Unwitnessed posterior initialization')
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        windows,std,binding=load_data(p,teacher)
        for w in windows:
            w['memories']=make_memories(teacher.base,w['states'])
            w['initial_memories']=make_memories(teacher.base,w['states'],True)
        source=dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
            codec_sha256=p['codec_checkpoint_sha256'],teacher_checkpoint_sha256=p['teacher_checkpoint_sha256'],
            teacher_tensor_digest=p['teacher_tensor_digest'],posterior_initial_digest=initial_digest,
            posterior_prefix_policy=p['prefix_policy'],decoder_frozen=True,quantized=False,code_shape=[2,9,16],
            states={str(w['start']):[binding.rows[f]['state_sha256'] for f in range(w['start'],w['start']+10)] for w in windows})
        write(out/'PROVENANCE.json',dict(source=source,trainable_parameters=sum(v.numel() for v in posterior.parameters()),
            memories={str(w['start']):[len(m.keys) for m in w['memories']] for w in windows},
            candidates={str(w['start']):len(w['condition'].coordinates) for w in windows},code_scale_rms=float(teacher.codes.square().mean().sqrt())))
        def frozen():
            bridge.assert_frozen()
            if digest(teacher)!=p['teacher_tensor_digest'] or any(v.requires_grad or v.grad is not None for v in teacher.parameters()):
                raise RuntimeError('Frozen learned decoder changed')
        def optimizer():return torch.optim.Adam(posterior.parameters(),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'])
        initial={k:cpu(v).clone() for k,v in posterior.state_dict().items()}
        posterior.train();opt=optimizer()
        probes=[update(posterior,teacher,bridge,windows,std,opt,'distill',0,p),
                update(posterior,teacher,bridge,windows,std,opt,'render',8,p,probe_rgb=True)]
        frozen();posterior.load_state_dict(initial,strict=True)
        if digest(posterior)!=initial_digest:raise RuntimeError('Preflight reset failed')
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=2,rows=probes,source=source,
            restored_exact_initialization=True,teacher_frozen=True,codec_frozen=True,rgb_to_qformer='passed'))
        del initial,opt;opt=optimizer()
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for step in range(p['total_steps']):
                if time.monotonic()-began>p['worker_max_seconds']:raise RuntimeError('Training deadline')
                phase='distill' if step<p['distillation_steps'] else 'render'
                index=step if phase=='distill' else step-p['distillation_steps'];context=dict(phase=phase,completed_updates=step)
                lr=p['final_lr']+.5*(p['lr']-p['final_lr'])*(1+math.cos(math.pi*step/max(p['total_steps']-1,1)))
                opt.param_groups[0]['lr']=lr
                row=update(posterior,teacher,bridge,windows,std,opt,phase,index,p)
                row.update(step=step+1,phase=phase,lr=lr,elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if step==0 or (step+1)%p['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase=phase,completed_updates=step+1,total_updates=p['total_steps'],
                        mean_loss=sum(r['loss'] for r in row['windows'])/2,elapsed_seconds=time.monotonic()-began),replace=True)
                if step+1 in p['checkpoints']:
                    frozen();save(out/f'step_{step+1:04d}.pt',dict(schema='prefix_posterior_checkpoint_v1',step=step+1,
                        posterior=posterior.state_dict(),optimizer=opt.state_dict(),source=source))
        frozen();posterior.eval()
        checkpoint_path=out/f"step_{p['total_steps']:04d}.pt"
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        reloaded=PrefixPosterior(teacher.base,p['code_projection_init_std']).cuda().eval()
        reloaded.load_state_dict(checkpoint['posterior'],strict=True)
        with torch.no_grad():
            memories=[w['memories'] for w in windows]
            before=posterior(memories)['codes'];after=reloaded(memories)['codes']
            if not torch.equal(before,after):raise RuntimeError('Posterior checkpoint replay differs')
        write(out/'TRAIN_DONE.json',dict(status='complete',steps=p['total_steps'],source=source,
            final_posterior_digest=digest(posterior),checkpoint_sha256=sha(checkpoint_path),
            teacher_frozen=True,codec_frozen=True,posterior_checkpoint_replay='passed',
            elapsed_seconds=time.monotonic()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        del posterior,opt,checkpoint,before,after
        context={'phase':'evaluation'}
        write(out/'STATUS.json',dict(status='running',phase='evaluation',completed_updates=p['total_steps']),replace=True)
        evaluate(teacher,reloaded,bridge,windows,p,out,source);frozen()
        write(out/'STATUS.json',dict(status='complete',completed_updates=p['total_steps'],native_replay='pending',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(out/'FAILED.json',dict(status='failed',context=context,error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
