import argparse,hashlib,json,math,os,time,traceback
import torch
from io_utils import ROOT,read,write,sha,verify
from full_model import build_models,frozen_digest,parameters,schedule,TRAIN_PREFIXES
from runtime import digest,render,save,cpu
from data import WindowStore
from support_decoder import state_loss,select_state
from core import image_losses


def update(decoder,posterior,bridge,store,std,opt,item,mode,p,probe=False):
    opt.zero_grad(set_to_none=True);rows=[]
    for member in item['batch']:
        w=store.get(member['start']);h=member['horizon']
        memory=w['memories'] if mode=='posterior' else w['initial_memories']
        codes=posterior([memory])['codes'][0]
        prediction=decoder(w['condition'],codes)
        latent,terms=state_loss(prediction,w['targets'],std);loss=latent
        row=dict(start=w['start'],latent_loss=float(latent.detach()),
                 occupancy_loss=float(terms['occupancy'].detach()),feature_loss=float(terms['feature'].detach()),
                 code_rms=float(codes.detach().square().mean().sqrt()))
        if item['phase']=='render':
            state=select_state(w['condition'].coordinates,prediction['features'][h-1],prediction['logits'][h-1],p['max_predicted_latent_nodes'])
            sample=store.sample(w['start'],h)
            images,gs,sh=render(bridge,state,sample['record'])
            rgb,_=image_losses(images[0],sample['record']['real'],w['roi'],sample['motion'])
            if probe:
                gradients=torch.autograd.grad(rgb,(posterior.queries.weight,posterior.project_in.weight,decoder.base.features.weight),retain_graph=True)
                g=dict(last_query=float(gradients[0][-1].norm()),posterior_projection=float(gradients[1].norm()),decoder_features=float(gradients[2].norm()))
                if not all(math.isfinite(v) and v>0 for v in g.values()):raise RuntimeError('Missing RGB gradient')
                row['rgb_only_gradients']=g
            loss=rgb+.1*latent
            row.update(horizon=h,rgb_loss=float(rgb.detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite full-window training loss')
        row['loss']=float(loss.detach());(loss/len(item['batch'])).backward();rows.append(row)
        if item['phase']=='render':del sample,state,images,gs,sh,rgb
        del w,memory,codes,prediction,latent,terms,loss
    trainable=parameters(decoder,posterior)
    if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in trainable):raise RuntimeError('Nonfinite gradient')
    query_norms=posterior.queries.weight.grad.norm(dim=-1)
    if probe and not bool((query_norms>0).all()):raise RuntimeError('Missing query gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(trainable,p['grad_clip']));opt.step()
    return dict(windows=rows,query_gradient_norms=query_norms.tolist(),preclip_gradient_norm=norm,clip_triggered=norm>p['grad_clip'])


def main(mode):
    p=verify()
    if mode not in p['modes']:raise ValueError('Unknown arm')
    out=ROOT/mode;out.mkdir(exist_ok=False);began=time.monotonic();context={'phase':'initializing'}
    write(out/'STARTED.json',dict(mode=mode,pid=os.getpid()))
    try:
        torch.set_num_threads(6);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one worker GPU')
        free,total=torch.cuda.mem_get_info()
        if free<p['minimum_free_gpu_gib']*2**30:raise RuntimeError('GPU memory unavailable')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
        decoder,posterior=build_models(p,'cuda');initial_decoder=digest(decoder);initial_posterior=digest(posterior)
        witness=read(ROOT/'CPU_TESTS.json')
        if initial_decoder!=witness['initial_decoder_digest'] or initial_posterior!=witness['initial_posterior_digest']:
            raise RuntimeError('Initial models differ from CPU witness')
        store=WindowStore(p);std=store.frames['std'].cuda();frozen=frozen_digest(decoder)
        if frozen!=store.manifest['frozen_model_digest']:raise RuntimeError('Cached encoder/context mismatch')
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval();items=schedule(p)
        schedule_sha=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if schedule_sha!=witness['schedule_sha256']:raise RuntimeError('Training schedule differs')
        source=dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
            codec_sha256=p['codec_checkpoint_sha256'],warm_decoder_sha256=p['teacher_checkpoint_sha256'],
            warm_posterior_sha256=p['posterior_checkpoint_sha256'],initial_decoder_digest=initial_decoder,
            initial_posterior_digest=initial_posterior,frozen_component_digest=frozen,
            cache_manifest_sha256=sha(ROOT/'cache/MANIFEST.json'),schedule_sha256=schedule_sha,
            mode=mode,training_frames=177,training_windows=168,code_shape=[4,9,16],quantized=False)
        write(out/'PROVENANCE.json',dict(source=source,trainable_parameters=sum(v.numel() for v in parameters(decoder,posterior))))
        def assert_frozen():
            bridge.assert_frozen()
            if frozen_digest(decoder)!=frozen:raise RuntimeError('Frozen state encoder or unused free codes changed')
            if any(v.grad is not None for name,v in decoder.named_parameters() if not name.startswith(TRAIN_PREFIXES)):
                raise RuntimeError('Gradient on frozen decoder component')
        def optimizer():return torch.optim.Adam(parameters(decoder,posterior),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'])
        initial={'decoder':{k:cpu(v).clone() for k,v in decoder.state_dict().items()},
                 'posterior':{k:cpu(v).clone() for k,v in posterior.state_dict().items()}}
        posterior.train();opt=optimizer();batch=[dict(start=s,horizon=9) for s in [0,56,112,167]]
        probes=[update(decoder,posterior,bridge,store,std,opt,dict(phase='latent',batch=batch),mode,p),
                update(decoder,posterior,bridge,store,std,opt,dict(phase='render',batch=batch),mode,p,probe=True)]
        assert_frozen();decoder.load_state_dict(initial['decoder']);posterior.load_state_dict(initial['posterior'])
        if digest(decoder)!=initial_decoder or digest(posterior)!=initial_posterior:raise RuntimeError('Preflight reset differs')
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=2,logical_batch_size=4,rows=probes,source=source,
            restored_initial_models=True,codec_and_state_encoder_frozen=True,rgb_to_qformer_and_motion_decoder=True))
        del initial,opt;opt=optimizer()
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for index,item in enumerate(items):
                if time.monotonic()-began>p['worker_max_seconds']:raise RuntimeError('Training deadline')
                context=dict(phase=item['phase'],completed_updates=index)
                lr=p['final_lr']+.5*(p['lr']-p['final_lr'])*(1+math.cos(math.pi*index/max(p['total_steps']-1,1)))
                opt.param_groups[0]['lr']=lr
                row=update(decoder,posterior,bridge,store,std,opt,item,mode,p)
                row.update(step=index+1,phase=item['phase'],lr=lr,elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if index==0 or (index+1)%p['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase=item['phase'],completed_updates=index+1,
                        total_updates=p['total_steps'],mean_loss=sum(v['loss'] for v in row['windows'])/4,
                        elapsed_seconds=time.monotonic()-began),replace=True)
                if index+1 in p['checkpoints']:
                    assert_frozen();save(out/f'step_{index+1:04d}.pt',dict(schema='joint_full177_continuous_v1',mode=mode,step=index+1,
                        decoder=decoder.state_dict(),posterior=posterior.state_dict(),optimizer=opt.state_dict(),source=source))
        assert_frozen();decoder.eval();posterior.eval()
        decoder_digest=digest(decoder);posterior_digest=digest(posterior)
        with torch.no_grad():
            w=store.get(35);key='memories' if mode=='posterior' else 'initial_memories'
            before=cpu(posterior([w[key]])['codes'])
        del decoder,posterior,opt
        decoder,posterior=build_models(p,'cuda',train_decoder=False)
        checkpoint_path=out/f"step_{p['total_steps']:04d}.pt"
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        decoder.load_state_dict(checkpoint['decoder']);posterior.load_state_dict(checkpoint['posterior']);decoder.eval();posterior.eval()
        if digest(decoder)!=decoder_digest or digest(posterior)!=posterior_digest:raise RuntimeError('Checkpoint model replay mismatch')
        with torch.no_grad():
            if not torch.equal(before,cpu(posterior([w[key]])['codes'])):raise RuntimeError('Checkpoint code replay mismatch')
        del checkpoint,before,w
        write(out/'TRAIN_DONE.json',dict(status='complete',steps=p['total_steps'],source=source,
            decoder_digest=decoder_digest,posterior_digest=posterior_digest,checkpoint_sha256=sha(checkpoint_path),
            codec_and_state_encoder_frozen=True,checkpoint_replay='passed',elapsed_seconds=time.monotonic()-began,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        context={'phase':'evaluation'}
        write(out/'STATUS.json',dict(status='running',phase='evaluation',completed_updates=p['total_steps']),replace=True)
        from evaluate import evaluate
        evaluate(decoder,posterior,bridge,store,p,out,source,mode);assert_frozen()
        write(out/'STATUS.json',dict(status='complete',completed_updates=p['total_steps'],native_replay='pending',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(out/'FAILED.json',dict(status='failed',mode=mode,context=context,error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True);main(parser.parse_args().mode)
