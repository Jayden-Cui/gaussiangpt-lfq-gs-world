from runtime_env import configure
configure()
import argparse,copy,json,math,random,time,traceback,os
from pathlib import Path
import numpy as np
import torch
from bindings import ROOT,ASSETS,read,write,sha,require,digest,reference_model,verify_assets,Store,NativeBridge,state_loss,select_state,image_losses,image_metrics
from capacity_model import CapacityDecoder,FreeCodes
from sampling import schedule,audit


def verify():
    for row in (ROOT/'CODE.sha256').read_text().splitlines():
        expected,name=row.split('  ',1);require(Path(name).name==name and sha(ROOT/name)==expected,'Code binding changed: '+name)
    p=read(ROOT/'protocol.json')
    require(sha(ASSETS/'MANIFEST.json')==p['assets_manifest_sha256'],'Dataset binding changed')
    return p


def sampling(p):
    full=schedule(7,p['base_seed'],p['replay_seed']);audit(full,7)
    require(full[:2115]==schedule(5,p['base_seed'],p['replay_seed']),'Prior sampling prefix differs')
    rows=full[2115:];require(len(rows)==846,'Wrong continuation budget')
    return rows


def setup(arm):
    p=verify();torch.set_num_threads(6);torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one CUDA device')
    free,total=torch.cuda.mem_get_info();require(free>=p['minimum_free_gpu_gib']*2**30,'Insufficient GPU memory')
    torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
    random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    verify_assets();reference,parent=reference_model();a=p['arms'][arm]
    model=CapacityDecoder(reference,a['code_dim'],a['width'],p['seed']).cuda()
    bank=FreeCodes(parent['codes'],a['code_dim']).cuda();store=Store();bridge=NativeBridge().eval()
    source=dict(arm=arm,config=a,code_sha256=sha(ROOT/'CODE.sha256'),protocol_sha256=sha(ROOT/'protocol.json'),
        asset_manifest_sha256=sha(ASSETS/'MANIFEST.json'),parent_checkpoint_sha256=sha(ASSETS/'initial.pt'),
        initial_decoder_digest=digest(model),initial_code_shape=list(bank.stack().shape),
        optimizer='fresh Adam for all four arms',seed=p['seed'],state_width=64,codec_frozen=True,initial_context_frozen=True,
        trainable_decoder_parameters=sum(v.numel() for v in model.parameters()),gpu=torch.cuda.get_device_name())
    return p,model,bank,store,bridge,reference,parent,source


def optimizer(model,bank,p):
    return torch.optim.Adam([dict(params=model.parameters(),lr=p['decoder_lr']),dict(params=bank.parameters(),lr=p['code_lr'])],betas=tuple(p['betas']),eps=p['eps'])


def train_step(model,bank,store,bridge,opt,batch,p,probe=False):
    require(len(batch)==4,'B4 accumulation required');opt.zero_grad(set_to_none=True);rows=[];std=store.frames['std'].cuda()
    for member in batch:
        start,h=member['start'],member['horizon'];w=store.get(start);code=bank.code(start)
        prediction=model(w['condition'],code);latent,terms=state_loss(prediction,w['targets'],std)
        state=select_state(w['condition'].coordinates,prediction['features'][h-1],prediction['logits'][h-1],p['max_nodes'])
        sample=store.sample(start,h);images,gs,sh=bridge.render(state,sample['record'])
        rgb,rgb_terms=image_losses(images[0],sample['record']['real'],w['roi'],sample['motion']);loss=rgb+.1*latent
        require(bool(torch.isfinite(loss)),'Nonfinite loss')
        row=dict(start=start,horizon=h,loss=float(loss.detach()),rgb=float(rgb.detach()),latent=float(latent.detach()),gs_count=len(gs))
        if probe:
            zgrad,wgrad=torch.autograd.grad(rgb,(code,model.backbone.Wf.weight),retain_graph=True)
            row.update(rgb_code_grad=float(zgrad.norm()),rgb_decoder_grad=float(wgrad.norm()))
            require(row['rgb_code_grad']>0 and row['rgb_decoder_grad']>0,'RGB gradients missing')
            if model.code_dim>16:
                row['extra_code_rgb_grad']=float(zgrad[:,16:].norm());require(row['extra_code_rgb_grad']>0,'Extra code dimensions inactive')
            if model.width==128:
                row['channel_gradient_asymmetry']=float((wgrad[:64]-wgrad[64:]).norm())
                require(row['channel_gradient_asymmetry']>0,'Duplicate channels cannot diverge')
        (loss/4).backward();rows.append(row)
        del prediction,state,sample,images,gs,sh,latent,terms,rgb,rgb_terms,loss,w,code
    params=list(model.parameters())+list(bank.parameters())
    require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in params),'Nonfinite gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(params,p['grad_clip']));opt.step()
    require(all(bool(torch.isfinite(v).all()) for v in params),'Nonfinite parameter')
    return dict(rows=rows,preclip_gradient_norm=norm)


@torch.no_grad()
def causality(model,bank,store,p):
    was=model.training;model.eval();rows=[]
    for start in (35,123):
        w=store.get(start);z=bank.code(start);original=model(w['condition'],z)
        for h in (1,3,7):
            changed=z.clone();changed[h:]+=torch.linspace(-.5,.5,changed[h:].numel(),device=z.device).reshape_as(changed[h:])
            result=model(w['condition'],changed)
            delta=max(float((original[k][:h]-result[k][:h]).abs().max()) for k in ('features','logits'))
            require(delta<=p['preflight']['causal_max'],'Future code changed an earlier output')
            rows.append(dict(start=start,horizon=h,max_abs=delta))
    model.train(was);return rows


@torch.no_grad()
def anchor_metrics(model,bank,store,bridge,p):
    was=model.training;model.eval();rows=[]
    for start in (35,123):
        w=store.get(start);pred=model(w['condition'],bank.code(start))
        for h in range(1,10):
            state=select_state(w['condition'].coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_nodes'])
            sample=store.sample(start,h);im,gs,sh=bridge.render(state,sample['record'])
            rows.append(dict(start=start,horizon=h,**image_metrics(im[0],sample['record']['real'],w['roi'],sample['motion'])))
    model.train(was);return dict(motion_mae=sum(x['motion_mae'] for x in rows)/18,rows=rows)


def run(arm,phase,out):
    out=Path(out).resolve();require(out.is_relative_to(ROOT) and not out.exists(),'Use isolated new output')
    out.mkdir(parents=True);began=time.monotonic();write(out/'STARTED.json',dict(pid=os.getpid(),arm=arm,phase=phase))
    try:
        p,model,bank,store,bridge,reference,parent,source=setup(arm);opt=optimizer(model,bank,p)
        if phase=='preflight':
            model.eval();reference=reference.cuda();parity=[]
            with torch.no_grad():
                for start in (0,35,123,167):
                    w=store.get(start);expected=reference(w['condition'],parent['codes'][start].cuda());got=model(w['condition'],bank.code(start))
                    errors={k:float((expected[k]-got[k]).abs().max()) for k in ('features','logits')}
                    require(errors['features']<=p['preflight']['feature_parity_max'] and errors['logits']<=p['preflight']['logit_parity_max'],'Warm function parity failed')
                    parity.append(dict(start=start,**errors));del w,expected,got
            del reference,parent;torch.cuda.empty_cache()
            initial=anchor_metrics(model,bank,store,bridge,p)
            expected=read(ASSETS/'parent_summary.json')['pilot_anchor_motion_mae']
            require(abs(initial['motion_mae']-expected)<=p['preflight']['baseline_parent_anchor_motion_mae_tolerance'],'Parent render not reproduced')
            causal=causality(model,bank,store,p)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};codes=bank.stack()
            largest=max(store.manifest['windows'],key=lambda r:r['bytes'])['start']
            batch=[dict(start=s,horizon=h) for s,h in ((largest,9),(35,3),(123,3),(123,6))]
            model.train();torch.cuda.synchronize();t=time.monotonic();probe=train_step(model,bank,store,bridge,opt,batch,p,True);torch.cuda.synchronize()
            seconds=time.monotonic()-t;bridge.assert_frozen()
            model.load_state_dict(state,strict=True)
            with torch.no_grad():
                for i,value in enumerate(bank.values):value.copy_(codes[i].cuda())
            require(digest(model)==source['initial_decoder_digest'] and torch.equal(codes,bank.stack()),'Preflight restore failed')
            write(out/'PREFLIGHT.json',dict(status='passed',source=source,parity=parity,causality=causal,initial_anchors=initial,
                probe=probe,update_seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                max_candidate_probe_start=largest,restored_weights_codes=True,optimizer_discarded=True,elapsed_seconds=time.monotonic()-began))
            print(json.dumps(dict(status='preflight_passed',arm=arm,update_seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True);return
        pre=read(ROOT/'preflight'/arm/'PREFLIGHT.json')
        require(pre['status']=='passed' and pre['source']==source,'Preflight differs from this launch')
        require(read(ROOT/'ENV_AGENT_WITNESS.json')['status']=='passed','Independent environment witness required')
        del reference,parent;model.train();items=sampling(p)
        with (out/'train.jsonl').open('x',buffering=1) as log:
            for index,item in enumerate(items,1):
                require(time.monotonic()-began<p['training_timeout_seconds'],'Training time budget exceeded')
                result=train_step(model,bank,store,bridge,opt,item['batch'],p)
                log.write(json.dumps(dict(local_step=index,**item,elapsed_seconds=time.monotonic()-began,**result),allow_nan=False)+'\n')
                if index==1 or index%p['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',arm=arm,updates=index,budget=846,cumulative_historical_plus_new=item['step'],elapsed_seconds=time.monotonic()-began),True)
                if index in p['checkpoints']:
                    bridge.assert_frozen();path=out/f'step_{index:04d}.pt'
                    with path.open('xb') as f:torch.save(dict(schema='gs_capacity_teacher_v1',source=source,step=index,model=model.state_dict(),codes=bank.stack(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),f)
                    write(out/f'step_{index:04d}.json',dict(status='complete',step=index,sha256=sha(path),model_digest=digest(model),anchors=anchor_metrics(model,bank,store,bridge,p)))
        final_causal=causality(model,bank,store,p);bridge.assert_frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',arm=arm,source=source,updates=846,checkpoint_sha256=sha(out/'step_0846.pt'),causality=final_causal,elapsed_seconds=time.monotonic()-began))
        write(out/'STATUS.json',dict(status='complete',updates=846,evaluation='pending'),True)
    except BaseException as e:
        write(out/'FAILED.json',dict(status='failed',arm=arm,phase=phase,error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','train']);p.add_argument('--arm',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.arm,a.phase,a.out)
