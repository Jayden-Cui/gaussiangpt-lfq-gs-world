"""Optimize only the per-window float codes; all decoding parameters stay frozen."""
import argparse
import math
import time
import traceback
from pathlib import Path
import torch
from torch import nn
from stage_context import ROOT,setup,setup_gpu,load_codes,load_decoder,dataset,source_record,assert_frozen,read,write,sha
from sampling import build,audit


class FreeCodes(nn.Module):
    def __init__(self, initial, starts, device):
        super().__init__()
        self.starts = list(starts)
        self.index = {start:i for i,start in enumerate(starts)}
        self.values = nn.ParameterList([nn.Parameter(initial[start].clone().to(device)) for start in starts])

    def code(self,start):
        return self.values[self.index[start]]

    def stack(self):
        return torch.stack([p.detach().cpu() for p in self.values])


def make_optimizer(bank,c):
    return torch.optim.Adam(bank.parameters(),lr=c['learning_rate'],betas=tuple(c['betas']),eps=c['eps'])


def step(bank,decoder,bridge,store,std,opt,item,p,probe=False):
    from core import image_losses
    from support_decoder import state_loss,select_state
    from runtime import render
    opt.zero_grad(set_to_none=True)
    rows=[]
    for member in item['batch']:
        start,h=member['start'],member['horizon']
        w=store.get(start);code=bank.code(start)
        prediction=decoder(w['condition'],code)
        latent,terms=state_loss(prediction,w['targets'],std)
        state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                           prediction['logits'][h-1],p['max_predicted_latent_nodes'])
        sample=store.sample(start,h)
        images,gs,sh=render(bridge,state,sample['record'])
        rgb,_=image_losses(images[0],sample['record']['real'],w['roi'],sample['motion'])
        if probe:
            gradient=torch.autograd.grad(rgb,code,retain_graph=True)[0]
            norm=float(gradient.norm())
            if not math.isfinite(norm) or norm<=0:raise RuntimeError('RGB gradient did not reach free code')
        loss=rgb+.1*latent
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite free-code loss')
        row=dict(start=start,horizon=h,loss=float(loss.detach()),rgb_loss=float(rgb.detach()),
                 latent_loss=float(latent.detach()),occupancy_loss=float(terms['occupancy'].detach()),
                 feature_loss=float(terms['feature'].detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
        if probe:row['rgb_to_code_gradient_norm']=norm
        (loss/len(item['batch'])).backward();rows.append(row)
        del w,prediction,latent,terms,state,sample,images,gs,sh,rgb,loss
    selected={m['start'] for m in item['batch']}
    for start in bank.starts:
        gradient=bank.code(start).grad
        if start not in selected and gradient is not None:raise RuntimeError('Unselected window acquired gradient')
        if gradient is not None and not bool(torch.isfinite(gradient).all()):raise RuntimeError('Nonfinite code gradient')
    norm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),p['grad_clip']))
    opt.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in bank.parameters()):
        raise RuntimeError('Nonfinite optimized code')
    return dict(windows=rows,preclip_gradient_norm=norm,clip_triggered=norm>p['grad_clip'])


def main(shard):
    c,helper,orig,p,paths,receipt,evaluation=setup()
    if shard not in (0,1):raise ValueError('Unknown shard')
    out=ROOT/f'shard_{shard}';out.mkdir()
    began=time.monotonic();context=dict(phase='initializing',completed_updates=0)
    write(out/'STARTED.json',dict(shard=shard,pid=__import__('os').getpid()))
    write(out/'STATUS.json',dict(status='running',**context))
    try:
        witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256'):
            raise RuntimeError('CPU witness changed')
        setup_gpu(p)
        decoder,modules=load_decoder(c,helper,orig,p,paths,receipt,'cuda')
        fm,data,runtime=modules
        store=dataset(modules,p,c);std=store.frames['std'].cuda()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        initial=load_codes(paths,evaluation)['correct']
        starts=c['shards'][str(shard)]
        bank=FreeCodes(initial,starts,'cuda');opt=make_optimizer(bank,c)
        sampler=audit(c)
        if sampler!=witness['sampling']:raise RuntimeError('Sampling witness differs')
        source=source_record(c,paths,receipt,sampler['shards'][str(shard)])
        write(out/'PROVENANCE.json',dict(source=source,starts=starts,only_free_codes_trainable=True))
        probes=([0,35,56,83] if shard==0 else [84,112,123,167])
        before=bank.stack().clone()
        probe=step(bank,decoder,bridge,store,std,opt,dict(batch=[dict(start=s,horizon=9) for s in probes]),p,probe=True)
        if torch.equal(before,bank.stack()):raise RuntimeError('Probe did not update free codes')
        assert_frozen(decoder,bridge,runtime,c['decoder_tensor_digest'])
        del bank,opt
        bank=FreeCodes(initial,starts,'cuda');opt=make_optimizer(bank,c)
        if not torch.equal(before,bank.stack()) or opt.state:raise RuntimeError('Probe reset failed')
        write(out/'PREFLIGHT.json',dict(status='passed',windows=probes,horizon=9,batch_size=4,
              discarded_updates=1,only_codes_trainable=True,decoder_and_codec_frozen=True,
              restored_initial_codes_and_optimizer=True,results=probe))
        items=build(c,shard)
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for index,item in enumerate(items):
                context=dict(phase='optimizing_codes',completed_updates=index,sweep=item['sweep'])
                if time.monotonic()-began>c['training_max_seconds_per_shard']:
                    raise RuntimeError('Training time budget reached')
                row=step(bank,decoder,bridge,store,std,opt,item,p)
                row.update(step=index+1,sweep=item['sweep'],elapsed_seconds=time.monotonic()-began)
                log.write(__import__('json').dumps(row,allow_nan=False)+'\n')
                if index==0 or (index+1)%c['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase='optimizing_codes',
                        completed_updates=index+1,total_updates=len(items),sweep=item['sweep'],
                        loss=sum(x['loss'] for x in row['windows'])/4,
                        elapsed_seconds=time.monotonic()-began),replace=True)
                if item['sweep'] in c['checkpoint_sweeps'] and (index+1)%189==0:
                    assert_frozen(decoder,bridge,runtime,c['decoder_tensor_digest'])
                    path=out/f"sweep_{item['sweep']:02d}.pt"
                    runtime.save(path,dict(schema='fixed_decoder_free_codes_shard_v1',shard=shard,
                        sweep=item['sweep'],starts=starts,codes=bank.stack(),optimizer=opt.state_dict(),source=source))
                    disk=torch.load(path,map_location='cpu',weights_only=True)
                    if (disk['starts']!=starts or disk['sweep']!=item['sweep']
                            or not torch.equal(disk['codes'],bank.stack())):
                        raise RuntimeError('Saved codes differ from optimized codes')
                    write(out/f"sweep_{item['sweep']:02d}.json",dict(status='complete',sweep=item['sweep'],
                        checkpoint_sha256=sha(path),shape=list(disk['codes'].shape),only_free_codes_trainable=True))
                    del disk
        assert_frozen(decoder,bridge,runtime,c['decoder_tensor_digest'])
        write(out/'DONE.json',dict(status='complete',updates=len(items),sweeps=c['sweeps'],
              windows=len(starts),source=source,decoder_and_codec_frozen=True,
              code_shape=[len(starts),9,16],elapsed_seconds=time.monotonic()-began,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(out/'STATUS.json',dict(status='complete',completed_updates=len(items),sweep=c['sweeps']),replace=True)
    except BaseException as error:
        failure=dict(status='failed',shard=shard,context=context,error=repr(error),traceback=traceback.format_exc())
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',required=True,type=int)
    main(parser.parse_args().shard)
