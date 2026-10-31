"""Seeded real full-state loss/backward + continuous decode witness, no weight update."""
import argparse,gc,time
from pathlib import Path
import torch
from common import CLIPS,read,write
from codec import load_codec
from train import load_frame,terms
from evaluate import api_check
from perceptual import Vgg19

def main(root):
    started=time.monotonic();torch.set_num_threads(6);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.60);torch.manual_seed(20261029)
    assert read(root/'prepare_done.json')['states']==177
    model=load_codec(root).train();vgg=Vgg19(root/'torch_cache').cuda()
    # First, middle and last ensure data-domain changes do not only work at frame0.
    reports=[]
    for t in [0,88,176]:
        model.zero_grad(set_to_none=True);p=load_frame(root,CLIPS[0],t)
        losses,info=terms(model,p,vgg,True)
        loss=sum(read(root/'protocol.json')['train']['loss'][k]*v for k,v in losses.items())
        assert torch.isfinite(loss);loss.backward();groups={}
        for name,module in [('encoder',model.ae.encoder),('decoder',model.ae.decoder),('heads',model.ae.feature_heads),('set_encoder',model.set_encoder)]:
            gs=[x.grad for x in module.parameters() if x.grad is not None]
            assert gs and all(torch.isfinite(g).all() for g in gs),name
            norm=float(sum(g.float().square().sum() for g in gs).sqrt());assert norm>0,name;groups[name]=norm
        reports.append({'frame':t,'loss':float(loss.detach()),'latent_tokens':info['latent_count'],
            'decoded_gaussians':info['output_count'],'gradient_norms':groups})
        del p,losses,info,loss
    model.zero_grad(set_to_none=True);del vgg;gc.collect();torch.cuda.empty_cache()
    api=api_check(model,load_frame(root,CLIPS[0],0))
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=48
    result={'status':'passed','reports':reports,'api':api,'peak_allocated_gib':peak,'elapsed_seconds':time.monotonic()-started,
        'weights_updated':False,'state_lfq_removed':isinstance(model.ae.vq,torch.nn.Identity),'scope':'Engineering witness with real NeoVerse states, not quality acceptance'}
    write(root/'preflight.json',result);print('NEOVERSE_STATE_WITNESS',result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
