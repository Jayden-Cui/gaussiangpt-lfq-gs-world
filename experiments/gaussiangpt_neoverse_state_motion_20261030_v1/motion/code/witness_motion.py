"""Validate actual new frozen state codec and real adjacent-pair QFormer gradients."""
import argparse,gc,time
from pathlib import Path
import torch
from motion_common import BASE,CLIPS,read,write,frozen_codec,load_pair,load_target,Vgg19
from qmotion import MotionQFormer
from losses_motion import all_losses
from train_motion import parameter_sha

def main(root,arm):
    cfg=read(root/'protocol.json');k=next(a['motion_tokens'] for a in cfg['arms'] if a['name']==arm)
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60);torch.manual_seed(cfg['seed'])
    started=time.monotonic();codec=frozen_codec(root);before=parameter_sha(codec)
    model=MotionQFormer(k).cuda();vgg=Vgg19(BASE/'torch_cache').cuda();reports=[]
    for start in [0,88,175]:
        end=start+cfg['frame_gap'];s,t,initial,condition,idx=load_pair(root,CLIPS[0],start,end,codec)
        assert abs(initial.dt-1/29.97)<1e-9
        assert isinstance(codec.ae.vq,torch.nn.Identity)
        with torch.no_grad():
            model.eval();z=model.encode_motion(s,t,initial.dt);prediction=model.restore(z['z_quantized'],condition)
            packet=model.packet(initial,z['indices'].long(),binding='witness');replay=model.replay(packet,codec,binding='witness')
            error=float((prediction['features']-replay['features']).abs().max());assert error<1e-5
            try:model.replay(dict(packet,future_coordinates=t['coordinates']),codec,binding='witness')
            except ValueError:pass
            else:raise RuntimeError('Future-side data bypass accepted')
        del z,prediction,replay
        model.train();model.zero_grad(set_to_none=True)
        z=model.encode_motion(s,t,initial.dt);prediction=model.restore(z['z_quantized'],condition)
        terms,aux=all_losses(codec,prediction,t,idx,load_target(CLIPS[0],end),vgg,
            force_target=True,max_tokens=cfg['support']['max_predicted_tokens'])
        loss=sum(cfg['train']['render_loss'][key]*value for key,value in terms.items());assert torch.isfinite(loss);loss.backward()
        norms={}
        for name,module in [('posterior',model.posterior),('fsq',model.quantizer),('restorer',model.restorer),('delta',model.delta),('occupancy',model.occupancy)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
            n=float(sum(g.float().square().sum() for g in grads).sqrt());assert n>0;norms[name]=n
        assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
        reports.append({'start':start,'end':end,'dt':initial.dt,'z_shape':list(z['z_quantized'].shape),
            'id_shape':list(z['indices'].shape),'packet_max_abs':error,'gradient_norms':norms,**codec._motion_decode_stats})
        del s,t,initial,condition,idx,z,prediction,terms,aux,loss;gc.collect();torch.cuda.empty_cache()
    assert parameter_sha(codec)==before
    peak=torch.cuda.max_memory_allocated()/2**30;assert peak<=48
    write(root/arm/'preflight.json',{'status':'passed','reports':reports,'peak_allocated_gib':peak,'elapsed_seconds':time.monotonic()-started})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();main(a.root,a.arm)
