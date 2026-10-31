"""GPU witness for native full10 L1/LPIPS and joint B4 discriminator, no training."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from torch import nn
from reconstruction_objective import (FrozenLPIPS,build_discriminator,generator_image_loss,
    discriminator_image_loss,pick_video_frames,sample_frame_indices)

def run(root):
    started=time.monotonic()
    try:
        torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60);torch.manual_seed(20261031)
        free,_=torch.cuda.mem_get_info()
        if free<55*2**30:raise RuntimeError('GPU free memory below55GiB')
        perceptual=FrozenLPIPS().cuda()
        discriminator=build_discriminator().cuda().train()
        generator=nn.Conv2d(3,3,1).cuda()
        real=torch.rand(4,10,3,252,448,device='cuda')*2-1
        fake=generator(real.flatten(0,1)).tanh().reshape_as(real)
        didx=sample_frame_indices(4,device='cuda')
        d=discriminator_image_loss(discriminator,pick_video_frames(fake,didx),pick_video_frames(real,didx))
        d['loss'].backward()
        dnorm=float(torch.stack([p.grad.detach().float().square().sum() for p in discriminator.parameters() if p.grad is not None]).sum().sqrt())
        if not 0<dnorm<float('inf'):raise AssertionError('No finite discriminator/GP gradient')
        discriminator.zero_grad(set_to_none=True);discriminator.requires_grad_(False)
        gidx=sample_frame_indices(4,device='cuda')
        g=generator_image_loss(discriminator,perceptual,fake,real,generator.weight,frame_indices=gidx)
        g['loss'].backward()
        gnorm=float(generator.weight.grad.norm())
        if not 0<gnorm<float('inf') or not bool(torch.isfinite(g['loss'])):raise AssertionError('Invalid full image objective gradient')
        if any(p.grad is not None for p in perceptual.parameters()):raise AssertionError('LPIPS weights must stay frozen')
        if perceptual.training or perceptual.metric.training:raise AssertionError('LPIPS must stay eval')
        result=dict(status='passed',scope='B4_full10_native_image_objective_primitives_gpu',
            optimizer_updates=0,video_shape=list(fake.shape),discriminator_input_shape=[4,3,128,128],
            discriminator_gradient_norm=dnorm,generator_gradient_norm=gnorm,
            discriminator_loss=float(d['loss'].detach()),gradient_penalty=float(d['gradient_penalty'].detach()),
            l1=float(g['l1'].detach()),lpips=float(g['lpips'].detach()),adversarial=float(g['adversarial'].detach()),
            adaptive_weight=float(g['adaptive_weight']),d_frame_indices=didx.cpu().tolist(),g_frame_indices=gidx.cpu().tolist(),
            lpips_weight_provenance=perceptual.weight_provenance,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
            elapsed_seconds=time.monotonic()-started,uses_real_gs=False,
            limitation='Synthetic image generator; this does not verify full B4 GS system integration or quality')
        (root/'GPU_OBJECTIVE_WITNESS.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        (root/'GPU_OBJECTIVE_FAILURE.json').write_text(json.dumps(dict(status='failed',error=repr(error),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

