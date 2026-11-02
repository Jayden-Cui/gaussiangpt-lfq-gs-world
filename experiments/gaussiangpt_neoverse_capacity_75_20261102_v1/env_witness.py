"""One seeded real-native-render witness; no optimizer or training updates."""
from runtime_env import configure
configure()
import argparse,json,time,traceback,sys
from pathlib import Path
import torch
from bindings import ROOT,ASSETS,NativeBridge,Store,write,sha

def main(output):
    output=Path(output);began=time.monotonic();torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(20261102)
    r=dict(schema='private_runtime_native_gpu_witness_v1',code_sha256={p.name:sha(p) for p in (ROOT/'env_witness.py',ROOT/'bindings.py',ROOT/'runtime_env.py')})
    try:
        a=torch.randn(16,16,device='cuda');r['matrix_finite']=bool(torch.isfinite(a@a).all())
        bridge=NativeBridge().eval();store=Store();state=store.get(35)['states'][0]
        features=state['features'].detach().clone().requires_grad_(True)
        sample=store.sample(35,0)
        images,gs,sh=bridge.render(dict(coordinates=state['coordinates'],features=features),sample['record'],views=(0,1))
        loss=(images[0]-sample['record']['real']).abs().mean();loss.backward()
        assert features.grad is not None and bool(torch.isfinite(features.grad).all()) and float(features.grad.norm())>0
        bridge.assert_frozen()
        r.update(status='passed',python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),render_shape=list(images.shape),gs_count=len(gs),loss=float(loss),
            feature_gradient_norm=float(features.grad.norm()),native_codec_frozen=True,elapsed_seconds=time.monotonic()-began,
            max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,training_updates=0)
    except BaseException as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    write(output,r);print('NATIVE_ENV_WITNESS '+json.dumps(r),flush=True)
    if r['status']!='passed':raise RuntimeError('Environment witness failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
