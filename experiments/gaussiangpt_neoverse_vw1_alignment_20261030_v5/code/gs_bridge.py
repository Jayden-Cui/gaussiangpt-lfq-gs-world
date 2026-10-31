"""Trainable continuous GaussianGPT bridge, preserving original source files.

The AE remains in eval mode to disable target-forced pruning and keep existing
BatchNorm running statistics fixed. Its weights are trainable: eval is not
requires_grad(False). Motion occupancy is a soft final-opacity factor.
"""
from pathlib import Path
import hashlib,json,sys
import torch
from torch import nn

STATE_ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state')
MOTION_ROOT=STATE_ROOT.parent/'motion'
V4_ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_vw1_h9_lfq_condentropy_20261030_v4')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def verify_native_sources():
    binding=json.loads((V4_ROOT/'source_binding.json').read_text())
    checked=[]
    for row in binding['dependencies']:
        path=row['path']
        if '/state/code/' in path or '/vendor/GaussianGPT-' in path or path.endswith('/state/selection.json') or path.endswith('/state/protocol.json'):
            if sha(path)!=row['sha256']:raise ValueError('Native dependency changed: '+path)
            checked.append(row)
    checkpoint=STATE_ROOT/'codec_final.pt'
    if sha(checkpoint)!=binding['state_checkpoint_sha256']:raise ValueError('State checkpoint changed')
    manifest=V4_ROOT/'cache_manifest.json'
    if sha(manifest)!=binding['cache_manifest_sha256']:raise ValueError('Cache manifest changed')
    initializer=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_state_vae_20261028_v1/checkpoints/vqvae_photoshape_official.ckpt')
    return dict(state_checkpoint_sha256=binding['state_checkpoint_sha256'],initializer_checkpoint_path=str(initializer),
        initializer_checkpoint_sha256=sha(initializer),
        cache_manifest_sha256=binding['cache_manifest_sha256'],dependencies=checked)

def verify_window_sources(spec):
    rows={v['path']:v for v in json.loads((V4_ROOT/'cache_manifest.json').read_text())}
    checked=[]
    for frame in spec.frames:
        for path in [STATE_ROOT/'cache'/spec.clip/f'{frame:03d}.pt',
                     MOTION_ROOT/'states'/spec.clip/f'{frame:03d}.pt']:
            row=rows[str(path)]
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('Window source changed: '+str(path))
            checked.append(row)
    return checked

class NativeGSBridge(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        self.binding=verify_native_sources()
        sys.path.insert(0,str(STATE_ROOT/'code'))
        from codec import load_codec
        from common import prediction_gs,render
        self.codec=load_codec(STATE_ROOT,STATE_ROOT/'codec_final.pt',device=device)
        self.codec.eval().requires_grad_(True)
        if not isinstance(self.codec.ae.vq,nn.Identity):raise RuntimeError('State quantization must remain removed')
        self._prediction_gs=prediction_gs;self._render=render
        self.last_stats={}
    def train(self,mode=True):
        super().train(mode)
        self.codec.eval()
        return self
    def render_prediction(self,coordinates,features,logits,record,view=0):
        from soft_occupancy import SoftOccupancyTracker
        if self.codec.ae.decoder.training:raise RuntimeError('Target-forced decoder mode is forbidden')
        if features.shape!=(len(coordinates),64) or logits.shape!=(len(coordinates),):
            raise ValueError('Invalid predicted latent support')
        if len(coordinates)>100000:raise RuntimeError('Initial candidate budget exceeded; no truncation')
        # No hard motion support mask, no target coordinate key, no future state.
        z=self.codec._sparse(coordinates,features,self.codec.ae.stride)
        input_coordinates=torch.cat([torch.zeros((len(coordinates),1),device=features.device,dtype=torch.int32),
            coordinates.to(device=features.device,dtype=torch.int32)*int(self.codec.ae.stride)],1)
        with SoftOccupancyTracker(self.codec.ae.decoder,z,logits,input_coordinates=input_coordinates) as tracker:
            points,_=self.codec.ae.decode(z,target_key=None)
            points=self.codec._world(points)
            gs,sh=self._prediction_gs(points)
            gate=tracker.gaussian_gate(self.codec.count)
            if len(gate)!=len(gs):raise RuntimeError('GS multiplier order/count does not match opacity gate')
            gs=torch.cat([gs[:,:13],(gs[:,13]*gate)[:,None]],1)
            stats=tracker.summary()
        image,_=self._render(gs,record['cameras'][view],record['intrinsics'],record['near'],record['far'],sh)
        if image.shape!=(252,448,3) or not bool(torch.isfinite(image).all()):
            raise RuntimeError('Invalid full-resolution renderer output')
        self.last_stats=dict(latent_candidates=len(coordinates),decoded_gaussians=len(gs),gate_min=float(gate.detach().min()),
            gate_max=float(gate.detach().max()),**stats)
        # Native image loss will use this range consistently for real/predicted.
        return image.clamp(0,1)

