"""Frozen continuous state-AE bridge; no LFQ, state IDs or quantizer inverse."""
import hashlib,sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
BASE=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state')
sys.path.insert(0,str(BASE/'code'))
from common import CLIPS,H,W,read,write,sha,render,prediction_gs,metrics,region_metrics
from codec import load_codec
from perceptual import Vgg19,ssim

def frozen_codec(root=None):
    model=load_codec(BASE,BASE/'codec_final.pt').eval().requires_grad_(False)
    assert isinstance(model.ae.vq,torch.nn.Identity)
    model._motion_decode_stats={}
    def reset(module,inputs):model._motion_decode_stats={}
    def candidate(module,inputs,output):
        model._motion_decode_stats['max_candidates']=max(model._motion_decode_stats.get('max_candidates',0),len(output.F))
    def output_count(module,inputs,output):model._motion_decode_stats['gaussians']=len(output.F)*model.count
    model.ae.decoder.register_forward_pre_hook(reset)
    for stage in model.ae.decoder.ups:stage.register_forward_hook(candidate,prepend=True)
    model.ae.decoder.out.register_forward_hook(output_count,prepend=True)
    return model

def state_path(root,clip,t):return root/'states'/clip/f'{t:03d}.pt'
def load_state(root,clip,t,device='cuda'):
    p=torch.load(state_path(root,clip,t),map_location='cpu',weights_only=True)
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in p.items()}
def load_target(clip,t,device='cuda'):
    p=torch.load(BASE/'cache'/clip/f'{t:03d}.pt',map_location='cpu',weights_only=True)
    return {k:p[k].to(device) if torch.is_tensor(p[k]) else p[k] for k in ['cameras','intrinsics','targets','real','near','far','normalization']}
def render_points(points,target,view=0):
    gs,sh=prediction_gs(points)
    return render(gs,target['cameras'][view],target['intrinsics'],target['near'],target['far'],sh)[0].clamp(0,1)

@dataclass
class InitialState:
    coordinates:torch.Tensor
    features:torch.Tensor
    dt:float
    def validate(self):
        assert self.coordinates.ndim==2 and self.coordinates.shape[1]==3
        assert self.coordinates.dtype in (torch.int32,torch.int64)
        assert self.features.shape==(len(self.coordinates),64) and self.features.is_floating_point()
        assert torch.isfinite(self.features).all() and np.isfinite(self.dt) and self.dt>0

def candidate_geometry(coordinates,radius=6,limit=100000):
    c=coordinates.detach().cpu().numpy().astype(np.int32)
    offsets=np.stack(np.meshgrid(*([np.arange(-radius,radius+1,dtype=np.int32)]*3),indexing='ij'),-1).reshape(-1,3)
    candidates=np.unique((c[:,None]+offsets[None]).reshape(-1,3),axis=0)
    assert len(candidates)<=limit,f'Candidate budget exceeded:{len(candidates)}'
    distance,neighbors=cKDTree(c).query(candidates,k=4,workers=1)
    weights=1/np.maximum(distance,1e-6)**2;weights/=weights.sum(1,keepdims=True)
    present=distance[:,0]==0;weights[present]=0;weights[present,0]=1
    return {'coordinates':torch.from_numpy(candidates),'neighbors':torch.from_numpy(neighbors.astype(np.int64)),
        'weights':torch.from_numpy(weights.astype(np.float32)),'initial_present':torch.from_numpy(present),
        'distance':torch.from_numpy(distance[:,0].astype(np.float32)),
        'source_coordinates_sha256':hashlib.sha256(c.tobytes()).hexdigest(),'radius':radius}

def make_condition(codec,initial,geometry=None,radius=6,limit=100000):
    initial.validate()
    if geometry is None:geometry=candidate_geometry(initial.coordinates,radius,limit)
    c=initial.coordinates.detach().cpu().numpy().astype(np.int32)
    assert geometry['source_coordinates_sha256']==hashlib.sha256(c.tobytes()).hexdigest() and geometry['radius']==radius
    g={k:v.to(initial.features.device) if torch.is_tensor(v) else v for k,v in geometry.items()}
    base=(initial.features[g['neighbors']]*g['weights'][:,:,None]).sum(1)
    return {'coordinates':g['coordinates'],'features':base,'base_features':base,'initial_present':g['initial_present'],
        'distance':g['distance'],'dt':float(initial.dt)}

def alignment_indices(candidates,future_coordinates):
    d,i=cKDTree(candidates.detach().cpu().numpy()).query(future_coordinates.detach().cpu().numpy(),k=1,workers=1)
    assert np.all(d==0),f'Future support outside fixed initial-only candidates:{np.count_nonzero(d)}'
    return torch.from_numpy(i.astype(np.int64))
def pair_ids(protocol):return [(c,t,t+protocol['frame_gap']) for c in CLIPS for t in range(177-protocol['frame_gap'])]
def load_pair(root,clip,start,end,codec):
    s,t=load_state(root,clip,start),load_state(root,clip,end)
    geometry=torch.load(root/'geometry'/clip/f'{start:03d}.pt',map_location='cpu',weights_only=True)
    initial=InitialState(s['coordinates'],s['features'],t['pts']-s['pts'])
    condition=make_condition(codec,initial,geometry)
    labels=torch.load(root/'labels'/clip/f'{start:03d}.pt',map_location='cpu',weights_only=True)
    return s,t,initial,condition,labels['future_candidate_indices'].cuda()

