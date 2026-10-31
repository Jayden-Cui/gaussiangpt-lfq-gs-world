"""Joint native-GS forward for full ten-frame image training.

The raw-GS encoder is re-executed with autograd whenever this function is
called with grad enabled. Both generator and discriminator passes use the
same complete-window/initial-only geometry contract.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
from aligned_ldm import InitialCondition,make_condition
from online_state import load_online_window
from gs_bridge import STATE_ROOT,MOTION_ROOT

class InitialGeometryCache:
    """Only deterministic coordinate geometry is cached, never learned features."""
    def __init__(self):
        self.values={}
    def condition(self,state,spec):
        c=state['coordinates'].detach().cpu().numpy().astype(np.int32)
        fingerprint=hashlib.sha256(c.tobytes()).hexdigest()
        key=(spec.clip,spec.start,fingerprint)
        if key not in self.values:
            source=MOTION_ROOT/'geometry'/spec.clip/f'{spec.start:03d}.pt'
            geometry=torch.load(source,map_location='cpu',weights_only=True) if source.exists() else None
            if geometry is not None and geometry['source_coordinates_sha256']==fingerprint and geometry['radius']==6:
                if len(geometry['coordinates'])>100000:raise RuntimeError('Candidate budget exceeded')
                self.values[key]=tuple(geometry[n].detach().cpu().clone() for n in ['coordinates','neighbors','weights'])
            else:
                condition=make_condition(state,radius=6)
                self.values[key]=(condition.coordinates.detach().cpu(),condition.neighbors.detach().cpu(),condition.weights.detach().cpu())
        coords,neighbors,weights=(v.to(state['features'].device) for v in self.values[key])
        if weights.shape!=neighbors.shape or not bool(torch.isfinite(weights).all()):raise ValueError('Invalid initial geometry')
        if not bool(torch.allclose(weights.sum(1),torch.ones(len(weights),device=weights.device),atol=1e-5,rtol=1e-5)):
            raise ValueError('Initial interpolation weights must sum to1')
        return InitialCondition(state,coords,neighbors,weights)

def encode_batch(model,bridge,specs,geometry_cache):
    records=[load_online_window(STATE_ROOT,MOTION_ROOT,spec,bridge.codec,'cuda') for spec in specs]
    states=[record['window'] for record in records]
    conditions=[geometry_cache.condition(window[0],spec) for window,spec in zip(states,specs)]
    z=model.encode_windows(states)
    predictions=model.decode_latents(z['z_quantized'],conditions,initial_contexts=z['initial_contexts'])
    return dict(records=records,states=states,conditions=conditions,z=z,predictions=predictions)

def render_sampled_frames(bridge,batch,frame_indices):
    if frame_indices.shape!=(len(batch['records']),):raise ValueError('Expected one sampled index per window')
    fake=[];real=[]
    for b,index in enumerate(frame_indices.detach().cpu().tolist()):
        if not 0<=index<10:raise ValueError('Sampled frame outside full10')
        p=batch['predictions'][b][index];record=batch['records'][b]['records'][index]
        fake.append(bridge.render_prediction(p['coordinates'],p['features'],p['occupancy_logits'],record).permute(2,0,1)*2-1)
        real.append(record['real'].permute(2,0,1)*2-1)
    return torch.stack(fake),torch.stack(real)

def render_full_videos(bridge,batch,*,checkpoint_renderer=True):
    fake=[];real=[];statistics=[]
    for b,sequence in enumerate(batch['predictions']):
        f=[];r=[];rows=[]
        for h,p in enumerate(sequence):
            record=batch['records'][b]['records'][h]
            # Bind this record now. A late-bound loop closure would reconstruct
            # every checkpoint with the final frame's camera during backward.
            def render(coordinates,features,logits,record=record):
                return bridge.render_prediction(coordinates,features,logits,record)
            if checkpoint_renderer and torch.is_grad_enabled():
                # Complete the hook context during recomputation as well.
                with set_checkpoint_early_stop(False):
                    image=checkpoint(render,p['coordinates'],p['features'],p['occupancy_logits'],use_reentrant=False)
            else:image=render(p['coordinates'],p['features'],p['occupancy_logits'])
            f.append(image.permute(2,0,1)*2-1);r.append(record['real'].permute(2,0,1)*2-1)
            rows.append(dict(horizon=h,source_frame=batch['records'][b]['frames'][h],**bridge.last_stats))
        fake.append(torch.stack(f));real.append(torch.stack(r));statistics.append(rows)
    return torch.stack(fake),torch.stack(real),statistics

