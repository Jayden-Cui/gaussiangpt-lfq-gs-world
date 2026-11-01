"""Teacher-independent C0/C1/C2 posterior components for continuous GS action codes."""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def aligned_fields(states):
    """Return fields on a common world-cell grid, with absent features set to zero.

    Coordinates, not array row numbers or Gaussian IDs, define correspondence.
    The caller must supply a prefix ending at the requested output horizon.
    """
    if not 1<=len(states)<=10:raise ValueError('Expected 1..10 prefix states')
    from aligned_ldm import validate_state
    for state in states:validate_state(state)
    times=[float(state['pts']) for state in states]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Times must increase')
    coords,inverse=torch.unique(torch.cat([state['coordinates'] for state in states]),
                                dim=0,sorted=True,return_inverse=True)
    dense=states[0]['features'].new_zeros((len(states),len(coords),64))
    present=torch.zeros((len(states),len(coords)),device=dense.device,dtype=torch.bool)
    offset=0
    for t,state in enumerate(states):
        indices=inverse[offset:offset+len(state['coordinates'])];offset+=len(indices)
        if indices.unique().numel()!=indices.numel():raise ValueError('Duplicate world cell in a frame')
        dense[t,indices]=state['features'];present[t,indices]=True
    return dict(coordinates=coords,features=dense,present=present,
                delta_features=dense-dense[0:1],
                delta_occupancy=present.to(dense.dtype)-present[0:1].to(dense.dtype),
                seen=present.to(torch.int32).cumsum(0)>0,times=times)


class PosteriorEncoderCopy(nn.Module):
    """Trainable copy; never shares decoder's frozen initial-state encoder."""
    def __init__(self,base):
        super().__init__()
        self.input=copy.deepcopy(base.input)
        self.encoder_backbone=copy.deepcopy(base.encoder_backbone)
        self.requires_grad_(True)
        source={id(parameter) for module in (base.input,base.encoder_backbone) for parameter in module.parameters()}
        if any(id(parameter) in source for parameter in self.parameters()):
            raise RuntimeError('Posterior and decoder encoders share parameters')

    def _encode_hidden(self,hidden,coordinates,active,neighbors):
        return self.encoder_backbone(hidden,coordinates,active)

    def prepare(self,states):
        from aligned_ldm import AlignedLDM
        return AlignedLDM.prepare(self,states)


class ContinuousPosterior(nn.Module):
    """C0 frozen memory, C1 trainable encoder, C2 trainable encoder plus time/deltas.

    C1 and C2 recompute from raw GS states; there is no cached/detached hidden
    feature on their learning path. All outputs are continuous [9,16] codes.
    """
    def __init__(self,base,posterior,variant,*,checkpoint_prefixes=True,include_time=True,include_change=True):
        super().__init__()
        if variant not in ('C0','C1','C2'):raise ValueError('Expected C0, C1 or C2')
        self.variant=variant
        self.posterior=copy.deepcopy(posterior).requires_grad_(True)
        self.encoder=PosteriorEncoderCopy(base) if variant!='C0' else None
        self.checkpoint_prefixes=bool(checkpoint_prefixes)
        self.include_time=bool(include_time) if variant=='C2' else False
        self.include_change=bool(include_change) if variant=='C2' else False
        if variant=='C2':
            # 64 aligned feature differences and four occupancy/time flags.
            if self.include_change:
                self.delta_projection=nn.Linear(68,64)
                nn.init.normal_(self.delta_projection.weight,std=.01)
                nn.init.zeros_(self.delta_projection.bias)
            if self.include_time:
                self.time_projection=nn.Linear(18,64,bias=False)
                nn.init.normal_(self.time_projection.weight,std=.01)

    @staticmethod
    def _time_features(states,device,dtype):
        t=torch.tensor([float(s['pts']) for s in states],device=device,dtype=dtype)
        relative=t-t[0];age=t[-1]-t
        pair=torch.stack((relative,age),-1)
        frequencies=2**torch.arange(4,device=device,dtype=dtype)
        phase=pair[...,None]*frequencies*torch.pi
        return torch.cat((pair,phase.sin().flatten(-2),phase.cos().flatten(-2)),-1)

    def memory_for_h(self,states,h):
        if self.variant=='C0':raise ValueError('C0 consumes existing frozen prefix memories')
        if not 1<=h<=9 or len(states)<h+1:raise ValueError('Invalid prefix horizon')
        prefix=states[:h+1]
        prepared=self.encoder.prepare(prefix)
        present=prepared['present'];coordinates=prepared['coordinates'];hidden=prepared['hidden']
        if self.variant=='C1':
            mask=present
            coords=coordinates[None].expand(h+1,-1,-1)[mask]
            return hidden[mask],self.posterior.position(coords),dict(mask=mask,coordinates=coordinates)
        fields=aligned_fields(prefix)
        if not torch.equal(fields['coordinates'],coordinates) or not torch.equal(fields['present'],present):
            raise RuntimeError('World-aligned fields differ from encoder input')
        # At time tau, include cells seen at or before tau, including disappeared cells.
        mask=fields['seen'] if self.include_change else present;n=len(coordinates)
        coords=coordinates[None].expand(h+1,-1,-1)[mask]
        time_ids=torch.arange(h+1,device=hidden.device)[:,None].expand(-1,n)[mask]
        now=present.to(hidden.dtype);initial=present[0:1].expand_as(present).to(hidden.dtype)
        latest=(torch.arange(h+1,device=hidden.device)==h)[:,None].expand(-1,n).to(hidden.dtype)
        extra=torch.cat((fields['delta_features'],now[...,None],initial[...,None],
                         fields['delta_occupancy'][...,None],latest[...,None]),-1)[mask]
        keys=hidden[mask]+self.delta_projection(extra) if self.include_change else hidden[mask]
        time=self._time_features(prefix,hidden.device,hidden.dtype)[time_ids]
        positions=self.posterior.position(coords)
        if self.include_time:positions=positions+self.time_projection(time)
        return keys,positions,dict(mask=mask,coordinates=coordinates,time_ids=time_ids,
                                   present=present,aligned=fields)

    def _code_for_h(self,states,h):
        from model_core import qformer_sequence
        keys,positions,_=self.memory_for_h(states,h)
        if not len(keys):raise RuntimeError('Empty Q-Former memory')
        valid=torch.ones((1,len(keys)),device=keys.device,dtype=torch.bool)
        query=self.posterior.queries.weight[h-1:h][None]
        vector=qformer_sequence(self.posterior.posteriors[h-1],keys[None],positions[None],query,valid)
        code=self.posterior.project_in(vector.float())[0,0]
        if code.shape!=(16,) or not bool(torch.isfinite(code).all()):raise RuntimeError('Invalid continuous code')
        return code

    def code_for_h(self,states,h):
        if self.variant=='C0':raise ValueError('Use forward_cached for C0')
        prefix=states[:h+1]
        if self.training and self.checkpoint_prefixes:
            # Non-reentrant checkpoint supports fixed input states and trainable module parameters.
            def calculate(*features):
                values=[dict(s,features=f) for s,f in zip(prefix,features)]
                return self._code_for_h(values,h)
            return checkpoint(calculate,*(s['features'] for s in prefix),use_reentrant=False)
        return self._code_for_h(prefix,h)

    def forward_cached(self,memories):
        if self.variant!='C0':raise ValueError('C1/C2 must recompute raw-state features')
        return self.posterior([memories])['codes'][0]

    def forward_states(self,states):
        if self.variant=='C0':raise ValueError('C0 uses frozen prefix memory')
        if len(states)!=10:raise ValueError('Expected ten adjacent states')
        return torch.stack([self.code_for_h(states,h) for h in range(1,10)])
