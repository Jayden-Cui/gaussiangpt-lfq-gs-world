"""64-wide causal decoder with initial-only candidates and predicted occupancy."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transfer import transfer_initial_features


@dataclass
class Condition:
    initial: dict
    coordinates: torch.Tensor
    features: torch.Tensor
    initial_present: torch.Tensor
    geometry: torch.Tensor
    context: torch.Tensor
    report: dict


def candidate_coordinates(initial_coordinates, radius, max_candidates):
    if type(radius)!=int or radius<0 or initial_coordinates.ndim!=2 or initial_coordinates.shape[1]!=3:
        raise ValueError('Invalid initial geometry/radius')
    a=initial_coordinates.detach().cpu().numpy().astype(np.int64)
    offsets=np.stack(np.meshgrid(*([np.arange(-radius,radius+1)]*3),indexing='ij'),-1).reshape(-1,3)
    result=np.unique((a[:,None]+offsets[None]).reshape(-1,3),axis=0)
    if not 0<len(result)<=max_candidates: raise ValueError('Candidate budget exceeded; no truncation')
    return torch.from_numpy(result).to(device=initial_coordinates.device,dtype=initial_coordinates.dtype)


@torch.no_grad()
def make_condition(base, initial, radius, max_candidates):
    candidates=candidate_coordinates(initial['coordinates'],radius,max_candidates)
    features,indices,report=transfer_initial_features(initial['coordinates'],initial['features'],candidates)
    source=initial['coordinates'][indices]
    offset=candidates-source
    present=(offset==0).all(-1)
    geometry=torch.cat((offset.float()/max(radius,1),present[:,None].float()),-1)
    # Frozen encoder sees exactly S0, never a union with future coordinates.
    context=base.encode_initial(initial)[:,indices].detach()
    return Condition(initial,candidates,features,present,geometry,context,report)


def target_labels(condition, states):
    """Only the loss caller receives these labels. They never enter Condition."""
    lookup={tuple(c):i for i,c in enumerate(condition.coordinates.detach().cpu().tolist())}
    labels=[];indices=[];features=[]
    for state in states:
        rows=[lookup.get(tuple(c),-1) for c in state['coordinates'].detach().cpu().tolist()]
        if min(rows)<0: raise ValueError('Target outside initial-only candidate set; no silent omission')
        index=torch.tensor(rows,device=condition.coordinates.device,dtype=torch.long)
        label=condition.features.new_zeros(len(condition.coordinates));label[index]=1
        labels.append(label);indices.append(index);features.append(state['features'].detach())
    return dict(occupancy=torch.stack(labels),indices=indices,features=features)


def select_state(coordinates, features, logits, max_nodes):
    keep=logits>0
    count=int(keep.sum())
    if not 0<count<=max_nodes: raise ValueError('Empty/oversized predicted support; no oracle fallback')
    return dict(coordinates=coordinates[keep],features=features[keep])


def state_loss(prediction, target, std):
    logits,features=prediction['logits'],prediction['features']
    y=target['occupancy']
    values=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    positive=(values*y).sum(1)/y.sum(1).clamp_min(1)
    negative=(values*(1-y)).sum(1)/(1-y).sum(1).clamp_min(1)
    occupancy=.5*(positive+negative).mean()
    feature=torch.stack([((features[h,index]-actual)/std.clamp_min(1e-6)).abs().mean()
                         for h,(index,actual) in enumerate(zip(target['indices'],target['features']))]).mean()
    return occupancy+feature,dict(occupancy=occupancy,feature=feature)


@torch.no_grad()
def support_metrics(logits, target):
    pred=logits>0;actual=target.bool()
    tp=int((pred&actual).sum());fp=int((pred&~actual).sum());fn=int((~pred&actual).sum())
    return dict(predicted_nodes=int(pred.sum()),target_nodes=int(actual.sum()),true_positive=tp,false_positive=fp,
                false_negative=fn,iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


class SupportDecoder(nn.Module):
    """Free codes diagnose capacity; fixed16 is separately trained without Z."""
    def __init__(self,base,mode,code_std=.1):
        super().__init__()
        if mode not in ('free16','fixed16'): raise ValueError(mode)
        self.base,self.mode=base,mode
        self.base.requires_grad_(False)
        for module in (base.decoder_backbone,base.output_norm,base.features,base.quantizer.project_out):
            module.requires_grad_(True)
        self.geometry=nn.Linear(4,64,bias=False)
        nn.init.normal_(self.geometry.weight,std=.01)
        self.occupancy=nn.Linear(64,1)
        nn.init.zeros_(self.occupancy.weight);nn.init.zeros_(self.occupancy.bias)
        self.codes=nn.Parameter(torch.randn(2,9,16)*code_std,requires_grad=mode=='free16')

    def action_codes(self,index):
        return self.codes[index] if self.mode=='free16' else torch.zeros_like(self.codes[index])

    def forward(self,condition,codes):
        if codes.shape!=(9,16) or codes.dtype!=torch.float32: raise ValueError('Expected actual continuous codes [9,16]')
        initial=condition.context+self.geometry(condition.geometry)[None]
        latent=self.base.quantizer.project_out(codes)
        n=len(condition.coordinates)
        sequence=torch.cat((initial,latent[:,None].expand(-1,n,-1)),0)
        active=torch.ones(sequence.shape[:2],device=sequence.device,dtype=torch.bool)
        hidden=self.base._decode_hidden(sequence,condition.coordinates,active,None)[4:]
        normalized=self.base.output_norm(hidden)
        raw=self.base.features(normalized)
        features=condition.features[None]+3*self.base.output_std*torch.tanh(raw)
        prior=torch.where(condition.initial_present,2.,-2.)
        logits=prior[None]+self.occupancy(normalized).squeeze(-1)
        return dict(features=features,logits=logits,raw_residual=raw)
