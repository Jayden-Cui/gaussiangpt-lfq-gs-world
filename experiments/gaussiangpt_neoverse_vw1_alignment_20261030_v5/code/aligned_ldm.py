"""Development GS-LDM: full windows, shared initial encoder, spatial mixing.

No training entry point is provided yet. The native 64 interface and LFQ remain;
this is not a claim of complete VideoWorld reproduction. Sparse spatial message
passing replaces dense image convolutions; original spatial AE integration and
the final optimizer/render objective remain separate work.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from upstream import load_classes
from spatial64 import SparseSpatialBlock64, build_neighbor_index
from model_core import Position3D,TemporalResidual64,AdaptiveWindowNorm64,qformer_sequence

def validate_state(state):
    c,f=state['coordinates'],state['features']
    if c.ndim!=2 or c.shape[1]!=3 or c.dtype not in (torch.int32,torch.int64):
        raise ValueError('Expected integer world coordinates [M,3]')
    if len(c)==0 or f.shape!=(len(c),64) or not f.is_floating_point() or not bool(torch.isfinite(f).all()):
        raise ValueError('Expected nonempty finite features [M,64]')
    if c.device!=f.device or not math.isfinite(float(state['pts'])):raise ValueError('Invalid state device or time')
    if len(torch.unique(c,dim=0))!=len(c):raise ValueError('Duplicate state coordinates')

@dataclass
class InitialCondition:
    """All geometry is computed from the provided initial state alone."""
    state:dict
    coordinates:torch.Tensor
    neighbors:torch.Tensor
    weights:torch.Tensor

@dataclass
class EncodedInitial:
    values:torch.Tensor
    coordinates:torch.Tensor
    features:torch.Tensor
    versions:tuple
    pts:float
    model_identity:int
    encoder_versions:tuple

    def validate(self,state,model):
        if (self.coordinates is not state['coordinates'] or self.features is not state['features']
                or self.versions!=(self.coordinates._version,self.features._version)
                or self.pts!=float(state['pts']) or self.model_identity!=id(model)
                or self.encoder_versions!=model.encoder_versions()):
            raise ValueError('Initial context belongs to a different or modified state/model')

def make_condition(initial,radius=6,max_candidates=100000):
    validate_state(initial)
    if not isinstance(radius,int) or radius<0:raise ValueError('Invalid radius')
    device=initial['features'].device
    c=initial['coordinates']
    offsets=torch.cartesian_prod(*[torch.arange(-radius,radius+1,device=c.device)]*3)
    candidates=c.new_empty((0,3))
    for start in range(0,len(c),64):
        expanded=(c[start:start+64,None]+offsets[None]).reshape(-1,3)
        candidates=torch.unique(torch.cat([candidates,expanded]),dim=0,sorted=True)
        if len(candidates)>max_candidates:raise ValueError('Candidate budget exceeded; no truncation')
    # Geometry only; no future coordinates/features, no learned-state detach.
    with torch.no_grad():
        ids=[]; weights=[]
        for start in range(0,len(candidates),512):
            distances=torch.cdist(candidates[start:start+512].float(),c.float())
            d,i=distances.topk(min(4,len(c)),largest=False,dim=1)
            w=d.clamp_min(1e-6).square().reciprocal()
            exact=d[:,0]==0
            w[exact]=0;w[exact,0]=1
            w=w/w.sum(1,keepdim=True)
            ids.append(i);weights.append(w)
    return InitialCondition(initial,candidates.to(device),torch.cat(ids).to(device),torch.cat(weights).to(device))

class AlignedLDM(nn.Module):
    width=64;horizon=9;vocabulary=65536;time_padding=3
    def __init__(self,*,coordinate_scale=.032,decoder_mode='window',blocks=2):
        super().__init__()
        if decoder_mode not in ('window','prefix'):raise ValueError('Unknown decoder mode')
        if blocks<1:raise ValueError('Empty backbone')
        self.decoder_mode=decoder_mode;self.blocks=blocks
        self.input=nn.Linear(65,64)
        self.encoder_space=nn.ModuleList([SparseSpatialBlock64() for _ in range(blocks)])
        self.encoder_time=nn.ModuleList([TemporalResidual64(1 if i%2==0 else 2) for i in range(blocks)])
        self.position=Position3D(coordinate_scale)
        qformer,_=load_classes()
        self.posteriors=nn.ModuleList([qformer.QFormer(2,64,4,64,attention_downsample_rate=2) for _ in range(9)])
        self.queries=nn.Embedding(9,64)
        self.decoder_space=nn.ModuleList([SparseSpatialBlock64() for _ in range(blocks)])
        self.decoder_time=nn.ModuleList([TemporalResidual64(1 if i%2==0 else 2,across_time=decoder_mode=='window') for i in range(blocks)])
        self.decoder_condition=nn.ModuleList([AdaptiveWindowNorm64() for _ in range(blocks)]) if decoder_mode=='window' else None
        self.features=nn.Linear(64,64)
        self.occupancy=nn.Linear(64,1)
        from vector_quantize_pytorch import LFQ
        self.quantizer=LFQ(dim=64,codebook_size=65536,num_codebooks=1,
            entropy_loss_weight=.1,commitment_loss_weight=0.,diversity_gamma=1.,
            soft_clamp_input_value=10.,projection_has_bias=False,frac_per_sample_entropy=1.,
            experimental_softplus_entropy_loss=True,entropy_loss_offset=5.,channel_first=False)

    def prepare(self,states):
        if not 1<=len(states)<=10:raise ValueError('Expected 1..10 real states')
        for state in states:validate_state(state)
        pts=[float(s['pts']) for s in states]
        if any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Non-increasing nominal times')
        coordinates,inverse=torch.unique(torch.cat([s['coordinates'] for s in states]),dim=0,sorted=True,return_inverse=True)
        n=len(coordinates);t=len(states)
        dense=states[0]['features'].new_zeros((t,n,64))
        present=torch.zeros((t,n),device=dense.device,dtype=torch.bool)
        offset=0;first_indices=None
        for j,state in enumerate(states):
            indices=inverse[offset:offset+len(state['coordinates'])];offset+=len(indices)
            dense[j,indices]=state['features'];present[j,indices]=True
            if j==0:first_indices=indices
        # Three zero-input frames exist only on the known initial spatial domain.
        pad_features=dense.new_zeros((3,n,64));pad_occ=present.new_zeros((3,n))
        x=torch.cat([torch.cat([pad_features,dense],0),
            torch.cat([pad_occ,present],0)[...,None].to(dense.dtype)],-1)
        active=torch.cat([present[0:1].expand(3,-1),present],0)
        hidden=self.input(x)*active[...,None]
        neighbors=build_neighbor_index(coordinates)
        hidden=self._encode_hidden(hidden,coordinates,active,neighbors)
        initial=hidden[:4,first_indices]
        return dict(hidden=hidden[3:],present=present,coordinates=coordinates,
            initial=initial,initial_coordinates=states[0]['coordinates'])

    def encode_initial(self,initial):
        return self.prepare([initial])['initial']

    def _encode_hidden(self,hidden,coordinates,active,neighbors):
        for spatial,temporal in zip(self.encoder_space,self.encoder_time):
            hidden=spatial(hidden,coordinates,active,neighbor_index=neighbors)
            hidden=temporal(hidden,active)
        return hidden

    def _decode_hidden(self,sequence,coordinates,active,neighbors):
        x=sequence
        for i,(spatial,temporal) in enumerate(zip(self.decoder_space,self.decoder_time)):
            if self.decoder_condition is not None:x=self.decoder_condition[i](x,sequence,active)
            x=spatial(x,coordinates,active,neighbor_index=neighbors)
            x=temporal(x,active)
        return x

    def _project_features(self,x):
        return self.features(x)

    def encoder_versions(self):
        return tuple(p._version for module in (self.input,self.encoder_space,self.encoder_time) for p in module.parameters())

    def encode_windows(self,windows):
        if not windows or any(len(w)!=10 for w in windows):raise ValueError('Training requires complete ten-frame windows')
        prepared=[self.prepare(w) for w in windows]
        vectors=[]
        for h in range(1,10):
            keys=[p['hidden'][:h+1][p['present'][:h+1]] for p in prepared]
            positions=[self.position(p['coordinates'][None].expand(h+1,-1,-1)[p['present'][:h+1]]) for p in prepared]
            k=pad_sequence(keys,batch_first=True);pos=pad_sequence(positions,batch_first=True)
            lengths=torch.tensor([len(v) for v in keys],device=k.device)
            valid=torch.arange(k.shape[1],device=k.device)[None]<lengths[:,None]
            query=self.queries.weight[h-1:h][None].expand(len(windows),-1,-1)
            vectors.append(qformer_sequence(self.posteriors[h-1],k,pos,query,valid))
        z=torch.cat(vectors,1)
        outputs=[]
        with torch.autocast(device_type=z.device.type,enabled=False):
            for h in range(9):
                outputs.append(self.quantizer(z[:,h:h+1].float(),inv_temperature=100.))
        return dict(z_prequant=z,z_quantized=torch.cat([o[0] for o in outputs],1),
            indices=torch.cat([o[1] for o in outputs],1),aux_loss=torch.stack([o[2] for o in outputs]).mean(),
            initial_contexts=[EncodedInitial(p['initial'],w[0]['coordinates'],w[0]['features'],
                (w[0]['coordinates']._version,w[0]['features']._version),float(w[0]['pts']),id(self),self.encoder_versions())
                for p,w in zip(prepared,windows)])

    def decode_latents(self,quantized,conditions,*,initial_contexts=None):
        if quantized.shape!=(len(conditions),9,64):raise ValueError('Expected quantized [B,9,64]')
        if initial_contexts is not None and len(initial_contexts)!=len(conditions):raise ValueError('Context batch mismatch')
        outputs=[]
        for b,condition in enumerate(conditions):
            if initial_contexts is None:initial=self.encode_initial(condition.state)
            else:
                initial_contexts[b].validate(condition.state,self)
                initial=initial_contexts[b].values
            if initial.shape!=(4,len(condition.state['coordinates']),64):raise ValueError('Initial context shape mismatch')
            context=(initial[:,condition.neighbors]*condition.weights[None,:,:,None]).sum(2)
            sequence=torch.cat([context,quantized[b,:,None].expand(-1,len(condition.coordinates),-1)],0)
            active=torch.ones(sequence.shape[:2],device=sequence.device,dtype=torch.bool)
            neighbors=build_neighbor_index(condition.coordinates)
            x=self._decode_hidden(sequence,condition.coordinates,active,neighbors)
            x=x[3:] # Learned initial output plus all nine learned futures.
            f=self._project_features(x);o=self.occupancy(x).squeeze(-1)
            outputs.append([dict(coordinates=condition.coordinates,features=f[h],occupancy_logits=o[h]) for h in range(10)])
        return outputs

    def decode_ids(self,ids,conditions):
        if ids.shape!=(len(conditions),9) or ids.dtype not in (torch.int32,torch.int64):
            raise ValueError('Expected integer IDs [B,9]')
        if not bool(((ids>=0)&(ids<65536)).all()):raise ValueError('Invalid LFQ ID')
        return self.decode_latents(self.quantizer.indices_to_codes(ids,project_out=True),conditions)

