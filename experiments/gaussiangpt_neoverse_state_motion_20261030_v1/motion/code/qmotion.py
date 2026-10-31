"""Two instantaneous states -> fixed Kx512 motion Z -> hard FSQ -> future state.

The decoder accepts initial continuous state features plus motion IDs; no future support or
feature fields exist in its packet API. Original VideoWorld QFormer and FSQ
implementations are loaded from the hash-verified vendor snapshot.
"""
import torch
from torch import nn
from upstream import load_classes
from motion_common import InitialState, make_condition


class Position3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies',2**torch.arange(6,dtype=torch.float32),persistent=False)
        self.project=nn.Linear(39,512,bias=False)

    def forward(self,coordinates):
        xyz=coordinates.float()*.032
        phase=xyz[:,:,None]*self.frequencies*torch.pi
        return self.project(torch.cat([xyz,phase.sin().flatten(1),phase.cos().flatten(1)],-1))


class MotionQFormer(nn.Module):
    levels=(8,8,8,5,5,5)
    vocabulary=64000

    def __init__(self,num_tokens=1):
        super().__init__()
        qformer,FSQ=load_classes()
        self.num_tokens=num_tokens
        # Modules shared across K=1/K=4 are initialized before any K-dependent tensors.
        self.endpoint=nn.Sequential(nn.LayerNorm(195),nn.Linear(195,512),nn.SiLU(),nn.Linear(512,512))
        self.position=Position3D()
        self.posterior=qformer.QFormer(2,512,4,512)
        self.quantizer=FSQ(levels=list(self.levels),dim=512)
        self.initial_query=nn.Sequential(nn.LayerNorm(67),nn.Linear(67,512),nn.SiLU(),nn.Linear(512,512))
        self.restorer=qformer.QFormer(2,512,4,512)
        self.delta=nn.Linear(512,64)
        self.occupancy=nn.Linear(512,1)
        nn.init.normal_(self.delta.weight,std=1e-4);nn.init.zeros_(self.delta.bias)
        nn.init.normal_(self.occupancy.weight,std=1e-4);nn.init.zeros_(self.occupancy.bias)
        self.queries=nn.Parameter(torch.randn(num_tokens,512)*.02)
        self.key_positions=nn.Parameter(torch.randn(num_tokens,512)*.02)

    def encode_motion(self,start,end,dt):
        """Posterior sees exactly two endpoints, not a temporal block of GS."""
        sc,tc=start['coordinates'],end['coordinates']
        union,inverse=torch.unique(torch.cat([sc,tc],0),dim=0,sorted=True,return_inverse=True)
        s=union.new_zeros((len(union),64),dtype=torch.float32)
        t=torch.zeros_like(s)
        occupied=s.new_zeros((len(union),2))
        s[inverse[:len(sc)]]=start['features']
        t[inverse[len(sc):]]=end['features']
        occupied[inverse[:len(sc)],0]=1
        occupied[inverse[len(sc):],1]=1
        context=torch.cat([s,t,t-s,occupied,s.new_full((len(s),1),float(dt)/.2)],-1)
        keys=self.endpoint(context)
        pe=self.position(union)
        query=self.queries[None]
        z,_=self.posterior(keys.T[None,:,None,None,:],pe.T[None,:,None,None,:],query)
        with torch.autocast(device_type=z.device.type,enabled=False):
            quantized,indices=self.quantizer(z.float())
        assert z.shape==(1,self.num_tokens,512) and indices.shape==(1,self.num_tokens)
        return {'z_prequant':z,'z_quantized':quantized,'indices':indices}

    def restore(self,quantized,condition):
        assert quantized.shape==(1,self.num_tokens,512)
        n=len(condition['coordinates'])
        local=torch.cat([condition['features'],condition['initial_present'][:,None].float(),
                         condition['distance'][:,None]/6,
                         condition['features'].new_full((n,1),condition['dt']/.2)],-1)
        query=self.initial_query(local)+self.position(condition['coordinates'])
        keys=quantized.transpose(1,2)[:,:,None,None,:]
        pe=self.key_positions.T[None,:,None,None,:]
        restored,_=self.restorer(keys,pe,query[None])
        restored=restored[0]
        features=condition['base_features']+self.delta(restored)
        prior=torch.where(condition['initial_present'],4.,-4.)
        logits=prior+self.occupancy(restored).reshape(-1)
        return {'coordinates':condition['coordinates'],'features':features,'occupancy_logits':logits}

    def decode_ids(self,indices,condition):
        if indices.shape!=(1,self.num_tokens) or indices.dtype not in (torch.int32,torch.int64):
            raise ValueError('Expected one integer motion index per fixed query')
        if not bool(((indices>=0)&(indices<self.vocabulary)).all()):
            raise ValueError('Motion index outside VideoWorld FSQ vocabulary')
        with torch.autocast(device_type=indices.device.type,enabled=False):
            quantized=self.quantizer.indices_to_codes(indices,project_out=True)
        return self.restore(quantized,condition)

    @torch.no_grad()
    def packet(self,initial,indices,*,binding,metadata=None):
        initial.validate()
        return {'schema':'gs-continuous-instant-pair-qformer-fsq-v1','binding':binding,
                'initial_coordinates':initial.coordinates.detach().cpu().clone(),
                'initial_features':initial.features.detach().cpu().clone(),
                'motion_indices':indices.detach().cpu().clone(),'dt':initial.dt,
                'metadata':metadata or {}}

    def replay(self,packet,codec,*,binding,device='cuda'):
        expected={'schema','binding','initial_coordinates','initial_features','motion_indices','dt','metadata'}
        if set(packet)!=expected or packet['schema']!='gs-continuous-instant-pair-qformer-fsq-v1' or packet['binding']!=binding:
            raise ValueError('Packet binding or schema mismatch; future-side fields are forbidden')
        initial=InitialState(packet['initial_coordinates'].to(device),packet['initial_features'].to(device),float(packet['dt']))
        condition=make_condition(codec,initial)
        return self.decode_ids(packet['motion_indices'].to(device),condition)


def select_support(prediction,target_indices=None,force_target=False,max_tokens=6000):
    keep=prediction['occupancy_logits']>0
    predicted_count=int(keep.sum())
    if force_target:
        assert target_indices is not None
        keep=keep.clone();keep[target_indices]=True
    count=int(keep.sum())
    if not 0<count<=max_tokens:
        raise RuntimeError(f'Support count {count} outside [1,{max_tokens}]: predicted={predicted_count}, '
                           f'forced_additions={count-predicted_count}, force_target={force_target}; no truncation')
    return prediction['coordinates'][keep],prediction['features'][keep],keep
