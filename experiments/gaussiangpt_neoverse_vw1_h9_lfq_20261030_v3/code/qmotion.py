"""VideoWorld1 H9 semantics with continuous GaussianGPT64 and LFQ actions.

One independent QFormer and one query for each future horizon. Only the
initial state and quantized codes enter the causal temporal sequence decoder.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from causal_encoder import TemporalBlock, RelativeTime
from native_core import Position3D, qformer_sequence, select_support
from upstream import load_classes
from motion_common import InitialState, make_condition


class WindowLDM(nn.Module):
    schema = 'gs-vw1-h9-native64-lfq-v3'
    vocabulary = 65536
    width = 64

    def __init__(self, horizon=9, candidate_chunk=2048):
        super().__init__()
        assert horizon == 9
        self.horizon = horizon
        self.candidate_chunk = candidate_chunk
        qformer, _ = load_classes()
        self.temporal_input = nn.Sequential(nn.LayerNorm(65), nn.Linear(65,64))
        self.temporal_cnn = nn.ModuleList([TemporalBlock(1), TemporalBlock(2)])
        self.position = Position3D()
        self.time_position = RelativeTime()
        # CALVIN QFormerMFSep: separate parameters for all nine horizons.
        # Width and MLP shrink together, with reference downsample_rate=2.
        self.posteriors = nn.ModuleList([
            qformer.QFormer(2,64,4,64,attention_downsample_rate=2) for _ in range(horizon)])
        self.queries = nn.Parameter(torch.randn(horizon,1,64)*.02)
        self.initial_query = nn.Sequential(nn.LayerNorm(67),nn.Linear(67,64),nn.SiLU(),nn.Linear(64,64))
        self.decoder_cnn = nn.ModuleList([TemporalBlock(1),TemporalBlock(2)])
        self.delta = nn.Linear(64,64)
        self.occupancy = nn.Linear(64,1)
        # A source residual is permitted, but avoid the old 1e-4 output weights
        # suppressing gradients to the new action pathway at initialization.
        nn.init.normal_(self.delta.weight,std=.01); nn.init.zeros_(self.delta.bias)
        nn.init.normal_(self.occupancy.weight,std=.01); nn.init.zeros_(self.occupancy.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261046)
            from vector_quantize_pytorch import LFQ
            self.quantizer=LFQ(dim=64,codebook_size=self.vocabulary,num_codebooks=1,
                entropy_loss_weight=.1,commitment_loss_weight=0.,diversity_gamma=1.,
                soft_clamp_input_value=10.,projection_has_bias=False,frac_per_sample_entropy=1.,
                experimental_softplus_entropy_loss=True,entropy_loss_offset=5.,channel_first=False)

    def prepare_window(self, states):
        assert 2 <= len(states) <= self.horizon+1
        sizes=[len(s['coordinates']) for s in states]
        union,inverse=torch.unique(torch.cat([s['coordinates'] for s in states]),dim=0,sorted=True,return_inverse=True)
        dense=states[0]['features'].new_zeros((len(states),len(union),64))
        present=torch.zeros(dense.shape[:2],device=dense.device,dtype=torch.bool)
        offset=0
        for j,(s,n) in enumerate(zip(states,sizes)):
            assert s['features'].shape==(n,64)
            index=inverse[offset:offset+n];offset+=n
            dense[j,index]=s['features'];present[j,index]=True
        seen=present.cumsum(0)>0
        pts=torch.tensor([float(s['pts']) for s in states],device=dense.device,dtype=torch.float64)
        assert bool((pts[1:]>pts[:-1]).all())
        hidden=self.temporal_input(torch.cat([dense,present[...,None].float()],-1))*seen[...,None]
        for block in self.temporal_cnn: hidden=block(hidden,seen)
        return dict(hidden=hidden,coordinates=union,seen=seen,pts=pts)

    def prefix_tokens(self, prepared, h):
        assert 1 <= h < len(prepared['pts'])
        valid=prepared['seen'][:h+1]
        keys=prepared['hidden'][:h+1][valid]
        coords=prepared['coordinates'][None].expand(h+1,-1,-1)[valid]
        ages=(prepared['pts'][:h+1]-prepared['pts'][h]).float()
        ages=ages[:,None].expand(-1,valid.shape[1])[valid]
        return keys,self.position(coords)+self.time_position(ages)

    def encode_windows(self, windows, horizons=None):
        """h is relative to the SAME window initial state, not a sliding pair."""
        hs=list(range(1,self.horizon+1)) if horizons is None else list(horizons)
        assert windows and all(1<=h<=self.horizon for h in hs)
        prepared=[self.prepare_window(w) for w in windows]
        features=[]
        for h in hs:
            tokens=[self.prefix_tokens(p,h) for p in prepared]
            keys=pad_sequence([v[0] for v in tokens],batch_first=True)
            positions=pad_sequence([v[1] for v in tokens],batch_first=True)
            lengths=torch.tensor([len(v[0]) for v in tokens],device=keys.device)
            valid=torch.arange(keys.shape[1],device=keys.device)[None]<lengths[:,None]
            query=self.queries[h-1:h].expand(len(windows),-1,-1)
            features.append(qformer_sequence(self.posteriors[h-1],keys,positions,query,valid))
        z=torch.cat(features,1)
        with torch.autocast(device_type=z.device.type,enabled=False):
            result,breakdown=self.quantizer(z.float(),inv_temperature=100.,return_loss_breakdown=True)
        quantized,indices,aux=result
        return dict(z_prequant=z,z_quantized=quantized,indices=indices,aux_loss=aux,
            per_sample_entropy=breakdown.per_sample_entropy,codebook_entropy=breakdown.batch_entropy)

    def _decode_chunk(self, quantized, initial_hidden):
        # [time, coordinate,64]. At time0 there is the initial spatial feature;
        # at all future times there is only the corresponding broadcast code.
        sequence=torch.cat([initial_hidden[None],quantized[:,None,:].expand(-1,len(initial_hidden),-1)],0)
        seen=torch.ones(sequence.shape[:2],device=sequence.device,dtype=torch.bool)
        for block in self.decoder_cnn: sequence=block(sequence,seen)
        return sequence[1:]

    def decode_latents(self, quantized, conditions):
        assert quantized.shape==(len(conditions),self.horizon,64)
        result=[]
        for b,c in enumerate(conditions):
            n=len(c['coordinates'])
            local=torch.cat([c['features'],c['initial_present'][:,None].float(),c['distance'][:,None]/6,
                c['features'].new_full((n,1),c['dt']/.2)],-1)
            initial=self.initial_query(local)+self.position(c['coordinates'])
            chunks=[]
            for start in range(0,n,self.candidate_chunk):
                arguments=(quantized[b],initial[start:start+self.candidate_chunk])
                h=checkpoint(self._decode_chunk,*arguments,use_reentrant=False) if self.training and torch.is_grad_enabled() else self._decode_chunk(*arguments)
                chunks.append(h)
            hidden=torch.cat(chunks,1)
            features=c['base_features'][None]+self.delta(hidden)
            prior=torch.where(c['initial_present'],4.,-4.)
            logits=prior[None]+self.occupancy(hidden).squeeze(-1)
            result.append([dict(coordinates=c['coordinates'],features=features[h],occupancy_logits=logits[h]) for h in range(self.horizon)])
        return result

    def decode_ids(self, indices, conditions):
        assert indices.shape==(len(conditions),self.horizon) and indices.dtype in (torch.int32,torch.int64)
        assert bool(((indices>=0)&(indices<self.vocabulary)).all())
        return self.decode_latents(self.quantizer.indices_to_codes(indices,project_out=True),conditions)

    @torch.no_grad()
    def packet(self, initial, indices, *, binding, metadata=None):
        initial.validate()
        assert indices.shape==(1,self.horizon)
        return dict(schema=self.schema,binding=binding,quantizer='lfq',horizon=self.horizon,queries_per_horizon=1,
            feature_width=64,initial_coordinates=initial.coordinates.detach().cpu().clone(),
            initial_features=initial.features.detach().cpu().clone(),motion_indices=indices.detach().cpu().clone(),
            dt=initial.dt,metadata=metadata or {})

    def replay(self, packet, codec, *, binding, device='cuda'):
        allowed={'schema','binding','quantizer','horizon','queries_per_horizon','feature_width','initial_coordinates',
            'initial_features','motion_indices','dt','metadata'}
        if (set(packet)!=allowed or packet['schema']!=self.schema or packet['binding']!=binding
            or packet['quantizer']!='lfq' or packet['horizon']!=9 or packet['queries_per_horizon']!=1 or packet['feature_width']!=64):
            raise ValueError('Sequence packet schema/binding mismatch; future state fields forbidden')
        initial=InitialState(packet['initial_coordinates'].to(device),packet['initial_features'].to(device),float(packet['dt']))
        return self.decode_ids(packet['motion_indices'].to(device),[make_condition(codec,initial)])[0]


def build_model(config):
    assert config['quantizer']=='lfq' and config['motion_tokens']==1
    return WindowLDM(config.get('horizon',9),config.get('candidate_chunk',2048))
