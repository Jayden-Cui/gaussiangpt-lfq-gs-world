"""Coordinate-aligned temporal causal CNN -> prefix QFormer64 -> LFQ.

Conditional causality is proved on GS states. Offline NeoVerse extraction is
outside that boundary. Four IDs describe one adjacent transition, not four times.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from native_core import MotionQFormer as NativeSpatialBridge, qformer_sequence, select_support


class TemporalBlock(nn.Module):
    """No operation mixes different coordinates or normalization time steps."""
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm1 = nn.LayerNorm(64); self.norm2 = nn.LayerNorm(64)
        self.conv1 = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(64, 64, 3, dilation=dilation)

    def convolve(self, x, norm, convolution):
        # x: T,U,64 -> independent coordinate streams U,64,T.
        y = F.silu(norm(x)).permute(1, 2, 0)
        y = convolution(F.pad(y, (2*self.dilation, 0)))
        return y.permute(2, 0, 1)

    def forward(self, x, seen):
        y = self.convolve(x, self.norm1, self.conv1) * seen[..., None]
        y = self.convolve(y, self.norm2, self.conv2)
        return (x+y) * seen[..., None]


class RelativeTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', 2**torch.arange(6,dtype=torch.float32), persistent=False)
        self.project = nn.Linear(13, 64, bias=False)

    def forward(self, seconds):
        age = seconds[..., None] / .2
        phase = age * self.frequencies * torch.pi
        return self.project(torch.cat([age, phase.sin(), phase.cos()], -1))


class CausalMotionQFormer(NativeSpatialBridge):
    schema = 'gs-causal64-prefix-lfq-action-v2'

    def __init__(self, num_tokens=4, quantizer_kind='lfq', max_context_frames=10):
        super().__init__(num_tokens, quantizer_kind)
        self.max_context_frames = max_context_frames
        self.temporal_input = nn.Sequential(nn.LayerNorm(65), nn.Linear(65,64))
        self.temporal_cnn = nn.ModuleList([TemporalBlock(1), TemporalBlock(2)])
        self.time_position = RelativeTime()

    def prepare_window(self, states):
        assert 2 <= len(states) <= self.max_context_frames
        coordinates = [s['coordinates'] for s in states]
        sizes = [len(c) for c in coordinates]
        union, inverse = torch.unique(torch.cat(coordinates), dim=0, sorted=True, return_inverse=True)
        dense = states[0]['features'].new_zeros((len(states), len(union), 64))
        present = torch.zeros((len(states),len(union)), device=dense.device, dtype=torch.bool)
        offset = 0
        for j, (state, size) in enumerate(zip(states, sizes)):
            assert state['features'].shape == (size,64)
            indices = inverse[offset:offset+size]; offset += size
            dense[j, indices] = state['features']; present[j,indices] = True
        seen = present.cumsum(0)>0
        # PTS is used only locally and relative to the requested prefix endpoint.
        pts = torch.tensor([float(s['pts']) for s in states], device=dense.device, dtype=torch.float64)
        assert bool((pts[1:]>pts[:-1]).all())
        hidden = self.temporal_input(torch.cat([dense,present[...,None].float()],-1)) * seen[...,None]
        for block in self.temporal_cnn: hidden = block(hidden,seen)
        previous = torch.cat([torch.zeros_like(dense[:1]),dense[:-1]],0)
        previous_present = torch.cat([torch.zeros_like(present[:1]),present[:-1]],0)
        dt = torch.cat([pts.new_zeros(1),pts[1:]-pts[:-1]]).float()/.2
        contexts = torch.cat([hidden,dense,dense-previous,present[...,None].float(),
            previous_present[...,None].float(),dt[:,None,None].expand(-1,len(union),1)],-1)
        return {'contexts':contexts,'coordinates':union,'seen':seen,'pts':pts,'temporal_features':hidden}

    def prefix_tokens(self, prepared, end):
        assert 1 <= end < len(prepared['pts'])
        # Remove all unseen/future-only columns before attention. Their insertion
        # into the indexing union cannot alter the ordered visible prefix keys.
        visible = prepared['seen'][:end+1]
        contexts = prepared['contexts'][:end+1][visible]
        coordinate_rows = prepared['coordinates'][None].expand(end+1,-1,-1)[visible]
        age = (prepared['pts'][:end+1]-prepared['pts'][end]).float()
        age_rows = age[:,None].expand(-1,visible.shape[1])[visible]
        keys = self.endpoint(contexts)
        positions = self.position(coordinate_rows)+self.time_position(age_rows)
        return keys, positions

    def posterior_keys(self, keys, positions, valid):
        query = self.queries[None].expand(len(keys),-1,-1)
        return qformer_sequence(self.posterior,keys,positions,query,valid)

    def encode_motion_batch(self, windows, prefix_ends=None):
        assert windows
        if prefix_ends is None: prefix_ends = [len(w)-1 for w in windows]
        assert len(prefix_ends)==len(windows)
        prepared = {}; tokens=[]
        for window, end in zip(windows,prefix_ends):
            identity=id(window)
            if identity not in prepared: prepared[identity]=self.prepare_window(window)
            tokens.append(self.prefix_tokens(prepared[identity],end))
        keys=pad_sequence([v[0] for v in tokens],batch_first=True)
        positions=pad_sequence([v[1] for v in tokens],batch_first=True)
        lengths=torch.tensor([len(v[0]) for v in tokens],device=keys.device)
        valid=torch.arange(keys.shape[1],device=keys.device)[None]<lengths[:,None]
        z=self.posterior_keys(keys,positions,valid)
        with torch.autocast(device_type=z.device.type,enabled=False):
            result,breakdown=self.quantizer(z.float(),inv_temperature=100.,return_loss_breakdown=True)
            quantized,indices,aux=result
        assert z.shape==(len(windows),self.num_tokens,64)
        assert indices.shape==(len(windows),self.num_tokens)
        return {'z_prequant':z,'z_quantized':quantized,'indices':indices,'aux_loss':aux,
            'per_sample_entropy':breakdown.per_sample_entropy,'codebook_entropy':breakdown.batch_entropy}

    def encode_prefix(self, window, end=None):
        return self.encode_motion_batch([window],None if end is None else [end])

    def encode_sequence(self, window):
        """T states -> T-1 groups, each K IDs, using a shared prefix encoder."""
        result=self.encode_motion_batch([window]*(len(window)-1),list(range(1,len(window))))
        return {k:v.unsqueeze(0) if k in ('z_prequant','z_quantized','indices') else v for k,v in result.items()}

    def encode_motion(self, *args, **kwargs):
        raise RuntimeError('Use encode_prefix(window); implicit two-frame endpoint API is disabled in causal version')


def build_model(config):
    return CausalMotionQFormer(config['motion_tokens'],config['quantizer'],config.get('max_context_frames',10))
