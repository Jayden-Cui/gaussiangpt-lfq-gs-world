"""Native64 motion codec. AR Z denotes integer IDs, not floating features.

The pinned VideoWorld1 QFormer weights and equations are reused at width64.
The only attention change is an explicit mask for padded endpoint keys.
Spatial coordinates stay separate from continuous64 state features.
"""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from upstream import load_classes
from motion_common import InitialState, make_condition


def masked_attention(module, q, k, v, valid):
    q = module._separate_heads(module.q_proj(q), module.num_heads)
    k = module._separate_heads(module.k_proj(k), module.num_heads)
    v = module._separate_heads(module.v_proj(v), module.num_heads)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if valid is not None:
        assert valid.shape == (k.shape[0], k.shape[2]) and valid.any(dim=1).all()
        logits = logits.masked_fill(~valid[:, None, None, :], float('-inf'))
    out = logits.softmax(-1) @ v
    return module.out_proj(module._recombine_heads(out))


def qformer_sequence(module, keys, positions, queries, valid=None):
    """Same equations as pinned QFormer; B,N,D API and padded-key masking."""
    assert keys.shape == positions.shape and keys.shape[-1] == queries.shape[-1] == 64
    for block in module.layers:
        attn = masked_attention(block.cross_attn_token_to_image, queries, keys + positions, keys, valid)
        queries = block.norm2(queries + attn)
        queries = block.norm3(queries + block.mlp(queries))
    attn = masked_attention(module.final_attn_token_to_image, queries + queries, keys + positions, keys, valid)
    return module.norm_final_attn(queries + attn)


class Position3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', 2 ** torch.arange(6, dtype=torch.float32), persistent=False)
        self.project = nn.Linear(39, 64, bias=False)

    def forward(self, coordinates):
        xyz = coordinates.float() * .032
        phase = xyz[..., None] * self.frequencies * torch.pi
        return self.project(torch.cat([xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1))


class MotionQFormer(nn.Module):
    width = 64
    schema = 'gs-native64-instant-pair-action-v1'

    def __init__(self, num_tokens=4, quantizer_kind='lfq'):
        super().__init__()
        if quantizer_kind != 'lfq':
            raise ValueError('This user-selected version supports LFQ motion IDs only')
        qformer, _ = load_classes()
        self.num_tokens, self.quantizer_kind = num_tokens, quantizer_kind
        self.vocabulary = 65536
        # Shared geometry and state modules precede quantizer creation.
        self.endpoint = nn.Sequential(nn.LayerNorm(195), nn.Linear(195, 64), nn.SiLU(), nn.Linear(64, 64))
        self.position = Position3D()
        self.posterior = qformer.QFormer(2, 64, 4, 256, attention_downsample_rate=1)
        self.initial_query = nn.Sequential(nn.LayerNorm(67), nn.Linear(67, 64), nn.SiLU(), nn.Linear(64, 64))
        self.restorer = qformer.QFormer(2, 64, 4, 256, attention_downsample_rate=1)
        self.delta = nn.Linear(64, 64)
        self.occupancy = nn.Linear(64, 1)
        nn.init.normal_(self.delta.weight, std=1e-4); nn.init.zeros_(self.delta.bias)
        nn.init.normal_(self.occupancy.weight, std=1e-4); nn.init.zeros_(self.occupancy.bias)
        self.queries = nn.Parameter(torch.randn(num_tokens, 64) * .02)
        self.key_positions = nn.Parameter(torch.randn(num_tokens, 64) * .02)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261046)
            from vector_quantize_pytorch import LFQ
            self.quantizer = LFQ(dim=64, codebook_size=65536, num_codebooks=1,
                entropy_loss_weight=.1, commitment_loss_weight=0., diversity_gamma=1.,
                soft_clamp_input_value=10., projection_has_bias=False,
                frac_per_sample_entropy=1., experimental_softplus_entropy_loss=True,
                entropy_loss_offset=5., channel_first=False)

    @staticmethod
    def endpoint_context(start, end, dt):
        sc, tc = start['coordinates'], end['coordinates']
        assert start['features'].shape == (len(sc), 64)
        assert end['features'].shape == (len(tc), 64)
        union, inverse = torch.unique(torch.cat([sc, tc]), dim=0, sorted=True, return_inverse=True)
        s = start['features'].new_zeros((len(union), 64)); t = torch.zeros_like(s)
        occupied = s.new_zeros((len(union), 2))
        s[inverse[:len(sc)]] = start['features']; t[inverse[len(sc):]] = end['features']
        occupied[inverse[:len(sc)], 0] = 1; occupied[inverse[len(sc):], 1] = 1
        context = torch.cat([s, t, t-s, occupied, s.new_full((len(s), 1), float(dt)/.2)], -1)
        return context, union

    def posterior_padded(self, contexts, coordinates, valid):
        keys = self.endpoint(contexts)
        positions = self.position(coordinates)
        query = self.queries[None].expand(len(contexts), -1, -1)
        return qformer_sequence(self.posterior, keys, positions, query, valid)

    def encode_motion_batch(self, starts, ends, dts):
        assert 0 < len(starts) == len(ends) == len(dts)
        items = [self.endpoint_context(s, t, dt) for s, t, dt in zip(starts, ends, dts)]
        contexts = pad_sequence([v[0] for v in items], batch_first=True)
        coordinates = pad_sequence([v[1] for v in items], batch_first=True)
        lengths = torch.tensor([len(v[0]) for v in items], device=contexts.device)
        valid = torch.arange(contexts.shape[1], device=contexts.device)[None] < lengths[:, None]
        z = self.posterior_padded(contexts, coordinates, valid)
        with torch.autocast(device_type=z.device.type, enabled=False):
            result, breakdown = self.quantizer(z.float(), inv_temperature=100., return_loss_breakdown=True)
            quantized, indices, aux = result
            entropy = breakdown.per_sample_entropy
            codebook_entropy = breakdown.batch_entropy
        assert z.shape == (len(starts), self.num_tokens, 64)
        assert indices.shape == (len(starts), self.num_tokens)
        return {'z_prequant': z, 'z_quantized': quantized, 'indices': indices,
            'aux_loss': aux, 'per_sample_entropy': entropy, 'codebook_entropy': codebook_entropy}

    def encode_motion(self, start, end, dt):
        return self.encode_motion_batch([start], [end], [dt])

    def restore_batch(self, quantized, conditions):
        assert quantized.shape == (len(conditions), self.num_tokens, 64)
        queries = []
        for c in conditions:
            n = len(c['coordinates'])
            local = torch.cat([c['features'], c['initial_present'][:, None].float(),
                c['distance'][:, None]/6, c['features'].new_full((n, 1), c['dt']/.2)], -1)
            queries.append(self.initial_query(local) + self.position(c['coordinates']))
        padded = pad_sequence(queries, batch_first=True)
        key_pe = self.key_positions[None].expand(len(conditions), -1, -1)
        restored = qformer_sequence(self.restorer, quantized, key_pe, padded)
        # There is no query self-attention: padded queries cannot influence valid ones.
        output = []
        for b, c in enumerate(conditions):
            h = restored[b, :len(c['coordinates'])]
            features = c['base_features'] + self.delta(h)
            prior = torch.where(c['initial_present'], 4., -4.)
            output.append({'coordinates': c['coordinates'], 'features': features,
                'occupancy_logits': prior + self.occupancy(h).squeeze(-1)})
        return output

    def restore(self, quantized, condition):
        return self.restore_batch(quantized, [condition])[0]

    def decode_ids(self, indices, condition):
        if indices.shape != (1, self.num_tokens) or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('Expected one integer action ID per query')
        if not bool(((indices >= 0) & (indices < self.vocabulary)).all()):
            raise ValueError('Action ID outside bound motion vocabulary')
        with torch.autocast(device_type=indices.device.type, enabled=False):
            quantized = self.quantizer.indices_to_codes(indices, project_out=True)
        return self.restore(quantized, condition)

    @torch.no_grad()
    def packet(self, initial, indices, *, binding, metadata=None):
        initial.validate()
        return {'schema': self.schema, 'binding': binding, 'quantizer': self.quantizer_kind,
            'query_count': self.num_tokens, 'feature_width': 64,
            'initial_coordinates': initial.coordinates.detach().cpu().clone(),
            'initial_features': initial.features.detach().cpu().clone(),
            'motion_indices': indices.detach().cpu().clone(), 'dt': initial.dt, 'metadata': metadata or {}}

    def replay(self, packet, codec, *, binding, device='cuda'):
        expected = {'schema','binding','quantizer','query_count','feature_width',
            'initial_coordinates','initial_features','motion_indices','dt','metadata'}
        if (set(packet) != expected or packet['schema'] != self.schema or packet['binding'] != binding
                or packet['quantizer'] != self.quantizer_kind or packet['query_count'] != self.num_tokens
                or packet['feature_width'] != 64):
            raise ValueError('Packet binding/schema mismatch; future-side fields are forbidden')
        initial = InitialState(packet['initial_coordinates'].to(device), packet['initial_features'].to(device), float(packet['dt']))
        return self.decode_ids(packet['motion_indices'].to(device), make_condition(codec, initial))


def build_model(config):
    return MotionQFormer(config['motion_tokens'], config['quantizer'])


def select_support(prediction, target_indices=None, force_target=False, max_tokens=6000):
    keep = prediction['occupancy_logits'] > 0
    predicted_count = int(keep.sum())
    if force_target:
        assert target_indices is not None
        keep = keep.clone(); keep[target_indices] = True
    count = int(keep.sum())
    if not 0 < count <= max_tokens:
        raise RuntimeError(f'Support count {count} outside [1,{max_tokens}]; predicted={predicted_count}; no truncation')
    return prediction['coordinates'][keep], prediction['features'][keep], keep
