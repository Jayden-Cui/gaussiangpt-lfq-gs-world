"""Initial-state anchored, native-64 GS motion adapter for bounded experiments.

The default four blocks per side are a deliberately smaller GS adaptation,
not the released twenty-block VideoWorld architecture. ``blocks=20`` retains
the pinned v5 multiscale block schedule. All initial coordinate rows survive;
there is no radius expansion, motion occupancy head, or motion support mask.
The downstream native GaussianGPT decoder may retain its own predicted pruning.

Ten latent outputs use F0 + TRAIN_std * residual_scale * tanh(head(hidden)).
The residual head starts at zero. Consequently the first reconstruction
backward updates that head, but its gradient to upstream motion modules is
zero until the head changes. This is expected, not a missing-gradient failure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from aligned_ldm import AlignedLDM, EncodedInitial, InitialCondition, validate_state
from model_core import qformer_sequence
from multiscale64 import MultiScaleBackbone64


def make_condition(initial):
    """Identity condition on every initial row, with no nearest-neighbor search."""
    validate_state(initial)
    coordinates = initial['coordinates']
    neighbors = torch.arange(len(coordinates), device=coordinates.device)[:, None]
    weights = initial['features'].new_ones((len(coordinates), 1))
    return InitialCondition(initial, coordinates, neighbors, weights)


class AnchoredBackbone64(MultiScaleBackbone64):
    """Use one block at each of four scales, or the full twenty-block schedule."""

    def __init__(self, mode, *, blocks=4, checkpoint_blocks=True):
        if type(blocks) is not int or blocks not in (4, 20):
            raise ValueError('blocks must be 4 or 20 per side')
        super().__init__(mode, checkpoint_blocks=checkpoint_blocks)
        if blocks == 4:
            # Stage-ending decoder blocks retain their original conditional
            # normalization. Discarded modules are not registered/optimized.
            selected = [self.blocks[index - 1] for index in self.stage_ends]
            self.blocks = nn.ModuleList(selected)
            self.scales = tuple(block.distance for block in selected)
            self.stage_ends = (1, 2, 3, 4)


class AnchoredLDM(AlignedLDM):
    """Nine independent posteriors and one LFQ, with initial-only GS anchoring.

    ``code_mode='constant'`` is the trained initial+h control. It never executes
    a future posterior or its entropy auxiliary objective. A single valid ID
    is broadcast across samples/horizons; only the shared inverse projection
    remains trainable on the quantizer. The decoder's time-slot structure can
    still distinguish horizons. This is not a zero-vector/OOD token ablation.

    The public encode/decode interfaces follow v5. Constant mode returns
    ``z_prequant=None`` and a detached scalar-zero ``aux_loss``. Predictions
    contain only ``coordinates`` and ``features``; no occupancy head exists.
    """

    def __init__(self, *, coordinate_scale=.032, blocks=4,
                 checkpoint_blocks=True, residual_scale=3.,
                 code_mode='posterior', constant_id=0, inv_temperature=1.):
        if code_mode not in ('posterior', 'constant'):
            raise ValueError('code_mode must be posterior or constant')
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('residual_scale must be finite and positive')
        if not math.isfinite(inv_temperature) or inv_temperature <= 0:
            raise ValueError('inv_temperature must be finite and positive')
        if type(constant_id) is not int or not 0 <= constant_id < 65536:
            raise ValueError('constant_id must be an integer in [0,65536)')
        if type(blocks) is not int or blocks not in (4, 20):
            raise ValueError('blocks must be 4 or 20 per side')
        super().__init__(coordinate_scale=coordinate_scale,
                         decoder_mode='window', blocks=1)
        self.code_mode = code_mode
        self.constant_id = constant_id
        self.residual_scale = float(residual_scale)
        self.inv_temperature = float(inv_temperature)
        self.blocks = blocks
        self.encoder_space = nn.ModuleList()
        self.encoder_time = nn.ModuleList()
        self.decoder_space = nn.ModuleList()
        self.decoder_time = nn.ModuleList()
        self.decoder_condition = None
        del self.occupancy
        self.encoder_backbone = AnchoredBackbone64(
            'encoder', blocks=blocks, checkpoint_blocks=checkpoint_blocks)
        self.decoder_backbone = AnchoredBackbone64(
            'decoder', blocks=blocks, checkpoint_blocks=checkpoint_blocks)
        self.output_norm = nn.LayerNorm(64)
        self.register_buffer('output_std', torch.ones(64))
        self.register_buffer('output_calibrated', torch.tensor(False))
        nn.init.zeros_(self.features.weight)
        nn.init.zeros_(self.features.bias)
        if code_mode == 'constant':
            self.posteriors.requires_grad_(False)
            self.queries.requires_grad_(False)
            self.quantizer.requires_grad_(False)
            self.quantizer.project_out.requires_grad_(True)

    @torch.no_grad()
    def initialize_output_units(self, train_reference_features):
        """Bind population std once from admitted TRAIN features, never targets.

        The caller records the exact calibration source in its protocol. A
        constant channel has std=0 and therefore no residual freedom in that
        channel; it is not silently assigned a different scale.
        """
        if bool(self.output_calibrated):
            raise RuntimeError('TRAIN output units are already bound')
        f = train_reference_features.detach().float()
        if (f.ndim != 2 or f.shape[1] != 64 or len(f) < 2
                or not bool(torch.isfinite(f).all())):
            raise ValueError('Expected finite TRAIN reference features [M,64]')
        std = f.std(0, unbiased=False)
        if not bool(torch.isfinite(std).all()) or not bool((std > 0).any()):
            raise ValueError('TRAIN reference has no usable finite channel scale')
        self.output_std.copy_(std)
        self.output_calibrated.fill_(True)

    def _encode_hidden(self, hidden, coordinates, active, neighbors):
        return self.encoder_backbone(hidden, coordinates, active)

    def _decode_hidden(self, sequence, coordinates, active, neighbors):
        return self.decoder_backbone(sequence, coordinates, active, condition=sequence)

    def encoder_versions(self):
        return tuple(parameter._version
                     for module in (self.input, self.encoder_backbone)
                     for parameter in module.parameters())

    def encode_windows(self, windows):
        if not windows or any(len(window) != 10 for window in windows):
            raise ValueError('Training requires complete ten-frame windows')
        if self.code_mode == 'posterior':
            # Same prefix/QFormer/shared-LFQ equations as AlignedLDM, with an
            # explicit auxiliary-entropy temperature instead of hardcoded 100.
            # Temperature does not change LFQ's sign-based forward hard codes.
            prepared = [self.prepare(window) for window in windows]
            vectors = []
            for h in range(1, 10):
                keys = [p['hidden'][:h + 1][p['present'][:h + 1]] for p in prepared]
                positions = [self.position(p['coordinates'][None].expand(h + 1, -1, -1)
                                           [p['present'][:h + 1]]) for p in prepared]
                keys_padded = pad_sequence(keys, batch_first=True)
                positions_padded = pad_sequence(positions, batch_first=True)
                lengths = torch.tensor([len(value) for value in keys], device=keys_padded.device)
                valid = torch.arange(keys_padded.shape[1], device=keys_padded.device)[None] < lengths[:, None]
                query = self.queries.weight[h - 1:h][None].expand(len(windows), -1, -1)
                vectors.append(qformer_sequence(self.posteriors[h - 1], keys_padded,
                                                positions_padded, query, valid))
            z = torch.cat(vectors, dim=1)
            with torch.autocast(device_type=z.device.type, enabled=False):
                quantized = [self.quantizer(z[:, h:h + 1].float(),
                                            inv_temperature=self.inv_temperature)
                             for h in range(9)]
            contexts = [EncodedInitial(
                p['initial'], w[0]['coordinates'], w[0]['features'],
                (w[0]['coordinates']._version, w[0]['features']._version),
                float(w[0]['pts']), id(self), self.encoder_versions())
                for p, w in zip(prepared, windows)]
            return dict(z_prequant=z,
                        z_quantized=torch.cat([value[0] for value in quantized], dim=1),
                        indices=torch.cat([value[1] for value in quantized], dim=1),
                        aux_loss=torch.stack([value[2] for value in quantized]).mean(),
                        initial_contexts=contexts, posterior_used=True)
        # Only initial states are read here. In particular, no tensor from
        # window[1:] can affect decoder conditioning in the constant control.
        initials = [window[0] for window in windows]
        contexts = []
        for state in initials:
            values = self.encode_initial(state)
            contexts.append(EncodedInitial(
                values, state['coordinates'], state['features'],
                (state['coordinates']._version, state['features']._version),
                float(state['pts']), id(self), self.encoder_versions()))
        ids = torch.full((len(windows), 9), self.constant_id,
                         device=initials[0]['features'].device, dtype=torch.long)
        codes = self.quantizer.indices_to_codes(ids, project_out=True)
        return dict(z_prequant=None, z_quantized=codes, indices=ids,
                    aux_loss=codes.new_zeros(()), initial_contexts=contexts,
                    posterior_used=False)

    @staticmethod
    def _validate_identity_condition(condition):
        state = condition.state
        validate_state(state)
        coordinates = state['coordinates']
        if (condition.coordinates.shape != coordinates.shape
                or not torch.equal(condition.coordinates, coordinates)):
            raise ValueError('Condition must preserve every initial coordinate in order')
        identity = torch.arange(len(coordinates), device=coordinates.device)[:, None]
        if not torch.equal(condition.neighbors, identity):
            raise ValueError('Anchored condition requires identity neighbors')
        if (condition.weights.shape != (len(coordinates), 1)
                or not torch.equal(condition.weights, torch.ones_like(condition.weights))):
            raise ValueError('Anchored condition requires identity weights')

    def decode_latents(self, quantized, conditions, *, initial_contexts=None):
        if not bool(self.output_calibrated):
            raise RuntimeError('Bind fixed TRAIN output units before decoding')
        if quantized.shape != (len(conditions), 9, 64):
            raise ValueError('Expected quantized [B,9,64]')
        if initial_contexts is not None and len(initial_contexts) != len(conditions):
            raise ValueError('Context batch mismatch')
        outputs = []
        for b, condition in enumerate(conditions):
            self._validate_identity_condition(condition)
            if initial_contexts is None:
                initial = self.encode_initial(condition.state)
            else:
                initial_contexts[b].validate(condition.state, self)
                initial = initial_contexts[b].values
            count = len(condition.coordinates)
            if initial.shape != (4, count, 64):
                raise ValueError('Initial context must contain four slots on all initial rows')
            sequence = torch.cat([
                initial, quantized[b, :, None].expand(-1, count, -1)], dim=0)
            active = torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.bool)
            hidden = self._decode_hidden(sequence, condition.coordinates, active, None)[3:]
            residual = torch.tanh(self.features(self.output_norm(hidden)))
            features = (condition.state['features'][None]
                        + self.output_std * self.residual_scale * residual)
            outputs.append([dict(coordinates=condition.coordinates, features=features[h])
                            for h in range(10)])
        return outputs

    @property
    def last_layer(self):
        # This is the latent residual head, not the final GS attribute head.
        return self.features.weight


__all__ = ['AnchoredLDM', 'AnchoredBackbone64', 'make_condition']
