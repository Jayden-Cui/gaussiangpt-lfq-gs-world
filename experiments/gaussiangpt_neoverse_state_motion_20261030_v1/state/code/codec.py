"""GS World continuous spatial autoencoder; state LFQ removed."""
import sys
import math
import numpy as np

import torch
from torch import nn

from common import VENDOR, CHECKPOINT, external_dict, prediction_gs

sys.path.insert(0, str(VENDOR))
import MinkowskiEngine as ME
from conf.dataclasses import PCKeys as K
from model.gaussian_vqvae import GaussianVQVAE


class ResidualMultiplier(nn.Module):
    """K=4 extension initialized as small spatial offsets of the pretrained output."""
    def __init__(self, dim=64, count=4):
        super().__init__()
        assert count == 4
        self.count = count
        self.offsets = nn.Parameter(torch.tensor([[-.2, -.2, -.2], [-.2, .2, .2],
                                                 [.2, -.2, .2], [.2, .2, -.2]]))
        self.residual = nn.Linear(dim, dim*count)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, points):
        f = points[K.EMBEDDINGS]
        return {K.BATCH: points[K.BATCH][:, None].expand(-1, self.count).reshape(-1),
                K.COORDS: (points[K.COORDS][:, None]+self.offsets[None]).reshape(-1, 3),
                K.EMBEDDINGS: (f[:, None]+self.residual(f).reshape(-1, self.count, f.shape[-1])).reshape(-1, f.shape[-1])}


class StateCodec(nn.Module):
    def __init__(self, selection, protocol, checkpoint=CHECKPOINT):
        super().__init__()
        official = GaussianVQVAE.load_from_checkpoint(str(checkpoint), map_location='cpu',
                                                       training_config=None, weights_only=False)
        self.ae = official.autoencoder
        # The state LFQ is removed, including its parameters. No quantize call remains.
        self.ae.vq = nn.Identity()
        self.selection = dict(selection)
        self.protocol = protocol
        self.grid = float(selection['grid'])
        self.ae.default_grid_size = self.grid
        self.ae.make_unique = False  # The input is already strictly unique.
        self.ae.decoder.prune_thresh = list(protocol['train']['prune_threshold'])
        self.set_encoder = None
        self.count = int(selection['gaussians_per_voxel'])
        if selection['mode'] == 'set_statistics':
            embedding_dim = sum(v['dimension']*v['dim_embedding'] for v in self.ae.features.values() if v['in_input'])
            self.set_encoder = nn.Sequential(nn.Linear(29, 128), nn.SiLU(), nn.Linear(128, embedding_dim))
            nn.init.zeros_(self.set_encoder[-1].weight)
            nn.init.zeros_(self.set_encoder[-1].bias)
            self.ae.gaussians_per_voxel = self.count
            self.ae.gaussian_multiplier = ResidualMultiplier(count=self.count)
        self._guards = []
        for stage in self.ae.decoder.ups:
            self._guards.append(stage.register_forward_hook(self._candidate_guard))
        self._guards.append(self.ae.decoder.out.register_forward_hook(self._output_guard))

    def _candidate_guard(self, module, inputs, output):
        n = len(output.F)
        if not 0 < n <= self.protocol['max_decoder_candidates']:
            raise RuntimeError(f'Decoder candidate occupancy {n} exceeds budget; no truncation')

    def _output_guard(self, module, inputs, output):
        n = len(output.F)*self.count
        if not 0 < n <= self.protocol['max_decoded_gaussians']:
            raise RuntimeError(f'Decoded GS count {n} exceeds budget or is empty; no truncation')

    def embed_state(self, representatives, set_statistics=None):
        if len(representatives) > self.protocol['max_input_voxels']:
            raise RuntimeError('Input voxel budget exceeded')
        # Quantize ONCE with the same CPU rule used by the cache producer.
        # Repeating round(x/h) on CUDA can change boundary voxels by one cell.
        xyz = torch.from_numpy(np.rint(representatives.detach().cpu().numpy()[:, :3]/self.grid).astype(np.int32)).to(representatives.device)
        voxel_coords = torch.cat([torch.zeros((len(xyz), 1), dtype=torch.int32, device=xyz.device), xyz], 1)
        points = external_dict(representatives)
        points[K.COORD_OFFSET] = representatives[:, :3]/self.grid-xyz.to(representatives.dtype)
        embedded = []
        for feature, params in sorted(self.ae.features.items()):
            if not params['in_input']:
                continue
            values = params['internal_representation'].to_internal_representation(points[feature])
            if values.ndim == 1:
                values = values[:, None]
            embedded.append(self.ae.embeddings[feature](values))
        x = ME.SparseTensor(features=torch.cat(embedded, -1), coordinates=voxel_coords,
                            device=representatives.device,
                            quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        assert len(x.F) == len(representatives), 'Input must be voxel-unique'
        if self.set_encoder is not None:
            assert set_statistics is not None and set_statistics.shape == (len(representatives), 29)
            # ME may reorder rows. Its explicit unique_index is the only valid correspondence.
            order = x.unique_index.long()
            assert torch.equal(x.C, voxel_coords[order])
            f = x.F + self.set_encoder(set_statistics[order])
            x = ME.SparseTensor(f, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        return x, voxel_coords

    def encode_state(self, representatives, set_statistics=None):
        x, _ = self.embed_state(representatives, set_statistics)
        z, _ = self.ae.encode(x)
        stride = int(z.tensor_stride[0])
        assert stride == self.ae.stride and (z.C[:, 1:] % stride == 0).all()
        return {'coordinates': z.C[:, 1:].clone()//stride,
                'features': z.F, 'stride': stride, 'grid': self.grid, 'sparse': z}

    def _world(self, points):
        points[K.COORDS] = (points[K.COORDS]+points[K.COORD_OFFSET])*self.grid
        if self.count > 1:
            # At initialization K copies approximate the pretrained alpha budget.
            points[K.OPACITIES] = points[K.OPACITIES]-math.log(self.count)
        return points

    def _sparse(self, coordinates, features, stride):
        assert int(stride) == self.ae.stride
        assert coordinates.ndim == 2 and coordinates.shape[1] == 3
        xyz = coordinates.clone().to(device=features.device, dtype=torch.int32)*int(stride)
        coord = torch.cat([torch.zeros((len(xyz), 1), dtype=torch.int32, device=features.device), xyz], 1)
        return ME.SparseTensor(features=features, coordinates=coord, device=features.device,
                               tensor_stride=[int(stride)]*3)

    def decode_state(self, coordinates, features, stride=None):
        """Continuous sparse state -> GS; no quantizer, no target occupancy."""
        if self.ae.decoder.training:
            raise RuntimeError('Use eval decoder for state decoding')
        assert features.ndim == 2 and features.shape == (len(coordinates), 64)
        assert torch.isfinite(features).all()
        z = self._sparse(coordinates, features, stride or self.ae.stride)
        points, _ = self.ae.decode(z, target_key=None)
        return self._world(points)

    def forward(self, representatives, set_statistics=None, force_target=True):
        x, voxel_coords = self.embed_state(representatives, set_statistics)
        target_key, _ = x.coordinate_manager.insert_and_map(voxel_coords, string_id='state_target')
        z, _ = self.ae.encode(x)
        # This explicitly separates weight training from occupancy forcing.
        self.ae.decoder.train(bool(force_target))
        points, decoder_info = self.ae.decode(z, target_key=target_key)
        points = self._world(points)
        occupancy = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(
            logits.F.reshape(-1), target.float()) for logits, target in
            zip(decoder_info['out_cls'], decoder_info['targets'])]).mean()
        return points, {'occupancy': occupancy, 'latent_count': len(z.F), 'output_count': len(points[K.COORDS])}


def save_state(path, encoded, metadata=None):
    torch.save({'schema': 'gsworld_continuous_state_v1',
                'coordinates': encoded['coordinates'].detach().cpu(),
                'features': encoded['features'].detach().cpu().float(),
                'stride': encoded['stride'], 'grid': encoded['grid'],
                'feature_dtype': 'float32', 'metadata': metadata or {}}, path)


def load_codec(root, checkpoint=None, device='cuda'):
    from common import read
    model = StateCodec(read(root/'selection.json'), read(root/'protocol.json'))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
    return model.to(device)
