"""Selected baseline native bridge with differentiable float64 decoder path."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from codec_binding import CHECKPOINT_SHA, checked_bytes, require, sha
from teacher_binding import BASE, STATE_ROOT, RENDERER_SHA

CODEC_SHA = 'e5eb7b948a70e9abea5fc0428ba7c37a35a71a17b5a49ed5da0578df3a719a2a'
PROTOCOL_SHA = '24add801f064c181658b43da98e1b9a42779c2b876105231ded2ef044b1205da'
SELECTION_SHA = '6b52157b31c7c12baf645c6f6b8c8c61c0179d13eeeefc5184a16c6432463119'
OFFICIAL_SHA = '5dab10ba82cec3737c152622f477fbc9ae970f7bbd86a644b21dbce6d1392f0e'
PRECISION_SHA = '2b6d4513afed3bc1d5f6d31d4a076a341c49a23ba500a49f257b65319609d7e8'
PRECISION = BASE/'gaussiangpt_neoverse_replay_probe_20261031_v7/precision.py'
OFFICIAL = BASE/'gaussiangpt_state_vae_20261028_v1/checkpoints/vqvae_photoshape_official.ckpt'


def _bound_module(name, path, digest):
    path = Path(path).resolve()
    content = checked_bytes(path, digest)
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path and sha(path) == digest, 'Conflicting module: '+name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(content, str(path), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class BoundNativeBridge(nn.Module):
    """Same native pruning, float64 decoder; optional same partial unfreeze per arm."""
    def __init__(self, *, partial=True, device='cuda'):
        super().__init__()
        root = STATE_ROOT
        for path, digest in ((root/'codec_final.pt',CHECKPOINT_SHA), (root/'code/codec.py',CODEC_SHA),
                             (root/'code/common.py',RENDERER_SHA), (root/'protocol.json',PROTOCOL_SHA),
                             (root/'selection.json',SELECTION_SHA), (OFFICIAL,OFFICIAL_SHA), (PRECISION,PRECISION_SHA)):
            require(sha(path) == digest, 'Codec dependency changed: '+str(path))
        common = _bound_module('common', root/'code/common.py', RENDERER_SHA)
        codec_module = _bound_module('codec', root/'code/codec.py', CODEC_SHA)
        precision_module = _bound_module('_baseline_decoder_precision', PRECISION, PRECISION_SHA)
        import MinkowskiEngine as ME
        self.ME = ME
        self._check_me()
        self.codec = codec_module.load_codec(root, root/'codec_final.pt', device=device)
        self.codec.eval().requires_grad_(False)
        require(self.codec.grid == .008 and self.codec.ae.stride == 4 and isinstance(self.codec.ae.vq, nn.Identity),
                'Codec latent shape/identity quantizer changed')
        self.precision = precision_module.DecoderPrecision(self.codec, 'float64')
        if partial:
            for module in (self.codec.ae.decoder.out, self.codec.ae.feature_heads, self.codec.ae.gaussian_multiplier):
                module.requires_grad_(True)
        self.partial = partial
        self.prediction_gs, self.render_fn = common.prediction_gs, common.render
        self.last_count = None
        self._digest_tensor = precision_module._tensor_digest
        self._frozen = self._frozen_digests()
        names = [name for name, p in self.codec.named_parameters() if p.requires_grad]
        allowed = ('ae.decoder.out.', 'ae.feature_heads.', 'ae.gaussian_multiplier.')
        require((bool(names) if partial else not names) and all(name.startswith(allowed) for name in names),
                'Unexpected native trainable parameters')

    def _check_me(self):
        require(self.ME.global_coordinate_manager() is None and self.ME.is_deterministic() is False
                and self.ME.sparse_tensor_operation_mode() == self.ME.SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER,
                'Sparse manager mode changed')

    def _frozen_digests(self):
        values = {name: value for name,value in self.codec.named_parameters() if not value.requires_grad}
        values.update({name: value for name,value in self.codec.named_buffers()})
        return {name: self._digest_tensor(value) for name,value in values.items()}

    def assert_frozen(self):
        require(self._frozen_digests() == self._frozen, 'Frozen codec parameter/buffer changed')
        require(not self.codec.training and not self.codec.ae.decoder.training, 'Codec left eval mode')
        self._check_me()

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def decode_points(self, coordinates, features):
        require(features.dtype == torch.float32 and tuple(features.shape) == (len(coordinates),64)
                and len(coordinates) > 0 and bool(torch.isfinite(features).all()), 'Invalid complete latent state')
        self._check_me()
        sparse = self.codec._sparse(coordinates, features, 4)
        prepared = self.precision.prepare_sparse(sparse, self.ME)
        points, _ = self.codec.ae.decode(prepared, target_key=None)
        return self.precision.to_render_points(self.codec._world(points))

    def render_views(self, prediction, record):
        points = self.decode_points(prediction['coordinates'], prediction['features'])
        gs, sh = self.prediction_gs(points)
        require(0 < len(gs) <= 2400000 and tuple(sh.shape) == (len(gs),4,3), 'Invalid decoded GS/SH')
        self.last_count = len(gs)
        images = [self.render_fn(gs, camera, record['intrinsics'], record['near'], record['far'], sh)[0].clamp(0,1)
                  for camera in record['cameras']]
        require(len(images) == 2 and all(tuple(image.shape) == (252,448,3) and bool(torch.isfinite(image).all())
                                        for image in images), 'Invalid complete two-view render')
        return torch.stack(images)

    def render(self, prediction, record):
        return self.render_views(prediction, record)[0]
