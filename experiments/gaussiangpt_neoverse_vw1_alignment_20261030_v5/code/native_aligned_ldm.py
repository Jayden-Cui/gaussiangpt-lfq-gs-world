"""Full-depth native64 GS adaptation, with explicit TRAIN-only output units.

Twenty multiscale spatial/temporal residual blocks per side preserve every
input/candidate coordinate. This substitutes dilated 3D sparse neighborhoods
for image-grid down/up-sampling; it is not dense Conv3d operator equivalence.
"""
import torch
from torch import nn
from aligned_ldm import AlignedLDM
from multiscale64 import MultiScaleBackbone64


class NativeAlignedLDM(AlignedLDM):
    def __init__(self,*,coordinate_scale=.032,checkpoint_blocks=True):
        super().__init__(coordinate_scale=coordinate_scale,decoder_mode='window',blocks=1)
        # Do not leave unused prototype parameters in the optimizer.
        self.encoder_space=nn.ModuleList();self.encoder_time=nn.ModuleList()
        self.decoder_space=nn.ModuleList();self.decoder_time=nn.ModuleList()
        self.decoder_condition=None
        self.encoder_backbone=MultiScaleBackbone64('encoder',checkpoint_blocks=checkpoint_blocks)
        self.decoder_backbone=MultiScaleBackbone64('decoder',checkpoint_blocks=checkpoint_blocks)
        self.output_norm=nn.LayerNorm(64)
        self.register_buffer('output_mean',torch.zeros(64))
        self.register_buffer('output_std',torch.ones(64))
        self.register_buffer('output_calibrated',torch.tensor(False))
        nn.init.orthogonal_(self.features.weight);nn.init.zeros_(self.features.bias)

    @torch.no_grad()
    def initialize_output_units(self,train_reference_features):
        if bool(self.output_calibrated):raise RuntimeError('Output units already bound; no per-target recalibration')
        f=train_reference_features.detach().float()
        if f.ndim!=2 or f.shape[1]!=64 or len(f)<2 or not bool(torch.isfinite(f).all()):
            raise ValueError('Expected finite admitted TRAIN reference features [M,64]')
        std=f.std(0,unbiased=False)
        if not bool((std>1e-6).all()):raise ValueError('Degenerate TRAIN channel scale')
        self.output_mean.copy_(f.mean(0));self.output_std.copy_(std);self.output_calibrated.fill_(True)

    def _encode_hidden(self,hidden,coordinates,active,neighbors):
        return self.encoder_backbone(hidden,coordinates,active)

    def _decode_hidden(self,sequence,coordinates,active,neighbors):
        return self.decoder_backbone(sequence,coordinates,active,condition=sequence)

    def _project_features(self,x):
        if not bool(self.output_calibrated):raise RuntimeError('Bind TRAIN-only latent output units before decoding')
        return self.output_mean+self.output_std*self.features(self.output_norm(x))

    def encoder_versions(self):
        return tuple(p._version for module in (self.input,self.encoder_backbone) for p in module.parameters())

    @property
    def last_layer(self):return self.features.weight
