"""Native-64, coordinate-preserving multiscale spatiotemporal GS backbone.

The state encoder has already compressed space. This adapter therefore keeps
ALL N coordinate rows and all time slots instead of repeating dense-image
down/up-sampling. Its six-axis neighbor distances are 1, 2, 4 and 8 integer
latent cells. It is a GS multiscale adaptation, not an operator-equivalent
reproduction of VideoWorld's dense Conv3d/downsample/upsample architecture.

Each of 20 blocks applies the existing SparseSpatialBlock64 message formula
with its scale-specific neighbor index, followed by TemporalResidual64(1).
Encoder normalization never crosses time. The decoder uses full-window
normalization and one AdaptiveWindowNorm64 at each of four stage ends; all
four receive the SAME original 13-slot concatenation, never evolving hidden
features as a replacement condition. No stem, truncation, pruning or pooling
of coordinate rows is implemented.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from model_core import AdaptiveWindowNorm64, TemporalResidual64
from spatial64 import AXIS_OFFSETS, SparseSpatialBlock64


ENCODER_SCALES = (1,) * 4 + (2,) * 4 + (4,) * 4 + (8,) * 8
DECODER_SCALES = tuple(reversed(ENCODER_SCALES))
SPATIAL_DISTANCES = (1, 2, 4, 8)
DECODER_TIME_SLOTS = 13


def _coordinate_lookup(coordinates: Tensor):
    if not torch.is_tensor(coordinates) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [N,3]")
    if coordinates.dtype not in (torch.int32, torch.int64):
        raise TypeError("coordinates must be int32 or int64")
    rows = [tuple(row) for row in coordinates.detach().cpu().tolist()]
    lookup = {row: index for index, row in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("coordinates must be unique")
    return rows, lookup


def _neighbors(rows, lookup, distance: int, device: torch.device) -> Tensor:
    indices = [
        [lookup.get((x + distance * dx, y + distance * dy, z + distance * dz), -1)
         for dx, dy, dz in AXIS_OFFSETS]
        for x, y, z in rows
    ]
    return torch.tensor(indices, dtype=torch.long, device=device).reshape(-1, 6)


def build_neighbor_index_at_distance(coordinates: Tensor, distance: int) -> Tensor:
    """Six exact axis offsets at distance 1/2/4/8, not radius or nearest-k.

    Missing edges are -1; every coordinate row remains in the output. Topology
    uses coordinates only, so absent/future-only nodes cannot change a prefix
    through feature-dependent neighbor selection. Activity is applied by the
    spatial block at each time. Rebuild indices after any row permutation.
    """
    if type(distance) is not int or distance not in SPATIAL_DISTANCES:
        raise ValueError("distance must be one of 1, 2, 4, 8")
    rows, lookup = _coordinate_lookup(coordinates)
    return _neighbors(rows, lookup, distance, coordinates.device)


class _SpatialTemporalBlock64(nn.Module):
    def __init__(self, distance: int, *, across_time: bool, condition_at_end: bool) -> None:
        super().__init__()
        self.distance = distance
        self.spatial = SparseSpatialBlock64()
        self.temporal = TemporalResidual64(dilation=1, across_time=across_time)
        self.condition_norm = AdaptiveWindowNorm64() if condition_at_end else None

    def forward(self, x, coordinates, active, neighbor_index, condition):
        x = self.spatial(x, coordinates, active, neighbor_index=neighbor_index)
        x = self.temporal(x, active)
        if self.condition_norm is not None:
            x = self.condition_norm(x, condition, active)
        return x


class MultiScaleBackbone64(nn.Module):
    """Twenty coordinate-preserving blocks, configured as encoder or decoder.

    ``mode='encoder'`` supports arbitrary positive prefix lengths and rejects
    external conditioning. ``mode='decoder'`` expects the full 13 slots (three
    left-padding contexts, initial context, nine quantized future codes). Its
    condition defaults to the original input x; explicit conditioning must have
    the same shape, dtype and device. Decoder normalization is intentionally
    across the complete quantized window, so it is not prefix-causal.

    Training with ``checkpoint_blocks=True`` checkpoints each complete block
    using use_reentrant=False. It also works on CPU for equivalence tests. Eval
    or no_grad calls use the direct path. Checkpointing changes memory use, not
    coordinates, time extent, update count or gradient connectivity.
    """

    width = 64

    def __init__(self, mode: str = "encoder", *, checkpoint_blocks: bool = False) -> None:
        super().__init__()
        if mode not in ("encoder", "decoder"):
            raise ValueError("mode must be 'encoder' or 'decoder'")
        if type(checkpoint_blocks) is not bool:
            raise TypeError("checkpoint_blocks must be bool")
        self.mode = mode
        self.checkpoint_blocks = checkpoint_blocks
        self.scales = ENCODER_SCALES if mode == "encoder" else DECODER_SCALES
        self.stage_ends = tuple(
            index + 1 for index, scale in enumerate(self.scales)
            if index == len(self.scales) - 1 or self.scales[index + 1] != scale
        )
        self.blocks = nn.ModuleList([
            _SpatialTemporalBlock64(
                scale,
                across_time=mode == "decoder",
                condition_at_end=mode == "decoder" and index + 1 in self.stage_ends,
            )
            for index, scale in enumerate(self.scales)
        ])

    def forward(
        self,
        x: Tensor,
        coords: Tensor,
        active: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("x must have shape [T,N,64]")
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("backbone requires nonempty time and coordinate axes")
        rows, lookup = _coordinate_lookup(coords)
        if len(rows) != x.shape[1] or coords.device != x.device:
            raise ValueError("coords must match all N input rows and the input device")
        if (not torch.is_tensor(active) or active.dtype != torch.bool
                or active.shape != x.shape[:2] or active.device != x.device):
            raise ValueError("active must be [T,N] bool on the input device")
        if self.mode == "encoder":
            if condition is not None:
                raise ValueError("encoder does not accept full-window conditioning")
        else:
            if x.shape[0] != DECODER_TIME_SLOTS:
                raise ValueError("decoder requires the original 13-slot concatenation")
            if condition is None:
                condition = x
            if (not torch.is_tensor(condition) or condition.shape != x.shape
                    or condition.dtype != x.dtype or condition.device != x.device):
                raise ValueError("decoder condition must match the full input shape/dtype/device")
        # Build once per distance and forward call, reusing the same CPU lookup.
        # No cross-call cache can silently retain stale row identities.
        neighbors = {distance: _neighbors(rows, lookup, distance, coords.device)
                     for distance in SPATIAL_DISTANCES}
        use_checkpoint = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        hidden = x
        for block in self.blocks:
            args = (hidden, coords, active, neighbors[block.distance], condition)
            hidden = checkpoint(block, *args, use_reentrant=False) if use_checkpoint else block(*args)
        return hidden


__all__ = [
    "ENCODER_SCALES", "DECODER_SCALES", "SPATIAL_DISTANCES",
    "build_neighbor_index_at_distance", "MultiScaleBackbone64",
]
