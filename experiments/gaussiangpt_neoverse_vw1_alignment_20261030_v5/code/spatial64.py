"""Masked, per-time spatial messages on a sparse integer world-coordinate grid.

This is a GS-specific spatial adapter, not an operator-equivalent replacement
for VideoWorld's dense spatiotemporal Conv3d backbone. Features remain 64 wide.
There is no time mixing, point removal, sampling, or learned support selection.
"""

from typing import Optional

import torch
from torch import Tensor, nn


# The order is fixed, independent of input row order or current visibility.
AXIS_OFFSETS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                (0, 1, 0), (0, 0, -1), (0, 0, 1))


def _validate_coordinates(coordinates: Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [N, 3]")
    if coordinates.dtype not in (torch.int32, torch.int64):
        raise TypeError("coordinates must use integer int32 or int64")


def build_neighbor_index(coordinates: Tensor) -> Tensor:
    """Return all six one-grid-step axis neighbors as [N, 6] int64 indices.

    A missing neighbor is -1. Only integer coordinates are inspected: neither
    features, time, masks, nor future targets affect this topology. Coordinates
    must be unique, but need not be sorted. CPU dictionary construction is O(N)
    memory/time for this fixed-degree graph; the result returns to the original
    device. Call once and reuse for multiple blocks on the same coordinate rows.
    """
    _validate_coordinates(coordinates)
    rows = [tuple(row) for row in coordinates.detach().cpu().tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("coordinates must be unique; duplicate rows are ambiguous")
    indices = [[lookup.get((x + dx, y + dy, z + dz), -1)
                for dx, dy, dz in AXIS_OFFSETS] for x, y, z in rows]
    return torch.tensor(indices, dtype=torch.long, device=coordinates.device).reshape(-1, 6)


class SparseSpatialBlock64(nn.Module):
    """Shared 64-wide residual messages, separately for every time slice.

    For active node i at time t:
      y[t,i] = x[t,i] + mean_{j in active axis-neighbors(i)} message(x[t,j]).
    An empty neighborhood contributes exactly zero. Inactive outputs are zero.
    LayerNorm acts only on each individual node's 64 channels; absent nodes
    enter neither the message sum nor its denominator. All six directions and
    all times share the same parameters. Stacking blocks extends spatial reach.

    ``present`` means occupancy at the CURRENT time, not cumulative ``seen``.
    Optional ``visible`` can additionally exclude not-yet-visible nodes. Use
    present & visible when a union includes coordinates belonging to the future.
    A supplied neighbor_index must come from build_neighbor_index for exactly
    these coordinate rows; it must be rebuilt after any coordinate permutation.
    """

    width = 64

    def __init__(self) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )

    def forward(
        self,
        features: Tensor,
        coordinates: Tensor,
        present: Tensor,
        *,
        visible: Optional[Tensor] = None,
        neighbor_index: Optional[Tensor] = None,
    ) -> Tensor:
        _validate_coordinates(coordinates)
        if features.ndim != 3 or features.shape[-1] != self.width:
            raise ValueError("features must have shape [T, N, 64]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if coordinates.shape[0] != features.shape[1]:
            raise ValueError("coordinate count must match feature rows")
        if coordinates.device != features.device:
            raise ValueError("features and coordinates must be on the same device")
        for name, mask in (("present", present), ("visible", visible)):
            if mask is None:
                if name == "present":
                    raise TypeError("present is required")
                continue
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must be a boolean mask")
            if mask.shape != features.shape[:2] or mask.device != features.device:
                raise ValueError(f"{name} must have shape [T, N] on the feature device")
        active = present if visible is None else present & visible
        # Sanitize BEFORE normalization, including masked NaN/Inf placeholders.
        clean = torch.where(active[..., None], features, torch.zeros_like(features))
        if neighbor_index is None:
            neighbor_index = build_neighbor_index(coordinates)
        else:
            if (neighbor_index.dtype != torch.long
                    or neighbor_index.shape != (features.shape[1], 6)
                    or neighbor_index.device != features.device):
                raise ValueError("neighbor_index must be [N, 6] int64 on the feature device")
            if bool(((neighbor_index < -1) | (neighbor_index >= features.shape[1])).any()):
                raise ValueError("neighbor_index contains an out-of-range row")
        if features.shape[0] == 0 or features.shape[1] == 0:
            return clean

        messages = self.message(clean)
        aggregate = torch.zeros_like(features)
        count = torch.zeros_like(present, dtype=features.dtype)
        # Six streaming gathers avoid a materialized [T,N,6,64] edge tensor.
        for direction in range(len(AXIS_OFFSETS)):
            source = neighbor_index[:, direction]
            safe_source = source.clamp_min(0)
            valid = (source[None] >= 0) & active & active.index_select(1, safe_source)
            gathered = messages.index_select(1, safe_source)
            aggregate = aggregate + torch.where(valid[..., None], gathered, 0.0)
            count = count + valid.to(features.dtype)
        return clean + aggregate / count.clamp_min(1)[..., None]


__all__ = ["AXIS_OFFSETS", "build_neighbor_index", "SparseSpatialBlock64"]
