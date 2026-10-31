"""Differentiable motion occupancy -> final GS opacity, without vendor edits.

GS bridge integration (one decoder call per context):
    z = codec._sparse(candidate_coordinates, predicted_features, codec.ae.stride)
    with SoftOccupancyGate(codec.ae.decoder, z, motion_logits) as tracker:
        points, info = codec.ae.decode(z, target_key=None)
        gate = tracker.gaussian_probabilities(len(points[K.OPACITIES]),
                                             multiplier=codec.count)
    gs, sh = prediction_gs(codec._world(points))
    gs = torch.cat([gs[:, :13], (gs[:, 13] * gate)[:, None]], dim=-1)

All candidate latents enter the AE. Its existing internal predicted pruning is
preserved. No floor-parent lookup, target coordinates, support truncation, or
modification of AE parameters is used. Gates are means over actual generative
kernel-map parents, not physical Gaussian identities or full receptive fields.
The original multiplier must preserve voxel-major row order (four rows/voxel
for the bound StateCodec). Multiply probabilities AFTER its opacity conversion.
"""

from dataclasses import asdict, dataclass
from weakref import WeakKeyDictionary

import torch


class GateCoverageError(RuntimeError):
    """An output coordinate lacks its required actual sparse-map predecessor."""


def mean_gate_from_kernel_map(probabilities, kernel_map, output_count):
    """ME dict values are [input_row_indices, output_row_indices]."""
    if output_count <= 0:
        raise GateCoverageError("Empty sparse output; no points are silently removed")
    total = probabilities.new_zeros(output_count)
    degree = torch.zeros(output_count, dtype=torch.long, device=probabilities.device)
    for source, destination in kernel_map.values():
        source = source.to(device=probabilities.device, dtype=torch.long)
        destination = destination.to(device=probabilities.device, dtype=torch.long)
        total = total.index_add(0, destination, probabilities.index_select(0, source))
        degree = degree.index_add(0, destination, torch.ones_like(destination))
    missing = degree == 0
    if bool(missing.any()):
        raise GateCoverageError(
            f"{int(missing.sum())}/{output_count} sparse output rows lack a parent; "
            f"first missing rows={missing.nonzero().flatten()[:8].tolist()}"
        )
    return total / degree, degree


@dataclass(frozen=True)
class StageGeometry:
    index: int
    module_path: str
    kernel_size: tuple
    stride: tuple
    dilation: tuple


def inspect_decoder_stages(decoder):
    """Find the real generative operator recursively; reject changed topology.

    No Sequential/list position is assumed. The supported bound vendor has one
    3D kernel3/stride2/dilation1 generative transpose per stage, and only stride1
    coordinate-preserving ordinary convolutions around it.
    """
    import MinkowskiEngine as ME

    if not hasattr(decoder, "ups") or len(decoder.ups) == 0:
        raise ValueError("Decoder must expose nonempty ups stages")
    if not isinstance(getattr(decoder, "pruning", None), ME.MinkowskiPruning):
        raise ValueError("Decoder must expose its shared MinkowskiPruning module")
    if not isinstance(getattr(decoder, "out", None), ME.MinkowskiConvolution):
        raise ValueError("Decoder.out must be a coordinate-preserving sparse convolution")
    result = []
    for index, stage in enumerate(decoder.ups):
        found = [(name, module) for name, module in stage.named_modules()
                 if isinstance(module, ME.MinkowskiGenerativeConvolutionTranspose)]
        if len(found) != 1:
            raise ValueError(f"Stage {index} must contain exactly one generative transpose")
        name, transpose = found[0]
        kg = transpose.kernel_generator
        geometry = StageGeometry(index, name, tuple(kg.kernel_size),
                                 tuple(kg.kernel_stride), tuple(kg.kernel_dilation))
        if (geometry.kernel_size != (3, 3, 3) or geometry.stride != (2, 2, 2)
                or geometry.dilation != (1, 1, 1) or kg.dimension != 3
                or kg.region_type != ME.RegionType.HYPER_CUBE or not kg.expand_coordinates):
            raise ValueError(f"Unsupported actual generative geometry: {geometry}")
        for module in stage.modules():
            if isinstance(module, ME.MinkowskiPruning):
                raise ValueError("Unexpected pruning inside an up stage")
            if isinstance(module, ME.MinkowskiConvolution):
                kernel = module.kernel_generator
                if tuple(kernel.kernel_stride) != (1, 1, 1) or kernel.expand_coordinates:
                    raise ValueError("Unexpected coordinate-changing convolution in an up stage")
        result.append((geometry, stage, transpose))
    final = decoder.out.kernel_generator
    if tuple(final.kernel_stride) != (1, 1, 1) or final.expand_coordinates:
        raise ValueError("Decoder.out changes spatial support")
    return result


_ACTIVE_DECODERS = WeakKeyDictionary()


class SoftOccupancyGate:
    """Context manager tracking one eval-mode sparse decoder forward.

    motion_logits is in the ORIGINAL coordinate row order used to construct z.
    SparseTensor.unique_index aligns it to z.C/z.F. Duplicate input coordinates
    are rejected. For an explicitly pre-aligned tensor, set logits_are_aligned.
    eval mode preserves normal predicted pruning without target forcing; it does
    not disable gradients or require the decoder weights to be frozen.
    """

    def __init__(self, decoder, latent, motion_logits, *, input_coordinates=None,
                 logits_are_aligned=False):
        self.decoder = decoder
        self.latent = latent
        self.cm = latent.coordinate_manager
        self.stages = inspect_decoder_stages(decoder)
        expected_stride = 2 ** len(self.stages)
        if tuple(latent.tensor_stride) != (expected_stride,) * 3:
            raise ValueError("Latent tensor_stride does not match the actual up stages")
        if motion_logits.ndim == 2 and motion_logits.shape[1] == 1:
            motion_logits = motion_logits[:, 0]
        if (motion_logits.ndim != 1 or len(motion_logits) != len(latent)
                or not motion_logits.is_floating_point()):
            raise ValueError("Require one floating motion logit per unique source latent")
        if motion_logits.device != latent.F.device or not bool(torch.isfinite(motion_logits).all()):
            raise ValueError("Motion logits must be finite and on the latent feature device")
        if not logits_are_aligned or input_coordinates is not None:
            order = getattr(latent, "unique_index", None)
            if not isinstance(order, torch.Tensor) or order.numel() != len(latent):
                raise ValueError("Missing unique_index; explicitly align logits or supply the original SparseTensor")
            order = order.to(device=motion_logits.device, dtype=torch.long)
            if not torch.equal(order.sort().values, torch.arange(len(latent), device=order.device)):
                raise ValueError("Duplicate/non-bijective input coordinate mapping is unsupported")
            if input_coordinates is not None:
                if (input_coordinates.shape != latent.C.shape
                        or input_coordinates.dtype not in (torch.int32, torch.int64)):
                    raise ValueError("input_coordinates must be original integer [N, batch+xyz] rows")
                supplied = input_coordinates.to(device=latent.C.device, dtype=latent.C.dtype)
                supplied = supplied.index_select(0, order.to(supplied.device))
                if not torch.equal(supplied, latent.C):
                    raise ValueError("input_coordinates do not match SparseTensor.unique_index alignment")
            if not logits_are_aligned:
                motion_logits = motion_logits.index_select(0, order)
        self._probabilities = motion_logits.sigmoid()
        self._key = latent.coordinate_map_key
        self._handles = []
        self._used = self._started = self._finished = False
        self._next_stage = 0
        self._transposed_stage = None
        self._awaiting_prune = False
        self._multiplier = None
        self.events = []

    @property
    def stage_geometries(self):
        return tuple(item[0] for item in self.stages)

    def probabilities_for(self, sparse):
        """Read-only exact-coordinate alignment to an existing same-stride subset."""
        if sparse.coordinate_manager is not self.cm:
            raise GateCoverageError("Coordinate manager changed during gate propagation")
        km = self.cm.kernel_map(self._key, sparse.coordinate_map_key,
                               stride=1, kernel_size=1, dilation=1, is_transpose=False)
        probabilities, degree = mean_gate_from_kernel_map(self._probabilities, km, len(sparse))
        if not bool((degree == 1).all()):
            raise GateCoverageError("Exact coordinate alignment must have exactly one predecessor")
        return probabilities

    def _begin(self, module, args, kwargs):
        if self._started:
            raise RuntimeError("Only one decoder forward is allowed per gate context")
        if module.training:
            raise ValueError("Use decoder.eval(); training mode may force target occupancy")
        if kwargs.get("target_key") is not None or (len(args) > 1 and args[1] is not None):
            raise ValueError("Soft occupancy decoding forbids target_key")
        if args[0] is not self.latent:
            raise ValueError("Decoder input is not the SparseTensor bound to this gate context")
        self._started = True

    def _transpose_hook(self, index, geometry, module, args, output):
        if not self._started or index != self._next_stage or self._awaiting_prune:
            raise RuntimeError("Unexpected generative transpose execution order")
        source = args[0]
        probabilities = self.probabilities_for(source)
        km = self.cm.kernel_map(source.coordinate_map_key, output.coordinate_map_key,
                               stride=list(geometry.stride), kernel_size=list(geometry.kernel_size),
                               dilation=list(geometry.dilation), is_transpose=True)
        self._probabilities, degree = mean_gate_from_kernel_map(probabilities, km, len(output))
        self._key = output.coordinate_map_key
        self._transposed_stage = index
        self.events.append(dict(kind="up", stage=index, input_count=len(source),
                                output_count=len(output), max_parent_count=int(degree.max())))

    def _stage_hook(self, index, module, args, output):
        if self._transposed_stage != index or index != self._next_stage:
            raise RuntimeError("Up stage did not execute its inspected generative transpose")
        if len(output) != len(self._probabilities):
            raise GateCoverageError("Up stage changed support after its generative operator")
        self._probabilities = self.probabilities_for(output)
        self._key = output.coordinate_map_key
        self._awaiting_prune = True

    def _prune_hook(self, module, args, output):
        if not self._awaiting_prune:
            raise RuntimeError("Unexpected decoder pruning call")
        self._probabilities = self.probabilities_for(output)
        self._key = output.coordinate_map_key
        self.events.append(dict(kind="prune", stage=self._next_stage,
                                input_count=len(args[0]), output_count=len(output)))
        self._next_stage += 1
        self._awaiting_prune = False

    def _out_hook(self, module, args, output):
        if self._next_stage != len(self.stages) or self._awaiting_prune:
            raise RuntimeError("Decoder.out reached before all up/prune stages completed")
        if len(output) != len(self._probabilities):
            raise GateCoverageError("Final sparse convolution changed support")
        self._probabilities = self.probabilities_for(output)
        self._key = output.coordinate_map_key
        self._finished = True
        self.events.append(dict(kind="out", output_count=len(output)))

    def gaussian_probabilities(self, gaussian_count, *, multiplier=4):
        """Return voxel-major repeated probabilities; assert the GS bridge count."""
        if not self._finished:
            raise RuntimeError("No completed decoder output for this gate context")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError("multiplier must be a positive integer")
        if gaussian_count != multiplier * len(self._probabilities):
            raise ValueError("GS count disagrees with voxel-major multiplier ordering")
        self._multiplier = multiplier
        return self._probabilities.repeat_interleave(multiplier)

    def gaussian_gate(self, multiplier=4):
        """GS bridge API; the bridge must also check len(gate) == len(final_gs)."""
        return self.gaussian_probabilities(multiplier * len(self._probabilities), multiplier=multiplier)

    def summary(self):
        """JSON-safe diagnostics with names that cannot clash with bridge stats."""
        return dict(soft_gate_source_rows=len(self.latent),
                    soft_gate_output_voxels=len(self._probabilities) if self._finished else None,
                    soft_gate_multiplier=self._multiplier,
                    soft_gate_complete=self._finished,
                    soft_gate_stages=[asdict(geometry) for geometry in self.stage_geometries],
                    soft_gate_events=[dict(event) for event in self.events])

    def __enter__(self):
        if self._used or self.decoder in _ACTIVE_DECODERS:
            raise RuntimeError("Gate contexts are single-use and cannot nest on one decoder")
        if self.decoder.training:
            raise ValueError("Use decoder.eval(); its internal predicted pruning stays enabled")
        self._used = True
        _ACTIVE_DECODERS[self.decoder] = self
        try:
            self._handles.append(self.decoder.register_forward_pre_hook(self._begin, with_kwargs=True))
            for geometry, stage, transpose in self.stages:
                self._handles.append(transpose.register_forward_hook(
                    lambda m, a, o, i=geometry.index, g=geometry: self._transpose_hook(i, g, m, a, o)))
                self._handles.append(stage.register_forward_hook(
                    lambda m, a, o, i=geometry.index: self._stage_hook(i, m, a, o)))
            self._handles.append(self.decoder.pruning.register_forward_hook(self._prune_hook))
            self._handles.append(self.decoder.out.register_forward_hook(self._out_hook))
        except BaseException:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if _ACTIVE_DECODERS.get(self.decoder) is self:
            del _ACTIVE_DECODERS[self.decoder]

    def __exit__(self, exc_type, exc, traceback):
        self._remove_hooks()
        if exc_type is None and not self._finished:
            raise RuntimeError("Gate context exited without a completed decoder forward")
        return False


# Preferred integration name; the original descriptive API remains compatible.
SoftOccupancyTracker = SoftOccupancyGate
