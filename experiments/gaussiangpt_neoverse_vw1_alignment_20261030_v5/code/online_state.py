"""Encode raw cached GS representatives online without detaching the encoder.

Sources are the state codec's ``cache/<clip>/NNN.pt`` payloads produced by
prepare_neoverse.py / common.voxelize: representatives [N,14], set_statistics
[N,29], raw voxel coordinates [N,3], and original RGB/camera supervision.
Motion ``states/<clip>/NNN.pt`` supplies time metadata only, never its latent.
An explicit per-frame time mapping may instead be supplied by the caller.

All .pt reads use weights_only=True on CPU. Source hashes are NOT checked here:
the caller must bind these files and any time mapping through its manifest.
No MinkowskiEngine, codec implementation, renderer, or LFQ is imported. The
caller supplies codec.encode_state, its grid and ae.stride, device and mode.
Autograd mode is inherited; this module never uses no_grad/detach or changes
codec.train()/eval()/requires_grad. Training callers must keep grad enabled.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import numpy as np

from sampling import FRAMES_PER_WINDOW, Window
from sequence_data import _scalar_time, _to_device, _validate_state


ONLINE_SCHEMA = "gs_ldm_online_state_window_v1"
RAW_REQUIRED = {
    "representatives", "coordinates", "set_statistics", "cameras", "intrinsics",
    "targets", "real", "near", "far", "normalization", "clip", "frame",
}


def _positive_grid(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive grid size")
    try:
        grid = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite positive grid size") from error
    if not math.isfinite(grid) or grid <= 0:
        raise ValueError(f"{label} must be a finite positive grid size")
    return grid


def _positive_stride(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer stride")
    return value


def _check_identity(payload: dict[str, Any], path: Path, spec: Window, frame: int) -> None:
    if "clip" in payload and payload["clip"] != spec.clip:
        raise ValueError(f"Cached clip does not match sampling identity: {path}")
    for key in ("frame", "frame_index"):
        if key in payload and (type(payload[key]) is not int or payload[key] != frame):
            raise ValueError(f"Cached {key} does not match sampling identity: {path}")
    try:
        padded = bool(payload.get("is_padding", False))
    except (ValueError, TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid padding flag: {path}") from error
    if padded:
        raise ValueError(f"Padded source is not a real frame: {path}")


def _validate_raw(payload: Any, path: Path, spec: Window, frame: int, grid: float) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a raw GS cache dictionary: {path}")
    missing = RAW_REQUIRED.difference(payload)
    if missing:
        raise ValueError(f"Missing raw GS cache fields {sorted(missing)}: {path}")
    _check_identity(payload, path, spec, frame)
    representatives = payload["representatives"]
    if (
        not torch.is_tensor(representatives) or representatives.ndim != 2
        or representatives.shape[1] != 14 or len(representatives) == 0
        or not representatives.is_floating_point()
        or not bool(torch.isfinite(representatives).all())
    ):
        raise ValueError(f"Expected finite representatives [N,14]: {path}")
    statistics = payload["set_statistics"]
    if (
        not torch.is_tensor(statistics) or statistics.shape != (len(representatives), 29)
        or not statistics.is_floating_point() or not bool(torch.isfinite(statistics).all())
    ):
        raise ValueError(f"Expected finite set_statistics [N,29]: {path}")
    coordinates = payload["coordinates"]
    if (
        not torch.is_tensor(coordinates) or coordinates.shape != (len(representatives), 3)
        or coordinates.dtype not in (torch.int32, torch.int64)
        or len(torch.unique(coordinates, dim=0)) != len(coordinates)
    ):
        raise ValueError(f"Expected unique integer raw voxel coordinates [N,3]: {path}")
    # The known producer uses float32 CPU round-to-nearest/even (np.rint).
    # Check this identity on CPU; never replace source coordinates or regrid
    # its sufficient statistics. Changing the grid requires a new raw cache.
    expected_coordinates = torch.from_numpy(np.rint(representatives.detach().cpu().numpy()[:, :3] / grid).astype(np.int64))
    if not torch.equal(coordinates.to(torch.int64), expected_coordinates):
        raise ValueError(f"Raw voxel coordinates do not match codec.grid: {path}")
    if "grid" in payload and _positive_grid(payload["grid"], "raw grid") != grid:
        raise ValueError(f"Raw cache grid does not match codec.grid: {path}")


def _read_time(
    motion_root: Path | None,
    spec: Window,
    frame: int,
    pts_by_frame: Mapping[int, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if pts_by_frame is None:
        path = motion_root / "states" / spec.clip / f"{frame:03d}.pt"
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict) or "pts" not in payload:
            raise ValueError(f"Missing motion-cache pts: {path}")
        _check_identity(payload, path, spec, frame)
        # Deliberately do not access coordinates/features/reconstruction.
        time = {key: payload[key] for key in ("pts", "source_pts") if key in payload}
        source = {"kind": "motion_cache_metadata", "path": str(path)}
    else:
        if frame not in pts_by_frame:
            raise ValueError(f"Explicit time mapping is missing frame {frame}")
        entry = pts_by_frame[frame]
        if isinstance(entry, Mapping):
            if "pts" not in entry:
                raise ValueError(f"Explicit time mapping is missing pts at frame {frame}")
            time = {key: entry[key] for key in ("pts", "source_pts") if key in entry}
        else:
            time = {"pts": entry}
        source = {"kind": "caller_explicit_mapping", "frame": frame}
    return time, source


def load_online_window(
    state_root: str | Path,
    motion_root: str | Path | None,
    window_spec: Window,
    codec: Any,
    device: str | torch.device = "cpu",
    *,
    pts_by_frame: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Return ten differentiable online states and unchanged supervision records.

    ``records`` holds the original raw-cache dictionaries, moving tensors to
    ``device`` without rescaling/clamping RGB, changing cameras, or relabeling
    fields. ``window`` contains fresh encode_state outputs with 64-wide features,
    integer latent coordinates, matching grid/stride, and original time values.
    ``initial`` is exactly ``window[0]``; construct any initial condition from
    that state alone. Future coordinates are not used to build an initial state.

    If pts_by_frame is given it is authoritative and motion_root is not read.
    Values may be original numeric timestamps or {pts, optional source_pts}.
    Nothing derives time from FPS or fills missing frames. The mapping's clip
    association and hash, like all raw sources, must be bound by the caller.
    """
    if not isinstance(window_spec, Window):
        raise TypeError("window_spec must be a sampling.Window")
    frames = window_spec.frames
    if (len(frames) != FRAMES_PER_WINDOW or len(set(frames)) != FRAMES_PER_WINDOW
            or any(b != a + 1 for a, b in zip(frames, frames[1:]))):
        raise ValueError("Online encoding requires ten distinct consecutive real frames")
    if not callable(getattr(codec, "encode_state", None)):
        raise TypeError("codec must provide encode_state(representatives, set_statistics)")
    grid = _positive_grid(getattr(codec, "grid", None), "codec.grid")
    stride = _positive_stride(getattr(getattr(codec, "ae", None), "stride", None), "codec.ae.stride")
    if pts_by_frame is not None and not isinstance(pts_by_frame, Mapping):
        raise TypeError("pts_by_frame must be an explicit frame-to-time mapping")
    if pts_by_frame is None and motion_root is None:
        raise ValueError("A motion cache or explicit time mapping is required")
    state_root = Path(state_root)
    motion_root = None if motion_root is None else Path(motion_root)
    target_device = torch.device(device)
    records, states, sources = [], [], []
    previous_pts = None
    for frame in frames:
        path = state_root / "cache" / window_spec.clip / f"{frame:03d}.pt"
        raw = torch.load(path, weights_only=True, map_location="cpu")
        _validate_raw(raw, path, window_spec, frame, grid)
        time, time_source = _read_time(motion_root, window_spec, frame, pts_by_frame)
        numeric_pts = _scalar_time(time["pts"], path)
        if previous_pts is not None and numeric_pts <= previous_pts:
            raise ValueError(f"Online window pts must be strictly increasing: {path}")
        previous_pts = numeric_pts
        record = _to_device(raw, target_device)
        # Keep the returned feature tensor itself: no clone/detach, cache
        # substitution, or no_grad scope around the external encoder call.
        encoded = codec.encode_state(record["representatives"], record["set_statistics"])
        if not isinstance(encoded, dict) or not {"coordinates", "features", "stride", "grid"} <= encoded.keys():
            raise ValueError("encode_state must return coordinates, features, stride and grid")
        if _positive_stride(encoded["stride"], "encoded stride") != stride:
            raise ValueError("Encoded stride does not match codec.ae.stride")
        if _positive_grid(encoded["grid"], "encoded grid") != grid:
            raise ValueError("Encoded grid does not match codec.grid")
        state = {
            "coordinates": encoded["coordinates"],
            "features": encoded["features"],
            "stride": stride,
            "grid": grid,
            "coordinate_scale": grid * stride,
            "coordinate_units": "integer latent cells, world spacing = grid * stride",
            "clip": window_spec.clip,
            "frame": frame,
            "frame_index": frame,
            "normalization": record["normalization"],
            **_to_device(time, target_device),
        }
        _validate_state(state, path, window_spec, frame)
        if (state["features"].device != record["representatives"].device
                or state["coordinates"].device != state["features"].device):
            raise ValueError("Encoded coordinates/features must use the input device")
        records.append(record)
        states.append(state)
        sources.append({"frame": frame, "raw_cache_path": str(path), "time_source": time_source})
    return {
        "schema": ONLINE_SCHEMA,
        "window": states,
        "records": records,
        "initial": states[0],
        "frames": list(frames),
        "window_id": window_spec.window_id,
        "identity": window_spec.to_dict(),
        "sources": sources,
        "source_verification": "caller_manifest_binding_required; this loader does not verify hashes",
        "encoding": "fresh codec.encode_state for every frame; motion latent is never used",
    }
