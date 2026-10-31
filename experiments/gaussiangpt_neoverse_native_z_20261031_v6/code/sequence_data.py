"""Load complete sampled windows from the existing continuous-state cache.

The sampler defines all frame identities. This module never repeats a missing
frame, synthesizes timestamps, or replaces ``source_pts``. Cached timestamps
remain the source's existing values; loading them does not establish that they
were measured container PTS. There is no renderer, MinkowskiEngine, codec, or
LFQ construction here.
"""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from sampling import FRAMES_PER_WINDOW, Window


def _scalar_time(value: Any, path: Path) -> float:
    """Convert only for validation, never replace the stored timestamp."""
    if torch.is_tensor(value):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"Expected a real scalar pts in {path}")
    elif not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"Expected a real scalar pts in {path}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Nonfinite pts in {path}")
    return converted


def _validate_state(state: Any, path: Path, window_spec: Window, frame: int) -> float:
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    required = {"coordinates", "features", "pts"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Missing state fields {sorted(missing)} in {path}")

    try:
        is_padding = bool(state.get("is_padding", False))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Invalid is_padding flag in {path}") from error
    if is_padding:
        raise ValueError(f"Padded cached state is not a real frame: {path}")

    coordinates, features = state["coordinates"], state["features"]
    if (
        not torch.is_tensor(coordinates)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.dtype not in (torch.int32, torch.int64)
        or len(coordinates) == 0
    ):
        raise ValueError(f"Expected nonempty integer coordinates [M,3] in {path}")
    if (
        not torch.is_tensor(features)
        or features.shape != (len(coordinates), 64)
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError(f"Expected finite floating features [M,64] in {path}")
    if len(torch.unique(coordinates, dim=0)) != len(coordinates):
        raise ValueError(f"Duplicate state coordinates in {path}")

    # Existing cache metadata must not disagree with the path selected by the
    # sampler. Do not silently relabel a cached state from another frame/clip.
    for key in ("frame", "frame_index"):
        if key in state and (type(state[key]) is not int or state[key] != frame):
            raise ValueError(f"Cached {key} disagrees with requested frame {frame}: {path}")
    if "clip" in state and state["clip"] != window_spec.clip:
        raise ValueError(f"Cached clip disagrees with requested clip {window_spec.clip!r}: {path}")
    return _scalar_time(state["pts"], path)


def _to_device(value: Any, device: torch.device) -> Any:
    """Move cached tensors, including nested tensors, without altering values."""
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def load_window(
    root: str | Path,
    window_spec: Window,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Read precisely ``Window.frames`` from ``root/states/<clip>/NNN.pt``.

    Deserialization always maps to CPU with ``weights_only=True``. After shape,
    provenance, padding and strictly increasing time checks, tensors move to the
    requested device. Missing files raise immediately; no fallback is attempted.
    ``frame_index`` is supplied from the validated sampling identity. Existing
    ``pts``, ``source_pts`` and all other cache fields retain their values.
    """
    if not isinstance(window_spec, Window):
        raise TypeError("window_spec must be a sampling.Window")
    frames = window_spec.frames
    if len(frames) != FRAMES_PER_WINDOW or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Expected a complete ten-frame, stride-one Window")
    root = Path(root)
    target_device = torch.device(device)
    states = []
    previous_pts = None
    for frame in frames:
        path = root / "states" / window_spec.clip / f"{frame:03d}.pt"
        state = torch.load(path, weights_only=True, map_location="cpu")
        pts = _validate_state(state, path, window_spec, frame)
        if previous_pts is not None and pts <= previous_pts:
            raise ValueError(f"State pts must be strictly increasing; frame {frame} in {path}")
        previous_pts = pts
        state = _to_device(state, target_device)
        state["frame_index"] = frame
        states.append(state)
    return states


def load_record(
    root: str | Path,
    window_spec: Window,
    device: str | torch.device = "cpu",
    *,
    radius: int = 6,
    max_candidates: int = 100000,
) -> dict[str, Any]:
    """Load a full window and construct geometry from its initial state only.

    Importing ``aligned_ldm`` does not instantiate its model or import LFQ.
    Future coordinates are never passed to ``make_condition`` and are not used
    to validate, expand, or truncate the initial candidate domain.
    """
    import aligned_ldm

    states = load_window(root, window_spec, device)
    condition = aligned_ldm.make_condition(states[0], radius=radius, max_candidates=max_candidates)
    return {
        "window": states,
        "condition": condition,
        "frames": list(window_spec.frames),
        "window_id": window_spec.window_id,
        "identity": window_spec.to_dict(),
    }
