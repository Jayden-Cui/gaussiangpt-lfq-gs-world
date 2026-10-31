"""Complete ten-frame, stride-one windows for the v5 GS-LDM experiment.

A segment has inclusive frame boundaries. Training first chooses an eligible
segment uniformly, then chooses one of its complete windows uniformly. Segment
draws use replacement, including the deliberate single-clip adaptation: one
clip does not create a length-one/drop-last loader or a shuffled-start epoch.
This does not reproduce the official multi-segment DefaultSampler epoch order.

Evaluation enumerates each (segment, start) once without consuming training RNG.
Overlapping annotations remain distinct sampling units. No frame is replicated,
and there are no padded targets. Only the Python standard library is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
import random
import sys
from typing import Any, Iterable, Iterator, Mapping


FRAMES_PER_WINDOW = 10
FRAME_STRIDE = 1
HORIZON = FRAMES_PER_WINDOW - 1
WINDOW_SCHEMA = "gs_ldm_complete_window_v1"
MANIFEST_SCHEMA = "gs_ldm_segment_sampling_manifest_v1"
STATE_SCHEMA = "gs_ldm_segment_sampler_state_v1"
HORIZON_SEMANTICS = (
    "h=1..9 selects frame start+h relative to the same initial frame start; "
    "a posterior at h may read frames start..start+h. The nine targets are "
    "not nine independent adjacent-frame pairs or predicted AR actions."
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Segment:
    """One annotation segment; ``start`` and ``end`` are both inclusive."""

    clip: str
    start: int
    end: int
    segment_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.clip, "clip")
        _integer(self.start, "start")
        _integer(self.end, "end")
        if self.end < self.start:
            raise ValueError("segment end must be >= start (inclusive boundaries)")
        if self.segment_id is None:
            # JSON avoids delimiter ambiguities in arbitrary clip names.
            object.__setattr__(self, "segment_id", _canonical_json([self.clip, self.start, self.end]))
        else:
            _nonempty(self.segment_id, "segment_id")

    @property
    def frame_count(self) -> int:
        return self.end - self.start + 1

    @property
    def complete_window_count(self) -> int:
        return max(0, self.frame_count - HORIZON)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "clip": self.clip,
            "start": self.start,
            "end": self.end,
            "bounds": "inclusive",
            "frame_count": self.frame_count,
            "complete_window_count": self.complete_window_count,
        }


@dataclass(frozen=True)
class Window:
    """An immutable complete window belonging to a specific segment."""

    segment: Segment
    start: int

    def __post_init__(self) -> None:
        if not isinstance(self.segment, Segment):
            raise TypeError("segment must be a Segment")
        _integer(self.start, "window start")
        if not self.segment.start <= self.start <= self.segment.end - HORIZON:
            raise ValueError("window must contain ten real frames within its segment")

    @property
    def clip(self) -> str:
        return self.segment.clip

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id  # type: ignore[return-value]

    @property
    def end(self) -> int:
        return self.start + HORIZON

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))

    @property
    def future_frames(self) -> tuple[int, ...]:
        return self.frames[1:]

    @property
    def horizon(self) -> int:
        return HORIZON

    @property
    def frame_stride(self) -> int:
        return FRAME_STRIDE

    @property
    def window_id(self) -> str:
        # Include the segment identity even if two annotations cover the same frames.
        return _canonical_json([self.segment_id, self.clip, self.start, self.end, FRAME_STRIDE])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WINDOW_SCHEMA,
            "window_id": self.window_id,
            "segment_id": self.segment_id,
            "segment": self.segment.to_dict(),
            "clip": self.clip,
            "start": self.start,
            "end": self.end,
            "bounds": "inclusive",
            "frames": list(self.frames),
            "frames_per_window": FRAMES_PER_WINDOW,
            "frame_stride": FRAME_STRIDE,
            "initial_frame": self.start,
            "horizon": HORIZON,
            "future_offsets": list(range(1, HORIZON + 1)),
            "future_frames": list(self.future_frames),
            "valid_future_mask": [True] * HORIZON,
            "padding_positions": 0,
            "horizon_semantics": HORIZON_SEMANTICS,
        }


class SegmentWindowSampler(Iterator[Window]):
    """Uniform segment draw followed by a uniform complete-window draw.

    ``seed`` is an integer used exclusively by this sampler's private RNG.
    ``state_dict`` is JSON-serializable and restores the exact next draw under
    the same Python implementation/major.minor version and ordered segment
    declarations. Loading a checkpoint may replace the constructor's seed.
    The sample stream is intentionally unbounded; callers set update budgets.
    """

    def __init__(
        self,
        segments: Iterable[Segment],
        seed: int,
        frames_per_window: int = FRAMES_PER_WINDOW,
    ) -> None:
        _integer(seed, "seed")
        if type(frames_per_window) is not int or frames_per_window != FRAMES_PER_WINDOW:
            raise ValueError("v5 sampling requires exactly ten frames per window")
        self.segments = tuple(segments)
        if any(not isinstance(segment, Segment) for segment in self.segments):
            raise TypeError("segments must contain only Segment objects")
        identities = [segment.segment_id for segment in self.segments]
        if len(set(identities)) != len(identities):
            raise ValueError("segment_id values must be unique, including filtered segments")
        self.valid_segments = tuple(s for s in self.segments if s.complete_window_count)
        self.filtered_segments = tuple(s for s in self.segments if not s.complete_window_count)
        if not self.valid_segments:
            raise ValueError("no eligible segment contains ten real consecutive frames")
        self.seed = seed
        self.draw_count = 0
        self._rng = random.Random(seed)
        self._contract = {
            "segments": [segment.to_dict() for segment in self.segments],
            "frames_per_window": FRAMES_PER_WINDOW,
            "frame_stride": FRAME_STRIDE,
            "segment_sampling": "uniform_with_replacement",
            "start_sampling": "uniform_over_complete_windows_in_selected_segment",
        }
        self._contract_sha256 = hashlib.sha256(_canonical_json(self._contract).encode("utf-8")).hexdigest()

    def __iter__(self) -> SegmentWindowSampler:
        return self

    def __next__(self) -> Window:
        return self.sample()

    def sample(self) -> Window:
        segment = self.valid_segments[self._rng.randrange(len(self.valid_segments))]
        start = segment.start + self._rng.randrange(segment.complete_window_count)
        self.draw_count += 1
        return Window(segment, start)

    def sample_batch(self, batch_size: int) -> list[Window]:
        _integer(batch_size, "batch_size", minimum=1)
        return [self.sample() for _ in range(batch_size)]

    def iter_evaluation_windows(self) -> Iterator[Window]:
        """Enumerate segment declaration order, then ascending complete starts."""
        for segment in self.valid_segments:
            for start in range(segment.start, segment.end - HORIZON + 1):
                yield Window(segment, start)

    def manifest(self) -> dict[str, Any]:
        count = len(self.valid_segments)
        windows = sum(segment.complete_window_count for segment in self.valid_segments)
        single_clip = len({segment.clip for segment in self.valid_segments}) == 1
        return {
            "schema": MANIFEST_SCHEMA,
            "contract_sha256": self._contract_sha256,
            "frames_per_window": FRAMES_PER_WINDOW,
            "frame_stride": FRAME_STRIDE,
            "horizon": HORIZON,
            "horizon_semantics": HORIZON_SEMANTICS,
            "sampling_unit": "annotation_segment",
            "segment_sampling": "uniform_with_replacement",
            "start_sampling": "uniform_over_complete_windows_in_selected_segment",
            "single_clip_adaptation": single_clip,
            "single_clip_note": (
                "Repeated independent draws from eligible segments in the single clip; "
                "no shuffled-start epoch and no length-one/drop-last DataLoader."
                if single_clip else None
            ),
            "official_epoch_order_reproduced": False,
            "counts": {
                "declared_segments": len(self.segments),
                "eligible_segments": count,
                "filtered_short_segments": len(self.filtered_segments),
                "eligible_clips": len({segment.clip for segment in self.valid_segments}),
                "evaluation_windows": windows,
                "real_future_positions": windows * HORIZON,
                "padding_positions": 0,
            },
            "evaluation_unit": "(segment_id, window_start); overlapping annotations retain distinct identities",
            "evaluation_weighting": "one row per complete window; future-position counts include repeated physical frames",
            "segments": [
                {
                    **segment.to_dict(),
                    "segment_probability": {"numerator": 1, "denominator": count},
                    "start_probability_given_segment": {
                        "numerator": 1,
                        "denominator": segment.complete_window_count,
                    },
                }
                for segment in self.valid_segments
            ],
            "filtered_segments": [
                {**segment.to_dict(), "reason": "fewer_than_ten_real_frames"}
                for segment in self.filtered_segments
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        version, internal, gaussian = self._rng.getstate()
        return {
            "schema": STATE_SCHEMA,
            "contract_sha256": self._contract_sha256,
            "python_implementation": platform.python_implementation(),
            "python_version": list(sys.version_info[:2]),
            "seed": self.seed,
            "draw_count": self.draw_count,
            "rng_state": [version, list(internal), gaussian],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate first; an invalid checkpoint never partially mutates state."""
        if not isinstance(state, Mapping):
            raise ValueError("sampler state must be a mapping")
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported sampler state schema")
        if state.get("contract_sha256") != self._contract_sha256:
            raise ValueError("sampler state belongs to different ordered segments or sampling rules")
        if state.get("python_implementation") != platform.python_implementation():
            raise ValueError("exact sampler restoration requires the same Python implementation")
        if state.get("python_version") != list(sys.version_info[:2]):
            raise ValueError("exact sampler restoration requires the same Python major.minor version")
        seed = _integer(state.get("seed"), "state seed")
        draws = _integer(state.get("draw_count"), "state draw_count")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, (list, tuple)) or len(rng_state) != 3:
            raise ValueError("invalid sampler RNG state")
        if not isinstance(rng_state[1], (list, tuple)):
            raise ValueError("invalid sampler RNG internal state")
        restored = random.Random(0)
        try:
            restored.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid sampler RNG state") from error
        self._rng = restored
        self.seed = seed
        self.draw_count = draws
