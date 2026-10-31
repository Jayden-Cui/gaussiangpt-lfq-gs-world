"""CPU cache-loader checks only; run: python -B -m unittest -v test_sequence_data.

All .pt files are tiny synthetic states inside a TemporaryDirectory. No actual
dataset, LFQ model, MinkowskiEngine, renderer, training, or evaluation is used.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import aligned_ldm
from sampling import Segment, Window
from sequence_data import load_record, load_window


class SequenceDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gs_ldm_sequence_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.clip = "synthetic_tower"
        self.window_spec = Window(Segment(self.clip, 0, 176, "tower"), 167)
        self.folder = self.root / "states" / self.clip
        self.folder.mkdir(parents=True)
        # Nonuniform, monotonic times deliberately disallow inferred FPS or
        # replacement by frame/29.97. source_pts deliberately differs from pts.
        self.times = {frame: 50.0 + index * .04 + index * index * .001
                      for index, frame in enumerate(self.window_spec.frames)}
        self.sources = {frame: 5000.0 + frame for frame in self.window_spec.frames}
        for frame in self.window_spec.frames:
            self.write(frame)

    def state(self, frame):
        return {
            "coordinates": torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.int32),
            "features": torch.full((3, 64), float(frame), dtype=torch.float32),
            "pts": self.times[frame],
            "source_pts": self.sources[frame],
            "clip": self.clip,
            "frame": frame,
            "is_padding": False,
            "metadata": {"note": "synthetic", "nested_tensor": torch.tensor([frame])},
        }

    def write(self, cache_frame, **updates):
        state = self.state(cache_frame)
        state.update(updates)
        torch.save(state, self.folder / f"{cache_frame:03d}.pt")

    def test_last_complete_window_preserves_frames_times_sources_and_cpu_load_options(self):
        original_load = torch.load
        with patch("sequence_data.torch.load", wraps=original_load) as loader:
            states = load_window(self.root, self.window_spec, "cpu")
        self.assertEqual(len(states), 10)
        self.assertEqual([state["frame_index"] for state in states], list(range(167, 177)))
        self.assertEqual([state["pts"] for state in states], list(self.times.values()))
        self.assertEqual([state["source_pts"] for state in states], list(self.sources.values()))
        self.assertEqual([call.args[0].name for call in loader.call_args_list],
                         [f"{frame:03d}.pt" for frame in range(167, 177)])
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs, {"weights_only": True, "map_location": "cpu"})
        for frame, state in zip(range(167, 177), states):
            self.assertEqual(state["coordinates"].shape, (3, 3))
            self.assertEqual(state["features"].shape, (3, 64))
            self.assertEqual(state["features"].device.type, "cpu")
            self.assertEqual(state["metadata"]["nested_tensor"].device.type, "cpu")
            torch.testing.assert_close(state["features"], self.state(frame)["features"])
        # The loader's added frame_index must not rewrite the source cache.
        self.assertNotIn("frame_index", original_load(self.folder / "167.pt", weights_only=True))

    def test_missing_frame_fails_at_that_frame_without_retry_or_replication(self):
        (self.folder / "171.pt").unlink()
        original_load = torch.load
        with patch("sequence_data.torch.load", wraps=original_load) as loader:
            with self.assertRaises(FileNotFoundError):
                load_window(self.root, self.window_spec, "cpu")
        self.assertEqual([call.args[0].name for call in loader.call_args_list],
                         ["167.pt", "168.pt", "169.pt", "170.pt", "171.pt"])

    def test_equal_or_decreasing_pts_fail_without_synthesis(self):
        for invalid in (self.times[170], self.times[170] - .1):
            with self.subTest(pts=invalid):
                self.write(171, pts=invalid)
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    load_window(self.root, self.window_spec, "cpu")
                stored = torch.load(self.folder / "171.pt", weights_only=True)
                self.assertEqual(stored["pts"], invalid)
                self.assertEqual(stored["source_pts"], self.sources[171])

    def test_nonfinite_or_missing_or_nonscalar_pts_fail(self):
        for invalid in (float("nan"), float("inf"), "50.0", True, torch.tensor([50.0])):
            with self.subTest(pts=invalid):
                self.write(167, pts=invalid)
                with self.assertRaisesRegex(ValueError, "pts"):
                    load_window(self.root, self.window_spec, "cpu")
        state = self.state(167)
        state.pop("pts")
        torch.save(state, self.folder / "167.pt")
        with self.assertRaisesRegex(ValueError, "Missing state fields"):
            load_window(self.root, self.window_spec, "cpu")

    def test_old_padding_is_rejected_even_when_its_pts_increases(self):
        for flag in (True, torch.tensor(True)):
            with self.subTest(flag=flag):
                self.write(176, is_padding=flag)
                with self.assertRaisesRegex(ValueError, "Padded cached state"):
                    load_window(self.root, self.window_spec, "cpu")

    def test_absent_source_pts_or_padding_flag_is_not_synthesized(self):
        state = self.state(167)
        state.pop("source_pts")
        state.pop("is_padding")
        torch.save(state, self.folder / "167.pt")
        loaded = load_window(self.root, self.window_spec, "cpu")[0]
        self.assertNotIn("source_pts", loaded)
        self.assertNotIn("is_padding", loaded)
        self.assertEqual(loaded["pts"], state["pts"])

    def test_invalid_coordinates_and_feature_shapes_fail(self):
        invalid_fields = [
            {"coordinates": torch.zeros((3, 2), dtype=torch.int32)},
            {"coordinates": torch.zeros((3, 3), dtype=torch.float32)},
            {"coordinates": torch.empty((0, 3), dtype=torch.int32)},
            {"coordinates": torch.zeros((3, 3), dtype=torch.int32)},
            {"features": torch.zeros((3, 63))},
            {"features": torch.zeros((2, 64))},
            {"features": torch.ones((3, 64), dtype=torch.int64)},
            {"features": torch.full((3, 64), float("nan"))},
        ]
        for updates in invalid_fields:
            with self.subTest(updates=list(updates)):
                self.write(167, **updates)
                with self.assertRaises(ValueError):
                    load_window(self.root, self.window_spec, "cpu")

    def test_existing_frame_or_clip_identity_mismatch_is_rejected(self):
        for updates in ({"frame": 166}, {"frame_index": 168}, {"clip": "another_clip"}):
            with self.subTest(updates=updates):
                self.write(167, **updates)
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    load_window(self.root, self.window_spec, "cpu")

    def test_condition_uses_only_initial_state_and_ignores_changed_future_contents(self):
        with patch("aligned_ldm.make_condition", wraps=aligned_ldm.make_condition) as condition_builder:
            first = load_record(self.root, self.window_spec, "cpu", radius=1, max_candidates=100)
        self.assertEqual(condition_builder.call_count, 1)
        self.assertIs(condition_builder.call_args.args[0], first["window"][0])
        self.assertEqual(len(condition_builder.call_args.args), 1)
        self.assertEqual(condition_builder.call_args.kwargs, {"radius": 1, "max_candidates": 100})
        self.assertIs(first["condition"].state, first["window"][0])
        for frame in self.window_spec.future_frames:
            # Distant, differently-sized target supports deliberately lie far
            # outside the initial candidate domain; loading must still succeed.
            coordinates = torch.tensor([[10000 + i, frame, 0] for i in range(5)], dtype=torch.int64)
            self.write(frame, coordinates=coordinates, features=torch.full((5, 64), -77.0))
        second = load_record(self.root, self.window_spec, "cpu", radius=1, max_candidates=100)
        for name in ("coordinates", "neighbors", "weights"):
            torch.testing.assert_close(getattr(first["condition"], name), getattr(second["condition"], name), rtol=0, atol=0)
        torch.testing.assert_close(first["condition"].state["features"], second["condition"].state["features"], rtol=0, atol=0)
        self.assertEqual(second["window"][1]["coordinates"].shape[0], 5)
        self.assertLess(int(second["condition"].coordinates.max()), 10000)

    def test_record_identity_and_candidate_budget_are_forwarded(self):
        record = load_record(self.root, self.window_spec, "cpu", radius=0, max_candidates=3)
        self.assertEqual(record["frames"], list(range(167, 177)))
        self.assertEqual(record["window_id"], self.window_spec.window_id)
        self.assertEqual(record["identity"], self.window_spec.to_dict())
        self.assertEqual(len(record["condition"].coordinates), 3)
        with self.assertRaisesRegex(ValueError, "Candidate budget exceeded"):
            load_record(self.root, self.window_spec, "cpu", radius=0, max_candidates=2)


if __name__ == "__main__":
    unittest.main()
