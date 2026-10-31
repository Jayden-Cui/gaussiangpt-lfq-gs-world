"""Only CPU fake-codec tests: python -B -m unittest -v test_online_state."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from online_state import ONLINE_SCHEMA, load_online_window
from sampling import Segment, Window


class FakeCodec(nn.Module):
    """Tiny trainable adapter matching the real StateCodec.encode_state API."""

    def __init__(self, *, encoded_stride=None, encoded_grid=None):
        super().__init__()
        self.grid = .008
        self.ae = SimpleNamespace(stride=4)
        self.weight = nn.Parameter(torch.linspace(.1, 1.0, 64))
        self.encoded_stride = encoded_stride
        self.encoded_grid = encoded_grid
        self.calls = []
        self.outputs = []

    def encode_state(self, representatives, set_statistics):
        self.calls.append((representatives, set_statistics))
        coordinates = torch.round(representatives[:, :3] / self.grid).to(torch.int32) // self.ae.stride
        base = representatives[:, 3:4] + set_statistics[:, :1]
        features = base * self.weight[None]
        self.outputs.append(features)
        return {
            "coordinates": coordinates,
            "features": features,
            "stride": self.ae.stride if self.encoded_stride is None else self.encoded_stride,
            "grid": self.grid if self.encoded_grid is None else self.encoded_grid,
        }


class OnlineStateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gs_ldm_online_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.state_root = self.root / "state"
        self.motion_root = self.root / "motion"
        self.clip = "synthetic_tower"
        self.spec = Window(Segment(self.clip, 0, 176, "tower"), 167)
        self.raw_dir = self.state_root / "cache" / self.clip
        self.motion_dir = self.motion_root / "states" / self.clip
        self.raw_dir.mkdir(parents=True)
        self.motion_dir.mkdir(parents=True)
        self.times = {frame: 20.0 + index * .04 + index * index * .001
                      for index, frame in enumerate(self.spec.frames)}
        for frame in self.spec.frames:
            self.write_raw(frame)
            self.write_motion(frame)

    def raw(self, frame):
        index = frame - self.spec.start
        coordinates = torch.tensor([[0, 0, 0], [4, 0, 0], [0, 4, 0]], dtype=torch.int32)
        reps = torch.zeros((3, 14))
        reps[:, :3] = coordinates.float() * .008
        reps[:, 3:6] = .2 + index * .01
        reps[:, 6:9] = .001
        reps[:, 9] = 1.0
        reps[:, 13] = .5
        return {
            "representatives": reps,
            "coordinates": coordinates,
            "set_statistics": torch.full((3, 29), .1 + index * .02),
            "source_indices": torch.arange(3),
            "counts": torch.ones(3, dtype=torch.int64),
            "n_input": 3,
            "cameras": torch.eye(4)[None].repeat(2, 1, 1),
            "intrinsics": torch.tensor([.8, 1.2, .5, .5]),
            # The pseudo-view target intentionally exceeds 1: preserve it,
            # do not silently clamp or normalize any teacher supervision.
            "targets": torch.full((2, 2, 3, 3), 1.25 + index * .01),
            "real": torch.full((2, 3, 3), index / 10),
            "near": .001,
            "far": 1000.0,
            "normalization": {"scale": .5, "fixed_for_all_frames": True},
            "clip": self.clip,
            "frame": frame,
            "metrics": {"pts_seconds": self.times[frame], "note": "synthetic"},
        }

    def write_raw(self, cache_frame, **changes):
        value = self.raw(cache_frame)
        value.update(changes)
        torch.save(value, self.raw_dir / f"{cache_frame:03d}.pt")

    def write_motion(self, cache_frame, **changes):
        value = {
            "pts": self.times[cache_frame], "source_pts": 8000.0 + cache_frame,
            "clip": self.clip, "frame": cache_frame,
            # Both old latent fields are intentionally unusable. Only metadata
            # may be read from this file, so these must not affect online output.
            "coordinates": torch.tensor([[90000, 90000, 90000]], dtype=torch.int32),
            "features": torch.full((1, 64), -999999.0),
            "reconstruction": torch.full((2, 2, 3, 3), -999.0),
        }
        value.update(changes)
        torch.save(value, self.motion_dir / f"{cache_frame:03d}.pt")

    def load(self, codec=None, **kwargs):
        return load_online_window(self.state_root, self.motion_root, self.spec,
                                  codec or FakeCodec(), "cpu", **kwargs)

    def test_all_ten_frames_reach_codec_parameter_gradients(self):
        codec = FakeCodec()
        record = self.load(codec)
        self.assertEqual(len(codec.calls), 10)
        per_frame_gradients = []
        for index, state in enumerate(record["window"]):
            self.assertIs(state["features"], codec.outputs[index])
            self.assertTrue(state["features"].requires_grad)
            self.assertIsNotNone(state["features"].grad_fn)
            state["features"].retain_grad()
            gradient, = torch.autograd.grad(state["features"].sum(), codec.weight, retain_graph=True)
            self.assertTrue(bool((gradient != 0).all()))
            per_frame_gradients.append(gradient)
        sum(state["features"].sum() for state in record["window"]).backward()
        torch.testing.assert_close(codec.weight.grad, torch.stack(per_frame_gradients).sum(0))
        for state in record["window"]:
            self.assertIsNotNone(state["features"].grad)
            self.assertTrue(bool((state["features"].grad != 0).all()))

    def test_fresh_features_ignore_wrong_old_latent_and_follow_encoder_updates(self):
        codec = FakeCodec()
        first = self.load(codec)
        for state in first["window"]:
            self.assertEqual(state["features"].shape, (3, 64))
            self.assertTrue(bool((state["features"] > 0).all()))
            self.assertEqual(state["coordinates"].shape, (3, 3))
            self.assertLess(int(state["coordinates"].max()), 90000)
        first_values = [state["features"].detach().clone() for state in first["window"]]
        with torch.no_grad():
            codec.weight.add_(1.0)
        second = self.load(codec)
        self.assertEqual(len(codec.calls), 20)
        self.assertTrue(all(not torch.equal(old, new["features"])
                            for old, new in zip(first_values, second["window"])))

    def test_stride_grid_coordinates_and_final_window_identity(self):
        codec = FakeCodec()
        result = self.load(codec)
        self.assertEqual(result["schema"], ONLINE_SCHEMA)
        self.assertEqual(result["frames"], list(range(167, 177)))
        self.assertEqual(len(set(result["frames"])), 10)
        self.assertEqual(result["identity"], self.spec.to_dict())
        self.assertIs(result["initial"], result["window"][0])
        expected = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.int32)
        for frame, state in zip(self.spec.frames, result["window"]):
            torch.testing.assert_close(state["coordinates"], expected)
            self.assertEqual((state["stride"], state["grid"], state["coordinate_scale"]), (4, .008, .032))
            self.assertEqual(state["frame_index"], frame)
            self.assertEqual(state["pts"], self.times[frame])
            self.assertEqual(state["source_pts"], 8000.0 + frame)

    def test_supervision_values_preserved_and_reads_are_safe_cpu_only(self):
        original_load = torch.load
        codec = FakeCodec().eval()
        with patch("online_state.torch.load", wraps=original_load) as loader:
            result = self.load(codec)
        self.assertFalse(codec.training)
        self.assertEqual(loader.call_count, 20)
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs, {"weights_only": True, "map_location": "cpu"})
        for frame, record in zip(self.spec.frames, result["records"]):
            expected = self.raw(frame)
            self.assertEqual(record.keys(), expected.keys())
            for name, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(record[name], value, rtol=0, atol=0)
                else:
                    self.assertEqual(record[name], value)
        self.assertIn("caller_manifest_binding_required", result["source_verification"])
        self.assertIn("does not verify hashes", result["source_verification"])

    def test_missing_raw_frame_never_falls_back_to_old_latent(self):
        (self.raw_dir / "167.pt").unlink()
        codec = FakeCodec()
        with self.assertRaises(FileNotFoundError):
            self.load(codec)
        self.assertEqual(len(codec.calls), 0)

    def test_missing_raw_representatives_or_statistics_rejected(self):
        for missing in ("representatives", "set_statistics"):
            with self.subTest(missing=missing):
                payload = self.raw(167)
                payload.pop(missing)
                torch.save(payload, self.raw_dir / "167.pt")
                codec = FakeCodec()
                with self.assertRaisesRegex(ValueError, "Missing raw GS cache fields"):
                    self.load(codec)
                self.assertEqual(len(codec.calls), 0)

    def test_mismatched_raw_grid_or_encoded_metadata_rejected(self):
        for codec in (FakeCodec(encoded_stride=8), FakeCodec(encoded_grid=.016)):
            with self.subTest(codec=(codec.encoded_stride, codec.encoded_grid)):
                with self.assertRaisesRegex(ValueError, "Encoded (stride|grid)"):
                    self.load(codec)
        wrong_grid_codec = FakeCodec()
        wrong_grid_codec.grid = .016
        with self.assertRaisesRegex(ValueError, "Raw voxel coordinates"):
            self.load(wrong_grid_codec)
        self.assertEqual(len(wrong_grid_codec.calls), 0)
        self.write_raw(167, grid=.016)
        with self.assertRaisesRegex(ValueError, "Raw cache grid"):
            self.load()

    def test_bad_raw_shapes_and_nonfinite_statistics_rejected(self):
        for update in (
            {"representatives": torch.zeros((3, 13))},
            {"set_statistics": torch.zeros((3, 28))},
            {"set_statistics": torch.full((3, 29), float("nan"))},
        ):
            with self.subTest(fields=list(update)):
                self.write_raw(167, **update)
                with self.assertRaises(ValueError):
                    self.load()

    def test_mapping_is_explicit_and_does_not_read_motion_files(self):
        mapping = {frame: {"pts": self.times[frame] + 100, "source_pts": -frame}
                   for frame in self.spec.frames}
        original_load = torch.load
        with patch("online_state.torch.load", wraps=original_load) as loader:
            result = load_online_window(self.state_root, None, self.spec, FakeCodec(),
                                        "cpu", pts_by_frame=mapping)
        self.assertEqual(loader.call_count, 10)
        self.assertTrue(all(call.args[0].parent == self.raw_dir for call in loader.call_args_list))
        self.assertEqual([state["pts"] for state in result["window"]],
                         [mapping[frame]["pts"] for frame in self.spec.frames])
        self.assertTrue(all(source["time_source"]["kind"] == "caller_explicit_mapping"
                            for source in result["sources"]))
        mapping.pop(167)
        with self.assertRaisesRegex(ValueError, "mapping is missing frame"):
            self.load(pts_by_frame=mapping)
        with self.assertRaisesRegex(ValueError, "mapping is required"):
            load_online_window(self.state_root, None, self.spec, FakeCodec())

    def test_no_time_synthesis_or_padding_acceptance(self):
        self.write_motion(168, pts=self.times[167])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.load()
        self.write_motion(168)
        self.write_motion(167, is_padding=True)
        with self.assertRaisesRegex(ValueError, "Padded source"):
            self.load()
        self.write_motion(167)
        self.write_raw(167, is_padding=True)
        with self.assertRaisesRegex(ValueError, "Padded source"):
            self.load()

    def test_initial_state_and_condition_are_independent_of_future_raw_contents(self):
        from aligned_ldm import make_condition

        first = self.load()
        condition_before = make_condition(first["initial"], radius=0, max_candidates=3)
        for frame in self.spec.future_frames:
            raw = self.raw(frame)
            shifted = raw["coordinates"] + 4000
            representatives = raw["representatives"].clone()
            representatives[:, :3] = shifted.float() * .008
            representatives[:, 3:6] = 900.0
            self.write_raw(frame, representatives=representatives, coordinates=shifted,
                           set_statistics=torch.full((3, 29), 700.0))
        second = self.load()
        condition_after = make_condition(second["initial"], radius=0, max_candidates=3)
        for name in ("coordinates", "neighbors", "weights"):
            torch.testing.assert_close(getattr(condition_before, name), getattr(condition_after, name), rtol=0, atol=0)
        torch.testing.assert_close(first["initial"]["features"], second["initial"]["features"], rtol=0, atol=0)
        self.assertGreater(int(second["window"][1]["coordinates"].min()), 100)
        self.assertIs(condition_after.state, second["window"][0])


if __name__ == "__main__":
    unittest.main()
