"""CPU-only invariants for complete-window sampling; no frequency thresholds.

Run from this directory: python -B -m unittest -v test_sampling
"""

import copy
import json
import unittest
from unittest.mock import patch

from sampling import (
    FRAME_STRIDE,
    FRAMES_PER_WINDOW,
    HORIZON,
    MANIFEST_SCHEMA,
    STATE_SCHEMA,
    WINDOW_SCHEMA,
    Segment,
    SegmentWindowSampler,
    Window,
)


class ScriptedRandom:
    """Select exact categorical outcomes and expose their domains, not rates."""

    def __init__(self, choices):
        self.choices = iter(choices)
        self.domains = []

    def randrange(self, stop):
        self.domains.append(stop)
        result = next(self.choices)
        if not 0 <= result < stop:
            raise AssertionError("scripted outcome is outside the requested domain")
        return result


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.clip = Segment("buildingblock_tower_17_176", 0, 176, "tower")

    def test_real_177_frame_boundary_counts_and_frame_weights(self):
        sampler = SegmentWindowSampler([self.clip], seed=20261030)
        windows = list(sampler.iter_evaluation_windows())
        self.assertEqual((FRAMES_PER_WINDOW, FRAME_STRIDE, HORIZON), (10, 1, 9))
        self.assertEqual(len(windows), 168)
        self.assertEqual([w.start for w in windows], list(range(168)))
        self.assertEqual(windows[0].frames, tuple(range(10)))
        self.assertEqual(windows[-1].frames, tuple(range(167, 177)))
        self.assertEqual(sum(len(w.future_frames) for w in windows), 1512)
        self.assertEqual(sum(176 in w.future_frames for w in windows), 1)
        self.assertEqual(sum(100 in w.future_frames for w in windows), 9)
        for window in windows:
            self.assertEqual(len(set(window.frames)), 10)
            self.assertTrue(all(b - a == 1 for a, b in zip(window.frames, window.frames[1:])))
            self.assertLessEqual(window.end, self.clip.end)
        counts = sampler.manifest()["counts"]
        self.assertEqual(counts["evaluation_windows"], 168)
        self.assertEqual(counts["real_future_positions"], 1512)
        self.assertEqual(counts["padding_positions"], 0)

    def test_last_legal_training_start_is_selectable(self):
        rng = ScriptedRandom([0, 167])
        with patch("sampling.random.Random", return_value=rng):
            sampler = SegmentWindowSampler([self.clip], seed=4)
        window = sampler.sample()
        self.assertEqual(rng.domains, [1, 168])
        self.assertEqual((window.start, window.end), (167, 176))

    def test_nonzero_inclusive_segment_and_exactly_ten_frames(self):
        segment = Segment("offset_clip", 30, 39)
        sampler = SegmentWindowSampler([segment], seed=9)
        self.assertEqual(segment.frame_count, 10)
        self.assertEqual(segment.complete_window_count, 1)
        self.assertEqual([w.frames for w in sampler.iter_evaluation_windows()], [tuple(range(30, 40))])
        self.assertEqual([w.start for w in sampler.sample_batch(4)], [30] * 4)

    def test_short_segments_are_reported_and_filtered_before_sampling(self):
        segments = [Segment("short", 5, 13), self.clip, Segment("single_frame", 8, 8)]
        sampler = SegmentWindowSampler(segments, seed=2)
        self.assertEqual(sampler.valid_segments, (self.clip,))
        self.assertEqual(len(sampler.filtered_segments), 2)
        manifest = sampler.manifest()
        self.assertEqual(manifest["counts"]["declared_segments"], 3)
        self.assertEqual(manifest["counts"]["eligible_segments"], 1)
        self.assertEqual(manifest["counts"]["filtered_short_segments"], 2)
        self.assertEqual([s["frame_count"] for s in manifest["filtered_segments"]], [9, 1])
        self.assertTrue(all(w.segment_id == "tower" for w in sampler.sample_batch(24)))

    def test_no_eligible_segments_fails_without_padding(self):
        for segments in ([], [Segment("short", 0, 8)], [Segment("one", 7, 7)]):
            with self.subTest(segments=segments):
                with self.assertRaisesRegex(ValueError, "no eligible segment"):
                    SegmentWindowSampler(segments, seed=1)

    def test_different_lengths_use_equal_segment_domains_then_local_start_domains(self):
        # First draw selects the one-window segment; second selects the last
        # window in the four-window segment. No empirical frequency tolerance.
        short = Segment("a", 20, 29, "short_valid")
        long = Segment("b", 100, 112, "long_valid")
        rng = ScriptedRandom([0, 0, 1, 3])
        with patch("sampling.random.Random", return_value=rng):
            sampler = SegmentWindowSampler([short, long], seed=1)
        selected = sampler.sample_batch(2)
        self.assertEqual(rng.domains, [2, 1, 2, 4])
        self.assertEqual([(w.segment_id, w.start, w.end) for w in selected],
                         [("short_valid", 20, 29), ("long_valid", 103, 112)])
        rows = sampler.manifest()["segments"]
        self.assertEqual([row["segment_probability"] for row in rows],
                         [{"numerator": 1, "denominator": 2}] * 2)
        self.assertEqual([row["start_probability_given_segment"]["denominator"] for row in rows], [1, 4])

    def test_overlapping_annotations_keep_distinct_window_identities(self):
        left = Segment("same_clip", 0, 9, "annotation_a")
        right = Segment("same_clip", 0, 9, "annotation_b")
        windows = list(SegmentWindowSampler([left, right], seed=1).iter_evaluation_windows())
        self.assertEqual(windows[0].frames, windows[1].frames)
        self.assertNotEqual(windows[0].window_id, windows[1].window_id)

    def test_duplicate_segment_identity_fails_even_for_filtered_segments(self):
        for second in (Segment("other", 0, 20, "tower"), Segment("short", 0, 1, "tower")):
            with self.assertRaisesRegex(ValueError, "segment_id"):
                SegmentWindowSampler([self.clip, second], seed=1)

    def test_same_seed_and_batch_partition_produce_same_stream(self):
        segments = [self.clip, Segment("second", 40, 60)]
        first = SegmentWindowSampler(segments, seed=42)
        second = SegmentWindowSampler(segments, seed=42)
        expected = first.sample_batch(31)
        actual = second.sample_batch(8) + [next(second)] + second.sample_batch(22)
        self.assertEqual(expected, actual)
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_json_checkpoint_resumes_exact_next_windows_and_seed(self):
        segments = [Segment("filtered", 0, 3), self.clip, Segment("second", 40, 79)]
        original = SegmentWindowSampler(segments, seed=71)
        original.sample_batch(7)
        snapshot = json.loads(json.dumps(original.state_dict()))
        expected = original.sample_batch(37)
        restored = SegmentWindowSampler(segments, seed=999)
        restored.load_state_dict(snapshot)
        self.assertEqual(restored.seed, 71)
        self.assertEqual(restored.draw_count, 7)
        self.assertEqual(restored.sample_batch(37), expected)
        self.assertEqual(restored.state_dict(), original.state_dict())

    def test_state_rejects_changed_or_reordered_segment_contract(self):
        second = Segment("second", 30, 49)
        state = SegmentWindowSampler([self.clip, second], seed=7).state_dict()
        for segments in ([second, self.clip], [Segment(self.clip.clip, 0, 175, "tower"), second]):
            sampler = SegmentWindowSampler(segments, seed=7)
            before = sampler.state_dict()
            with self.assertRaisesRegex(ValueError, "different ordered segments"):
                sampler.load_state_dict(state)
            self.assertEqual(sampler.state_dict(), before)

    def test_failed_restore_is_atomic(self):
        sampler = SegmentWindowSampler([self.clip], seed=7)
        sampler.sample_batch(3)
        before = sampler.state_dict()
        mutations = {
            "schema": "wrong_schema",
            "draw_count": -1,
            "seed": True,
            "rng_state": [3, [0], None],
            "python_version": [0, 0],
            "python_implementation": "different_runtime",
        }
        for key, value in mutations.items():
            broken = copy.deepcopy(before)
            broken[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    sampler.load_state_dict(broken)
                self.assertEqual(sampler.state_dict(), before)

    def test_evaluation_and_manifest_do_not_advance_rng(self):
        sampler = SegmentWindowSampler([self.clip], seed=12)
        sampler.sample_batch(5)
        before = sampler.state_dict()
        list(sampler.iter_evaluation_windows())
        sampler.manifest()
        self.assertEqual(sampler.state_dict(), before)

    def test_window_and_manifest_exports_have_unambiguous_semantics(self):
        sampler = SegmentWindowSampler([self.clip], seed=3)
        exported = Window(self.clip, 167).to_dict()
        self.assertEqual(exported["schema"], WINDOW_SCHEMA)
        self.assertEqual(exported["initial_frame"], 167)
        self.assertEqual(exported["future_frames"], list(range(168, 177)))
        self.assertEqual(exported["future_offsets"], list(range(1, 10)))
        self.assertEqual(exported["valid_future_mask"], [True] * 9)
        self.assertIn("same initial frame", exported["horizon_semantics"])
        manifest = sampler.manifest()
        self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
        self.assertTrue(manifest["single_clip_adaptation"])
        self.assertFalse(manifest["official_epoch_order_reproduced"])
        self.assertEqual(sampler.state_dict()["schema"], STATE_SCHEMA)
        self.assertEqual(json.loads(json.dumps(exported)), exported)
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_invalid_segments_window_starts_and_batch_sizes(self):
        for args in (("", 0, 9), ("x", 10, 9), ("x", -1, 9), ("x", True, 9), ("x", 0, 9.0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    Segment(*args)
        for start in (-1, 168, 175, True):
            with self.assertRaises(ValueError):
                Window(self.clip, start)
        sampler = SegmentWindowSampler([self.clip], seed=1)
        before = sampler.state_dict()
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                sampler.sample_batch(size)
        self.assertEqual(sampler.state_dict(), before)
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            SegmentWindowSampler([self.clip], seed=1, frames_per_window=9)


if __name__ == "__main__":
    unittest.main()
