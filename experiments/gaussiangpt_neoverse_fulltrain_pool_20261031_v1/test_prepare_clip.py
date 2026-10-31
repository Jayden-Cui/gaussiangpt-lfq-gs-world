"""CPU contract tests for the new pool adapter, without codec/vendor execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import prepare_clip as prep


ROOT = Path(__file__).resolve().parent
POOL = json.loads((ROOT / "INPUT_POOL.json").read_text(encoding="utf-8"))
ITEM = next(row for row in POOL["clips"] if row["requires_new_full177_export"])
CLIP = ITEM["clip_id"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def blob(path, value=b"fixture-only"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return prep.fingerprint(path)


def completed_fixture(directory):
    """Tiny mock payloads test admission hashes, never stand in for real tensors."""
    item = copy.deepcopy(ITEM)
    teacher = directory / "teacher" / CLIP
    records = [blob(teacher / name) for name in
               ("provenance.json", "cameras.npz", "input_rgb.npy", "rendered_rgb.npy")]
    records += [blob(teacher / "gaussians" / f"native_group_{t:03d}.npz", f"native-{t}".encode())
                for t in range(prep.FRAMES)]
    source = blob(directory / "fixture_source" / f"{CLIP}.mp4")
    renderer = blob(directory / "fixture_renderer.py")
    empty_initializer = blob(directory / "fixture_vendor" / "__init__.py", b"")
    records += [source, renderer]
    binding = {"files": [source, renderer, empty_initializer], "expected_teacher":
               {"source_video_path": source["path"], "renderer_path": renderer["path"]}}
    save(directory / "input_binding.json", binding)
    binding_record = prep.fingerprint(directory / "input_binding.json")
    save(directory / "inputs_hashes.json", {"schema": "gs_transfer_inputs_v1", "binding": binding_record,
         "static_files": [source, renderer, empty_initializer], "teacher_and_source_files": records,
         "teacher_hashes_computed_after_export": True})
    normal = {"source_frame": 0, "fixed_for_all_frames": True, "uses_future_frames": False,
              "changes_codec_output_std": False, "center": [0.0, 0.0, 0.0], "scale": 0.5}
    save(directory / "normalization.json", normal)
    save(directory / "mappings" / f"{CLIP}.json", {"clip": CLIP, "source_sha256": item["expected_video_sha256"],
         "source_frame_indices": list(range(177)), "model_timestamps": list(range(177)),
         "adjacent_pairs": [[t, t + 1] for t in range(176)], "source_pts_seconds": [t / prep.FPS for t in range(177)]})
    metrics = {key: {"psnr": 20.0, "mae": 0.05} for key in prep.QUALITY_KEYS}
    rows = []
    for t in range(177):
        cache = blob(directory / "cache" / CLIP / f"{t:03d}.pt", f"cache-{t}".encode())
        state = blob(directory / "states" / CLIP / f"{t:03d}.pt", f"state-{t}".encode())
        row = {"clip": CLIP, "frame": t, "source_count": prep.POINTS, "pts_seconds": t / prep.FPS,
               "native_gs_sha256": records[4 + t]["sha256"], "input_voxels": 20,
               "max_decoder_candidates": 80, "decoded_gaussians": 100, "latent_positions": 8,
               "native_render_parity": {"psnr": 100.0, "mae": 0.0},
               "similarity_parity": {"psnr": 100.0, "mae": 0.0}, **copy.deepcopy(metrics),
               "cache_sha256": cache["sha256"], "cache_bytes": cache["bytes"],
               "state_sha256": state["sha256"], "state_bytes": state["bytes"]}
        rows.append(row)
        save(directory / "quality" / CLIP / f"{t:03d}.json", row)
    save(directory / "quality" / f"{CLIP}.json", {"frames": rows, "summary": metrics})
    summary = {"status": "complete", "preparation_contract_passed": True, "scientific_transfer_passed": None,
               "clip": CLIP, "frames": 177, "adjacent_pairs": 176, "source_points_per_frame": prep.POINTS,
               "all_input_points_aggregated": True, "codec_frozen": True,
               "state_quantization": "Identity; continuous float32 Nx64", "training_performed": False,
               "input_binding_sha256": binding_record["sha256"], "normalization": normal, "quality": metrics}
    save(directory / "SUMMARY.json", summary)
    return item, binding_record["sha256"]


class RecipeTests(unittest.TestCase):
    def test_all_48_commands_preserve_full_export(self):
        exported = [row for row in POOL["clips"] if row["requires_new_full177_export"]]
        self.assertEqual(len(exported), 48)
        for row in exported:
            argv = prep.teacher_command(ROOT, row["clip_id"])
            self.assertEqual(argv[:4], [prep.RUNTIME, "-B", "-u", prep.RENDERER])
            options = dict(zip(argv[4::2], argv[5::2]))
            self.assertEqual(options, {"--repo": prep.REPO, "--checkpoint": prep.CHECKPOINT,
                "--input": row["video_path"], "--output": str(ROOT / "clips" / row["clip_id"] / "teacher" / row["clip_id"]),
                "--num-frames": "177", "--start-frame": "0", "--frame-stride": "1", "--fps": "29.97",
                "--height": "252", "--width": "448", "--save-gs": "177", "--novel-view-shift": "0.03",
                "--device": "cuda:0", "--dtype": "bfloat16"})
            self.assertNotIn("--use-motion", argv)

    def test_reuse_and_eval_clips_cannot_get_export_command(self):
        for clip in ("buildingblock_tower_10_176", "buildingblock_tower_17_176", "buildingblock_tower_14_0", "../escape"):
            with self.subTest(clip=clip), self.assertRaises(RuntimeError):
                prep.teacher_command(ROOT, clip)

    def test_adapter_only_changes_three_helper_constants(self):
        helper = SimpleNamespace(CLIP="old", RUN="old", SOURCE_SHA256="old", function=object(), frames=177)
        original = dict(vars(helper))
        prep.bind_helper(helper, ITEM)
        changed = {key for key in original if original[key] != vars(helper)[key]}
        self.assertEqual(changed, {"CLIP", "RUN", "SOURCE_SHA256"})
        self.assertEqual(helper.RUN, CLIP)
        self.assertEqual(helper.SOURCE_SHA256, ITEM["expected_video_sha256"])

    def test_pinned_empty_static_initializer_is_distinct_from_empty_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            path = directory / "__init__.py"
            record = blob(path, b"")
            self.assertEqual(prep.check_record(record)["bytes"], 0)
            with self.assertRaisesRegex(RuntimeError, "Missing/empty"):
                prep.owned_file(path, directory)

    def test_no_overwrite_or_retry_even_without_completed_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prep.assert_fresh(directory)
            for name in prep.OUTPUT_NAMES:
                path = directory / name
                path.write_text("incomplete", encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Refusing overwrite or retry"):
                    prep.assert_fresh(directory)
                path.unlink()

    def test_teacher_sparse_export_and_pts_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            teacher = directory / "teacher" / CLIP
            for t in range(59):
                blob(teacher / "gaussians" / f"native_group_{t:03d}.npz")
            helper = SimpleNamespace(validate_provenance=mock.Mock(return_value=[t / prep.FPS for t in range(177)]))
            with self.assertRaisesRegex(RuntimeError, "no sparse export"):
                prep.admit_teacher(directory, ITEM, {"expected_teacher": {}}, helper)
            helper.validate_provenance.assert_not_called()
            for t in range(59, 177):
                blob(teacher / "gaussians" / f"native_group_{t:03d}.npz")
            for name in ("cameras.npz", "input_rgb.npy", "rendered_rgb.npy"):
                blob(teacher / name)
            save(teacher / "provenance.json", {})
            self.assertEqual(prep.admit_teacher(directory, ITEM, {"expected_teacher": {}}, helper), teacher)
            changed = copy.deepcopy(ITEM)
            changed["probe"]["pts_seconds"][176] += 0.1
            with self.assertRaisesRegex(RuntimeError, "PTS disagrees"):
                prep.admit_teacher(directory, changed, {"expected_teacher": {}}, helper)


class OutputAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.item, self.binding_sha256 = completed_fixture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self):
        return prep.verify_outputs(self.directory, self.item, self.binding_sha256)

    def test_complete_fixture_reports_actual_coverage_and_precise_replay_scope(self):
        result = self.audit()
        self.assertEqual(result["actual_frames"], 177)
        self.assertEqual(result["actual_source_points_total"], 177 * 112896)
        self.assertEqual(result["actual_adjacent_pairs"], 176)
        self.assertTrue(result["checks"]["serialized_coordinates_and_features_equal_in_memory"])
        self.assertFalse(result["checks"]["disk_reloaded_state_rendered_again"])
        self.assertEqual(len(result["output_files"]), 4 + 177 * 3)

    def test_missing_frame_rejected_even_with_complete_summary(self):
        (self.directory / "states" / CLIP / "176.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "states coverage"):
            self.audit()

    def test_actual_cache_hash_mismatch_rejected(self):
        (self.directory / "cache" / CLIP / "044.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
            self.audit()

    def test_same_frame_count_but_reduced_points_rejected(self):
        path = self.directory / "quality" / f"{CLIP}.json"
        quality = prep.read_json(path)
        quality["frames"][44]["source_count"] = 112895
        save(path, quality)
        with self.assertRaisesRegex(RuntimeError, "point count"):
            self.audit()

    def test_other_clip_mapping_and_future_normalization_rejected(self):
        path = self.directory / "mappings" / f"{CLIP}.json"
        mapping = prep.read_json(path)
        mapping["clip"] = "buildingblock_tower_14_0"
        save(path, mapping)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            self.audit()
        mapping["clip"] = CLIP
        save(path, mapping)
        normal_path = self.directory / "normalization.json"
        normal = prep.read_json(normal_path)
        normal["uses_future_frames"] = True
        save(normal_path, normal)
        with self.assertRaisesRegex(RuntimeError, "first frame"):
            self.audit()

    def test_changed_teacher_after_helper_rejected(self):
        (self.directory / "teacher" / CLIP / "gaussians" / "native_group_132.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "hash/size changed"):
            self.audit()


class AttemptTests(unittest.TestCase):
    def test_failure_is_persisted_and_second_attempt_does_not_call_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / prep.RUN
            directory = root / "clips" / CLIP
            directory.mkdir(parents=True)
            for name in ("INPUT_POOL.json", "CODE_MANIFEST.json"):
                save(root / name, {"fixture": True})
            save(directory / "input_binding.json", {"fixture": True})
            helper = SimpleNamespace(CLIP="old", RUN="old", SOURCE_SHA256="old")
            with mock.patch.object(prep, "__file__", str(root / "prepare_clip.py")), \
                 mock.patch.object(prep.bindings, "verify_manifest"), \
                 mock.patch.object(prep.bindings, "get_clip", return_value=ITEM), \
                 mock.patch.object(prep.bindings, "clip_root", return_value=directory), \
                 mock.patch.object(prep.bindings, "validate_binding"), \
                 mock.patch.object(prep, "load_helper", return_value=helper) as loader, \
                 mock.patch.object(prep, "admit_teacher", side_effect=RuntimeError("teacher incomplete")):
                with self.assertRaisesRegex(RuntimeError, "teacher incomplete"):
                    prep.run(root, CLIP)
                self.assertEqual(prep.read_json(directory / "ADAPTER_FAILED.json")["status"], "failed")
                self.assertFalse((directory / "PREPARATION_RECEIPT.json").exists())
                with self.assertRaisesRegex(RuntimeError, "Refusing overwrite or retry"):
                    prep.run(root, CLIP)
                self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
