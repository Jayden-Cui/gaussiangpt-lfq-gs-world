"""CPU-only negative contract checks; no production binding or data artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bindings as b

ROOT = Path(__file__).resolve().parent
NEW_CLIP = "buildingblock_tower_0_0"


class PoolContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = b.load_pool(ROOT)

    def rejected(self, mutate):
        pool = copy.deepcopy(self.pool)
        mutate(pool)
        with self.assertRaises(RuntimeError):
            b.validate_pool(pool)

    def test_real_audited_pool_has_exact_membership(self):
        self.assertEqual(len(self.pool["clips"]), 50)
        self.assertEqual(sum(r["original_role"] == "TRAIN" for r in self.pool["clips"]), 49)
        self.assertEqual(sum(r["requires_new_full177_export"] for r in self.pool["clips"]), 48)

    def test_historical_dev_role_cannot_be_erased(self):
        def mutate(pool):
            row = next(r for r in pool["clips"] if r["clip_id"] == b.AUTHORIZED_DEV)
            row.update(original_role="TRAIN", role_basis="original_authoritative_TRAIN", source_group_assignment="train")
        self.rejected(mutate)

    def test_unapproved_dev_and_excluded_group_rejected(self):
        self.rejected(lambda p: p["clips"][0].update(original_role="DEV", source_group_assignment="dev"))
        self.rejected(lambda p: p["clips"][0].update(clip_id="buildingblock_tower_14_0", source_group="buildingblock_tower_14"))
        self.rejected(lambda p: p["clips"][0].update(source_group="buildingblock_tower_17"))

    def test_sparse_or_reordered_frames_rejected(self):
        self.rejected(lambda p: p["clips"][0]["planned_native_export"].update(source_frame_indices=list(range(0, 177, 11))))
        self.rejected(lambda p: p["clips"][0]["probe"].update(frame_indices=list(range(177))[::-1]))
        self.rejected(lambda p: p["clips"][0]["probe"].update(frame_count=93))

    def test_full_native_dimensions_and_points_enforced(self):
        self.rejected(lambda p: p["clips"][0]["planned_native_export"].update(gs_per_frame=112895))
        self.rejected(lambda p: p["clips"][0]["planned_native_export"].update(width=224))
        self.rejected(lambda p: p["clips"][0].update(expected_source_shape_HW=[252, 448]))

    def test_pts_gap_duplicate_nan_and_evidence_mismatch_rejected(self):
        for value in (0.0, 0.08, float("nan")):
            with self.subTest(value=value):
                self.rejected(lambda p: p["clips"][0]["probe"]["pts_seconds"].__setitem__(1, value))
        self.rejected(lambda p: p["clips"][0]["probe"]["frame_timestamp_records"][1].update(best_effort_timestamp_time="0.5"))

    def test_fps_and_source_hash_mismatch_rejected(self):
        self.rejected(lambda p: p["clips"][0]["probe"].update(fps=30.0))
        self.rejected(lambda p: p["clips"][0].update(video_sha256="0" * 64))
        self.rejected(lambda p: p["authority"]["source_assignments.json"].update(sha256="0" * 64))

    def test_reuse_cannot_be_promoted_to_new_export(self):
        def mutate(pool):
            next(r for r in pool["clips"] if r["clip_id"] in b.REUSE)["requires_new_full177_export"] = True
        self.rejected(mutate)
        for clip in b.REUSE:
            with self.subTest(clip=clip), self.assertRaises(RuntimeError):
                b.get_clip(ROOT, clip)


class BindingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = b.load_pool(ROOT)
        cls.row = next(r for r in cls.pool["clips"] if r["clip_id"] == NEW_CLIP)

    def declaration(self):
        files, expected, record = b._declared_inputs(ROOT, self.row, self.pool)
        return {"schema": "gs_frozen_codec_transfer_binding_v1", "files": files,
                "expected_teacher": expected, "input_pool_sha256": record["sha256"],
                "verification": b._verification(self.row)}

    def test_declaration_preserves_pinned_dependency_closure(self):
        binding = self.declaration()
        b.validate_binding(ROOT, NEW_CLIP, binding)
        self.assertEqual(len(binding["files"]), 96)
        videos = [r["path"] for r in binding["files"] if r["path"].endswith(".mp4")]
        self.assertEqual(videos, [self.row["video_path"]])

    def test_changed_checkpoint_removed_dependency_and_role_fail(self):
        for alteration in ("checkpoint", "remove", "role", "pool"):
            binding = self.declaration()
            if alteration == "checkpoint":
                binding["expected_teacher"]["checkpoint_sha256"] = "0" * 64
            elif alteration == "remove":
                binding["files"].pop(0)
            elif alteration == "role":
                binding["verification"]["original_role"] = "DEV"
            else:
                binding["input_pool_sha256"] = "0" * 64
            with self.subTest(alteration=alteration), self.assertRaises(RuntimeError):
                b.validate_binding(ROOT, NEW_CLIP, binding)

    def test_duplicate_or_forged_static_dependency_fails(self):
        for duplicate in (False, True):
            binding = self.declaration()
            if duplicate:
                binding["files"][-1] = copy.deepcopy(binding["files"][0])
            else:
                binding["files"][0]["sha256"] = "0" * 64
            with self.subTest(duplicate=duplicate), self.assertRaises(RuntimeError):
                b.validate_binding(ROOT, NEW_CLIP, binding)

    def test_unsafe_clip_path_rejected(self):
        for clip in ("../old", "/tmp/out", "buildingblock_tower_14_0", NEW_CLIP + "/../other"):
            with self.subTest(clip=clip), self.assertRaises(RuntimeError):
                b.clip_root(ROOT, clip)

    def test_missing_source_does_not_write_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [{"path": str(root / "absent-input"), "bytes": 1, "sha256": "0" * 64}]
            with mock.patch.object(b, "verify_manifest"), mock.patch.object(b, "load_pool", return_value=self.pool), \
                 mock.patch.object(b, "_declared_inputs", return_value=(files, {}, {})):
                with self.assertRaisesRegex(RuntimeError, "not ready"):
                    b.build_binding(root, NEW_CLIP)
            self.assertFalse((root / "clips").exists())

    def test_actual_hash_mismatch_does_not_write_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "unit-test-input.txt"
            source.write_bytes(b"CPU test input, not a model artifact")
            files = [{"path": str(source), "bytes": source.stat().st_size, "sha256": "0" * 64}]
            with mock.patch.object(b, "verify_manifest"), mock.patch.object(b, "load_pool", return_value=self.pool), \
                 mock.patch.object(b, "_declared_inputs", return_value=(files, {}, {})):
                with self.assertRaisesRegex(RuntimeError, "hash/bytes mismatch"):
                    b.build_binding(root, NEW_CLIP)
            self.assertFalse((root / "clips").exists())

    def test_existing_output_blocks_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = b.clip_root(root, NEW_CLIP)
            target.mkdir(parents=True)
            sentinel = target / "FAILED.json"
            sentinel.write_text("unit test sentinel", encoding="utf-8")
            with mock.patch.object(b, "verify_manifest"), mock.patch.object(b, "load_pool", return_value=self.pool), \
                 mock.patch.object(b, "_declared_inputs") as inputs:
                with self.assertRaisesRegex(RuntimeError, "Refusing overwrite"):
                    b.build_binding(root, NEW_CLIP)
                inputs.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unit test sentinel")

    def test_reuse_reference_keeps_limitations_and_never_hashes_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Only a copied source pool is placed in the temporary test directory.
            (root / "INPUT_POOL.json").write_bytes((ROOT / "INPUT_POOL.json").read_bytes())
            with mock.patch.object(b, "verify_manifest"), mock.patch.object(b, "fingerprint") as fp, \
                 mock.patch.object(b, "_write_new") as writer:
                for clip in b.REUSE:
                    reference = b.build_reuse_reference(root, clip)
                    self.assertEqual(reference["status"], "reference_only_requires_admission")
                    self.assertFalse(reference["freshly_verified"])
                    self.assertFalse(reference["teacher_or_cache_verified"])
                    self.assertEqual(set(reference["paths_requiring_admission"]), {"teacher", "cache"})
                    self.assertEqual(reference["original_role"], "DEV" if clip == b.AUTHORIZED_DEV else "TRAIN")
                fp.assert_not_called()
                self.assertEqual(writer.call_count, 2)
            self.assertFalse((root / "clips").exists())

    def test_manifest_rejects_tampering_and_unbound_python_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / b.RUN
            root.mkdir()
            # Minimal temporary manifest fixtures; external pins are checked before IO.
            manifest = {name: "0" * 64 for name in
                        ("bindings.py", "test_bindings.py", "prepare_clip.py", "test_prepare_clip.py", "INPUT_POOL.json")}
            manifest.update({b.HELPER_RELATIVE: b.HELPER_SHA256, b.TEMPLATE_RELATIVE: b.TEMPLATE_SHA256})
            (root / "unexpected.py").write_text("# unit test", encoding="utf-8")
            (root / "CODE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing required"):
                b.verify_manifest(root)
            manifest["unexpected.py"] = hashlib.sha256(b"# unit test").hexdigest()
            manifest[b.HELPER_RELATIVE] = "0" * 64
            (root / "CODE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest.*differs"):
                b.verify_manifest(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
