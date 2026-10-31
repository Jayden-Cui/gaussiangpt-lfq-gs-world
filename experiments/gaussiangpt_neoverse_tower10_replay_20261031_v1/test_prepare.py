"""CPU-only wrapper tests: no remote filesystem, CUDA, torch, or teacher export."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tower10_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / prepare.RUN
        self.root.mkdir()
        for name in ("prepare.py", "protocol.json", "input_binding.json"):
            shutil.copyfile(HERE / name, self.root / name)
        self.helper_path = self.root / prepare.HELPER_RELATIVE
        self.helper_path.parent.mkdir(parents=True)
        shutil.copyfile(HERE / prepare.HELPER_RELATIVE, self.helper_path)
        (self.root / "teacher" / prepare.CLIP).mkdir(parents=True)
        self.protocol = prepare.read_json(self.root / "protocol.json")
        self.binding = prepare.read_json(self.root / "input_binding.json")
        self.freeze()

    def freeze(self):
        names = ("prepare.py", "protocol.json", "input_binding.json", prepare.HELPER_RELATIVE)
        self.manifest = {name: prepare.sha256(self.root / name) for name in names}
        self.write(self.root / "CODE_MANIFEST.json", self.manifest)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_valid_bound_config(self):
        prepare.validate_config(self.protocol, self.binding)
        prepare.verify_manifest(self.root)

    def test_fixed_protocol_rejects_changes(self):
        for key, replacement in {
            "schema": "other", "run": "old_run", "clip": "buildingblock_tower_14_0",
            "frames": 17, "height": 504, "width": 896, "stride": 11, "fps": 30.0,
            "source_video_sha256": "0" * 64, "prepare_helper_sha256": "0" * 64,
            "prepare_helper_relative": "../other.py", "codec_root": "/tmp/codec",
            "preparation_seconds": 1201, "source_role": "TEST",
            "official_independent_test_split": True, "automatic_retry": True,
            "training_performed": True, "scientific_transfer_passed": True,
        }.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(self.protocol)
                changed[key] = replacement
                with self.assertRaises(RuntimeError):
                    prepare.validate_config(changed, self.binding)

    def test_binding_identity_and_duplicate_rejected(self):
        changed = copy.deepcopy(self.binding)
        changed["expected_teacher"]["source_video_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "expected_teacher.source_video_sha256"):
            prepare.validate_config(self.protocol, changed)
        changed = copy.deepcopy(self.binding)
        changed["files"].append(changed["files"][0])
        with self.assertRaisesRegex(RuntimeError, "Duplicate binding path"):
            prepare.validate_config(self.protocol, changed)

    def test_every_required_manifest_input_is_checked(self):
        for name in self.manifest:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(RuntimeError, "CODE_MANIFEST mismatch"):
                    prepare.verify_manifest(self.root)
                path.write_bytes(original)
        missing = dict(self.manifest)
        del missing[prepare.HELPER_RELATIVE]
        self.write(self.root / "CODE_MANIFEST.json", missing)
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            prepare.verify_manifest(self.root)

    def test_existing_outputs_refuse_overwrite(self):
        for name in prepare.OUTPUT_NAMES:
            with self.subTest(name=name):
                sentinel = self.root / name
                sentinel.write_text("preserve", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Refusing overwrite"):
                    prepare.assert_fresh(self.root)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
                sentinel.unlink()

    def test_unchanged_helper_import_is_cpu_only_and_read_only(self):
        original = self.helper_path.read_bytes()
        with patch.dict(sys.modules, {"torch": None, "numpy": None}):
            helper = prepare.load_helper(self.root)
        self.assertTrue(callable(helper.main))
        self.assertEqual(helper.CLIP, "buildingblock_tower_14_0")
        self.assertEqual(original, self.helper_path.read_bytes())
        self.assertFalse((self.helper_path.parent / "__pycache__").exists())

    def test_stub_gets_exact_three_constants_and_original_cli(self):
        observed = {}
        stub = types.SimpleNamespace(CLIP="old", RUN="old", SOURCE_SHA256="old", untouched=object())
        original_summary = {"status": "complete", "preparation_contract_passed": True,
                            "scientific_transfer_passed": None, "training_performed": False,
                            "codec_frozen": True, "clip": prepare.CLIP, "frames": 177,
                            "scope": "old scope", "quality": {"sentinel": 1.25}}

        def fake_main():
            observed["argv"] = list(sys.argv)
            self.write(self.root / "SUMMARY.json", original_summary)

        stub.main, stub.write_json = fake_main, self.write
        initial = dict(vars(stub))
        saved_argv = sys.argv
        with patch.object(prepare, "__file__", str(self.root / "prepare.py")), \
                patch.object(prepare, "load_helper", return_value=stub), \
                patch.dict(sys.modules, {"torch": None, "numpy": None}):
            prepare.run(self.root)
        self.assertIs(sys.argv, saved_argv)
        self.assertEqual({key for key in vars(stub) if vars(stub)[key] != initial[key]},
                         {"CLIP", "RUN", "SOURCE_SHA256"})
        self.assertEqual((stub.CLIP, stub.RUN, stub.SOURCE_SHA256),
                         (prepare.CLIP, prepare.RUN, prepare.SOURCE_SHA256))
        self.assertEqual(observed["argv"][1:], [
            "--root", str(self.root.resolve()), "--teacher", str(self.root / "teacher" / prepare.CLIP),
            "--codec-root", prepare.CODEC_ROOT, "--clip", prepare.CLIP, "--max-seconds", "1200"])
        summary = prepare.read_json(self.root / "SUMMARY.json")
        for key in original_summary.keys() - {"scope"}:
            self.assertEqual(summary[key], original_summary[key])
        self.assertEqual(summary["source_role"], "TRAIN")
        self.assertEqual(summary["source_group"], "buildingblock_tower_10")
        self.assertIs(summary["official_independent_test_split"], False)
        self.assertIs(summary["goal_complete"], False)
        self.assertEqual(summary["preparation_helper_scope_original"], original_summary["scope"])
        self.assertEqual(prepare.sha256(self.helper_path), prepare.HELPER_SHA256)

    def test_helper_failure_propagates_and_restores_argv(self):
        def fail():
            raise TimeoutError("stub budget exhausted")

        stub = types.SimpleNamespace(main=fail)
        saved_argv = sys.argv
        with patch.object(prepare, "__file__", str(self.root / "prepare.py")), \
                patch.object(prepare, "load_helper", return_value=stub):
            with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
                prepare.run(self.root)
        self.assertIs(sys.argv, saved_argv)
        self.assertFalse((self.root / "SUMMARY.json").exists())


if __name__ == "__main__":
    unittest.main()
