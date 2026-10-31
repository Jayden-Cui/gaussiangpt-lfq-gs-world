"""Bind the unchanged frozen-codec preparer to one complete TRAIN clip.

This adapter performs data preparation only. It does not train a model or
establish generalization, an official holdout, or completion of a later goal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
RUN = "gaussiangpt_neoverse_tower10_replay_20261031_v1"
CLIP = "buildingblock_tower_10_176"
SOURCE_SHA256 = "a8d76842e1d7de31cb8bedfe437264a28b3203c9f3bfbaebf2a6f0864a051dab"
HELPER_RELATIVE = "../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py"
HELPER_SHA256 = "3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85"
CODEC_ROOT = "/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1/state"
OUTPUT_NAMES = ("cache", "states", "SUMMARY.json", "FAILED.json", "progress.json",
                "inputs_hashes.json", "normalization.json", "quality", "mappings")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(protocol, binding):
    """Validate local declarations; the unchanged helper rehashes remote inputs."""
    fixed = {
        "schema": "gs_tower10_full_preparation_v1", "run": RUN, "clip": CLIP,
        "source_group": "buildingblock_tower_10", "source_role": "TRAIN",
        "official_independent_test_split": False,
        "frames": 177, "height": 252, "width": 448, "stride": 1, "fps": 29.97,
        "source_video_sha256": SOURCE_SHA256,
        "prepare_helper_relative": HELPER_RELATIVE, "prepare_helper_sha256": HELPER_SHA256,
        "codec_root": CODEC_ROOT, "preparation_seconds": 1200,
        "automatic_retry": False, "training_performed": False,
        "scientific_transfer_passed": None, "goal_complete": False,
    }
    for key, value in fixed.items():
        require(key in protocol and type(protocol[key]) is type(value) and protocol[key] == value,
                f"protocol.{key} must be {value!r}")
    require(binding.get("schema") == "gs_frozen_codec_transfer_binding_v1",
            "Unexpected input_binding schema")
    expected = binding.get("expected_teacher", {})
    for key, value in {"clip": CLIP, "frames": 177, "height": 252, "width": 448,
                       "frame_stride": 1, "use_motion": False,
                       "source_video_sha256": protocol["source_video_sha256"]}.items():
        require(type(expected.get(key)) is type(value) and expected[key] == value,
                f"input_binding.expected_teacher.{key} must be {value!r}")
    source = PurePosixPath(expected.get("source_video_path", ""))
    require(source.is_absolute() and source.name == CLIP + ".mp4",
            "Binding must name the original target video")
    rows = binding.get("files")
    require(isinstance(rows, list) and bool(rows), "Binding must contain files")
    paths = [row.get("path") for row in rows]
    require(len(set(paths)) == len(paths), "Duplicate binding path")
    matching = [row for row in rows if row.get("path") == str(source)]
    require(len(matching) == 1 and matching[0].get("sha256") == SOURCE_SHA256
            and matching[0].get("bytes") == 652682, "Original video static binding differs")
    verification = binding.get("verification", {})
    require(verification.get("source_role") == "TRAIN"
            and verification.get("source_group") == "buildingblock_tower_10"
            and verification.get("official_independent_test_split") is False,
            "Binding must preserve the original TRAIN source identity")


def verify_manifest(root):
    """Validate the frozen local manifest, including the exact external helper."""
    manifest = read_json(root / "CODE_MANIFEST.json")
    require(isinstance(manifest, dict) and manifest, "CODE_MANIFEST must map paths to SHA256")
    required = {"prepare.py", "protocol.json", "input_binding.json", HELPER_RELATIVE}
    require(required.issubset(manifest), "CODE_MANIFEST missing required preparation inputs")
    require(manifest[HELPER_RELATIVE] == HELPER_SHA256, "Manifest helper SHA256 differs")
    for relative, digest in manifest.items():
        require(isinstance(relative, str) and isinstance(digest, str)
                and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Invalid CODE_MANIFEST entry")
        # Other external helpers may be bound by the queue. Read them only.
        require(not PurePosixPath(relative).is_absolute() and "\\" not in relative,
                f"Manifest paths must be relative POSIX paths: {relative}")
        path = (root / relative).resolve(strict=True)
        require(path.is_file() and sha256(path) == digest, f"CODE_MANIFEST mismatch: {relative}")
    return manifest


def assert_fresh(root):
    for name in OUTPUT_NAMES:
        path = root / name
        require(not path.exists() and not path.is_symlink(),
                f"Refusing overwrite or automatic retry: {path}")


def load_helper(root):
    """Execute only the bytes just checked; never generate bytecode in the old run."""
    path = (root / HELPER_RELATIVE).resolve(strict=True)
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == HELPER_SHA256, "Preparation helper changed")
    spec = importlib.util.spec_from_file_location("_tower10_bound_frozen_codec_preparer", path)
    require(spec is not None and spec.loader is not None, "Cannot import bound preparation helper")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def run(root):
    root = Path(root).resolve(strict=True)
    require(root.name == RUN, f"Output root must be the dedicated {RUN} directory")
    require(Path(__file__).resolve().parent == root, "Run the prepare.py located in the output root")
    verify_manifest(root)
    protocol = read_json(root / "protocol.json")
    binding = read_json(root / "input_binding.json")
    validate_config(protocol, binding)
    assert_fresh(root)
    teacher = root / "teacher" / CLIP
    require(teacher.resolve(strict=True).is_relative_to(root), "Teacher must remain inside this new run")
    helper = load_helper(root)
    # These are the only modifications to the imported helper. All algorithms,
    # integrity checks, model guards, parity thresholds and budgets stay intact.
    helper.CLIP = protocol["clip"]
    helper.RUN = protocol["run"]
    helper.SOURCE_SHA256 = binding["expected_teacher"]["source_video_sha256"]
    old_argv = sys.argv
    try:
        sys.argv = [str(root / HELPER_RELATIVE), "--root", str(root),
                    "--teacher", str(teacher), "--codec-root", protocol["codec_root"],
                    "--clip", protocol["clip"], "--max-seconds", str(protocol["preparation_seconds"])]
        helper.main()
    finally:
        sys.argv = old_argv
    verify_manifest(root)
    summary_path = root / "SUMMARY.json"
    summary = read_json(summary_path)
    require(summary.get("status") == "complete" and summary.get("preparation_contract_passed") is True
            and summary.get("training_performed") is False
            and summary.get("scientific_transfer_passed", "missing") is None,
            "Bound helper did not report a completed frozen preparation")
    summary.update({"preparation_helper_scope_original": summary.get("scope"),
                    "source_role": "TRAIN", "source_group": "buildingblock_tower_10",
                    "official_independent_test_split": False, "goal_complete": False,
                    "scope": "Original project TRAIN source data preparation only; no training, "
                             "no generalization claim, and no official independent holdout"})
    helper.write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
