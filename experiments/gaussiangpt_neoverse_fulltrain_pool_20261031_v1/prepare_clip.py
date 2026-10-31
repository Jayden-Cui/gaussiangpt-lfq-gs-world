"""Prepare one predeclared full TRAIN clip using the unchanged frozen codec.

The CLI consumes an already completed native teacher export.  teacher_command()
only constructs the fixed export argv for a later queue; it never starts it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import bindings


RUN = "gaussiangpt_neoverse_fulltrain_pool_20261031_v1"
HELPER_RELATIVE = "../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py"
HELPER_SHA256 = "3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85"
BASE = "/data/CJH/AGS-V6Direct/outputs"
PREVIEW = BASE + "/neoverse_gs_preview_20261028_v1"
RUNTIME = PREVIEW + "/runtime/bin/python"
REPO = PREVIEW + "/NeoVerse_source"
CHECKPOINT = PREVIEW + "/weights_cache/models/Yuppie1204--NeoVerse/snapshots/master/reconstructor.ckpt"
RENDERER = BASE + "/neoverse_tower177_fps30_20261029_v1/render_neoverse.py"
CODEC_ROOT = BASE + "/gaussiangpt_neoverse_state_motion_20261030_v1/state"
FRAMES, HEIGHT, WIDTH, POINTS, FPS = 177, 252, 448, 112896, 29.97
OUTPUT_NAMES = ("cache", "states", "SUMMARY.json", "FAILED.json", "progress.json",
                "inputs_hashes.json", "normalization.json", "quality", "mappings",
                "PREPARATION_STARTED.json", "PREPARATION_RECEIPT.json", "ADAPTER_FAILED.json",
                "HELPER_SUMMARY.json")
QUALITY_KEYS = ("teacher_real", "representative_real", "representative_teacher",
                "codec_real", "codec_teacher", "pseudo_view_teacher")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Exclusive creation leaves even an interrupted attempt non-restartable."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    # Several pre-bound vendor __init__.py files are legitimately empty.
    # Nonempty data artifacts are enforced by owned_file(), not generic hashing.
    require(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def teacher_command(root, clip):
    """Return the verified tower10 export recipe, changing only input/output."""
    item = bindings.get_clip(root, clip, require_export=True)
    teacher = bindings.clip_root(root, clip) / "teacher" / clip
    return [RUNTIME, "-B", "-u", RENDERER, "--repo", REPO, "--checkpoint", CHECKPOINT,
            "--input", item["video_path"], "--output", str(teacher),
            "--num-frames", "177", "--start-frame", "0", "--frame-stride", "1",
            "--fps", "29.97", "--height", "252", "--width", "448", "--save-gs", "177",
            "--novel-view-shift", "0.03", "--device", "cuda:0", "--dtype", "bfloat16"]


def assert_fresh(directory):
    for name in OUTPUT_NAMES:
        path = directory / name
        require(not path.exists() and not path.is_symlink(),
                f"Refusing overwrite or retry of a preparation attempt: {path}")


def owned_file(path, directory):
    path, directory = Path(path), Path(directory).resolve(strict=True)
    require(path.resolve(strict=True).is_relative_to(directory), f"Output escapes clip root: {path}")
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f"Missing/empty/symlink output: {path}")
    for parent in path.parents:
        if parent == directory:
            break
        require(not parent.is_symlink(), f"Symlink output directory: {parent}")
    return path


def load_helper(root):
    """Execute exactly the checked bytes, with no old-run bytecode writes."""
    path = (Path(root) / HELPER_RELATIVE).resolve(strict=True)
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == HELPER_SHA256, "Immutable preparation helper changed")
    spec = importlib.util.spec_from_file_location("_fulltrain_bound_frozen_codec_preparer", path)
    require(spec is not None and spec.loader is not None, "Cannot load bound preparation helper")
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def bind_helper(helper, item):
    # The helper requires root.name == RUN; this run's root is clips/<clip>.
    # These are deliberately the only modifications to the immutable module.
    helper.CLIP = item["clip_id"]
    helper.RUN = item["clip_id"]
    helper.SOURCE_SHA256 = item["expected_video_sha256"]
    return helper


def admit_teacher(directory, item, binding, helper):
    teacher = directory / "teacher" / item["clip_id"]
    require(teacher.resolve(strict=True).is_relative_to(directory.resolve(strict=True))
            and not teacher.is_symlink(), "Teacher must be inside this clip's new root")
    expected = {teacher / "gaussians" / f"native_group_{t:03d}.npz" for t in range(FRAMES)}
    require(set((teacher / "gaussians").glob("*.npz")) == expected,
            "Teacher coverage must be exactly native_group_000..176; no sparse export")
    for path in expected | {teacher / name for name in
                            ("provenance.json", "cameras.npz", "input_rgb.npy", "rendered_rgb.npy")}:
        owned_file(path, directory)
    times = helper.validate_provenance(read_json(teacher / "provenance.json"),
                                       binding["expected_teacher"], teacher.resolve(strict=True))
    measured = item["probe"]["pts_seconds"]
    require(len(times) == len(measured) == FRAMES
            and all(abs(a - b) <= 1e-6 for a, b in zip(times, measured)),
            "Teacher PTS disagrees with the complete source-video CPU probe")
    return teacher


def check_record(record):
    current = fingerprint(record["path"])
    require(current["bytes"] == record["bytes"] and current["sha256"] == record["sha256"],
            f"Recorded file hash/size changed: {record['path']}")
    return current


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_outputs(directory, item, binding_sha256):
    """Audit the actual helper outputs without importing Torch or a new codec."""
    clip = item["clip_id"]
    summary = read_json(owned_file(directory / "SUMMARY.json", directory))
    fixed = {"status": "complete", "preparation_contract_passed": True,
             "scientific_transfer_passed": None, "clip": clip, "frames": FRAMES,
             "adjacent_pairs": 176, "source_points_per_frame": POINTS,
             "all_input_points_aggregated": True, "codec_frozen": True,
             "state_quantization": "Identity; continuous float32 Nx64", "training_performed": False,
             "input_binding_sha256": binding_sha256}
    for key, value in fixed.items():
        require(type(summary.get(key)) is type(value) and summary[key] == value,
                f"Incomplete/incompatible helper summary: {key}")
    normal_path = owned_file(directory / "normalization.json", directory)
    normal = read_json(normal_path)
    require(normal == summary["normalization"] and normal.get("source_frame") == 0
            and normal.get("fixed_for_all_frames") is True
            and normal.get("uses_future_frames") is False
            and normal.get("changes_codec_output_std") is False
            and normal.get("center") == [0.0, 0.0, 0.0]
            and _finite_number(normal.get("scale")) and normal["scale"] > 0,
            "Normalization must be fixed using this clip's first frame only")
    mapping_path = owned_file(directory / "mappings" / f"{clip}.json", directory)
    mapping = read_json(mapping_path)
    require(mapping.get("clip") == clip and mapping.get("source_sha256") == item["expected_video_sha256"]
            and mapping.get("source_frame_indices") == list(range(FRAMES))
            and mapping.get("model_timestamps") == list(range(FRAMES))
            and mapping.get("adjacent_pairs") == [[i, i + 1] for i in range(FRAMES - 1)],
            "Frame/PTS mapping lost full consecutive source coverage")
    times = mapping.get("source_pts_seconds", [])
    require(len(times) == FRAMES and all(_finite_number(t) and abs(t - i / FPS) <= 1e-9
                                        for i, t in enumerate(times)), "Mapping PTS changed")
    quality_path = owned_file(directory / "quality" / f"{clip}.json", directory)
    quality = read_json(quality_path)
    rows = quality.get("frames", [])
    require(len(rows) == FRAMES and [row.get("frame") for row in rows] == list(range(FRAMES)),
            "Quality must cover exactly 177 ordered frames")
    require(quality.get("summary") == summary["quality"], "Quality aggregate disagrees with summary")
    inputs_path = owned_file(directory / "inputs_hashes.json", directory)
    inputs = read_json(inputs_path)
    require(inputs.get("schema") == "gs_transfer_inputs_v1"
            and inputs.get("teacher_hashes_computed_after_export") is True,
            "Missing actual post-export input hashes")
    require(inputs["binding"]["sha256"] == binding_sha256
            and Path(inputs["binding"]["path"]).resolve() == directory / "input_binding.json",
            "Helper consumed a different input binding")
    check_record(inputs["binding"])
    input_binding = read_json(directory / "input_binding.json")
    static = {str(Path(row["path"]).resolve()): (row["sha256"], row["bytes"])
              for row in inputs["static_files"]}
    expected_static = {str(Path(row["path"]).resolve()): (row["sha256"], row["bytes"])
                       for row in input_binding["files"]}
    require(len(static) == len(inputs["static_files"]) and static == expected_static,
            "Actual static inputs differ from admitted input binding")
    teacher = directory / "teacher" / clip
    expected_teacher = {str((teacher / name).resolve()) for name in
                        ("provenance.json", "cameras.npz", "input_rgb.npy", "rendered_rgb.npy")}
    expected_teacher.update(str((teacher / "gaussians" / f"native_group_{t:03d}.npz").resolve())
                            for t in range(FRAMES))
    expected_teacher.update(str(Path(input_binding["expected_teacher"][key]).resolve())
                            for key in ("source_video_path", "renderer_path"))
    teacher_rows = inputs["teacher_and_source_files"]
    actual_teacher = {str(Path(row["path"]).resolve()): row for row in teacher_rows}
    require(len(actual_teacher) == len(teacher_rows) and set(actual_teacher) == expected_teacher,
            "Actual teacher input hashes must cover all 177 native GS and the original source")
    for record in inputs["static_files"] + teacher_rows:
        check_record(record)
    artifacts = [fingerprint(path) for path in (normal_path, mapping_path, quality_path, inputs_path)]
    for kind in ("cache", "states"):
        require(set((directory / kind / clip).glob("*.pt")) ==
                {directory / kind / clip / f"{t:03d}.pt" for t in range(FRAMES)},
                f"{kind} coverage must be exactly 000..176")
    require(set((directory / "quality" / clip).glob("*.json")) ==
            {directory / "quality" / clip / f"{t:03d}.json" for t in range(FRAMES)},
            "Per-frame quality coverage must be exactly 000..176")
    for t, row in enumerate(rows):
        require(row.get("clip") == clip and row.get("source_count") == POINTS
                and _finite_number(row.get("pts_seconds")) and abs(row["pts_seconds"] - times[t]) <= 1e-9,
                f"Frame {t} clip/point count/PTS mismatch")
        native_path = str((teacher / "gaussians" / f"native_group_{t:03d}.npz").resolve())
        require(row.get("native_gs_sha256") == actual_teacher[native_path]["sha256"],
                f"Frame {t} references a different teacher")
        for key in ("native_render_parity", "similarity_parity"):
            metric = row.get(key, {})
            require(_finite_number(metric.get("psnr")) and metric["psnr"] >= 60
                    and _finite_number(metric.get("mae")) and 0 <= metric["mae"] <= 1e-5,
                    f"Frame {t} parity did not pass: {key}")
        for key in QUALITY_KEYS:
            require(isinstance(row.get(key), dict) and bool(row[key])
                    and all(_finite_number(v) for v in row[key].values()),
                    f"Frame {t} missing/nonfinite quality: {key}")
        for key, limit in (("input_voxels", 80000), ("max_decoder_candidates", 600000),
                           ("decoded_gaussians", 2400000)):
            require(type(row.get(key)) is int and 0 < row[key] <= limit,
                    f"Frame {t} violates unchanged codec guard: {key}")
        require(type(row.get("latent_positions")) is int and row["latent_positions"] > 0,
                f"Frame {t} has no continuous state")
        for kind, prefix in (("cache", "cache"), ("states", "state")):
            path = owned_file(directory / kind / clip / f"{t:03d}.pt", directory)
            record = check_record({"path": str(path), "sha256": row[f"{prefix}_sha256"],
                                   "bytes": row[f"{prefix}_bytes"]})
            artifacts.append(record)
        frame_quality = owned_file(directory / "quality" / clip / f"{t:03d}.json", directory)
        require(read_json(frame_quality) == row, f"Per-frame quality differs: {t}")
        artifacts.append(fingerprint(frame_quality))
    for key in QUALITY_KEYS:
        for metric, value in summary["quality"][key].items():
            require(_finite_number(value) and math.isclose(value, math.fsum(row[key][metric] for row in rows) / FRAMES,
                                                           rel_tol=1e-10, abs_tol=1e-10),
                    f"Quality aggregate does not match all 177 frames: {key}.{metric}")
    return {"actual_frames": len(rows), "actual_source_points_per_frame": POINTS,
            "actual_source_points_total": sum(row["source_count"] for row in rows),
            "actual_cache_files": FRAMES, "actual_state_files": FRAMES,
            "actual_native_gs_files": FRAMES, "actual_adjacent_pairs": len(mapping["adjacent_pairs"]),
            "input_files_rehashed_after_preparation": len(static) + len(teacher_rows),
            "output_files": artifacts, "quality": summary["quality"],
            "checks": {"native_render_parity": True, "scale_only_similarity_parity": True,
                       "first_frame_fixed_normalization": True, "all_source_points_aggregated": True,
                       "two_view_decoding_rendered": True,
                       "serialized_coordinates_and_features_equal_in_memory": True,
                       "disk_reloaded_state_rendered_again": False,
                       "two_view_note": "Original camera plus teacher pseudo-view; no new-view ground truth",
                       "serialization_evidence": "Unchanged bound helper reloads saved coordinates/features and checks exact tensor equality"}}


def run(root, clip):
    root = Path(root).resolve(strict=True)
    require(root.name == RUN and Path(__file__).resolve().parent == root,
            "Run prepare_clip.py from its dedicated fulltrain pool root")
    bindings.verify_manifest(root)
    item = bindings.get_clip(root, clip, require_export=True)
    directory = bindings.clip_root(root, clip).resolve(strict=True)
    require(directory.is_relative_to(root) and directory.name == clip, "Invalid clip output root")
    assert_fresh(directory)
    source_hashes = {name: fingerprint(path)["sha256"] for name, path in
                     (("code_manifest_sha256", root / "CODE_MANIFEST.json"),
                      ("input_pool_sha256", root / "INPUT_POOL.json"),
                      ("input_binding_sha256", directory / "input_binding.json"))}
    started = {"schema": "gs_fulltrain_clip_preparation_attempt_v1", "status": "started",
               "clip": clip, "started_utc": datetime.now(timezone.utc).isoformat(),
               "automatic_retry": False, "training_performed": False, **source_hashes}
    write_new(directory / "PREPARATION_STARTED.json", started)
    helper = None
    try:
        binding = read_json(directory / "input_binding.json")
        bindings.validate_binding(root, clip, binding)
        helper = bind_helper(load_helper(root), item)
        teacher = admit_teacher(directory, item, binding, helper)
        old_argv = sys.argv
        try:
            sys.argv = [str(root / HELPER_RELATIVE), "--root", str(directory), "--teacher", str(teacher),
                        "--codec-root", CODEC_ROOT, "--clip", clip, "--max-seconds", "1200"]
            helper.main()
        finally:
            sys.argv = old_argv
        write_new(directory / "HELPER_SUMMARY.json", read_json(directory / "SUMMARY.json"))
        audited = verify_outputs(directory, item, source_hashes["input_binding_sha256"])
        bindings.verify_manifest(root)
        for key, path in (("code_manifest_sha256", root / "CODE_MANIFEST.json"),
                          ("input_pool_sha256", root / "INPUT_POOL.json"),
                          ("input_binding_sha256", directory / "input_binding.json")):
            require(fingerprint(path)["sha256"] == source_hashes[key], f"Preparation input changed: {key}")
        summary = read_json(directory / "SUMMARY.json")
        summary.update({"schema": "gs_fulltrain_clip_preparation_v1", "preparation_helper_scope_original": summary.get("scope"),
                        "source_role": item["current_role"], "original_role": item["original_role"],
                        "source_group": item["source_group"], "role_basis": item["role_basis"],
                        "data_origin": "new_full177_export", "existing_sequence_reused": False,
                        "official_independent_test_split": False, "goal_complete": False,
                        "training_performed": False, "helper_sha256": HELPER_SHA256, **source_hashes,
                        "actual_coverage": {key: value for key, value in audited.items() if key.startswith("actual_")},
                        "replay_checks": audited["checks"],
                        "scope": "Predeclared original TRAIN data preparation only; no model training or generalization claim"})
        helper.write_json(directory / "SUMMARY.json", summary)
        receipt = {"schema": "gs_fulltrain_clip_preparation_receipt_v1", "status": "complete",
                   "clip": clip, "source_group": item["source_group"], "original_role": item["original_role"],
                   "current_role": item["current_role"], "data_origin": "new_full177_export",
                   "training_performed": False, "goal_complete": False, "helper_sha256": HELPER_SHA256,
                   **source_hashes, **audited, "summary_sha256": fingerprint(directory / "SUMMARY.json")["sha256"],
                   "helper_summary_sha256": fingerprint(directory / "HELPER_SUMMARY.json")["sha256"]}
        write_new(directory / "PREPARATION_RECEIPT.json", receipt)
        return receipt
    except BaseException as error:
        failure = {"schema": "gs_fulltrain_clip_preparation_failure_v1", "status": "failed", "clip": clip,
                   "exception": type(error).__name__, "reason": str(error), "automatic_retry": False,
                   "training_performed": False, "goal_complete": False, **source_hashes}
        write_new(directory / "ADAPTER_FAILED.json", failure)
        if not (directory / "FAILED.json").exists():
            write_new(directory / "FAILED.json", failure)
        # A completed helper alone is insufficient if post-export admission fails.
        if helper is not None and (directory / "SUMMARY.json").is_file():
            summary = read_json(directory / "SUMMARY.json")
            summary.update({"helper_status_before_adapter_failure": summary.get("status"), "status": "failed",
                            "preparation_contract_passed": False, "adapter_failure": failure,
                            "training_performed": False, "goal_complete": False})
            helper.write_json(directory / "SUMMARY.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--clip", required=True)
    args = parser.parse_args()
    run(args.root, args.clip)


if __name__ == "__main__":
    main()
