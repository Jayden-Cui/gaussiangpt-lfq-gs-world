"""Fail-closed CPU bindings for the audited 50-clip pool; never export or train.

Binding creation hashes actual static source files. Existing teacher/cache/state
references remain unadmitted until a separate full integrity check is performed.
"""
from __future__ import annotations

import copy
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

RUN = "gaussiangpt_neoverse_fulltrain_pool_20261031_v1"
HELPER_RELATIVE = "../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py"
HELPER_SHA256 = "3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85"
TEMPLATE_RELATIVE = "../gaussiangpt_neoverse_tower10_replay_20261031_v1/input_binding.json"
TEMPLATE_SHA256 = "af208495ed3df26487a59913cdd306f45038ee2f2bb47bee552f5228702c6b79"
BASE = "/data/CJH/AGS-V6Direct/outputs"
DATASET = BASE + "/buildingblock_vw2_h4_20261011_v1"
VIDEO_DIRECTORY = DATASET + "/datasets/Video-CraftBench/Paper_and_Block_clips"
CODEC_ROOT = BASE + "/gaussiangpt_neoverse_state_motion_20261030_v1/state"
REUSE = {"buildingblock_tower_10_176", "buildingblock_tower_17_176"}
AUTHORIZED_DEV = "buildingblock_tower_17_176"
AUTHORITY = {
    "dataset_index.json": (1546532, "6f0eedbb4dcfcbcd739c733239cee12e218422c94bd7d69f5e509b6907ff4cb4"),
    "source_assignments.json": (2083, "eaed258ffc89b783d33cda69bd50a58a412e2331bc449d509974b202f3f66a15"),
}
# The audited authoritative membership, not a filename scan or quality ranking.
GROUP_OFFSETS = {
    0: (0, 176, 352), 1: (0, 176), 3: (0, 176, 352), 4: (0, 176, 352),
    5: (0, 176, 352), 6: (0, 176, 352), 7: (0, 176), 8: (0, 176, 352),
    9: (0, 176, 352), 10: (0, 176), 11: (0, 176), 12: (0, 176),
    13: (0, 176), 15: (0, 176), 16: (0, 176), 17: (176,),
    18: (0, 176, 352, 528), 20: (0, 176, 352, 528), 21: (0, 176, 352, 528),
}
EXPECTED_CLIPS = {f"buildingblock_tower_{g}_{o}": (f"buildingblock_tower_{g}", o)
                  for g, offsets in GROUP_OFFSETS.items() for o in offsets}
OUTPUT_NAMES = ("input_binding.json", "reuse_reference.json", "cache", "states", "SUMMARY.json",
                "FAILED.json", "progress.json", "inputs_hashes.json", "normalization.json", "quality", "mappings")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"Not a regular file: {path}")
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest, "mtime_ns": after.st_mtime_ns}


def _exact(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, f"{label} differs: expected {expected!r}")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_pool(pool):
    """Validate membership, historical roles, full source coverage and audit evidence."""
    require(isinstance(pool, dict), "INPUT_POOL must be an object")
    for key, value in {"schema": "gs_fulltrain_input_pool_v1", "status": "verified",
                       "official_independent_holdout": False, "goal_complete": False}.items():
        _exact(pool.get(key), value, f"pool.{key}")
    policy = pool.get("selection_policy", {})
    for key, value in {"fixed_original_TRAIN_clips": 49, "fixed_original_TRAIN_groups": 18,
                       "authorized_additional_clip": AUTHORIZED_DEV, "total_clips": 50,
                       "total_groups": 19, "excluded_group": "buildingblock_tower_14",
                       "based_on_visual_or_reconstruction_outcomes": False,
                       "split_files_modified": False}.items():
        _exact(policy.get(key), value, f"selection_policy.{key}")
    for name, (size, digest) in AUTHORITY.items():
        row = pool.get("authority", {}).get(name, {})
        for key, value in {"path": DATASET + "/" + name, "sha256": digest,
                           "expected_sha256": digest, "bytes": size, "matched": True}.items():
            _exact(row.get(key), value, f"authority.{name}.{key}")
    clips = pool.get("clips")
    require(isinstance(clips, list) and len(clips) == 50, "Pool must contain exactly 50 clips")
    require(all(isinstance(row, dict) for row in clips), "Invalid clip record")
    ids = [row.get("clip_id") for row in clips]
    require(all(isinstance(clip, str) for clip in ids), "Invalid clip identity")
    require(len(set(ids)) == 50 and set(ids) == set(EXPECTED_CLIPS), "Authoritative clip membership differs")
    for row in clips:
        clip = row["clip_id"]
        group, offset = EXPECTED_CLIPS[clip]
        dev = clip == AUTHORIZED_DEV
        fixed = {"source_group": group, "index_window_offset": offset,
                 "original_role": "DEV" if dev else "TRAIN", "current_role": "TRAIN",
                 "official_role": "released_training_clip",
                 "role_basis": "user_authorized_historical_DEV_tower17_176" if dev else "original_authoritative_TRAIN",
                 "source_group_assignment": "dev" if dev else "train",
                 "video_path": VIDEO_DIRECTORY + "/" + clip + ".mp4",
                 "official_archive_member": "Paper_and_Block_clips/" + clip + ".mp4",
                 "expected_frame_count": 177, "expected_source_shape_HW": [720, 1280],
                 "exists": True, "errors": [], "sha256_matches_index": True,
                 "bytes_match_index": True, "status": "verified",
                 "requires_new_full177_export": clip not in REUSE}
        for key, value in fixed.items():
            _exact(row.get(key), value, f"{clip}.{key}")
        require(_sha(row.get("video_sha256")) and row["video_sha256"] == row.get("expected_video_sha256"),
                f"{clip}: video hash mismatch")
        size = row.get("video_bytes")
        require(type(size) is int and size > 0 and size == row.get("expected_video_bytes")
                == row.get("source_stat", {}).get("size_bytes"), f"{clip}: video size mismatch")
        plan = row.get("planned_native_export", {})
        for key, value in {"width": 448, "height": 252, "frames": 177, "frame_stride": 1,
                           "start_frame": 0, "source_frame_indices": list(range(177)), "gs_per_frame": 112896}.items():
            _exact(plan.get(key), value, f"{clip}.planned_native_export.{key}")
        require(all(type(i) is int for i in plan["source_frame_indices"]), f"{clip}: invalid planned index type")
        probe = row.get("probe", {})
        for key, value in {"frame_count": 177, "width": 1280, "height": 720,
                           "frame_indices": list(range(177)), "pts_strictly_increasing": True,
                           "original_index_93_sampled_pts_agreement": True,
                           "source_resolution_consistent_all_frames": True}.items():
            _exact(probe.get(key), value, f"{clip}.probe.{key}")
        require(all(type(i) is int for i in probe["frame_indices"]), f"{clip}: invalid decoded index type")
        for fps in [row.get("expected_fps_from_index"), probe.get("fps")]:
            require(_finite(fps) and abs(fps - 29.97) <= 1e-9, f"{clip}: FPS mismatch")
        try:
            rational = float(Fraction(probe["fps_rational"]))
            stream = probe["ffprobe_stream"]
            require(all(abs(float(Fraction(stream[k])) - 29.97) <= 1e-9
                        for k in ("r_frame_rate", "avg_frame_rate")), f"{clip}: stream FPS mismatch")
            require(int(stream["nb_read_frames"]) == int(stream["nb_frames"]) == 177
                    and stream["width"] == 1280 and stream["height"] == 720, f"{clip}: stream shape mismatch")
            require(abs(rational - 29.97) <= 1e-9, f"{clip}: rational FPS mismatch")
            times = probe["pts_seconds"]
            strings, fields, records = [probe[k] for k in
                                        ("pts_decimal_strings", "pts_source_fields", "frame_timestamp_records")]
            require(all(isinstance(a, list) and len(a) == 177 for a in (times, strings, fields, records)),
                    f"{clip}: incomplete PTS evidence")
            for i, (t, s, field, record) in enumerate(zip(times, strings, fields, records)):
                require(_finite(t) and abs(t - i / 29.97) <= 1e-6 and abs(float(s) - t) <= 1e-12,
                        f"{clip}: PTS mismatch at frame {i}")
                require(field in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time", "pkt_dts_time")
                        and abs(float(record[field]) - t) <= 1e-12, f"{clip}: PTS evidence mismatch at {i}")
            intervals = [b - a for a, b in zip(times, times[1:])]
            require(min(intervals) > 0 and max(abs(d - 1 / 29.97) for d in intervals) <= 1e-6,
                    f"{clip}: nonconsecutive PTS")
            for key, value in (("pts_interval_min_seconds", min(intervals)),
                               ("pts_interval_max_seconds", max(intervals))):
                require(_finite(probe.get(key)) and abs(probe[key] - value) <= 1e-12,
                        f"{clip}: {key} mismatch")
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
            raise RuntimeError(f"{clip}: malformed frame/PTS evidence") from error
        reuse = row.get("existing_full_sequence_reuse")
        if clip in REUSE:
            require(isinstance(reuse, dict) and reuse.get("clip_id") == clip
                    and reuse.get("source_group") == group
                    and reuse.get("source_video_sha256") == row["video_sha256"], f"{clip}: reuse identity mismatch")
            for key, value in {"selection": {"start_frame": 0, "frame_stride": 1, "num_frames": 177},
                               "native_width": 448, "native_height": 252, "points_per_group": 112896,
                               "checkpoint_rehashed_in_this_audit": False,
                               "source_video_hash_matches_authoritative_record": True}.items():
                _exact(reuse.get(key), value, f"{clip}.reuse.{key}")
            _exact(reuse.get("path"), _reuse_paths(clip)["teacher"], f"{clip}.reuse.path")
            require(bool(row.get("reuse_evidence_scope")), f"{clip}: missing reuse limitations")
        else:
            require(reuse is None and row.get("reuse_evidence_scope") is None, f"{clip}: unexpected reuse")
    summary = pool.get("summary", {})
    for key, value in {"exists_count": 50, "missing_count": 0, "verified_count": 50, "blocked_count": 0,
                       "anomaly_clip_count": 0, "anomaly_count": 0, "planned_clip_internal_frames_total": 8850,
                       "measured_decoded_frames_total": 8850, "new_full177_exports_needed": 48,
                       "existing_full177_reuse_candidates": 2, "cross_clip_physical_frame_unique_count": None}.items():
        _exact(summary.get(key), value, f"summary.{key}")
    require(pool.get("inventory_evidence", {}).get("reuse_artifacts_freshly_revalidated") is False,
            "Pool must not claim freshly validated reuse artifacts")


def load_pool(root):
    pool = read_json(Path(root) / "INPUT_POOL.json")
    validate_pool(pool)
    return pool


def get_clip(root, clip, require_export=True):
    require(clip in EXPECTED_CLIPS, f"Clip outside authorized pool: {clip}")
    row = next(row for row in load_pool(root)["clips"] if row["clip_id"] == clip)
    require(not require_export or row["requires_new_full177_export"],
            f"Existing full177 reuse clip must not be exported again: {clip}")
    return row


def clip_root(root, clip):
    require(isinstance(clip, str) and clip in EXPECTED_CLIPS, "Unsafe or unauthorized clip id")
    root = Path(root).resolve(strict=True)
    parent = root / "clips"
    target = parent / clip
    require(not parent.is_symlink() and not target.is_symlink(), "Symlinked output directory forbidden")
    require(parent.resolve() == root / "clips" and target.resolve() == root / "clips" / clip,
            "Clip output escapes root/clips/clip")
    return target


def verify_manifest(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "CODE_MANIFEST.json")
    require(isinstance(manifest, dict) and bool(manifest), "CODE_MANIFEST must be a flat path-to-SHA256 map")
    required = {"bindings.py", "test_bindings.py", "prepare_clip.py", "test_prepare_clip.py", "INPUT_POOL.json",
                HELPER_RELATIVE, TEMPLATE_RELATIVE} | {p.name for p in root.glob("*.py")}
    require(required <= set(manifest), "CODE_MANIFEST missing required inputs or root Python source")
    for relative, fixed in ((HELPER_RELATIVE, HELPER_SHA256), (TEMPLATE_RELATIVE, TEMPLATE_SHA256)):
        _exact(manifest[relative], fixed, f"manifest.{relative}")
    for relative, digest in manifest.items():
        require(isinstance(relative, str) and _sha(digest), "Invalid CODE_MANIFEST entry")
        posix = PurePosixPath(relative)
        require(not posix.is_absolute() and "\\" not in relative and ":" not in relative
                and (".." not in posix.parts or relative in (HELPER_RELATIVE, TEMPLATE_RELATIVE)),
                f"Unsafe manifest path: {relative}")
        path = (root / relative).resolve(strict=True)
        require(relative in (HELPER_RELATIVE, TEMPLATE_RELATIVE) or path.is_relative_to(root),
                f"Manifest path escapes root: {relative}")
        require(path.is_file() and sha256(path) == digest, f"CODE_MANIFEST mismatch: {relative}")
    return manifest


def _template(root):
    path = Path(root) / TEMPLATE_RELATIVE
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == TEMPLATE_SHA256, "Pinned static binding template changed")
    return json.loads(raw)


def _declared_inputs(root, row, pool):
    template = _template(root)
    old_video = template["expected_teacher"]["source_video_path"]
    # Retain the successful template's entire immutable dependency closure.
    files = [{key: item[key] for key in ("path", "bytes", "sha256")}
             for item in template["files"] if item["path"] != old_video]
    for name, record in pool["authority"].items():
        existing = [r for r in files if r["path"] == record["path"]]
        expected = {k: record[k] for k in ("path", "bytes", "sha256")}
        require(existing == [expected], f"Authority differs from pinned template: {name}")
    pool_record = fingerprint(Path(root) / "INPUT_POOL.json")
    files += [{"path": row["video_path"], "bytes": row["video_bytes"], "sha256": row["video_sha256"]},
              {k: pool_record[k] for k in ("path", "bytes", "sha256")}]
    expected = copy.deepcopy(template["expected_teacher"])
    expected.update(clip=row["clip_id"], source_video_path=row["video_path"], source_video_sha256=row["video_sha256"])
    return files, expected, pool_record


def _verification(row):
    return {"source_role": "TRAIN", "original_role": row["original_role"], "current_role": "TRAIN",
            "source_group": row["source_group"], "role_basis": row["role_basis"],
            "source_group_assignment": row["source_group_assignment"],
            "official_independent_test_split": False, "static_sources_freshly_verified": True,
            "teacher_export_hashes": "Unknown until completed export is independently checked",
            "teacher_or_cache_verified": False, "gpu_ready": False}


def validate_binding(root, clip, binding):
    """Check declarations against pool and exact template; helper rehashes all data."""
    row = get_clip(root, clip)
    files, expected, pool_record = _declared_inputs(root, row, load_pool(root))
    _exact(binding.get("schema"), "gs_frozen_codec_transfer_binding_v1", "binding.schema")
    _exact(binding.get("expected_teacher"), expected, "binding.expected_teacher")
    _exact(binding.get("input_pool_sha256"), pool_record["sha256"], "binding.input_pool_sha256")
    declared = binding.get("files")
    require(isinstance(declared, list) and len(declared) == len(files), "Missing/extra bound static dependencies")
    actual = {}
    for item in declared:
        require(isinstance(item, dict) and isinstance(item.get("path"), str)
                and item["path"] not in actual and _sha(item.get("sha256"))
                and type(item.get("bytes")) is int, "Invalid or duplicate binding file")
        actual[item["path"]] = {k: item[k] for k in ("path", "bytes", "sha256")}
    _exact(actual, {item["path"]: item for item in files}, "binding.files")
    _exact(binding.get("verification"), _verification(row), "binding.verification")


def _assert_fresh(target):
    for name in OUTPUT_NAMES:
        require(not (target / name).exists() and not (target / name).is_symlink(),
                f"Refusing overwrite or automatic retry: {target / name}")


def _write_new(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def build_binding(root, clip):
    """Create static binding only after every source is present and matches its pin."""
    verify_manifest(root)
    row = get_clip(root, clip)
    target = clip_root(root, clip)
    _assert_fresh(target)
    files, expected, pool_record = _declared_inputs(root, row, load_pool(root))
    # Preflight all paths before hashing any large checkpoint; no output yet.
    for item in files:
        require(Path(item["path"]).is_file(), f"Static source not ready: {item['path']}")
    verified = []
    for item in files:
        record = fingerprint(item["path"])
        require(record["sha256"] == item["sha256"] and record["bytes"] == item["bytes"],
                f"Static source hash/bytes mismatch: {item['path']}")
        verified.append(record)
    binding = {"schema": "gs_frozen_codec_transfer_binding_v1",
               "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
               "expected_teacher": expected, "input_pool_sha256": pool_record["sha256"],
               "verification": _verification(row)}
    validate_binding(root, clip, binding)
    for record in verified:
        stat = Path(record["path"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                f"Static source changed before binding write: {record['path']}")
    verify_manifest(root)
    _assert_fresh(target)
    _write_new(target / "input_binding.json", binding)
    return binding


def _reuse_paths(clip):
    if clip == "buildingblock_tower_10_176":
        old = BASE + "/gaussiangpt_neoverse_tower10_replay_20261031_v1"
        return {"teacher": old + "/teacher/" + clip, "cache": old + "/cache/" + clip}
    require(clip == AUTHORIZED_DEV, "Not an authorized reuse clip")
    old = BASE + "/gaussiangpt_neoverse_state_motion_20261030_v1"
    return {"teacher": BASE + "/neoverse_tower177_fps30_20261029_v1/native448",
            "cache": old + "/state/cache/" + clip}


def build_reuse_reference(root, clip):
    """Record prior receipts and fixed locations only; no readiness or hash claim."""
    verify_manifest(root)
    row = get_clip(root, clip, require_export=False)
    require(clip in REUSE and row["requires_new_full177_export"] is False, "Clip requires a new export")
    target = clip_root(root, clip)
    _assert_fresh(target)
    reference = {"schema": "gs_fulltrain_reuse_reference_v1", "clip_id": clip,
                 "status": "reference_only_requires_admission", "freshly_verified": False,
                 "requires_new_full177_export": False, "original_role": row["original_role"],
                 "current_role": row["current_role"], "source_group": row["source_group"],
                 "role_basis": row["role_basis"], "paths_requiring_admission": _reuse_paths(clip),
                 "prior_pool_evidence": copy.deepcopy(row["existing_full_sequence_reuse"]),
                 "reuse_evidence_scope": row["reuse_evidence_scope"],
                 "cache_evidence": "Fixed location from original preparation code; not hashed or admitted",
                 "input_pool_sha256": sha256(Path(root) / "INPUT_POOL.json"),
                 "teacher_or_cache_verified": False, "training_performed": False, "goal_complete": False}
    _write_new(target / "reuse_reference.json", reference)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--clip", required=True, choices=sorted(EXPECTED_CLIPS))
    parser.add_argument("--reuse-reference", action="store_true",
                        help="Record an unadmitted reference for one of the two existing full177 clips")
    args = parser.parse_args()
    if args.reuse_reference:
        build_reuse_reference(args.root, args.clip)
        filename = "reuse_reference.json"
    else:
        build_binding(args.root, args.clip)
        filename = "input_binding.json"
    print(json.dumps({"written": str(clip_root(args.root, args.clip) / filename),
                      "teacher_or_cache_verified": False, "training_performed": False}))


if __name__ == "__main__":
    main()
