"""Prepare one complete cross-source clip with an immutable, frozen GS codec.

No model training, teacher inference, package installation, or old-run writes.
The caller must create input_binding.json before exporting the new teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
CLIP = "buildingblock_tower_14_0"
RUN = "gaussiangpt_neoverse_tower14_transfer_20261031_v1"
FRAMES, H, W, N, GRID, FPS = 177, 252, 448, 112896, 0.008, 29.97
SOURCE_SHA256 = "456d178c177d6790196f2e27f3bf5e1f5eebbb9fdaecc716794be90fbbab57f8"
BINDING_SCHEMA = "gs_frozen_codec_transfer_binding_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError(f"Preparation exceeded {self.seconds} seconds; no retry")

    def elapsed(self):
        return time.monotonic() - self.started


def fingerprint(path, budget):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            budget.check()
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest(),
            "mtime_ns": after.st_mtime_ns}


def verify_binding(root, codec_root, budget):
    path = root / "input_binding.json"
    binding_record = fingerprint(path, budget)
    binding = read_json(path)
    require(binding.get("schema") == BINDING_SCHEMA, "Unexpected input_binding schema")
    rows = binding.get("files")
    require(isinstance(rows, list) and rows, "Binding must contain files")
    records = {}
    for row in rows:
        p = Path(row["path"])
        require(p.is_absolute(), f"Binding path must be absolute: {p}")
        actual = fingerprint(p, budget)
        key = Path(actual["path"])
        require(key not in records, f"Duplicate binding path: {key}")
        require(actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"],
                f"Static binding mismatch: {key}")
        records[key] = actual
    for p in [codec_root / "codec_final.pt", codec_root / "selection.json",
              codec_root / "protocol.json", codec_root / "code/common.py", codec_root / "code/codec.py"]:
        require(p.resolve() in records, f"Required codec input is not pre-bound: {p}")
    expected = binding.get("expected_teacher", {})
    fixed = {"clip": CLIP, "frames": FRAMES, "height": H, "width": W,
             "frame_stride": 1, "use_motion": False, "source_video_sha256": SOURCE_SHA256}
    for key, value in fixed.items():
        require(expected.get(key) == value, f"expected_teacher.{key} must be {value!r}")
    source = Path(expected["source_video_path"]).resolve(strict=True)
    require(source.name == CLIP + ".mp4", "Binding source is not the target clip")
    renderer = Path(expected["renderer_path"]).resolve(strict=True)
    for key in ["renderer_sha256", "checkpoint_sha256"]:
        require(isinstance(expected.get(key), str) and len(expected[key]) == 64,
                f"Missing expected_teacher.{key}")
    renderer_record = fingerprint(renderer, budget)
    require(renderer_record["sha256"] == expected["renderer_sha256"], "Renderer changed")
    source_record = fingerprint(source, budget)
    require(source_record["sha256"] == SOURCE_SHA256, "Original video SHA256 mismatch")
    return binding, binding_record, records, [source_record, renderer_record]


def validate_provenance(prov, expected, teacher):
    """Validate the actual render_neoverse.py provenance structure, without CUDA."""
    require(prov.get("status") == "complete", "Teacher export is incomplete")
    require(prov.get("script_sha256") == expected["renderer_sha256"], "Teacher renderer SHA differs")
    checkpoint = prov["checkpoint"]
    require(checkpoint.get("sha256") == expected["checkpoint_sha256"], "Teacher checkpoint SHA differs")
    require(checkpoint.get("strict_load") is True and checkpoint.get("missing_keys") == []
            and checkpoint.get("unexpected_keys") == [], "Teacher did not strictly load its checkpoint")
    inp, arguments = prov["input"], prov["arguments"]
    require(Path(inp["path"]).resolve() == Path(expected["source_video_path"]).resolve(), "Teacher source path differs")
    require(inp.get("kind") == "video" and inp.get("source_video_sha256") == SOURCE_SHA256,
            "Teacher input identity differs")
    require(math.isclose(float(inp["source_fps"]), FPS, rel_tol=0, abs_tol=1e-9), "Source FPS differs")
    require(inp.get("selection") == {"start_frame": 0, "frame_stride": 1, "num_frames": FRAMES},
            "Teacher must use all 177 consecutive original frames")
    require([v["index"] for v in inp["selected_frames"]] == list(range(FRAMES)), "Missing/reordered source frames")
    times = [float(v) for v in inp["source_timestamps_seconds"]]
    require(len(times) == FRAMES and all(math.isfinite(t) for t in times), "Invalid PTS sequence")
    require(all(abs(t - i / FPS) <= 1e-9 for i, t in enumerate(times)), "PTS does not match original FPS")
    for key, value in {"num_frames": FRAMES, "height": H, "width": W, "start_frame": 0,
                       "frame_stride": 1, "use_motion": False, "save_gs": FRAMES, "rgb_npz": None}.items():
        require(arguments.get(key) == value, f"Teacher argument {key} differs")
    require(Path(arguments["input"]).resolve() == Path(expected["source_video_path"]).resolve(), "Teacher input argument differs")
    require(Path(arguments["output"]).resolve() == teacher, "Teacher output directory differs")
    groups, saved = prov["gaussian_groups"], prov["saved_gaussians"]
    require(len(groups) == len(saved) == FRAMES, "Expected 177 complete GS groups")
    for t, (group, item) in enumerate(zip(groups, saved)):
        require(group["timestamp"] == t and group["points"] == N
                and group["has_forward_motion"] is False and group["has_backward_motion"] is False,
                f"Invalid native GS group {t}")
        require(item["group_index"] == item["timestamp"] == t and item["points"] == N,
                f"Invalid saved GS metadata {t}")
        require(Path(item["path"]).resolve() == teacher / "gaussians" / f"native_group_{t:03d}.npz",
                f"Saved GS path mismatch at {t}")
    return times


def normalization_from_frame_zero(gs):
    """Same TRAIN algorithm, calibrated solely on the new clip's first frame."""
    import numpy as np
    xyz = gs[gs[:, 13] > 0.01, :3]
    require(len(xyz) > 0, "No frame-zero GS above calibration opacity threshold")
    lo, hi = np.quantile(xyz, [0.01, 0.99], axis=0)
    span = float(np.max(hi - lo))
    require(math.isfinite(span) and span > 0, "Invalid frame-zero robust span")
    robust_scale = 0.9 / span
    scale = float(2.0 ** np.floor(np.log2(robust_scale)))
    require(math.isfinite(scale) and scale > 0, "Invalid dyadic normalization scale")
    return {"center": [0.0, 0.0, 0.0], "scale": scale, "robust_scale_reference": robust_scale,
            "robust_center_reference": ((lo + hi) / 2).tolist(), "source_frame": 0,
            "fixed_for_all_frames": True, "calibration_only_alpha_threshold": 0.01,
            "rule": "Power-of-two scale only, no translation; same algorithm as TRAIN, clip-local frame-zero gauge",
            "uses_future_frames": False, "changes_codec_output_std": False}


class BoundSourceGuard(importlib.abc.MetaPathFinder):
    """Reject unbound codec/vendor Python modules before their execution."""
    def __init__(self, roots, records, budget):
        self.roots, self.records, self.budget = roots, records, budget

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.origin and spec.origin not in ("built-in", "frozen"):
            origin = Path(spec.origin).resolve()
            if any(origin.is_relative_to(root) for root in self.roots):
                require(origin in self.records, f"Imported source is not pre-bound: {origin}")
                require(fingerprint(origin, self.budget)["sha256"] == self.records[origin]["sha256"],
                        f"Imported source changed: {origin}")
        return None


def load_bound_codec(codec_root, records, budget):
    require("common" not in sys.modules and "codec" not in sys.modules, "Module name collision before codec import")
    sys.path.insert(0, str(codec_root / "code"))
    # common.py is already SHA-bound and contains only CPU imports and definitions.
    common = importlib.import_module("common")
    require(Path(common.__file__).resolve() == codec_root / "code/common.py", "Unexpected common module")
    require(Path(common.CHECKPOINT).resolve() in records, "Official initialization checkpoint is not pre-bound")
    vendor = Path(common.VENDOR).resolve(strict=True)
    guard = BoundSourceGuard([codec_root / "code", vendor], records, budget)
    sys.meta_path.insert(0, guard)
    codec = importlib.import_module("codec")
    require(Path(codec.__file__).resolve() == codec_root / "code/codec.py", "Unexpected codec module")
    return common, codec, vendor


def state_digest(model, budget):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        budget.check()
        data = value.detach().cpu().contiguous()
        digest.update((name + str(tuple(data.shape)) + str(data.dtype)).encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(torch, payload, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def check_parity(result, label, t):
    require(result["psnr"] >= 60.0 and result["mae"] <= 1e-5,
            f"{label} failed at frame {t}: {result}")


def prepare(args, budget):
    import numpy as np
    import torch

    root, teacher, codec_root = args.root, args.teacher, args.codec_root
    binding, binding_record, bound, extra_inputs = verify_binding(root, codec_root, budget)
    prov = read_json(teacher / "provenance.json")
    times = validate_provenance(prov, binding["expected_teacher"], teacher)
    expected_gs = [teacher / "gaussians" / f"native_group_{t:03d}.npz" for t in range(FRAMES)]
    require(set((teacher / "gaussians").glob("*.npz")) == set(expected_gs), "GS file coverage must be exactly 000..176")
    input_paths = [teacher / name for name in ["provenance.json", "cameras.npz", "input_rgb.npy", "rendered_rgb.npy"]] + expected_gs
    inputs = {Path(row["path"]): row for row in [fingerprint(p, budget) for p in input_paths] + extra_inputs}
    write_json(root / "inputs_hashes.json", {"schema": "gs_transfer_inputs_v1", "binding": binding_record,
               "static_files": list(bound.values()), "teacher_and_source_files": list(inputs.values()),
               "teacher_hashes_computed_after_export": True})
    selection, protocol = read_json(codec_root / "selection.json"), read_json(codec_root / "protocol.json")
    require(selection["grid"] == GRID and selection["mode"] == "set_statistics"
            and selection["gaussians_per_voxel"] == 4, "Codec selection contract changed")
    for key, value in {"max_input_voxels": 80000, "max_decoder_candidates": 600000,
                       "max_decoded_gaussians": 2400000}.items():
        require(protocol[key] == value, f"Codec guard changed: {key}")
    require(protocol["train"]["prune_threshold"] == [-1.0, -3.0], "Decoder pruning configuration changed")
    with np.load(teacher / "cameras.npz", allow_pickle=False) as archive:
        cameras = archive["camera_to_world"].copy()
        intrinsics = archive["intrinsics"].copy()
        timestamps = archive["timestamps"].copy()
    require(cameras.shape == (FRAMES, 4, 4) and intrinsics.shape == (FRAMES, 3, 3), "Camera shape mismatch")
    require(np.array_equal(timestamps, np.arange(FRAMES)), "Camera timestamps mismatch")
    require(np.isfinite(cameras).all() and np.isfinite(intrinsics).all(), "Nonfinite camera data")
    require(np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            and np.all(np.abs(np.linalg.det(cameras[:, :3, :3])) > 1e-6), "Invalid camera transforms")
    require(np.all(intrinsics[:, 0, 0] > 0) and np.all(intrinsics[:, 1, 1] > 0), "Invalid focal length")
    rgb = np.load(teacher / "input_rgb.npy", mmap_mode="r", allow_pickle=False)
    reference = np.load(teacher / "rendered_rgb.npy", mmap_mode="r", allow_pickle=False)
    require(rgb.shape == reference.shape == (FRAMES, H, W, 3) and rgb.dtype == np.uint8,
            "RGB must contain all 177 native448 frames")
    require(np.issubdtype(reference.dtype, np.floating), "Teacher rendered RGB must be floating point")
    common, codec, _vendor = load_bound_codec(codec_root, bound, budget)
    require((common.H, common.W, common.N) == (H, W, N), "Bound renderer dimensions changed")
    require(torch.cuda.is_available(), "Real CUDA runtime is required; CPU inspection is not readiness")
    torch.set_num_threads(6)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.60)
    torch.manual_seed(int(protocol["train"]["seed"]))
    model = codec.load_codec(codec_root, codec_root / "codec_final.pt", device="cuda").eval().requires_grad_(False)
    require(isinstance(model.ae.vq, torch.nn.Identity), "State quantization must remain Identity")
    require(int(model.ae.stride) == 4, "Unexpected sparse state stride")
    initial_state_digest = state_digest(model, budget)
    decode_counts = {}

    def reset_counts(module, values):
        decode_counts.clear()

    def candidates(module, values, output):
        decode_counts["max_candidates"] = max(decode_counts.get("max_candidates", 0), len(output.F))

    def output_count(module, values, output):
        decode_counts["gaussians"] = len(output.F) * model.count

    model.ae.decoder.register_forward_pre_hook(reset_counts)
    for stage in model.ae.decoder.ups:
        stage.register_forward_hook(candidates, prepend=True)
    model.ae.decoder.out.register_forward_hook(output_count, prepend=True)
    cache_dir, state_dir = root / "cache" / CLIP, root / "states" / CLIP
    cache_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    write_json(root / "mappings" / f"{CLIP}.json", {"clip": CLIP, "source_pts_seconds": times,
               "source_frame_indices": list(range(FRAMES)), "model_timestamps": list(range(FRAMES)),
               "source_sha256": SOURCE_SHA256, "adjacent_pairs": [[t, t + 1] for t in range(FRAMES - 1)],
               "note": "Original consecutive frames; model timestamps are integer indices, PTS are seconds"})
    rows, normal = [], None
    with torch.no_grad():
        for t, path in enumerate(expected_gs):
            budget.check()
            require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()),
                    "Codec must remain eval and frozen")
            require(isinstance(model.ae.vq, torch.nn.Identity), "State quantizer changed")
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata_json"]))
                require(meta["timestamp"] == t and meta["forward_vel"] is None and meta["backward_vel"] is None,
                        f"GS timestamp/motion metadata mismatch at {t}")
                shapes = {"means": (N, 3), "harmonics": (N, 1, 3), "scales": (N, 3),
                          "rotations": (N, 4), "opacities": (N,)}
                for key, shape in shapes.items():
                    require(archive[key].shape == shape and np.isfinite(archive[key]).all(), f"Invalid {key} at {t}")
                require(np.all(np.linalg.norm(archive["rotations"], axis=1) > 1e-12), "Degenerate GS quaternion")
                gs_array = np.concatenate([archive["means"], archive["harmonics"][:, 0] * common.C0 + 0.5,
                                           archive["scales"], archive["rotations"], archive["opacities"][:, None]], axis=-1)
            gs = torch.from_numpy(gs_array).float().cuda()
            require(tuple(gs.shape) == (N, 14) and bool(torch.isfinite(gs).all())
                    and bool((gs[:, 6:9] > 0).all()) and bool(((gs[:, 13] >= 0) & (gs[:, 13] <= 1)).all()),
                    f"Invalid GS state {t}")
            gs[:, 9:13] = common.canonical_quat(gs[:, 9:13])
            camera = torch.from_numpy(cameras[t].copy()).float().cuda()
            k = intrinsics[t]
            intr = torch.tensor([k[0, 0] / W, k[1, 1] / H, k[0, 2] / W, k[1, 2] / H], dtype=torch.float32, device="cuda")
            native, _ = common.render(gs, camera, intr, 0.01, 1e10)
            old = torch.from_numpy(np.array(reference[t])).float().cuda()
            require(bool(torch.isfinite(native).all()) and bool(torch.isfinite(old).all()), "Nonfinite native render")
            replay = common.metrics(native.clamp(0, 1), old)
            check_parity(replay, "Native render parity", t)
            if normal is None:
                require(t == 0, "Normalization can only use frame zero")
                normal = normalization_from_frame_zero(gs.cpu().numpy())
                write_json(root / "normalization.json", normal)
            scale = normal["scale"]
            gs[:, :3] *= scale
            gs[:, 6:9] *= scale
            camera[:3, 3] *= scale
            near, far = 0.01 * scale, 1e10 * scale
            require(bool(torch.isfinite(gs).all()) and bool((gs[:, 6:9] > 0).all()), "Normalization overflow/underflow")
            normalized, _ = common.render(gs, camera, intr, near, far)
            parity = common.metrics(normalized.clamp(0, 1), native.clamp(0, 1))
            check_parity(parity, "Similarity parity", t)
            # Same fixed +0.02 normalized-world camera-local x offset as original preparation.
            nearby = camera.clone()
            nearby[:3, 3] += camera[:3, 0] * 0.02
            pseudo, _ = common.render(gs, nearby, intr, near, far)
            require(bool(torch.isfinite(normalized).all()) and bool(torch.isfinite(pseudo).all()), "Nonfinite teacher render")
            real = torch.from_numpy(np.array(rgb[t])).float().cuda() / 255
            require(float((gs[:, :3] / GRID).abs().max()) < np.iinfo(np.int32).max - 8, "Voxel coordinate overflow")
            vox = common.voxelize(gs, GRID, with_stats=True)
            count = len(vox["representatives"])
            require(vox["n_input"] == N and int(vox["counts"].sum()) == N, "Some source GS did not enter voxel aggregation")
            require(0 < count <= protocol["max_input_voxels"], "Input voxel guard exceeded")
            require(vox["set_statistics"].shape == (count, 29), "Set-statistics shape mismatch")
            if t in protocol["calibration_frames"]:
                again = common.voxelize(gs, GRID, with_stats=True)
                for key in ["coordinates", "set_statistics", "source_indices", "counts"]:
                    require(torch.equal(vox[key], again[key]), f"Nondeterministic voxel aggregation: {key}")
                del again
            representative, _ = common.render(vox["representatives"].cuda(), camera, intr, near, far)
            row = {"clip": CLIP, "frame": t, "pts_seconds": times[t], "source_count": N,
                   "native_render_parity": replay, "similarity_parity": parity,
                   "teacher_real": common.metrics(normalized.clamp(0, 1), real),
                   "representative_real": common.metrics(representative.clamp(0, 1), real),
                   "representative_teacher": common.metrics(representative.clamp(0, 1), normalized.clamp(0, 1)),
                   "input_voxels": count, "native_gs_sha256": inputs[path]["sha256"]}
            payload = {**vox, "cameras": torch.stack([camera, nearby]).cpu(), "intrinsics": intr.cpu(),
                       "targets": torch.stack([normalized, pseudo]).cpu(), "real": real.cpu(),
                       "near": near, "far": far, "clip": CLIP, "frame": t, "pts": times[t],
                       "normalization": normal, "metrics": dict(row)}
            cache_path = cache_dir / f"{t:03d}.pt"
            atomic_torch_save(torch, payload, cache_path)
            cache_record = fingerprint(cache_path, budget)
            encoded = model.encode_state(vox["representatives"].cuda(), vox["set_statistics"].cuda())
            coords, features = encoded["coordinates"], encoded["features"]
            require(coords.dtype in (torch.int32, torch.int64) and tuple(coords.shape) == (len(features), 3)
                    and tuple(features.shape) == (len(coords), 64) and features.dtype == torch.float32
                    and bool(torch.isfinite(features).all()) and len(coords) > 0,
                    "Continuous Nx64 state contract failed")
            require(encoded["stride"] == 4 and encoded["grid"] == GRID, "Sparse state units changed")
            require(len(torch.unique(coords, dim=0)) == len(coords), "Duplicate sparse latent coordinates")
            decoded = model.decode_state(coords, features, encoded["stride"])
            predicted_gs, predicted_sh = common.prediction_gs(decoded)
            require(0 < len(predicted_gs) <= protocol["max_decoded_gaussians"], "Decoded GS guard exceeded")
            require(bool(torch.isfinite(predicted_gs).all()) and bool(torch.isfinite(predicted_sh).all()), "Nonfinite decoded GS")
            require(decode_counts["max_candidates"] <= protocol["max_decoder_candidates"], "Decoder candidate guard exceeded")
            images = torch.stack([common.render(predicted_gs, view, intr, near, far, predicted_sh)[0].clamp(0, 1)
                                  for view in (camera, nearby)])
            require(bool(torch.isfinite(images).all()), "Nonfinite codec rendering")
            row.update({"codec_real": common.metrics(images[0], real),
                        "codec_teacher": common.metrics(images[0], normalized.clamp(0, 1)),
                        "pseudo_view_teacher": common.metrics(images[1], pseudo.clamp(0, 1)),
                        "latent_positions": len(coords), "decoded_gaussians": len(predicted_gs),
                        "max_decoder_candidates": decode_counts["max_candidates"], "cache_sha256": cache_record["sha256"]})
            metadata = {"clip": CLIP, "frame": t, "pts": times[t], "normalization": normal,
                        "source_cache_sha256": cache_record["sha256"], "native_gs_sha256": inputs[path]["sha256"],
                        "codec_checkpoint_sha256": bound[codec_root / "codec_final.pt"]["sha256"],
                        "input_binding_sha256": binding_record["sha256"]}
            state_path = state_dir / f"{t:03d}.pt"
            temporary = state_path.with_suffix(".pt.tmp")
            codec.save_state(temporary, encoded, metadata)
            state = torch.load(temporary, map_location="cpu", weights_only=True)
            require(state["schema"] == "gsworld_continuous_state_v1" and state["feature_dtype"] == "float32",
                    "Serialized codec state schema mismatch")
            require(torch.equal(state["coordinates"], coords.cpu()) and torch.equal(state["features"], features.cpu()),
                    "Serialized latent changed")
            # Official save_state fields plus the actual motion-side state fields.
            # No fabricated source_state_sha256: this is the first state for this new clip.
            state.update({"reconstruction": images.cpu(), "pts": times[t], "clip": CLIP, "frame": t,
                          "normalization": normal, "baseline_real": row["codec_real"],
                          "source_cache_sha256": cache_record["sha256"]})
            atomic_torch_save(torch, state, state_path)
            state_record = fingerprint(state_path, budget)
            row.update({"state_sha256": state_record["sha256"], "state_bytes": state_record["bytes"],
                        "cache_bytes": cache_record["bytes"]})
            write_json(root / "quality" / CLIP / f"{t:03d}.json", row)
            rows.append(row)
            write_json(root / "progress.json", {"status": "preparing", "frames": len(rows), "total": FRAMES,
                       "elapsed_seconds": budget.elapsed(), "scientific_transfer_passed": None})
            print(f"PREPARE_FROZEN_CODEC {t + 1}/{FRAMES} voxels={count} latent={len(coords)}", flush=True)
            del gs, native, old, normalized, pseudo, real, representative, vox, payload, encoded
            del coords, features, decoded, predicted_gs, predicted_sh, images, state
    require(len(rows) == FRAMES, "Incomplete frame coverage")
    require(state_digest(model, budget) == initial_state_digest, "Frozen codec parameters/buffers changed")
    for record in [binding_record] + list(bound.values()) + list(inputs.values()):
        current = Path(record["path"]).stat()
        require((current.st_size, current.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                f"Input changed during preparation: {record['path']}")
    budget.check()
    keys = ["teacher_real", "representative_real", "representative_teacher", "codec_real", "codec_teacher", "pseudo_view_teacher"]
    averages = {key: {metric: float(np.mean([r[key][metric] for r in rows])) for metric in rows[0][key]} for key in keys}
    summary = {"status": "complete", "preparation_contract_passed": True, "scientific_transfer_passed": None,
               "scope": "Project cross-source transfer candidate; not an official or historically unexposed holdout",
               "clip": CLIP, "frames": FRAMES, "adjacent_pairs": FRAMES - 1, "source_points_per_frame": N,
               "all_input_points_aggregated": True, "normalization": normal, "codec_frozen": True,
               "state_quantization": "Identity; continuous float32 Nx64", "training_performed": False,
               "codec_state_sha256_before_and_after": initial_state_digest, "quality": averages,
               "min_native_parity_psnr": min(r["native_render_parity"]["psnr"] for r in rows),
               "min_similarity_parity_psnr": min(r["similarity_parity"]["psnr"] for r in rows),
               "max_native_parity_mae": max(r["native_render_parity"]["mae"] for r in rows),
               "max_similarity_parity_mae": max(r["similarity_parity"]["mae"] for r in rows),
               "max_input_voxels": max(r["input_voxels"] for r in rows),
               "max_decoder_candidates": max(r["max_decoder_candidates"] for r in rows),
               "max_decoded_gaussians": max(r["decoded_gaussians"] for r in rows),
               "elapsed_seconds": budget.elapsed(), "input_binding_sha256": binding_record["sha256"],
               "nearby_view": "Teacher pseudo-view, camera-local x +0.02 in normalized world; no new-view ground truth",
               "region_metrics": "Fixed central workspace crop only; no copied tower17 semantic boxes"}
    write_json(root / "quality" / f"{CLIP}.json", {"frames": rows, "summary": averages})
    write_json(root / "SUMMARY.json", summary)
    print("FROZEN_CODEC_PREPARATION_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--codec-root", required=True, type=Path)
    parser.add_argument("--clip", choices=[CLIP], default=CLIP)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    require(0 < args.max_seconds <= 1200, "Time budget must be in 1..1200 seconds")
    args.root, args.teacher, args.codec_root = [p.resolve(strict=True) for p in (args.root, args.teacher, args.codec_root)]
    require(args.root.name == RUN, f"Output root must be the dedicated {RUN} directory")
    require(not args.root.is_relative_to(args.codec_root) and not args.root.is_relative_to(args.teacher),
            "Output root must not be inside old codec data or teacher data")
    for name in ["cache", "states", "SUMMARY.json", "FAILED.json", "progress.json", "inputs_hashes.json", "normalization.json", "quality", "mappings"]:
        require(not (args.root / name).exists(), f"Refusing overwrite or automatic retry: {args.root / name}")
    budget = Budget(args.max_seconds)

    def alarm(signum, frame):
        raise TimeoutError("Preparation wall-clock budget exceeded; no retry")

    if hasattr(signal, "setitimer"):
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        prepare(args, budget)
    except Exception as error:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
        write_json(args.root / "FAILED.json", {"status": "failed", "preparation_contract_passed": False,
                   "scientific_transfer_passed": None, "exception": type(error).__name__, "reason": str(error),
                   "elapsed_seconds": budget.elapsed(), "automatic_retries": 0})
        raise
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
