"""Two-arm, full-frame state-codec experiment. Never loads old optimizer or latents.

Runtime imports of the bound original codec/renderer are delegated to sources.py.
The pure helpers below deliberately work on CPU without MinkowskiEngine.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

SEED = 20261031
STEPS = 300
FORCING_STEPS = 240
MAX_SECONDS = 1800
PREFLIGHT_SECONDS = 900
MAX_MEMORY = 48 * 2**30
INITIAL_SHA256 = "a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917"
ARMS = ("decoder_only", "full_codec")
TRAINABLE_DECODER = ("ae.decoder", "ae.feature_heads", "ae.gaussian_multiplier")
LOSS_WEIGHTS = {"real_l1": 1., "real_vgg19": .1, "teacher_l1": .5,
                "pseudo_view_l1": .25, "workspace_roi_l1": .25, "occupancy_bce": 1.}
SCREENSHOT_FRAMES = (0, 44, 88, 132, 176)
ROI = (67, 38, 380, 227)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def in_decoder(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in TRAINABLE_DECODER)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def snapshot(model, *, encoder_only=False, parameters_only=False):
    items = model.named_parameters() if parameters_only else model.state_dict().items()
    return {name: tensor_hash(value) for name, value in items
            if not encoder_only or not in_decoder(name)}


def digest_snapshot(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assert_frozen(model, expected):
    current = snapshot(model, encoder_only=True)
    changed = sorted(name for name in current.keys() | expected.keys() if current.get(name) != expected.get(name))
    if changed:
        raise RuntimeError("Frozen encoder parameters/buffers changed: " + ", ".join(changed[:12]))


def configure_arm(model, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if not isinstance(model.ae.vq, torch.nn.Identity) or list(model.ae.vq.parameters()):
        raise RuntimeError("State VQ must remain parameter-free Identity")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(arm == "full_codec" or in_decoder(name))
        parameter.grad = None
    if arm == "full_codec":
        model.train()
    else:
        # Encoding modules (including normalization buffers) stay in eval mode.
        # Original forward remains responsible for decoder target forcing.
        model.eval()
        for prefix in TRAINABLE_DECODER:
            model.get_submodule(prefix).train()
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Empty trainable set")
    assert_training_modes(model, arm)


def assert_training_modes(model, arm):
    if arm == "decoder_only":
        unexpected = [name for name, module in model.named_modules() if module.training and not in_decoder(name)]
        if unexpected:
            raise RuntimeError("Frozen modules entered train mode: " + ", ".join(unexpected[:12]))


def fresh_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-4, betas=(.9, .95), weight_decay=0.)


def forcing_for_step(step):
    if not 1 <= step <= STEPS:
        raise ValueError("Production step must lie in [1, 300]")
    return step <= FORCING_STEPS


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def group_name(name):
    if in_decoder(name):
        return next(prefix for prefix in TRAINABLE_DECODER if name.startswith(prefix + "."))
    if name.startswith("ae.encoder."):
        return "ae.encoder"
    if name.startswith("ae.embeddings."):
        return "ae.embeddings"
    if name.startswith("set_encoder."):
        return "set_encoder"
    return "other_encoding"


def gradient_report(model):
    report = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("Frozen parameter received a gradient: " + name)
            continue
        entry = report.setdefault(group_name(name), {"parameters": 0, "with_grad": 0, "nonzero_grad": 0})
        entry["parameters"] += 1
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("Nonfinite gradient: " + name)
            entry["with_grad"] += 1
            entry["nonzero_grad"] += int(bool(parameter.grad.detach().count_nonzero()))
    return report


def b2_update(model, optimizer, pair, frame_loader, terms_fn, vgg, force, *, guard=lambda: None):
    """One complete frame from each TRAIN clip; arithmetic mean of two losses."""
    if len(pair) != 2 or pair[0][0] == pair[1][0]:
        raise ValueError("B2 update requires exactly two distinct TRAIN clips")
    optimizer.zero_grad(set_to_none=True)
    records = []
    for clip, frame in pair:
        guard()
        payload = frame_loader(clip, frame)
        losses, info = terms_fn(model, payload, vgg, force)
        if set(losses) != set(LOSS_WEIGHTS):
            raise RuntimeError("Original loss terms differ from fixed protocol")
        weighted = sum(LOSS_WEIGHTS[name] * value for name, value in losses.items())
        if not bool(torch.isfinite(weighted)):
            raise RuntimeError("Nonfinite training loss")
        (weighted / 2.).backward()
        guard()
        records.append({"clip": clip, "frame": frame, "loss": float(weighted.detach()),
                        "terms": {name: float(value.detach()) for name, value in losses.items()},
                        "latent_count": int(info["latent_count"]), "output_count": int(info["output_count"])})
        del payload, losses, info, weighted
    gradients = gradient_report(model)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
    if float(norm) <= 0:
        raise RuntimeError("Zero gradient update")
    guard()
    optimizer.step()
    guard()
    return {"frames": records, "force_target": bool(force), "loss": sum(x["loss"] for x in records) / 2.,
            "terms": {key: sum(x["terms"][key] for x in records) / 2. for key in LOSS_WEIGHTS},
            "gradient_norm": float(norm), "gradient_groups": gradients}


class Budget:
    def __init__(self, seconds=MAX_SECONDS):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if torch.cuda.max_memory_allocated() > MAX_MEMORY or torch.cuda.max_memory_reserved() > MAX_MEMORY:
                raise RuntimeError("48 GiB GPU memory guard exceeded; no truncation")
        if self.elapsed() > self.seconds:
            raise TimeoutError(f"Phase exceeded {self.seconds} second wall-time budget")

    def elapsed(self):
        return time.monotonic() - self.started

    def evidence(self):
        return {"elapsed_seconds": self.elapsed(), "max_seconds": self.seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0}


def setup_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Runtime phases require CUDA; use test_worker.py for CPU-only tests")
    torch.set_num_threads(6)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., MAX_MEMORY / total), 0)
    torch.cuda.reset_peak_memory_stats()


def bound_evidence(bundle):
    from sources import balanced_plan
    validate_protocol(bundle.protocol)
    if sha(bundle.checkpoint) != INITIAL_SHA256:
        raise RuntimeError("Initial codec checkpoint hash mismatch")
    plan = balanced_plan(SEED, STEPS)
    return {"binding_sha256": bundle.binding_sha256, "initial_checkpoint_sha256": INITIAL_SHA256,
            "protocol_sha256": sha(bundle.root / "protocol.json"),
            "worker_sha256": sha(Path(__file__)), "sources_sha256": sha(Path(__file__).with_name("sources.py")),
            "plan_sha256": hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest(),
            "inherited_codec_steps": 3000, "additional_codec_updates": 300, "seed": SEED,
            "batch_frames": 2, "forcing_updates": 240, "unforced_updates": 60,
            "goal_complete": False, "scope": "Codec reconstruction comparison only; no action/transfer conclusion"}


def validate_protocol(protocol):
    """Reject configuration drift rather than silently running hard-coded defaults."""
    from sources import ALL_CLIPS, TRAIN_CLIPS
    checks = {
        ("seed",): SEED,
        ("source", "checkpoint_sha256"): INITIAL_SHA256,
        ("source", "original_completed_steps"): 3000,
        ("train", "steps"): STEPS, ("train", "seed"): SEED,
        ("train", "batch_size"): 2, ("train", "one_frame_per_train_clip_each_step"): True,
        ("train", "learning_rate"): 1e-4, ("train", "betas"): [.9, .95],
        ("train", "weight_decay"): 0., ("train", "grad_clip"): 1.,
        ("train", "loss"): LOSS_WEIGHTS, ("train", "save_steps"): [150, 300],
        ("train", "force_target_steps"): 240, ("train", "no_force_steps"): 60,
        ("train", "max_seconds"): MAX_SECONDS,
        ("preflight", "max_seconds"): PREFLIGHT_SECONDS,
        ("preflight", "updates_per_arm"): 2, ("preflight", "force_target_sequence"): [True, False],
        ("preflight", "frames_per_clip"): [0, 44],
        ("evaluation", "max_seconds"): MAX_SECONDS,
        ("evaluation", "snapshot_frames"): list(SCREENSHOT_FRAMES),
        ("evaluation", "workspace_roi_xyxy"): list(ROI),
        ("evaluation", "records_per_model"): 531, ("evaluation", "target_forcing"): False,
        ("evaluation", "saved_state_replay_psnr_min"): 60.,
        ("evaluation", "saved_state_replay_mae_max"): 1e-5,
        ("model", "latent_width"): 64, ("model", "stride"): 4,
        ("model", "grid"): .008, ("model", "gaussians_per_voxel"): 4,
        ("model", "state_quantization"): False,
        ("model", "max_input_voxels"): 80000,
        ("model", "max_decoder_candidates"): 600000,
        ("model", "max_decoded_gaussians"): 2400000,
        ("data", "train_clips"): list(TRAIN_CLIPS),
        ("data", "evaluation_clips"): list(ALL_CLIPS),
        ("data", "frames_per_clip"): 177, ("data", "teacher_points_per_frame"): 112896,
        ("data", "height"): 252, ("data", "width"): 448,
        ("resources", "peak_limit_gib"): 48, ("resources", "cpu_threads"): 6,
    }
    for keys, expected in checks.items():
        actual = protocol
        for key in keys:
            actual = actual[key]
        if actual != expected:
            raise RuntimeError("Protocol drift at " + ".".join(keys))


def require_preflight(root, evidence):
    path = Path(root) / "PREFLIGHT.json"
    if not path.is_file():
        raise RuntimeError("Successful complete two-arm preflight is required")
    gate = read_json(path)
    if gate.get("status") != "passed" or set(gate.get("arms", {})) != set(ARMS):
        raise RuntimeError("Both preflight arms must pass")
    arms = gate["arms"]
    if not gate.get("initial_arms_identical") or any(
            not isinstance(arms[arm], dict) or arms[arm].get("passed") is not True
            or arms[arm].get("discarded_updates") != 2 or arms[arm].get("production_updates") != 0
            for arm in ARMS):
        raise RuntimeError("Preflight did not complete both discarded B2 checks")
    if arms[ARMS[0]].get("initial_model_sha256") != arms[ARMS[1]].get("initial_model_sha256"):
        raise RuntimeError("Preflight arms did not initialize identically")
    if gate.get("source") != evidence:
        raise RuntimeError("Preflight binding is stale: code, source or protocol changed")
    return gate


def initial_model(bundle):
    seed_all()
    return bundle.codec.load_codec(bundle.codec_root, bundle.checkpoint)


def image_metrics(image, target):
    difference = image.float() - target.float()
    x0, y0, x1, y1 = ROI
    roi = difference[y0:y1, x0:x1]
    return {"mae": float(difference.abs().mean()),
            "psnr": float(-10 * difference.square().mean().clamp_min(1e-12).log10()),
            "workspace_roi_mae": float(roi.abs().mean()),
            "workspace_roi_psnr": float(-10 * roi.square().mean().clamp_min(1e-12).log10())}


def compare_render(a, b):
    result = image_metrics(a, b)
    if not all(np.isfinite(value) for value in result.values()) or result["psnr"] < 60 or result["mae"] > 1e-5:
        raise RuntimeError("State render replay failed: " + json.dumps(result))
    return result


def state_packet(encoded, metadata):
    coordinates = encoded["coordinates"].detach().cpu().clone()
    features = encoded["features"].detach().cpu().clone()
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("State coordinates must be integer Nx3")
    if features.dtype != torch.float32 or features.shape != (len(coordinates), 64) or not bool(torch.isfinite(features).all()):
        raise RuntimeError("Fresh encoder state must be finite Nx64 float32")
    return {"schema": "multisource_continuous_encoder_state_v1", "coordinates": coordinates,
            "features": features, "stride": int(encoded["stride"]), "grid": float(encoded["grid"]),
            "feature_dtype": "float32", "metadata": metadata}


@torch.no_grad()
def encode_replay(model, payload, render_prediction, metadata, state_path=None):
    model.eval()
    if model.ae.decoder.training:
        raise RuntimeError("Evaluation decoder must be eval")
    encoded = model.encode_state(payload["representatives"], payload["set_statistics"])
    packet = state_packet(encoded, metadata)
    # Direct sparse decode is explicitly target-free; saved replay rebuilds its coordinates.
    direct, _ = model.ae.decode(encoded["sparse"], target_key=None)
    direct = model._world(direct)
    packet["metadata"]["decoded_gaussians"] = len(next(iter(direct.values())))
    image = render_prediction(direct, payload, 0)
    nearby = render_prediction(direct, payload, 1)
    del direct, encoded
    if state_path is not None:
        atomic_save(packet, state_path)
        replay = torch.load(state_path, map_location="cpu", weights_only=True)
    else:
        replay = packet  # Explicit independent CPU tensor copies from state_packet.
    device = payload["representatives"].device
    reconstructed = model.decode_state(replay["coordinates"].to(device), replay["features"].to(device), replay["stride"])
    replay_image = render_prediction(reconstructed, payload, 0)
    replay_nearby = render_prediction(reconstructed, payload, 1)
    replay_metrics = {"real_view": compare_render(image, replay_image), "nearby_view": compare_render(nearby, replay_nearby)}
    return image, nearby, packet, replay_metrics


def preflight(bundle, requested_arm=None):
    from sources import TRAIN_CLIPS, balanced_plan, load_frame, verify_sources
    root = bundle.root
    evidence = bound_evidence(bundle)
    output = root / ("PREFLIGHT.json" if requested_arm is None else f"PREFLIGHT_{requested_arm}.json")
    if any(root.glob("PREFLIGHT*")):
        raise RuntimeError("Preflight already attempted; preserve evidence and use a separately authorized new attempt/root")
    # One attempt per root. An incomplete or failed diagnostic never authorizes training.
    write_json(output, {"status": "running", "source": evidence, "goal_complete": False})
    if requested_arm is not None:
        write_json(root / "PREFLIGHT.json", {"status": "incomplete", "source": evidence,
                   "reason": "Single-arm diagnostic cannot authorize training; use a new attempt/root", "goal_complete": False})
    budget = Budget(PREFLIGHT_SECONDS)
    results, initial_images = {}, {}
    initial_digest = None
    try:
        plan = [tuple((clip, frame) for clip in TRAIN_CLIPS) for frame in bundle.protocol["preflight"]["frames_per_clip"]]
        fixed = [entry for pair in plan[:2] for entry in pair]
        if len(set(fixed)) != 4 or {clip for clip, _ in fixed} != set(TRAIN_CLIPS):
            raise RuntimeError("Preflight requires four distinct TRAIN frames")
        for arm in ARMS if requested_arm is None else (requested_arm,):
            model = initial_model(bundle)
            complete_digest = digest_snapshot(snapshot(model))
            if initial_digest is not None and initial_digest != complete_digest:
                raise RuntimeError("Initial full model differs between arms")
            initial_digest = complete_digest
            states = []
            for clip, frame in fixed:
                budget.check()
                payload = load_frame(bundle.sources[clip], frame)
                image, nearby, packet, replay = encode_replay(model, payload, bundle.train.render_prediction,
                                                              {"clip": clip, "frame": frame, "phase": "discarded_preflight"})
                key = (clip, frame)
                real_quality = image_metrics(image, payload["real"])
                consistency = None
                if key in initial_images:
                    first_image, first_nearby, first_quality = initial_images[key]
                    consistency = {"real_view": compare_render(image.cpu(), first_image),
                                   "nearby_view": compare_render(nearby.cpu(), first_nearby),
                                   "real_psnr_difference": abs(real_quality["psnr"] - first_quality["psnr"])}
                    if consistency["real_psnr_difference"] > bundle.protocol["preflight"]["required_initial_render_psnr_tolerance"]:
                        raise RuntimeError("Initial arm reconstruction PSNR differs")
                initial_images[key] = (image.cpu(), nearby.cpu(), real_quality)
                states.append({"clip": clip, "frame": frame, "shape": list(packet["features"].shape),
                               "replay": replay, "initial_arm_consistency": consistency,
                               "codec_real": real_quality})
                del payload, image, nearby, packet
            configure_arm(model, arm)
            frozen = snapshot(model, encoder_only=True)
            before_parameters = snapshot(model, parameters_only=True)
            optimizer = fresh_optimizer(model)
            if optimizer.state:
                raise RuntimeError("Optimizer must be fresh")
            vgg = bundle.perceptual.Vgg19(bundle.codec_root / "torch_cache").cuda()
            updates = []
            for pair, force in zip(plan[:2], (True, False)):
                updates.append(b2_update(model, optimizer, pair,
                               lambda clip, frame: load_frame(bundle.sources[clip], frame),
                               bundle.train.terms, vgg, force, guard=budget.check))
                assert_training_modes(model, arm)
                if arm == "decoder_only":
                    assert_frozen(model, frozen)
            changed = [name for name, value in snapshot(model, parameters_only=True).items()
                       if before_parameters[name] != value]
            groups = {group_name(name) for name in changed}
            required = set(TRAINABLE_DECODER)
            if arm == "full_codec":
                required.update(group_name(name) for name, _ in model.named_parameters() if not in_decoder(name))
            if not required.issubset(groups):
                raise RuntimeError("No parameter updates in required groups: " + str(sorted(required - groups)))
            for group in required:
                if not any(update["gradient_groups"].get(group, {}).get("nonzero_grad", 0) for update in updates):
                    raise RuntimeError("Missing finite nonzero gradient group: " + group)
            results[arm] = {"passed": True, "discarded_updates": 2, "production_updates": 0,
                            "initial_model_sha256": complete_digest, "states": states, "updates": updates,
                            "initial_encoder_sha256": digest_snapshot(frozen),
                            "final_encoder_sha256": digest_snapshot(snapshot(model, encoder_only=True)),
                            "changed_parameter_groups": sorted(groups), "changed_parameters": len(changed),
                            "frozen_encoder_parameters_and_buffers_unchanged": arm == "decoder_only"}
            del optimizer, vgg, model
            gc.collect()
            torch.cuda.empty_cache()
        verify_sources(bundle)
        budget.check()
        write_json(output, {"status": "passed", "source": evidence, "arms": results,
                            "initial_arms_identical": requested_arm is None, "goal_complete": False, **budget.evidence()})
    except BaseException as error:
        write_json(output, {"status": "failed", "source": evidence, "arms": results,
                            "error": repr(error), "goal_complete": False, **budget.evidence()})
        raise


def train(bundle, arm):
    from sources import TRAIN_CLIPS, balanced_plan, load_frame, verify_sources
    evidence = bound_evidence(bundle)
    gate = require_preflight(bundle.root, evidence)
    folder = bundle.root / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ("trace.jsonl", "TRAIN_DONE.json", "FAILURE.json")):
        raise RuntimeError("Arm already attempted; use a new run root, never silently resume or overwrite")
    budget = Budget()
    write_json(folder / "SOURCE.json", {**evidence, "arm": arm, "preflight_sha256": sha(bundle.root / "PREFLIGHT.json")})
    model, optimizer = None, None
    completed, seen, no_force, context, event = 0, set(), 0, {}, "initialization"
    initial_model_hash = None

    def checkpoint(path, eligible=False, failed=False):
        if model is None:
            return
        atomic_save({"schema": "multisource_codec_checkpoint_v1", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict() if optimizer is not None else {},
                     "arm": arm, "step": completed, "inherited_codec_steps": 3000,
                     "additional_codec_updates": completed, "initial_checkpoint_sha256": INITIAL_SHA256,
                     "initial_model_sha256": initial_model_hash, "source": evidence,
                     "encoder_sha256": digest_snapshot(snapshot(model, encoder_only=True)),
                     "eligible_for_final_evaluation": eligible, "failed": failed,
                     "resumable": False, "frames_seen": sorted(seen), "no_target_forcing_steps": no_force,
                     **budget.evidence()}, path)

    try:
        model = initial_model(bundle)
        initial_model_hash = digest_snapshot(snapshot(model))
        if initial_model_hash != gate["arms"][arm]["initial_model_sha256"]:
            raise RuntimeError("Training initial model differs from preflight")
        configure_arm(model, arm)
        frozen = snapshot(model, encoder_only=True)
        encoder_parameters = snapshot(model, encoder_only=True, parameters_only=True)
        optimizer = fresh_optimizer(model)
        if optimizer.state:
            raise RuntimeError("Old optimizer state present")
        vgg = bundle.perceptual.Vgg19(bundle.codec_root / "torch_cache").cuda()
        plan = balanced_plan(SEED, STEPS)
        with (folder / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for step, pair in enumerate(plan, 1):
                if {clip for clip, _ in pair} != set(TRAIN_CLIPS):
                    raise RuntimeError("Evaluation clip or unbalanced training pair")
                context = {"attempted_step": step, "pair": pair}
                event = "update_in_progress"
                row = b2_update(model, optimizer, pair,
                                lambda clip, frame: load_frame(bundle.sources[clip], frame),
                                bundle.train.terms, vgg, forcing_for_step(step), guard=budget.check)
                completed = step
                event = "update_complete"
                no_force += int(not forcing_for_step(step))
                seen.update(f"{clip}/{frame:03d}" for clip, frame in pair)
                assert_training_modes(model, arm)
                if arm == "decoder_only" and (step == 1 or step % 25 == 0):
                    assert_frozen(model, frozen)
                row.update({"step": step, "arm": arm, "seen_frames": len(seen), **budget.evidence()})
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step == 177 and len(seen) != 354:
                    raise RuntimeError("First 177 updates must cover all 354 TRAIN frames")
                if step == 150:
                    checkpoint(folder / "step_0150.pt")
                if step == 1 or step % 10 == 0:
                    write_json(folder / "PROGRESS.json", row)
                    print("CODEC_TRAIN", arm, step, row["loss"], flush=True)
        event = "final_validation"
        if completed != 300 or len(seen) != 354 or no_force != 60:
            raise RuntimeError("Incomplete production training")
        if arm == "decoder_only":
            assert_frozen(model, frozen)
        elif snapshot(model, encoder_only=True, parameters_only=True) == encoder_parameters:
            raise RuntimeError("Full codec encoder parameters did not update")
        verify_sources(bundle)
        budget.check()
        checkpoint(folder / "step_0300.pt", eligible=True)
        budget.check()
        receipt = {"status": "complete", "arm": arm, "source": evidence,
                   "steps": 300, "inherited_codec_steps": 3000, "additional_codec_updates": 300,
                   "frames_seen": 354, "first_epoch_coverage_at_step": 177, "no_target_forcing_steps": 60,
                   "initial_model_sha256": initial_model_hash, "checkpoint_sha256": sha(folder / "step_0300.pt"),
                   "initial_encoder_sha256": digest_snapshot(frozen),
                   "final_encoder_sha256": digest_snapshot(snapshot(model, encoder_only=True)),
                   "initial_encoder_parameters_sha256": digest_snapshot(encoder_parameters),
                   "final_encoder_parameters_sha256": digest_snapshot(snapshot(model, encoder_only=True, parameters_only=True)),
                   "fresh_optimizer_verified": True, "trace_sha256": sha(folder / "trace.jsonl"),
                   "checkpoint_0150_sha256": sha(folder / "step_0150.pt"),
                   "frozen_encoder_verified": arm == "decoder_only", "goal_complete": False}
        budget.check()
        receipt.update(budget.evidence())
        write_json(folder / "TRAIN_DONE.json", receipt)
    except BaseException as error:
        saved = False
        try:
            checkpoint(folder / "FAILED_DO_NOT_EVALUATE.pt", failed=True)
            saved = model is not None
        except BaseException:
            pass
        write_json(folder / "FAILURE.json", {"status": "failed", "error": repr(error), "event": event,
                   "context": context, "last_completed_step": completed, "failed_checkpoint_saved": saved,
                   "eligible_for_final_evaluation": False, "goal_complete": False, **budget.evidence()})
        raise


def evaluate(bundle, arm=None):
    from sources import ALL_CLIPS, TRAIN_CLIPS, load_frame, verify_sources
    evidence = bound_evidence(bundle)
    require_preflight(bundle.root, evidence)
    label = arm or "baseline"
    folder = bundle.root / "evaluations" / label
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ("targets.jsonl", "EVALUATION.json", "FAILURE.json")):
        raise RuntimeError("Evaluation already attempted; do not overwrite evidence")
    budget = Budget()
    checkpoint_path = bundle.checkpoint
    step = 0
    rows = []
    context = {}
    write_json(folder / "SOURCE.json", {**evidence, "arm": label})
    try:
        if arm is not None:
            arm_folder = bundle.root / "arms" / arm
            if (arm_folder / "FAILURE.json").exists():
                raise RuntimeError("Failed training arm cannot be evaluated as complete")
            done = read_json(arm_folder / "TRAIN_DONE.json")
            checkpoint_path = arm_folder / "step_0300.pt"
            if done.get("status") != "complete" or done.get("steps") != 300 or done.get("source") != evidence:
                raise RuntimeError("Complete, correctly bound 300-update training required")
            if sha(checkpoint_path) != done["checkpoint_sha256"]:
                raise RuntimeError("Final checkpoint changed after training")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved.get("arm") != arm or saved.get("failed") or not saved.get("eligible_for_final_evaluation"):
                raise RuntimeError("Ineligible checkpoint")
            if saved.get("step") != 300 or saved.get("source") != evidence:
                raise RuntimeError("Checkpoint provenance mismatch")
            del saved
            step = 300
        seed_all()
        checkpoint_sha = sha(checkpoint_path)
        model = bundle.codec.load_codec(bundle.codec_root, checkpoint_path).eval()
        encoder_sha = digest_snapshot(snapshot(model, encoder_only=True))
        from PIL import Image
        with (folder / "targets.jsonl").open("x", encoding="utf-8", buffering=1) as metrics_log, \
             (folder / "state_manifest.jsonl").open("x", encoding="utf-8", buffering=1) as manifest, \
             (folder / "screenshot_manifest.jsonl").open("x", encoding="utf-8", buffering=1) as screenshot_manifest:
            for clip in ALL_CLIPS:
                states = folder / "states" / clip
                screenshots = folder / "screenshots" / clip
                states.mkdir(parents=True, exist_ok=True)
                screenshots.mkdir(parents=True, exist_ok=True)
                for frame in range(177):
                    context = {"clip": clip, "frame": frame}
                    budget.check()
                    payload = load_frame(bundle.sources[clip], frame)
                    metadata = {"clip": clip, "frame": frame, "arm": label,
                                "split": "TRAIN" if clip in TRAIN_CLIPS else "EVAL",
                                "normalization": payload["normalization"], "pts": payload["pts"],
                                "source_cache_sha256": payload["_source_cache_sha256"],
                                "checkpoint_sha256": checkpoint_sha, "encoder_sha256": encoder_sha,
                                "binding_sha256": bundle.binding_sha256, "inherited_codec_steps": 3000,
                                "additional_codec_updates": step, "decoder_target_key": None,
                                "fresh_encode": True, "frame_count": 177}
                    path = states / f"{frame:03d}.pt"
                    image, nearby, packet, replay = encode_replay(model, payload, bundle.train.render_prediction, metadata, path)
                    teacher = payload["targets"][0].clamp(0, 1)
                    row = {"clip": clip, "frame": frame, "split": metadata["split"], "pts": payload["pts"],
                           "codec_real": image_metrics(image, payload["real"]),
                           "teacher_real": image_metrics(teacher, payload["real"]),
                           "codec_teacher": image_metrics(image, teacher),
                           "pseudo_view_teacher": image_metrics(nearby, payload["targets"][1].clamp(0, 1)),
                           "latent_positions": len(packet["coordinates"]), "latent_channels": 64,
                           "input_gaussians": int(payload["n_input"]),
                           "input_voxels": len(payload["representatives"]),
                           "decoded_gaussians": metadata["decoded_gaussians"],
                           "replay": replay, "checkpoint_sha256": checkpoint_sha,
                           "encoder_sha256": encoder_sha, "additional_codec_updates": step}
                    rows.append(row)
                    metrics_log.write(json.dumps(row, allow_nan=False) + "\n")
                    entry = {**metadata, "state_path": path.relative_to(folder).as_posix(), "state_sha256": sha(path),
                             "state_bytes": path.stat().st_size, "shape": list(packet["features"].shape),
                             "saved_packet_replay": replay}
                    manifest.write(json.dumps(entry, allow_nan=False) + "\n")
                    if frame in SCREENSHOT_FRAMES:
                        for name, picture in (("real", payload["real"]), ("teacher", teacher), ("codec", image),
                                              ("nearby", nearby), ("nearby_teacher", payload["targets"][1])):
                            pixels = (picture.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                            screenshot = screenshots / f"{frame:03d}_{name}.png"
                            Image.fromarray(pixels).save(screenshot)
                            screenshot_manifest.write(json.dumps({"clip": clip, "frame": frame, "kind": name,
                                "path": screenshot.relative_to(folder).as_posix(), "bytes": screenshot.stat().st_size,
                                "sha256": sha(screenshot)}, allow_nan=False) + "\n")
                    del payload, image, nearby, packet, teacher
                    budget.check()
                    if frame % 20 == 0 or frame == 176:
                        write_json(folder / "PROGRESS.json", {"frames_complete": len(rows), "total": 531, **context, **budget.evidence()})
                        print("CODEC_EVAL", label, clip, frame, flush=True)
        if len(rows) != 531 or len({(row["clip"], row["frame"]) for row in rows}) != 531:
            raise RuntimeError("Full all-source 531-frame evaluation required")
        verify_sources(bundle)
        budget.check()
        def average(selected):
            return {key: {metric: float(np.mean([row[key][metric] for row in selected])) for metric in selected[0][key]}
                    for key in ("codec_real", "teacher_real", "codec_teacher", "pseudo_view_teacher")}
        summary = {clip: average([row for row in rows if row["clip"] == clip]) for clip in ALL_CLIPS}
        receipt = {"status": "evaluated", "arm": label, "frames": 531,
                   "frames_per_clip": 177, "additional_codec_updates": step, "inherited_codec_steps": 3000,
                   "checkpoint_sha256": checkpoint_sha, "encoder_sha256": encoder_sha, "source": evidence,
                   "summary_by_clip": summary, "summary_all": average(rows), "roi_xyxy": list(ROI),
                   "roi_definition": "Fixed central workspace; no semantic hand mask",
                   "every_packet_saved_replay_passed": True, "all_frames_freshly_encoded": True,
                   "decoder_eval": True, "decoder_target_key": None,
                   "targets_sha256": sha(folder / "targets.jsonl"),
                   "state_manifest_sha256": sha(folder / "state_manifest.jsonl"),
                   "screenshot_manifest_sha256": sha(folder / "screenshot_manifest.jsonl"),
                   "goal_complete": False}
        budget.check()
        receipt.update(budget.evidence())
        write_json(folder / "EVALUATION.json", receipt)
    except BaseException as error:
        write_json(folder / "FAILURE.json", {"status": "failed", "error": repr(error), "context": context,
                   "frames_complete": len(rows), "valid_complete_evaluation": False,
                   "goal_complete": False, **budget.evidence()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("preflight", "baseline", "train", "eval"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode in ("train", "eval") and args.arm is None:
        parser.error("--arm is required for train/eval")
    if args.mode == "baseline" and args.arm is not None:
        parser.error("baseline is shared and does not accept --arm")
    from sources import load_verified
    bundle = load_verified(args.root.resolve())
    setup_gpu()
    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")
    prior = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == "preflight":
            preflight(bundle, args.arm)
        elif args.mode == "train":
            train(bundle, args.arm)
        else:
            evaluate(bundle, args.arm)
    finally:
        signal.signal(signal.SIGTERM, prior)


if __name__ == "__main__":
    main()
