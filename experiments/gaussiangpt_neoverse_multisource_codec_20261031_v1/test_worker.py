"""CPU-only regression tests: no native GS, MinkowskiEngine, SSH, or GPU jobs."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import sources
import worker


class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = nn.Module()
        self.ae.vq = nn.Identity()
        self.ae.embeddings = nn.Linear(4, 4)
        self.ae.encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.SiLU())
        self.ae.decoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.SiLU())
        self.ae.feature_heads = nn.Linear(4, 2)
        self.ae.gaussian_multiplier = nn.Linear(2, 2)
        self.set_encoder = nn.Linear(4, 4)

    def forward(self, payload, force):
        encoded = self.ae.encoder(self.ae.embeddings(payload) + self.set_encoder(payload))
        self.ae.decoder.train(force)
        return self.ae.gaussian_multiplier(self.ae.feature_heads(self.ae.decoder(encoded)))


def tiny_loader(clip, frame):
    return torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31 + frame * .1


def tiny_terms(model, payload, vgg, force):
    predicted = model(payload, force)
    value = .01 * (predicted - .7).square().mean()
    return {name: value for name in worker.LOSS_WEIGHTS}, {"latent_count": len(payload), "output_count": len(payload)}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(worker.SEED)
        torch.set_num_threads(1)

    def test_balanced_sampling_is_same_for_arms_and_covers354(self):
        first = sources.balanced_plan(worker.SEED)
        second = sources.balanced_plan(worker.SEED)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 300)
        self.assertEqual(len({entry for pair in first[:177] for entry in pair}), 354)
        self.assertEqual({clip for pair in first for clip, _ in pair}, set(sources.TRAIN_CLIPS))
        self.assertTrue(all(len(pair) == 2 and {entry[0] for entry in pair} == set(sources.TRAIN_CLIPS) for pair in first))
        self.assertNotEqual(first, sources.balanced_plan(worker.SEED + 1))

    def test_forcing_is_exact_240_60_independent_of_wallclock(self):
        forces = [worker.forcing_for_step(step) for step in range(1, 301)]
        self.assertEqual(forces, [True] * 240 + [False] * 60)
        for step in (0, 301):
            with self.assertRaises(ValueError):
                worker.forcing_for_step(step)

    def test_decoder_arm_freezes_parameters_buffers_and_training_modes(self):
        model = TinyCodec()
        worker.configure_arm(model, "decoder_only")
        frozen = worker.snapshot(model, encoder_only=True)
        before = worker.snapshot(model, parameters_only=True)
        optimizer = worker.fresh_optimizer(model)
        for force in (True, False):
            report = worker.b2_update(model, optimizer, ((sources.TRAIN_CLIPS[0], 0), (sources.TRAIN_CLIPS[1], 44)),
                                      tiny_loader, tiny_terms, None, force)
            worker.assert_frozen(model, frozen)
            worker.assert_training_modes(model, "decoder_only")
            self.assertEqual(model.ae.decoder.training, force)
            self.assertFalse(model.ae.encoder[1].training)
            self.assertTrue(all(value["nonzero_grad"] > 0 for value in report["gradient_groups"].values()))
        changed = {name for name, value in worker.snapshot(model, parameters_only=True).items() if value != before[name]}
        self.assertTrue(changed)
        self.assertTrue(all(worker.in_decoder(name) for name in changed))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not worker.in_decoder(n)))

    def test_full_codec_updates_encoder_embeddings_adapter_and_decoder(self):
        model = TinyCodec()
        worker.configure_arm(model, "full_codec")
        before = worker.snapshot(model, parameters_only=True)
        buffer_before = model.ae.encoder[1].running_mean.clone()
        optimizer = worker.fresh_optimizer(model)
        worker.b2_update(model, optimizer, ((sources.TRAIN_CLIPS[0], 0), (sources.TRAIN_CLIPS[1], 44)),
                         tiny_loader, tiny_terms, None, False)
        groups = {worker.group_name(name) for name, value in worker.snapshot(model, parameters_only=True).items() if value != before[name]}
        self.assertEqual(groups, set(worker.TRAINABLE_DECODER) | {"ae.encoder", "ae.embeddings", "set_encoder"})
        self.assertFalse(torch.equal(buffer_before, model.ae.encoder[1].running_mean))
        self.assertTrue(model.ae.encoder[1].training)
        self.assertFalse(model.ae.decoder.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_frozen_buffer_drift_is_detected(self):
        model = TinyCodec()
        frozen = worker.snapshot(model, encoder_only=True)
        model.ae.encoder[1].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "Frozen encoder"):
            worker.assert_frozen(model, frozen)

    def test_prefix_does_not_admit_lookalike_encoder_parameters(self):
        self.assertTrue(worker.in_decoder("ae.decoder.0.weight"))
        self.assertFalse(worker.in_decoder("ae.decoder_shadow.weight"))
        self.assertFalse(worker.in_decoder("set_encoder.weight"))

    def test_parameterized_quantizer_is_rejected(self):
        model = TinyCodec()
        model.ae.vq = nn.Linear(4, 4)
        with self.assertRaisesRegex(RuntimeError, "Identity"):
            worker.configure_arm(model, "full_codec")

    def test_new_optimizer_has_exact_hyperparameters_and_empty_state(self):
        model = TinyCodec()
        worker.configure_arm(model, "decoder_only")
        optimizer = worker.fresh_optimizer(model)
        self.assertEqual(dict(optimizer.state), {})
        group = optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"]), (1e-4, (.9, .95), 0.))
        self.assertEqual({id(p) for p in group["params"]}, {id(p) for p in model.parameters() if p.requires_grad})

    def test_b2_gradient_and_loss_are_arithmetic_mean(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        seen = []
        def loader(clip, frame):
            seen.append((clip, frame))
            return torch.tensor([[.1 if clip == "a" else .3]])
        def terms(model, target, vgg, force):
            loss = (model(torch.ones(1, 1)) - target).square().mean()
            losses = {key: loss * 0 for key in worker.LOSS_WEIGHTS}
            losses["real_l1"] = loss
            return losses, {"latent_count": 1, "output_count": 1}
        row = worker.b2_update(model, optimizer, (("a", 0), ("b", 1)), loader, terms, None, True)
        self.assertEqual(seen, [("a", 0), ("b", 1)])
        self.assertAlmostEqual(row["loss"], .05, places=6)
        self.assertAlmostEqual(row["gradient_norm"], .4, places=6)
        self.assertAlmostEqual(float(model.weight.detach()), .04, places=6)

    def test_invalid_pair_is_rejected_before_loading(self):
        model = TinyCodec()
        worker.configure_arm(model, "decoder_only")
        with self.assertRaises(ValueError):
            worker.b2_update(model, worker.fresh_optimizer(model), (("a", 0), ("a", 1)),
                             lambda *_: self.fail("must not load"), tiny_terms, None, True)

    def test_state_packet_is_independent_compact_cpu_copy(self):
        encoded = {"coordinates": torch.zeros(2, 3, dtype=torch.int32), "features": torch.ones(2, 64),
                   "stride": 4, "grid": .008, "sparse": object(), "old_inputs": torch.zeros(100)}
        metadata = {"checkpoint_sha256": "c" * 64, "encoder_sha256": "e" * 64,
                    "clip": sources.TRAIN_CLIPS[0], "frame": 0, "additional_codec_updates": 300}
        packet = worker.state_packet(encoded, metadata)
        encoded["features"].zero_()
        self.assertTrue(torch.equal(packet["features"], torch.ones(2, 64)))
        self.assertNotIn("sparse", packet)
        self.assertNotIn("old_inputs", packet)
        self.assertEqual(packet["metadata"], metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000.pt"
            worker.atomic_save(packet, path)
            loaded = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(loaded["features"], packet["features"]))

    def test_state_packet_rejects_stale_or_invalid_feature_schema(self):
        encoded = {"coordinates": torch.zeros(2, 3, dtype=torch.int32), "features": torch.ones(2, 32), "stride": 4, "grid": .008}
        with self.assertRaisesRegex(RuntimeError, "Nx64"):
            worker.state_packet(encoded, {})
        encoded["features"] = torch.full((2, 64), float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Nx64"):
            worker.state_packet(encoded, {})

    def test_replay_checks_both_mae_and_psnr_thresholds(self):
        image = torch.zeros(252, 448, 3)
        self.assertGreaterEqual(worker.compare_render(image, image)["psnr"], 60)
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            worker.compare_render(image, image + 1e-4)
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            worker.compare_render(image, image + float("nan"))

    def test_eval_reencodes_and_renders_actual_saved_packet(self):
        class ReplayAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(1, 1)
                self.targets = []
            def decode(self, sparse, target_key):
                self.targets.append(target_key)
                return {"points": sparse}, {}
        class ReplayCodec(nn.Module):
            def __init__(self):
                super().__init__()
                self.ae = ReplayAE()
                self.encode_calls = 0
                self.replay_calls = 0
            def encode_state(self, representatives, statistics):
                self.encode_calls += 1
                features = torch.full((2, 64), self.encode_calls / 10.)
                return {"coordinates": torch.zeros(2, 3, dtype=torch.int32), "features": features,
                        "sparse": features, "stride": 4, "grid": .008}
            def _world(self, points):
                return points
            def decode_state(self, coordinates, features, stride):
                self.replay_calls += 1
                self.asserted = not self.ae.decoder.training and stride == 4
                return {"points": features}
        model = ReplayCodec()
        payload = {"representatives": torch.zeros(2, 14), "set_statistics": torch.zeros(2, 29)}
        def render(points, payload, view):
            return points["points"].mean().expand(252, 448, 3) + view * .01
        with tempfile.TemporaryDirectory() as directory:
            for frame in range(2):
                path = Path(directory) / f"{frame}.pt"
                image, nearby, packet, replay = worker.encode_replay(model, payload, render,
                                                                     {"frame": frame, "checkpoint_sha256": "c" * 64}, path)
                actual = torch.load(path, weights_only=True)
                self.assertTrue(torch.equal(actual["features"], torch.full((2, 64), (frame + 1) / 10.)))
                self.assertEqual(actual["metadata"]["frame"], frame)
                self.assertEqual(replay["real_view"]["mae"], 0.)
                self.assertEqual(replay["nearby_view"]["mae"], 0.)
        self.assertEqual((model.encode_calls, model.replay_calls), (2, 2))
        self.assertTrue(model.asserted)
        self.assertEqual(model.ae.targets, [None, None])

    def test_training_failure_cannot_create_done_or_eligible_checkpoint(self):
        model = TinyCodec()
        initial_digest = worker.digest_snapshot(worker.snapshot(model))
        evidence = {"binding_sha256": "test"}
        gate = {"arms": {"decoder_only": {"initial_model_sha256": initial_digest}}}
        vgg = SimpleNamespace(cuda=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker.write_json(root / "PREFLIGHT.json", gate)
            bundle = SimpleNamespace(root=root, codec_root=root, sources={}, train=SimpleNamespace(terms=None),
                                     perceptual=SimpleNamespace(Vgg19=lambda *_: vgg))
            with patch.object(worker, "bound_evidence", return_value=evidence), \
                 patch.object(worker, "require_preflight", return_value=gate), \
                 patch.object(worker, "initial_model", return_value=model), \
                 patch.object(worker, "b2_update", side_effect=TimeoutError("time budget")):
                with self.assertRaises(TimeoutError):
                    worker.train(bundle, "decoder_only")
            folder = root / "arms" / "decoder_only"
            self.assertFalse((folder / "TRAIN_DONE.json").exists())
            failure = worker.read_json(folder / "FAILURE.json")
            self.assertEqual(failure["last_completed_step"], 0)
            self.assertFalse(failure["eligible_for_final_evaluation"])
            checkpoint = torch.load(folder / "FAILED_DO_NOT_EVALUATE.pt", weights_only=True)
            self.assertTrue(checkpoint["failed"])
            self.assertFalse(checkpoint["eligible_for_final_evaluation"])

    def test_preflight_missing_failed_partial_or_stale_cannot_train(self):
        evidence = {"binding_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                worker.require_preflight(root, evidence)
            for gate in ({"status": "failed", "arms": dict.fromkeys(worker.ARMS), "source": evidence},
                         {"status": "passed", "arms": {"decoder_only": {}}, "source": evidence},
                         {"status": "passed", "arms": dict.fromkeys(worker.ARMS), "source": {"binding_sha256": "b" * 64}}):
                worker.write_json(root / "PREFLIGHT.json", gate)
                with self.assertRaises(RuntimeError):
                    worker.require_preflight(root, evidence)
            gate = {"status": "passed", "arms": {arm: {"passed": True, "discarded_updates": 2,
                    "production_updates": 0, "initial_model_sha256": "m" * 64} for arm in worker.ARMS},
                    "initial_arms_identical": True, "source": evidence}
            worker.write_json(root / "PREFLIGHT.json", gate)
            self.assertEqual(worker.require_preflight(root, evidence), gate)

    def test_protocol_has_exact_fixed_contract_and_detects_drift(self):
        protocol = worker.read_json(Path(__file__).with_name("protocol.json"))
        worker.validate_protocol(protocol)
        changed = copy.deepcopy(protocol)
        changed["train"]["force_target_steps"] = 239
        with self.assertRaisesRegex(RuntimeError, "Protocol drift"):
            worker.validate_protocol(changed)

    def test_preflight_never_overwrites_prior_attempt(self):
        for filename in ("PREFLIGHT.json", "PREFLIGHT_full_codec.json", "PREFLIGHT.json.tmp"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = SimpleNamespace(root=root)
                worker.write_json(root / filename, {"status": "failed", "original": "retain"})
                before = (root / filename).read_bytes()
                with patch.object(worker, "bound_evidence", return_value={}):
                    for arm in (None, "decoder_only"):
                        with self.assertRaisesRegex(RuntimeError, "already attempted"):
                            worker.preflight(bundle, arm)
                self.assertEqual((root / filename).read_bytes(), before)

    def test_deadline_failure_does_not_depend_on_step_schedule(self):
        with patch.object(worker.time, "monotonic", side_effect=[0., 1801.]):
            budget = worker.Budget(1800)
            with self.assertRaises(TimeoutError):
                budget.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
