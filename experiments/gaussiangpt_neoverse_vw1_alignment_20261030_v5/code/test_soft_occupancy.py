"""Actual MinkowskiEngine + bound GaussianGPT decoder CPU tests.

Run in the existing gaussiangpt Python environment:
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -B test_soft_occupancy.py -v
GS_VENDOR_ROOT may override the existing bound vendor directory. No installation,
checkpoint, renderer, GPU, training run, or test artifact output is needed.

验证记录（2026-10-31）：将这两个源文件只读传入134现有环境执行，
/data/CJH/envs/gaussiangpt/bin/python，MinkowskiEngine 0.5.8，CPU。
Ran 11 tests in 0.231s; OK; 0 failures, 0 errors, 0 skipped.
真实vendor stage +预测剪枝夹具：2→45→36→441→252体素，repeat4→1008 GS。
该记录仅覆盖门控和opacity代理梯度，不表示已验证真实RGB renderer。
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

from soft_occupancy import (GateCoverageError, SoftOccupancyGate, SoftOccupancyTracker,
                            mean_gate_from_kernel_map, inspect_decoder_stages)


VENDOR = Path(os.environ.get("GS_VENDOR_ROOT",
    "/data/CJH/AGS-V6Direct/outputs/gaussiangpt_state_vae_20261028_v1/vendor/"
    "GaussianGPT-00658a4ab3f48daed3bdf4ba50805bb2b7dc5e14"))
AVAILABLE = importlib.util.find_spec("MinkowskiEngine") is not None and VENDOR.is_dir()
if AVAILABLE:
    import MinkowskiEngine as ME
    sys.path.insert(0, str(VENDOR))
    from model.vqvae.cnn.minkowski_decoder import SparseConvDecoder

    class CoordinateClassifier(nn.Module):
        """Deterministic test-only predicted pruning, without target coordinates."""
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            scores = torch.where(x.C[:, self.axis] >= 0, 1., -1.).reshape(-1, 1)
            return ME.SparseTensor(features=scores, coordinate_map_key=x.coordinate_map_key,
                                   coordinate_manager=x.coordinate_manager)


@unittest.skipUnless(AVAILABLE, "Needs the existing MinkowskiEngine and bound GaussianGPT vendor")
class SoftOccupancyCPUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261031)
        torch.set_num_threads(1)
        # Actual vendor stage structure, reduced channels only for a tiny CPU fixture.
        self.decoder = SparseConvDecoder(out_c=4, embed_dim=1, max_channel=4,
                                          stages=2, prune_thresh=[0., 0.]).cpu().eval()
        self.decoder.up_cls[0] = CoordinateClassifier(axis=1)
        self.decoder.up_cls[1] = CoordinateClassifier(axis=2)
        self.decoder.requires_grad_(False)
        self.input_coords = torch.tensor([[0, 4, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
        self.latent = ME.SparseTensor(features=torch.tensor([[1.], [2.]]),
                                      coordinates=self.input_coords,
                                      tensor_stride=[4, 4, 4], device="cpu")
        self.logits = torch.tensor([-.3, .4], requires_grad=True)

    def hook_counts(self):
        return {id(m): (len(m._forward_hooks), len(m._forward_pre_hooks))
                for m in self.decoder.modules()}

    def test_actual_vendor_structure_and_unique_index_alignment(self):
        specs = inspect_decoder_stages(self.decoder)
        self.assertEqual(len(specs), 2)
        self.assertTrue(all(s[0].kernel_size == (3, 3, 3) and s[0].stride == (2, 2, 2)
                            and s[0].module_path for s in specs))
        order = self.latent.unique_index.long()
        self.assertTrue(torch.equal(self.latent.C, self.input_coords[order]))
        tracker = SoftOccupancyGate(self.decoder, self.latent, self.logits)
        torch.testing.assert_close(tracker.probabilities_for(self.latent), self.logits.sigmoid()[order])

    def test_actual_pruning_coverage_multiplier_and_source_gradient(self):
        before = self.hook_counts()
        with SoftOccupancyGate(self.decoder, self.latent, self.logits) as tracker:
            out, _ = self.decoder(self.latent, target_key=None)
            gate = tracker.gaussian_probabilities(4 * len(out), multiplier=4)
        self.assertEqual(self.hook_counts(), before)
        self.assertEqual(str(out.F.device), "cpu")
        self.assertEqual([e["output_count"] for e in tracker.events], [45, 36, 441, 252, 252])
        self.assertEqual(len(gate), 1008)
        self.assertEqual([e["max_parent_count"] for e in tracker.events if e["kind"] == "up"], [2, 8])
        self.assertTrue(bool(torch.isfinite(gate).all()))
        self.assertTrue(bool(((gate > 0) & (gate < 1)).all()))
        torch.testing.assert_close(gate.reshape(-1, 4), gate.reshape(-1, 4)[:, :1].expand(-1, 4))
        loss = (.7 * gate).square().mean()  # Opacity proxy, not a renderer/RGB test.
        loss.backward()
        self.assertTrue(bool(torch.isfinite(self.logits.grad).all()))
        self.assertTrue(bool((self.logits.grad > 0).all()))

    def test_overlap_receives_gradient_from_both_actual_parents(self):
        captured = []
        with SoftOccupancyGate(self.decoder, self.latent, self.logits) as tracker:
            def capture(module, inputs, output):
                km = output.coordinate_manager.kernel_map(
                    inputs[0].coordinate_map_key, output.coordinate_map_key,
                    stride=2, kernel_size=3, dilation=1, is_transpose=True)
                _, degree = mean_gate_from_kernel_map(torch.ones(len(inputs[0])), km, len(output))
                captured.append(tracker.probabilities_for(output)[degree == 2][0])
            handle = self.decoder.ups[0].register_forward_hook(capture)
            try:
                self.decoder(self.latent, target_key=None)
            finally:
                handle.remove()
        gradients = torch.autograd.grad(captured[0], self.logits)[0]
        expected = self.logits.sigmoid() * (1 - self.logits.sigmoid()) / 2
        torch.testing.assert_close(gradients, expected)
        self.assertTrue(bool((gradients > 0).all()))

    def test_hooks_removed_on_exception_and_decoder_reusable(self):
        before = self.hook_counts()
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with SoftOccupancyGate(self.decoder, self.latent, self.logits):
                raise RuntimeError("sentinel")
        self.assertEqual(self.hook_counts(), before)
        with SoftOccupancyGate(self.decoder, self.latent, self.logits) as tracker:
            out, _ = self.decoder(self.latent)
            self.assertEqual(len(tracker.gaussian_probabilities(4 * len(out))), 4 * len(out))
        self.assertEqual(self.hook_counts(), before)

    def test_missing_map_coverage_and_empty_output_are_errors(self):
        with self.assertRaises(GateCoverageError):
            mean_gate_from_kernel_map(self.logits.sigmoid(), {0: torch.tensor([[0], [0]])}, 2)
        with self.assertRaises(GateCoverageError):
            mean_gate_from_kernel_map(self.logits.sigmoid(), {}, 0)

    def test_incorrect_stage_structure_rejected_without_hooks(self):
        self.decoder.ups[0] = nn.Identity()
        before = self.hook_counts()
        with self.assertRaisesRegex(ValueError, "generative transpose"):
            SoftOccupancyGate(self.decoder, self.latent, self.logits)
        self.assertEqual(self.hook_counts(), before)

    def test_target_forcing_and_training_mode_rejected(self):
        self.decoder.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            with SoftOccupancyGate(self.decoder, self.latent, self.logits):
                pass
        self.decoder.eval()
        before = self.hook_counts()
        with self.assertRaisesRegex(ValueError, "target_key"):
            with SoftOccupancyGate(self.decoder, self.latent, self.logits):
                self.decoder(self.latent, target_key=self.latent.coordinate_map_key)
        self.assertEqual(self.hook_counts(), before)

    def test_mismatched_multiplier_and_second_forward_rejected(self):
        with SoftOccupancyGate(self.decoder, self.latent, self.logits) as tracker:
            out, _ = self.decoder(self.latent)
            with self.assertRaisesRegex(ValueError, "GS count"):
                tracker.gaussian_probabilities(4 * len(out) + 1)
            with self.assertRaisesRegex(RuntimeError, "Only one"):
                self.decoder(self.latent)

    def test_no_forward_context_removes_hooks(self):
        before = self.hook_counts()
        with self.assertRaisesRegex(RuntimeError, "without a completed"):
            with SoftOccupancyGate(self.decoder, self.latent, self.logits):
                pass
        self.assertEqual(self.hook_counts(), before)

    def test_gs_bridge_compatible_api_and_statistics(self):
        with SoftOccupancyTracker(self.decoder, self.latent, self.logits,
                                  input_coordinates=self.input_coords) as tracker:
            out, _ = self.decoder(self.latent, target_key=None)
            gate = tracker.gaussian_gate(4)
            stats = tracker.summary()
        self.assertEqual(len(gate), 4 * len(out))
        self.assertEqual(stats['soft_gate_output_voxels'], len(out))
        self.assertEqual(stats['soft_gate_multiplier'], 4)
        self.assertTrue(stats['soft_gate_complete'])
        self.assertTrue(all(key.startswith('soft_gate_') for key in stats))

    def test_wrong_original_coordinate_order_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique_index alignment'):
            SoftOccupancyTracker(self.decoder, self.latent, self.logits,
                                 input_coordinates=self.input_coords.flip(0))


if __name__ == "__main__":
    unittest.main()
