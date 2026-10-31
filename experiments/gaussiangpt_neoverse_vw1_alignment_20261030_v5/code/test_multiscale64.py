"""Small CPU tests only: python -B -m unittest -v test_multiscale64.

No dataset, renderer, MinkowskiEngine, LFQ, GPU, or training entry point is used.
Losses use random weighted residuals, not a constant sum of normalized channels.
"""

import copy
import unittest
from unittest.mock import patch

import torch

from multiscale64 import (
    DECODER_SCALES, ENCODER_SCALES, MultiScaleBackbone64,
    build_neighbor_index_at_distance,
)
from spatial64 import AXIS_OFFSETS


class MultiScale64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261031)
        # Each requested scale has at least one true axis edge in this support.
        self.coords = torch.tensor([[x, 0, 0] for x in (0, 1, 2, 4, 8, 9, 10, 12, 16)],
                                   dtype=torch.int64)

    def features(self, times):
        return torch.randn(times, len(self.coords), 64, dtype=torch.float64)

    def weighted_loss(self, output, target, weights):
        return ((output - target) * weights).square().mean() + .03 * (output * weights).mean()

    def test_twenty_blocks_scales_and_normalization_axes(self):
        encoder = MultiScaleBackbone64("encoder")
        decoder = MultiScaleBackbone64("decoder")
        self.assertEqual(ENCODER_SCALES, (1,) * 4 + (2,) * 4 + (4,) * 4 + (8,) * 8)
        self.assertEqual(DECODER_SCALES, tuple(reversed(ENCODER_SCALES)))
        self.assertEqual(len(encoder.blocks), 20)
        self.assertEqual(len(decoder.blocks), 20)
        self.assertEqual(encoder.stage_ends, (4, 8, 12, 20))
        self.assertEqual(decoder.stage_ends, (8, 12, 16, 20))
        self.assertEqual([b.distance for b in encoder.blocks], list(ENCODER_SCALES))
        self.assertEqual([b.distance for b in decoder.blocks], list(DECODER_SCALES))
        for model in (encoder, decoder):
            for block in model.blocks:
                self.assertEqual(block.temporal.dilation, 1)
                self.assertEqual(block.temporal.norm1.across_time, model.mode == "decoder")
                self.assertEqual(block.temporal.norm2.across_time, model.mode == "decoder")
        self.assertFalse(any(block.condition_norm is not None for block in encoder.blocks))
        self.assertEqual([i + 1 for i, block in enumerate(decoder.blocks) if block.condition_norm is not None],
                         [8, 12, 16, 20])

    def test_exact_six_axis_edges_at_each_distance(self):
        for distance in (1, 2, 4, 8):
            with self.subTest(distance=distance):
                coords = torch.tensor([(0, 0, 0),
                                       *[(distance * x, distance * y, distance * z) for x, y, z in AXIS_OFFSETS],
                                       (distance, distance, 0)], dtype=torch.int32)
                indices = build_neighbor_index_at_distance(coords, distance)
                self.assertEqual(indices.shape, (8, 6))
                self.assertEqual(indices[0].tolist(), [1, 2, 3, 4, 5, 6])
                self.assertNotIn(7, indices[0].tolist())

    def test_scales_two_four_eight_really_mix_neighbors(self):
        model = MultiScaleBackbone64("encoder").double()
        for distance in (2, 4, 8):
            with self.subTest(distance=distance):
                block = next(block for block in model.blocks if block.distance == distance)
                coords = torch.tensor([[0, 0, 0], [distance, 0, 0], [30, 30, 30]])
                x = torch.randn(2, 3, 64, dtype=torch.float64)
                active = torch.ones(2, 3, dtype=torch.bool)
                changed = x.clone()
                changed[:, 1] += torch.linspace(-2, 3, 64, dtype=x.dtype)
                neighbors = build_neighbor_index_at_distance(coords, distance)
                baseline = block.spatial(x, coords, active, neighbor_index=neighbors)
                result = block.spatial(changed, coords, active, neighbor_index=neighbors)
                self.assertGreater(float((result[:, 0] - baseline[:, 0]).abs().max().detach()), 1e-6)
                torch.testing.assert_close(result[:, 2], baseline[:, 2], rtol=0, atol=0)
                # At distance one these two points have no edge: the behavior
                # above cannot be caused by the block ignoring the scale index.
                unit = build_neighbor_index_at_distance(coords, 1)
                unchanged = block.spatial(changed, coords, active, neighbor_index=unit)
                reference = block.spatial(x, coords, active, neighbor_index=unit)
                torch.testing.assert_close(unchanged[:, 0], reference[:, 0], rtol=0, atol=0)

    def test_encoder_prefix_causality_forward_and_gradient(self):
        model = MultiScaleBackbone64("encoder").double()
        x = self.features(6).requires_grad_()
        active = torch.ones(6, len(self.coords), dtype=torch.bool)
        active[:3, -1] = False
        full = model(x, self.coords, active)
        prefix = model(x[:3], self.coords, active[:3])
        torch.testing.assert_close(full[:3], prefix, rtol=1e-10, atol=1e-10)
        changed = x.detach().clone()
        changed[3:] = torch.randn_like(changed[3:]) * 100
        changed_active = active.clone()
        changed_active[3:, 1:4] = False
        perturbed = model(changed, self.coords, changed_active)
        torch.testing.assert_close(full[:3], perturbed[:3], rtol=1e-10, atol=1e-10)
        target, weights = torch.randn_like(full[:3]), torch.randn_like(full[:3])
        self.weighted_loss(full[:3], target, weights).backward()
        torch.testing.assert_close(x.grad[3:], torch.zeros_like(x.grad[3:]), rtol=0, atol=0)
        self.assertGreater(float(x.grad[:3].abs().sum()), 0)

    def test_coordinate_permutation_equivariance_and_no_row_or_time_loss(self):
        permutation = torch.tensor([8, 1, 6, 4, 0, 7, 3, 2, 5])
        for mode, times in (("encoder", 5), ("decoder", 13)):
            with self.subTest(mode=mode):
                model = MultiScaleBackbone64(mode).double()
                x = self.features(times)
                active = torch.ones(times, len(self.coords), dtype=torch.bool)
                active[0, 2:5] = False
                condition = torch.randn_like(x) if mode == "decoder" else None
                expected = model(x, self.coords, active, condition)
                permuted_condition = condition[:, permutation] if condition is not None else None
                actual = model(x[:, permutation], self.coords[permutation], active[:, permutation], permuted_condition)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected[:, permutation], rtol=1e-9, atol=1e-9)

    def test_four_decoder_stages_reuse_original_thirteen_slot_condition(self):
        model = MultiScaleBackbone64("decoder").double()
        x = self.features(13)
        active = torch.ones(13, len(self.coords), dtype=torch.bool)
        seen = []
        handles = []
        for block in model.blocks:
            if block.condition_norm is not None:
                handles.append(block.condition_norm.register_forward_pre_hook(
                    lambda module, args: seen.append(args[1])))
        try:
            output = model(x, self.coords, active)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(output.shape, (13, len(self.coords), 64))
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(value is x for value in seen))

    def test_every_parameter_in_all_twenty_blocks_gets_finite_gradient(self):
        for mode, times in (("encoder", 5), ("decoder", 13)):
            with self.subTest(mode=mode):
                model = MultiScaleBackbone64(mode).double()
                x = self.features(times).requires_grad_()
                active = torch.ones(times, len(self.coords), dtype=torch.bool)
                output = model(x, self.coords, active)
                loss = self.weighted_loss(output, torch.randn_like(output), torch.randn_like(output))
                loss.backward()
                self.assertEqual(len(model.blocks), 20)
                for index, block in enumerate(model.blocks):
                    for name, parameter in block.named_parameters():
                        label = f"{mode}/block{index}/{name}"
                        self.assertIsNotNone(parameter.grad, label)
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), label)
                        self.assertGreater(float(parameter.grad.abs().sum()), 0, label)

    def test_nonreentrant_checkpoint_matches_direct_output_and_all_gradients(self):
        import multiscale64

        for mode, times in (("encoder", 5), ("decoder", 13)):
            with self.subTest(mode=mode):
                direct = MultiScaleBackbone64(mode).double().train()
                checked = copy.deepcopy(direct)
                checked.checkpoint_blocks = True
                x = self.features(times).requires_grad_()
                xc = x.detach().clone().requires_grad_()
                active = torch.ones(times, len(self.coords), dtype=torch.bool)
                condition = torch.randn_like(x, requires_grad=True) if mode == "decoder" else None
                condition_c = condition.detach().clone().requires_grad_() if condition is not None else None
                expected = direct(x, self.coords, active, condition)
                checkpoint_impl = multiscale64.checkpoint
                with patch("multiscale64.checkpoint", wraps=checkpoint_impl) as checkpoint_spy:
                    actual = checked(xc, self.coords, active, condition_c)
                self.assertEqual(checkpoint_spy.call_count, 20)
                self.assertTrue(all(call.kwargs == {"use_reentrant": False} for call in checkpoint_spy.call_args_list))
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                target, weights = torch.randn_like(expected), torch.randn_like(expected)
                self.weighted_loss(expected, target, weights).backward()
                self.weighted_loss(actual, target, weights).backward()
                torch.testing.assert_close(xc.grad, x.grad, rtol=1e-10, atol=1e-10)
                if condition is not None:
                    torch.testing.assert_close(condition_c.grad, condition.grad, rtol=1e-10, atol=1e-10)
                for (name, parameter), (name_c, parameter_c) in zip(direct.named_parameters(), checked.named_parameters()):
                    self.assertEqual(name, name_c)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(parameter_c.grad, name)
                    torch.testing.assert_close(parameter_c.grad, parameter.grad, rtol=1e-10, atol=1e-10, msg=name)

    def test_masked_nan_placeholders_do_not_pollute_outputs(self):
        model = MultiScaleBackbone64("encoder").double()
        x = self.features(4)
        active = torch.ones(4, len(self.coords), dtype=torch.bool)
        active[:, -1] = False
        reference = model(x, self.coords, active)
        x[:, -1] = float("nan")
        output = model(x, self.coords, active)
        self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]), rtol=0, atol=0)

    def test_invalid_contracts_are_rejected(self):
        x = self.features(5)
        active = torch.ones(5, len(self.coords), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "mode"):
            MultiScaleBackbone64("other")
        with self.assertRaisesRegex(ValueError, "distance"):
            build_neighbor_index_at_distance(self.coords, 3)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_neighbor_index_at_distance(torch.tensor([[0, 0, 0], [0, 0, 0]]), 2)
        with self.assertRaisesRegex(ValueError, "conditioning"):
            MultiScaleBackbone64("encoder").double()(x, self.coords, active, x)
        with self.assertRaisesRegex(ValueError, "13-slot"):
            MultiScaleBackbone64("decoder").double()(x, self.coords, active)
        with self.assertRaisesRegex(ValueError, "active"):
            MultiScaleBackbone64("encoder").double()(x, self.coords, active.float())


if __name__ == "__main__":
    unittest.main()
