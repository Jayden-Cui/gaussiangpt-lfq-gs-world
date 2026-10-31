"""Small CPU contract tests; run: python -B test_spatial64.py -v.

Uses only PyTorch and the standard-library unittest. No GPU, training run,
checkpoint, renderer, dataset, or filesystem output is required.
"""

import unittest

import torch

from spatial64 import AXIS_OFFSETS, SparseSpatialBlock64, build_neighbor_index


class SparseSpatial64Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261030)
        torch.set_num_threads(1)
        self.block = SparseSpatialBlock64().double().cpu()
        self.coords = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [7, 7, 7]],
                                   dtype=torch.int64, device="cpu")

    def random_features(self, times=3, nodes=4):
        return torch.randn(times, nodes, 64, dtype=torch.float64, device="cpu")

    def assert_close(self, a, b):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_real_neighbor_interaction_and_isolated_node(self):
        x = self.random_features()
        present = torch.ones(3, 4, dtype=torch.bool)
        baseline = self.block(x, self.coords, present)
        changed = x.clone()
        changed[:, 1] += torch.linspace(-3, 3, 64, dtype=x.dtype)
        result = self.block(changed, self.coords, present)
        self.assertGreater(float((baseline[:, 0] - result[:, 0]).abs().max().detach()), 1e-5)
        self.assert_close(result[:, 3], baseline[:, 3])
        self.assertEqual(result.shape, x.shape)

    def test_all_six_axis_neighbors_are_kept(self):
        coords = torch.tensor([(0, 0, 0), *AXIS_OFFSETS, (1, 1, 0)], dtype=torch.int32)
        index = build_neighbor_index(coords)
        self.assertEqual(index[0].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(index.shape, (8, 6))
        self.assertNotIn(7, index[0].tolist())  # Diagonal is intentionally not an axis edge.

    def test_future_only_insertion_and_nan_mask_do_not_leak(self):
        coords = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.long)
        x = self.random_features(times=3, nodes=2)
        present = torch.ones(3, 2, dtype=torch.bool)
        baseline = self.block(x, coords, present)
        # Insert a future-only point adjacent to the center, changing row indices.
        augmented_coords = torch.tensor([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        augmented = torch.cat([torch.full((3, 1, 64), float("nan"), dtype=x.dtype), x], 1)
        augmented_present = torch.ones(3, 3, dtype=torch.bool)
        visible = torch.ones_like(augmented_present)
        visible[:2, 0] = False
        augmented[2, 0] = torch.randn(64, dtype=x.dtype)
        output = self.block(augmented, augmented_coords, augmented_present, visible=visible)
        self.assert_close(output[:2, 1:], baseline[:2])
        self.assert_close(output[:2, 0], torch.zeros_like(output[:2, 0]))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertGreater(float((output[2, 1] - baseline[2, 0]).abs().max().detach()), 1e-5)

    def test_current_absence_overrides_cumulative_visibility(self):
        x = self.random_features()
        present = torch.ones(3, 4, dtype=torch.bool)
        present[:, 1] = False
        visible = torch.ones_like(present)
        baseline = self.block(x, self.coords, present, visible=visible)
        changed = x.clone()
        changed[:, 1] = float("inf")
        result = self.block(changed, self.coords, present, visible=visible)
        self.assert_close(result, baseline)
        self.assert_close(result[:, 1], torch.zeros_like(result[:, 1]))

    def test_time_slices_are_independent_forward_and_backward(self):
        x = self.random_features().requires_grad_()
        present = torch.ones(3, 4, dtype=torch.bool)
        baseline = self.block(x, self.coords, present)
        changed = x.detach().clone()
        changed[2] *= 100
        changed_present = present.clone()
        changed_present[2, 1] = False
        output = self.block(changed, self.coords, changed_present)
        self.assert_close(output[:2], baseline[:2])
        baseline[0, 0].square().sum().backward()
        self.assert_close(x.grad[1:], torch.zeros_like(x.grad[1:]))
        self.assertGreater(float(x.grad[0, 1].abs().sum()), 0)

    def test_coordinate_permutation_equivariance(self):
        x = self.random_features()
        present = torch.tensor([[True, True, True, False], [True, False, True, True],
                                [False, True, True, True]])
        permutation = torch.tensor([3, 1, 0, 2])
        expected = self.block(x, self.coords, present)[:, permutation]
        actual = self.block(x[:, permutation], self.coords[permutation], present[:, permutation])
        self.assert_close(actual, expected)

    def test_no_neighbors_and_fully_masked_slice_are_finite(self):
        coords = torch.tensor([[0, 0, 0], [10, 10, 10], [-9, -9, -9]], dtype=torch.long)
        x = self.random_features(times=2, nodes=3)
        present = torch.tensor([[True, True, True], [False, False, False]])
        x[1] = float("nan")
        output = self.block(x, coords, present)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assert_close(output[0], x[0])
        self.assert_close(output[1], torch.zeros_like(output[1]))

    def test_input_and_message_parameter_gradients(self):
        x = self.random_features().requires_grad_()
        present = torch.ones(3, 4, dtype=torch.bool)
        present[:, 2] = False
        output = self.block(x, self.coords, present)
        output[:, 0].square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(x.grad[:, 1].abs().sum()), 0)
        self.assert_close(x.grad[:, 2], torch.zeros_like(x.grad[:, 2]))
        for name, parameter in self.block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_precomputed_neighbors_match_and_reject_invalid_indices(self):
        x = self.random_features()
        present = torch.ones(3, 4, dtype=torch.bool)
        neighbors = build_neighbor_index(self.coords)
        self.assert_close(self.block(x, self.coords, present),
                          self.block(x, self.coords, present, neighbor_index=neighbors))
        broken = neighbors.clone()
        broken[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.block(x, self.coords, present, neighbor_index=broken)

    def test_empty_support_and_invalid_contracts(self):
        empty = torch.empty(2, 0, 64, dtype=torch.float64, requires_grad=True)
        output = self.block(empty, torch.empty(0, 3, dtype=torch.long), torch.empty(2, 0, dtype=torch.bool))
        self.assertEqual(output.shape, (2, 0, 64))
        output.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_neighbor_index(torch.tensor([[0, 0, 0], [0, 0, 0]]))
        with self.assertRaises(TypeError):
            build_neighbor_index(self.coords.float())
        with self.assertRaisesRegex(ValueError, "64"):
            self.block(torch.ones(2, 4, 32), self.coords, torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaises(TypeError):
            self.block(self.random_features(), self.coords, torch.ones(3, 4))


if __name__ == "__main__":
    unittest.main()
