"""CPU-only shortcut tests, without LFQ, MinkowskiEngine or checkpoint imports."""
import unittest

import torch
from torch import nn

from local_skip import LocalSkipBackbone, attach_local_skip, local_skip_grad_norm


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=False).double()

    def forward(self, x, coords, active, condition=None):
        return (x + self.linear(x).tanh()) * active[..., None]


class LocalSkipTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261031)
        self.x = torch.randn(13, 5, 64, dtype=torch.float64)
        self.coords = torch.arange(15).reshape(5, 3)
        self.active = torch.ones(13, 5, dtype=torch.bool)
        self.base = ToyBackbone().eval()

    def test_zero_initialization_exact_and_rng_preserved(self):
        expected = self.base(self.x, self.coords, self.active)
        rng = torch.get_rng_state().clone()
        wrapper = LocalSkipBackbone(self.base, True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(wrapper(self.x, self.coords, self.active), expected))
        self.assertEqual(wrapper.P.weight.dtype, torch.float64)
        self.assertFalse(wrapper.training)
        self.assertTrue(bool((wrapper.P.weight == 0).all()))

    def test_only_future_slots_change_and_anchor_is_per_row(self):
        wrapper = LocalSkipBackbone(self.base, True)
        with torch.no_grad():
            wrapper.P.weight.copy_(torch.eye(64, dtype=torch.float64))
        condition = torch.zeros_like(self.x)
        condition[3] = torch.arange(5, dtype=torch.float64)[:, None]
        expected = self.base(self.x, self.coords, self.active)
        actual = wrapper(self.x, self.coords, self.active, condition)
        self.assertTrue(torch.equal(actual[:4], expected[:4]))
        torch.testing.assert_close(actual[4:]-expected[4:], condition[3:4].expand(9, -1, -1))
        changed_future = condition.clone(); changed_future[4:] = 1e6
        self.assertTrue(torch.equal(actual, wrapper(self.x, self.coords, self.active, changed_future)))
        # A missing condition must use x[3], not a pooled x or the final slot.
        implicit = wrapper(self.x, self.coords, self.active)
        explicit = wrapper(self.x, self.coords, self.active, self.x)
        self.assertTrue(torch.equal(implicit, explicit))

    def test_coordinate_row_permutation_and_inactive_rows(self):
        wrapper = LocalSkipBackbone(self.base, True)
        with torch.no_grad():
            wrapper.P.weight.copy_(torch.randn_like(wrapper.P.weight))
        active = self.active.clone(); active[8:, 2] = False
        order = torch.tensor([3, 1, 4, 0, 2])
        expected = wrapper(self.x, self.coords, active)
        actual = wrapper(self.x[:, order], self.coords[order], active[:, order])
        torch.testing.assert_close(actual, expected[:, order], rtol=1e-13, atol=1e-13)
        self.assertTrue(bool((expected[8:, 2] == 0).all()))

    def test_gradients_reach_projection_and_only_initial_anchor(self):
        wrapper = LocalSkipBackbone(self.base, True)
        with torch.no_grad():
            wrapper.P.weight.copy_(torch.eye(64, dtype=torch.float64))
        condition = self.x.clone().requires_grad_(True)
        output = wrapper(self.x, self.coords, self.active, condition)
        output[4:].sum().backward()
        self.assertIsNotNone(wrapper.P.weight.grad)
        self.assertTrue(bool(torch.isfinite(wrapper.P.weight.grad).all()))
        self.assertGreater(float(wrapper.P.weight.grad.norm()), 0.)
        torch.testing.assert_close(condition.grad[3], torch.full_like(condition[3], 9.))
        self.assertTrue(bool((condition.grad[:3] == 0).all()))
        self.assertTrue(bool((condition.grad[4:] == 0).all()))
        # The zero-initialized projection must still receive a first-step signal.
        zero = LocalSkipBackbone(ToyBackbone(), True).double()
        zero(self.x, self.coords, self.active)[4:].square().mean().backward()
        self.assertGreater(float(zero.P.weight.grad.norm()), 0.)

    def test_disabled_is_exact_even_with_nonzero_projection(self):
        wrapper = LocalSkipBackbone(self.base, False)
        with torch.no_grad():
            wrapper.P.weight.fill_(19.)
        x = self.x.clone().requires_grad_(True)
        actual = wrapper(x, self.coords, self.active)
        self.assertTrue(torch.equal(actual, self.base(x, self.coords, self.active)))
        actual.sum().backward()
        self.assertFalse(wrapper.P.weight.requires_grad)
        self.assertIsNone(wrapper.P.weight.grad)

    def test_attach_parameter_api_state_dict_and_repeat_rejection(self):
        model = nn.Module(); model.decoder_backbone = self.base
        original_requires_grad = [p.requires_grad for p in self.base.parameters()]
        added = attach_local_skip(model, True)
        self.assertEqual(len(added), 1)
        self.assertIs(added[0], model.decoder_backbone.P.weight)
        self.assertIn('decoder_backbone.base.linear.weight', model.state_dict())
        self.assertIn('decoder_backbone.P.weight', model.state_dict())
        self.assertEqual(original_requires_grad, [p.requires_grad for p in model.decoder_backbone.base.parameters()])
        self.assertEqual(local_skip_grad_norm(model), 0.)
        with self.assertRaises(ValueError):
            attach_local_skip(model, True)
        restored = nn.Module(); restored.decoder_backbone = ToyBackbone()
        self.assertEqual(attach_local_skip(restored, False), [])
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored.decoder_backbone.P.weight, model.decoder_backbone.P.weight))


if __name__ == '__main__':
    unittest.main()
