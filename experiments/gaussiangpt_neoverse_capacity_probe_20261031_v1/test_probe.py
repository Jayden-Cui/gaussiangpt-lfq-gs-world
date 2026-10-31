"""CPU numerical checks only: no native codec, checkpoint, remote or GPU access."""
import unittest

import torch

from probe import assert_frozen, normalized_residual, parity, project_residual_, staged_backward


class CapacityProbeTests(unittest.TestCase):
    def test_residual_recovers_posterior_and_zero_channel(self):
        torch.manual_seed(47)
        initial = torch.randn(7, 4, dtype=torch.float64)
        units = torch.tensor([.1, 2., 0., .0001], dtype=torch.float64)
        r = torch.randn(10, 7, 4, dtype=torch.float64).tanh()
        features = initial[None] + units * r
        recovered = normalized_residual(features, initial, units)
        torch.testing.assert_close(initial[None] + units * recovered, features, rtol=1e-13, atol=1e-15)
        self.assertTrue(bool((recovered[..., 2] == 0).all()))
        features[0, 0, 2] += 1e-6
        with self.assertRaises(ValueError):
            normalized_residual(features, initial, units)

    def test_projection_retains_interior_and_preserves_parameter(self):
        value = torch.nn.Parameter(torch.tensor([-4., -1., -.3, 0., .8, 1., 3.]))
        identity = id(value)
        project_residual_(value)
        self.assertEqual(id(value), identity)
        self.assertTrue(value.requires_grad)
        torch.testing.assert_close(value, torch.tensor([-1., -1., -.3, 0., .8, 1., 1.]))

    def test_staged_vjp_matches_joint_loss_with_frozen_decoder(self):
        torch.manual_seed(97)
        decoder = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)).double()
        decoder.eval().requires_grad_(False)
        parameter = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        replica = parameter.detach().clone().requires_grad_(True)
        mixing = torch.randn(10, 6, 3, dtype=torch.float64)
        target = torch.randn(10, 6, 3, dtype=torch.float64)
        shared = torch.einsum('tnc,cd->tnd', mixing, parameter)
        outputs = shared.sin() + .2 * shared.square()
        direct = sum((decoder(outputs[h]) - target[h]).square().mean() * (h+1)/55 for h in range(10))
        direct.backward()
        replica_shared = torch.einsum('tnc,cd->tnd', mixing, replica)
        staged_outputs = replica_shared.sin() + .2 * replica_shared.square()
        total, gradients = staged_backward(list(staged_outputs),
            lambda h, leaf: (decoder(leaf)-target[h]).square().mean()*(h+1)/55)
        self.assertAlmostEqual(total, float(direct.detach()), places=13)
        self.assertEqual(len(gradients), 10)
        torch.testing.assert_close(replica.grad, parameter.grad, rtol=1e-12, atol=1e-14)
        assert_frozen(decoder)
        decoder[0].weight.grad = torch.zeros_like(decoder[0].weight)
        with self.assertRaises(RuntimeError):
            assert_frozen(decoder)

    def test_nonfinite_leaf_gradient_fails(self):
        parameter = torch.tensor([0.], requires_grad=True)
        with self.assertRaises(RuntimeError):
            staged_backward([parameter + 0.], lambda h, leaf: leaf.sqrt().sum())

    def test_parity_checks_every_h_and_preserves_tolerances(self):
        reference = [dict(mae=.03, psnr=25.) for _ in range(10)]
        actual = [dict(mae=.03001, psnr=25.001) for _ in range(10)]
        parity(actual, reference, dict(mae=1e-4, psnr=.02))
        actual[-1]['mae'] = .03011
        with self.assertRaises(RuntimeError):
            parity(actual, reference, dict(mae=1e-4, psnr=.02))
        with self.assertRaises(ValueError):
            parity(actual[:-1], reference, dict(mae=1e-4, psnr=.02))


if __name__ == '__main__':
    unittest.main()
