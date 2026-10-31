"""CPU-only semantic checks. No optimizer, GPU allocation, install or download.

python -B test_reconstruction_objective.py              # local D/GP primitives
python -B test_reconstruction_objective.py --lpips-only # existing 134 caches
"""
import argparse
import math
import time
import unittest
import torch
from torch import nn
import reconstruction_objective as objective


class PrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_l1_includes_initial_frame_and_native_pixels(self):
        real = torch.zeros(1, 10, 3, 252, 448)
        fake = real.clone()
        fake[:, 0] = 1
        self.assertAlmostEqual(float(objective.full_video_l1(fake, real)), .1, places=6)
        with self.assertRaises(ValueError):
            objective.full_video_l1(fake[:, 1:], real[:, 1:])

    def test_uniform_sampling_covers_initial_and_last(self):
        indices = objective.sample_frame_indices(20000, generator=torch.Generator().manual_seed(57))
        counts = torch.bincount(indices, minlength=10)
        self.assertTrue(bool(((counts > 1800) & (counts < 2200)).all()), counts)
        frames = torch.arange(10.).reshape(1, 10, 1, 1, 1).expand(2, 10, 3, 252, 448)
        selected = objective.pick_video_frames(frames, torch.tensor([0, 9]))
        self.assertEqual(tuple(selected.shape), (2, 3, 252, 448))
        self.assertEqual(float(selected[0].mean()), 0.)
        self.assertEqual(float(selected[1].mean()), 9.)

    def test_hinge_sign_and_half_weights(self):
        # real scores [2,0] give hinge [0,1], fake [-2,0] give [0,1].
        self.assertEqual(float(objective.hinge_discriminator_loss(
            torch.tensor([-2., 0.]), torch.tensor([2., 0.]))), .5)
        self.assertEqual(float(objective.generator_adversarial_loss(torch.tensor([2., 4.]))), -3.)

    def test_gp_is_norm_to_one_and_supports_parameter_second_derivative(self):
        images = torch.randn(4, 1, 2, 2, requires_grad=True)
        weight = nn.Parameter(torch.tensor(1.))
        score = weight * images.flatten(1).sum(1)
        penalty = objective.norm_to_one_gradient_penalty(images, score)
        self.assertAlmostEqual(float(penalty.detach()), 1.)  # (sqrt(4)-1)^2
        penalty.backward()
        self.assertAlmostEqual(float(weight.grad), 4.)  # d(2w-1)^2/dw at w=1

    def test_adaptive_weight_includes_nll_and_is_detached(self):
        weight = nn.Parameter(torch.tensor([3., 4.]))
        result = objective.adaptive_adversarial_weight((weight**2).sum(), 3*weight.sum(), weight)
        self.assertAlmostEqual(float(result), .75*10/(math.sqrt(18)+1e-4), places=6)
        self.assertFalse(result.requires_grad)
        clamped = objective.adaptive_adversarial_weight(1e8*weight.sum(), 0*weight.sum(), weight)
        self.assertEqual(float(clamped), 7500.)

    def test_joint_batch_is_required_and_g_uses_shared_sample(self):
        class Discriminator(nn.Module):
            def forward(self, x):
                self.shape = tuple(x.shape)
                return x.mean((1, 2, 3))
        class Perceptual(nn.Module):
            def forward(self, fake, real):
                self.shape, self.observed = tuple(fake.shape), fake[:, 0, 0, 0].detach()
                return (fake-real).square().mean()
        weight = nn.Parameter(torch.tensor(.1))
        basis = torch.arange(1., 11.).reshape(1, 10, 1, 1, 1).expand(4, 10, 3, 252, 448)
        fake, real = weight*basis, torch.zeros_like(basis)
        discriminator, perceptual = Discriminator(), Perceptual()
        indices = torch.tensor([0, 3, 6, 9])
        losses = objective.generator_image_loss(discriminator, perceptual, fake, real,
                                                weight, frame_indices=indices)
        self.assertEqual(discriminator.shape, (4, 3, 128, 128))
        self.assertEqual(perceptual.shape, (4, 3, 252, 448))
        torch.testing.assert_close(perceptual.observed, torch.tensor([.1, .4, .7, 1.]))
        losses['loss'].backward()
        self.assertTrue(bool(torch.isfinite(weight.grad)))
        with self.assertRaises(ValueError):
            objective.discriminator_input(torch.zeros(1, 3, 252, 448))


class DiscriminatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_mbstd_couples_the_four_samples(self):
        from vendor_gan.loader import load_classes
        layer = load_classes()['ModMBStddevLayer'](group_size=4, channel_groups=1)
        data = torch.arange(4.).reshape(4, 1, 1, 1).expand(4, 2, 4, 4)
        joint = layer(data)
        independent = torch.cat([layer(row[None]) for row in data])
        self.assertGreater(float(joint[0, -1, 0, 0]), 1.)
        self.assertLess(float(independent[0, -1, 0, 0]), .001)

    def test_blur_matches_independent_depthwise_fir(self):
        from vendor_gan.compat import upfirdn2d
        x = torch.randn(2, 3, 8, 8, dtype=torch.double, requires_grad=True)
        kernel = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.double) / 10
        actual = upfirdn2d(x, kernel, padding=(1, 2))
        expected = torch.cat([torch.nn.functional.conv2d(
            torch.nn.functional.pad(x[:, c:c+1], (1, 1, 2, 2)),
            kernel.flip((0, 1))[None, None]) for c in range(3)], dim=1)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradgradcheck(
            lambda z: upfirdn2d(z, kernel, padding=1),
            (x[:1, :1, :3, :3],), fast_mode=True))

    def test_original_128_discriminator_b4_gp_double_backward(self):
        torch.manual_seed(321)
        discriminator = objective.build_discriminator()
        self.assertEqual(discriminator.__class__.__name__, 'StyleGAN2Discriminator')
        self.assertEqual(discriminator.mbstd_layer.group_size, 4)
        fake = torch.randn(4, 3, 252, 448, requires_grad=True)
        real = torch.randn_like(fake)
        losses = objective.discriminator_image_loss(discriminator, fake, real)
        self.assertEqual(tuple(losses['real_scores'].shape), (4, 1))
        self.assertTrue(bool(torch.isfinite(losses['loss'])))
        # GP-only backward proves double derivatives reach original D weights.
        losses['gradient_penalty'].backward()
        first_weight = next(discriminator.parameters())
        self.assertIsNotNone(first_weight.grad)
        self.assertGreater(float(first_weight.grad.norm()), 0.)
        self.assertTrue(bool(torch.isfinite(first_weight.grad).all()))
        self.assertIsNone(fake.grad)


class LPIPSCacheTests(unittest.TestCase):
    def test_native_resolution_calibrated_frozen_deterministic_lpips(self):
        torch.set_num_threads(2)
        metric = objective.FrozenLPIPS()
        metric.train(True)
        self.assertFalse(metric.training)
        self.assertTrue(all(not p.requires_grad for p in metric.parameters()))
        self.assertTrue(all(not m.training for m in metric.modules()))
        torch.manual_seed(71)
        fake = torch.rand(1, 3, 252, 448).mul(2).sub(1).requires_grad_(True)
        real = torch.rand_like(fake).mul(2).sub(1)
        with torch.no_grad():
            first, second = metric(fake, real), metric(fake, real)
            identical = metric(real, real)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertAlmostEqual(float(identical), 0., places=7)
        result = metric(fake, real)
        result.backward()
        self.assertGreater(float(fake.grad.norm()), 0.)
        self.assertTrue(bool(torch.isfinite(fake.grad).all()))
        self.assertTrue(all(p.grad is None for p in metric.parameters()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lpips-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    classes = [LPIPSCacheTests] if args.lpips_only else [PrimitiveTests, DiscriminatorTests]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'CPU checks elapsed {time.monotonic()-started:.3f}s; CUDA initialized={torch.cuda.is_initialized()}')
    raise SystemExit(0 if result.wasSuccessful() else 1)
