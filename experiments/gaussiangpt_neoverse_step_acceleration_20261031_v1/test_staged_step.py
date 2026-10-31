"""CPU-only independent nonlinear gradient references; no native dependencies."""

import copy
import sys
import unittest

import torch
from torch import nn

from staged_step import staged_rgb_backward


class MotionToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.queries = nn.Embedding(3, 4)
        self.decoder = nn.Linear(4, 5)
        self.auxiliary_only = nn.Parameter(torch.tensor(.37))
        self.frozen = nn.Linear(4, 4).requires_grad_(False)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        hidden = torch.tanh(self.encoder(self.frozen(inputs)) + self.queries.weight[None, :, None])
        z = torch.sin(hidden.mean(2))
        z.retain_grad()
        # Both time and batch outputs share this graph; a per-output copy of
        # the model would not be equivalent. The native side is also nonlinear.
        features = self.decoder(torch.tanh(z[:, :, None] + .2*z.mean((0, 1)) + .1*inputs))
        outputs = [features[b, t] for b in range(2) for t in range(3)]
        for feature in outputs:
            feature.retain_grad()
        auxiliary = .4*hidden.square().mean() + .3*z.pow(3).mean() + self.auxiliary_only.square()
        return outputs, z, auxiliary


class SharedNativeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return torch.sin(self.network(features))


def parameter_gradients(model, native):
    return {prefix + name: None if p.grad is None else p.grad.detach().clone()
            for prefix, module in [('motion.', model), ('native.', native)]
            for name, p in module.named_parameters()}


def snapshot(model, native, z):
    return dict(parameters=parameter_gradients(model, native), z=z.grad.detach().clone(),
                queries=model.queries.weight.grad.detach().clone())


class StagedGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261031)
        self.model = MotionToy().double()
        self.native = SharedNativeToy().double()
        self.inputs = torch.randn(2, 3, 2, 4, dtype=torch.float64)
        self.targets = torch.randn(6, 2, 3, dtype=torch.float64)

    def run_path(self, path, *, native_trainable=True, auxiliary_weight=.17):
        model, native = copy.deepcopy(self.model), copy.deepcopy(self.native)
        native.requires_grad_(native_trainable)
        outputs, z, auxiliary = model(self.inputs)
        auxiliary = auxiliary * auxiliary_weight
        seen = []

        def loss(index, feature):
            seen.append(index)
            image = native(feature)
            # Different per-frame weights make omission/duplicate weighting
            # visible; the batch division is deliberately outside this loss.
            return (image-self.targets[index]).square().mean()*(index+1)/7 + .13*image.cos().mean()

        if path == 'staged':
            result = staged_rgb_backward(outputs, loss, batch_size=2, auxiliary=auxiliary,
                                         observe_rgb=lambda: snapshot(model, native, z))
            rgb = result.rgb_observation
            feature_gradients = result.feature_gradients
            rgb_loss = result.rgb_loss
        else:
            if path == 'sum':
                losses = [loss(index, feature)/2 for index, feature in enumerate(outputs)]
                total = torch.stack(losses).sum()
                total.backward(retain_graph=True)
                rgb_loss = float(total.detach())
            elif path == 'per_loss':
                rgb_loss = 0.
                for index, feature in enumerate(outputs):
                    value = loss(index, feature)/2
                    value.backward(retain_graph=True)
                    rgb_loss += float(value.detach())
            else:
                raise AssertionError(path)
            rgb = snapshot(model, native, z)
            feature_gradients = tuple(feature.grad.detach().clone() for feature in outputs)
            auxiliary.backward()
        total_gradients = parameter_gradients(model, native)
        total_z_gradient = z.grad.detach().clone()
        parameters = list(model.parameters()) + list(native.parameters())
        clipped_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer = torch.optim.AdamW([p for p in parameters if p.requires_grad], lr=.003, weight_decay=0.)
        optimizer.step()
        return dict(rgb=rgb, gradients=total_gradients, feature_gradients=feature_gradients,
                    total_z=total_z_gradient, rgb_loss=rgb_loss, clipped_norm=clipped_norm,
                    updated={prefix+name:p.detach().clone()
                             for prefix, module in [('motion.',model),('native.',native)]
                             for name,p in module.named_parameters()},
                    seen=seen, forward_calls=(model.calls,native.calls))

    def assert_mapping_close(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            with self.subTest(parameter=name):
                if actual[name] is None or expected[name] is None:
                    self.assertIs(actual[name], expected[name])
                else:
                    torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-12)

    def test_all_gradients_match_sum_and_original_per_loss_backward(self):
        staged = self.run_path('staged')
        for path in ('sum', 'per_loss'):
            reference = self.run_path(path)
            self.assert_mapping_close(staged['gradients'], reference['gradients'])
            self.assert_mapping_close(staged['rgb']['parameters'], reference['rgb']['parameters'])
            self.assert_mapping_close(staged['updated'], reference['updated'])
            for key in ('z', 'queries'):
                torch.testing.assert_close(staged['rgb'][key], reference['rgb'][key], rtol=1e-10, atol=1e-12)
            for actual, expected in zip(staged['feature_gradients'], reference['feature_gradients']):
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(staged['clipped_norm'], reference['clipped_norm'], rtol=1e-10, atol=1e-12)
            self.assertAlmostEqual(staged['rgb_loss'], reference['rgb_loss'], places=12)
        self.assertEqual(staged['seen'], list(range(6)))
        self.assertEqual(staged['forward_calls'], (1, 6))
        self.assertEqual(tuple(staged['rgb']['z'].shape[:2]), (2, 3))

    def test_auxiliary_once_and_rgb_observation_is_isolated(self):
        with_aux = self.run_path('staged', auxiliary_weight=.17)
        no_aux = self.run_path('staged', auxiliary_weight=0.)
        self.assert_mapping_close(with_aux['rgb']['parameters'], no_aux['rgb']['parameters'])
        torch.testing.assert_close(with_aux['rgb']['z'], no_aux['rgb']['z'])
        self.assertIsNone(with_aux['rgb']['parameters']['motion.auxiliary_only'])
        expected = .17*2*self.model.auxiliary_only.detach()
        torch.testing.assert_close(with_aux['gradients']['motion.auxiliary_only'], expected)
        self.assertFalse(torch.allclose(with_aux['total_z'], with_aux['rgb']['z']))
        for name, gradient in with_aux['gradients'].items():
            if name.startswith('native.'):
                torch.testing.assert_close(gradient, with_aux['rgb']['parameters'][name])
                self.assertGreater(float(gradient.norm()), 0.)
            if name.startswith('motion.frozen.'):
                self.assertIsNone(gradient)

    def test_frozen_native_still_returns_upstream_gradient(self):
        staged = self.run_path('staged', native_trainable=False)
        reference = self.run_path('sum', native_trainable=False)
        self.assert_mapping_close(staged['gradients'], reference['gradients'])
        self.assertGreater(float(staged['rgb']['z'].norm()), 0.)
        for name, gradient in staged['gradients'].items():
            if name.startswith('native.'):
                self.assertIsNone(gradient)

    def test_import_does_not_load_native_pipeline(self):
        self.assertNotIn('native_experiment', sys.modules)
        self.assertNotIn('MinkowskiEngine', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
