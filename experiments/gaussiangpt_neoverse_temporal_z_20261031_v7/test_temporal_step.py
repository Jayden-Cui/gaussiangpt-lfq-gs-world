"""Independent CPU full-loss reference for the temporal first-order VJP."""
import copy
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from temporal_step import staged_temporal_backward, step_native


class Motion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.queries = nn.Embedding(3, 4)
        self.head = nn.Linear(4, 5)
        self.aux_only = nn.Parameter(torch.tensor(.37))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        hidden = torch.tanh(self.encoder(inputs) + self.queries.weight[None, :, None])
        z = torch.sin(hidden.mean(2)); z.retain_grad()
        features = self.head(torch.tanh(z[:, :, None] + .2*z.mean((0, 1)) + .1*inputs))
        outputs = [[features[b, h] for h in range(3)] for b in range(2)]
        for row in outputs:
            for feature in row: feature.retain_grad()
        return outputs, z, .4*hidden.square().mean() + .3*z.pow(3).mean() + self.aux_only.square()


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
        self.calls = []

    def forward(self, b, h, features):
        self.calls.append((b, h))
        return torch.sin(self.network(features))


def gradients(model, native):
    return {prefix + name: None if p.grad is None else p.grad.detach().clone()
            for prefix, module in [('motion.', model), ('native.', native)]
            for name, p in module.named_parameters()}


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261031)
        self.model, self.native = Motion().double(), Native().double()
        self.inputs = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        self.targets = torch.randn(2, 3, 4, 3, dtype=torch.float64)

    def run_path(self, staged, *, auxiliary_weight=.17, base_weight=1., delta_weight=1., native_trainable=True):
        model, native = copy.deepcopy(self.model), copy.deepcopy(self.native)
        native.requires_grad_(native_trainable)
        outputs, z, auxiliary = model(self.inputs)
        auxiliary = auxiliary * auxiliary_weight

        def base(b, h, image):
            error = (image-self.targets[b, h]).abs().mean(-1)
            return base_weight*(error.mean()/3 + error[:2].mean()/3 + (error[1:].mean()/2 if h else 0))

        def delta(b, h, image, initial):
            error = ((image-initial)-(self.targets[b, h]-self.targets[b, 0])).abs().mean(-1)
            return error.mean() + error[:2].mean()

        def observe():
            return dict(parameters=gradients(model, native), z=z.grad.detach().clone(),
                        queries=model.queries.weight.grad.detach().clone())

        if staged:
            result = staged_temporal_backward(outputs, native, base, delta, delta_weight=delta_weight,
                                             auxiliary=auxiliary, observe_rgb=observe)
            rgb, rgb_loss = result.rgb_observation, result.rgb_loss
            feature_gradients, initial_gradients = result.feature_gradients, result.initial_rgb_gradients
        else:
            images = [[native(b, h, f) for h, f in enumerate(row)] for b, row in enumerate(outputs)]
            for row in images: row[0].retain_grad()
            losses = [base(b, h, image)/2 for b, row in enumerate(images) for h, image in enumerate(row)]
            losses += [delta_weight*delta(b, h, images[b][h], images[b][0])/4 for b in range(2) for h in (1, 2)]
            loss = torch.stack(losses).sum()
            loss.backward(retain_graph=True)
            rgb, rgb_loss = observe(), float(loss.detach())
            feature_gradients = tuple(tuple(f.grad.detach().clone() for f in row) for row in outputs)
            initial_gradients = tuple(row[0].grad.detach().clone() for row in images)
            auxiliary.backward()
        total = gradients(model, native)
        all_parameters = list(model.parameters()) + list(native.parameters())
        clip = torch.nn.utils.clip_grad_norm_(all_parameters, 1., error_if_nonfinite=True)
        optimizer = torch.optim.AdamW([p for p in all_parameters if p.requires_grad], lr=.003, weight_decay=0.)
        optimizer.step()
        updated = {prefix+name:p.detach().clone() for prefix, module in [('motion.',model),('native.',native)]
                   for name,p in module.named_parameters()}
        return dict(rgb=rgb, rgb_loss=rgb_loss, gradients=total, features=feature_gradients,
                    initial=initial_gradients, clip=clip, updated=updated,
                    calls=(model.calls, native.calls))

    def assert_mapping(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            with self.subTest(parameter=key):
                if actual[key] is None or expected[key] is None:
                    self.assertIs(actual[key], expected[key])
                else:
                    torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-12)

    def assert_paths(self, **kwargs):
        staged, full = self.run_path(True, **kwargs), self.run_path(False, **kwargs)
        for key in ('gradients', 'updated'): self.assert_mapping(staged[key], full[key])
        self.assert_mapping(staged['rgb']['parameters'], full['rgb']['parameters'])
        for key in ('z', 'queries'):
            torch.testing.assert_close(staged['rgb'][key], full['rgb'][key], rtol=1e-10, atol=1e-12)
        for a, b in zip(staged['features'], full['features']):
            for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-12)
        for a, b in zip(staged['initial'], full['initial']):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(staged['clip'], full['clip'], rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(staged['rgb_loss'], full['rgb_loss'], places=12)
        self.assertEqual(staged['calls'], (1, [(b,h) for b in range(2) for h in range(3)]))
        return staged

    def test_full_temporal_loss_all_parameter_gradients_and_updates(self):
        self.assert_paths()

    def test_difference_alone_restores_initial_image_and_feature_derivatives(self):
        result = self.assert_paths(base_weight=0., auxiliary_weight=0.)
        for image, features in zip(result['initial'], result['features']):
            self.assertGreater(float(image.norm()), 0.)
            self.assertGreater(float(features[0].norm()), 0.)
        self.assertGreater(float(result['rgb']['z'].norm()), 0.)

    def test_auxiliary_once_and_rgb_observation_excludes_it(self):
        with_aux = self.run_path(True)
        without_aux = self.run_path(True, auxiliary_weight=0.)
        self.assert_mapping(with_aux['rgb']['parameters'], without_aux['rgb']['parameters'])
        torch.testing.assert_close(with_aux['rgb']['z'], without_aux['rgb']['z'])
        self.assertIsNone(with_aux['rgb']['parameters']['motion.aux_only'])
        torch.testing.assert_close(with_aux['gradients']['motion.aux_only'], .17*2*self.model.aux_only.detach())
        for name, grad in with_aux['gradients'].items():
            if name.startswith('native.'):
                torch.testing.assert_close(grad, with_aux['rgb']['parameters'][name])
                self.assertGreater(float(grad.norm()), 0.)

    def test_frozen_native_keeps_upstream_and_initial_gradients(self):
        self.assert_paths(native_trainable=False)

    def test_zero_delta_native_dispatches_original_staged_step(self):
        arguments = [object() for _ in range(6)]
        sentinel = object()
        fake = types.ModuleType('staged_step')
        called = []
        def original(*args): called.append(args); return sentinel
        fake.step_native = original
        with patch.dict(sys.modules, {'staged_step': fake}):
            self.assertIs(step_native(*arguments, delta_weight=0.), sentinel)
        self.assertEqual(called, [tuple(arguments)])

    def test_import_does_not_load_native_pipeline(self):
        self.assertNotIn('native_experiment', sys.modules)
        self.assertNotIn('MinkowskiEngine', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
