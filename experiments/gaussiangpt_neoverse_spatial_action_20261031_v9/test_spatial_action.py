"""CPU-only branch invariants; no LFQ, MinkowskiEngine, renderer or checkpoints.

Tests measure wrapper_output - base_output so a toy backbone's own pathways
cannot masquerade as effects of the new spatial action branch.
"""
import copy
import unittest
from unittest import mock

import torch
from torch import nn

from spatial_action import (SpatialActionBackbone, attach_spatial_action,
                            spatial_action_grad_norm, spatial_action_gradients)


class ToyBackbone(nn.Module):
    """Deliberately mixes future slots and adds a non-equivariant row offset."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=False, device='cpu', dtype=torch.float64)
        self.unused_mode_probe = nn.Dropout(.1)

    def forward(self, x, coords, active, condition=None):
        source = x if condition is None else condition
        self.last_condition = condition
        pooled = source.mean((0, 1))
        row_offset = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[None, :, None] * .01
        return (x + self.linear(x).tanh() + pooled + row_offset) * active[..., None]


class SpatialActionTests(unittest.TestCase):
    def setUp(self):
        torch.random.default_generator.manual_seed(82731)
        self.x = torch.randn(13, 5, 64, dtype=torch.float64)
        self.condition = torch.randn_like(self.x)
        self.condition[4:] = torch.randn(9, 1, 64, dtype=torch.float64).expand(9, 5, 64)
        self.coords = torch.tensor([[0, 1, 3], [2, 4, 6], [-3, 8, 10], [11, 5, 7], [9, -2, 4]])
        self.active = torch.ones(13, 5, dtype=torch.bool)
        self.base = ToyBackbone().eval()

    def wrapper(self, enabled=True, activate=False):
        wrapped = SpatialActionBackbone(self.base, enabled)
        if activate:
            with torch.no_grad():
                wrapped.Wo.weight.copy_(torch.eye(64, dtype=torch.float64))
        return wrapped

    def delta(self, wrapper, *, x=None, coords=None, active=None, condition=None):
        x = self.x if x is None else x
        coords = self.coords if coords is None else coords
        active = self.active if active is None else active
        condition = self.condition if condition is None else condition
        actual = wrapper(x, coords, active, condition=condition)
        original = wrapper.base(x, coords, active, condition=condition)
        return actual - original

    def test_zero_initial_output_exact_and_cpu_rng_unchanged_without_cuda_calls(self):
        expected = self.base(self.x, self.coords, self.active, self.condition)
        before = torch.random.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA seeding')):
            with mock.patch('torch.cuda.get_rng_state', side_effect=AssertionError('CUDA RNG access')):
                wrapped = self.wrapper()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(wrapped(self.x, self.coords, self.active, self.condition), expected))
        self.assertIs(self.base.last_condition, self.condition)
        self.assertTrue(bool((wrapped.Wo.weight == 0).all()))
        self.assertEqual(wrapped.Wo.weight.dtype, torch.float64)
        self.assertFalse(wrapped.training)

    def test_fixed_seed_initialization_independent_of_external_rng(self):
        first = self.wrapper()
        torch.rand(179)
        rng = torch.random.get_rng_state().clone()
        second = SpatialActionBackbone(copy.deepcopy(self.base), True)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for (name_a, a), (name_b, b) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(name_a, name_b)
            if not name_a.startswith('base.'):
                self.assertTrue(torch.equal(a, b))

    def test_same_action_different_initial_features_changes_only_branch(self):
        wrapped = self.wrapper(activate=True)
        source = self.condition.clone()
        source[3] = source[3, :1].expand(5, 64)
        coords = self.coords[:1].expand(5, 3)
        original = self.delta(wrapped, condition=source, coords=coords)
        torch.testing.assert_close(original[4:, 0], original[4:, 1], rtol=0, atol=1e-14)
        source[3, 1] = torch.randn(64, dtype=torch.float64)
        changed = self.delta(wrapped, condition=source, coords=coords)
        self.assertGreater(float((changed[4:, 1] - original[4:, 1]).norm()), .01)
        torch.testing.assert_close(changed[:, [0, 2, 3, 4]], original[:, [0, 2, 3, 4]], rtol=0, atol=1e-14)

    def test_same_action_and_initial_different_coordinates_changes_branch(self):
        wrapped = self.wrapper(activate=True)
        source = self.condition.clone()
        source[3] = source[3, :1].expand(5, 64)
        coords = self.coords[:1].expand(5, 3).clone()
        original = self.delta(wrapped, condition=source, coords=coords)
        coords[1] += torch.tensor([2, -3, 4])
        changed = self.delta(wrapped, condition=source, coords=coords)
        self.assertGreater(float((changed[4:, 1] - original[4:, 1]).norm()), .01)
        torch.testing.assert_close(changed[:, [0, 2, 3, 4]], original[:, [0, 2, 3, 4]], rtol=0, atol=1e-14)

    def test_same_spatial_condition_different_action_changes_branch(self):
        wrapped = self.wrapper(activate=True)
        original = self.delta(wrapped)
        changed_source = self.condition.clone()
        changed_source[7] = torch.randn(1, 64, dtype=torch.float64).expand(5, 64)
        changed = self.delta(wrapped, condition=changed_source)
        self.assertGreater(float((changed[7] - original[7]).norm()), .01)
        slots = [i for i in range(13) if i != 7]
        torch.testing.assert_close(changed[slots], original[slots], rtol=0, atol=1e-14)

    def test_future_branch_reads_own_action_and_slot3_only_despite_mixing_base(self):
        wrapped = self.wrapper(activate=True)
        original = self.delta(wrapped)
        source = self.condition.clone()
        source[:3] += torch.randn_like(source[:3]) * 20
        source[10] = torch.randn_like(source[10]) * 7
        changed = self.delta(wrapped, condition=source)
        unaffected = [i for i in range(13) if i != 10]
        torch.testing.assert_close(changed[unaffected], original[unaffected], rtol=0, atol=1e-14)
        self.assertGreater(float((changed[10] - original[10]).norm()), .01)
        base_a = wrapped.base(self.x, self.coords, self.active, self.condition)
        base_b = wrapped.base(self.x, self.coords, self.active, source)
        self.assertGreater(float((base_a[6] - base_b[6]).norm()), .01)
        # Explicit conditioning insulates the branch from other backbone input.
        changed_x = self.x + torch.randn_like(self.x)
        torch.testing.assert_close(self.delta(wrapped, x=changed_x), original, rtol=0, atol=1e-14)

    def test_branch_row_permutation_equivariance_not_inherited_from_toy_base(self):
        wrapped = self.wrapper(activate=True)
        order = torch.tensor([3, 1, 4, 0, 2])
        original = self.delta(wrapped)
        changed = self.delta(wrapped, x=self.x[:, order], coords=self.coords[order],
                             active=self.active[:, order], condition=self.condition[:, order])
        torch.testing.assert_close(changed, original[:, order], rtol=0, atol=1e-14)
        original_base = wrapped.base(self.x, self.coords, self.active, self.condition)
        permuted_base = wrapped.base(self.x[:, order], self.coords[order], self.active[:, order], self.condition[:, order])
        self.assertFalse(torch.allclose(permuted_base, original_base[:, order]))

    def test_mask_and_first_four_slots_are_unchanged(self):
        wrapped = self.wrapper(activate=True)
        active = self.active.clone()
        active[7:, 2] = False
        active[3, 4] = False
        delta = self.delta(wrapped, active=active)
        self.assertTrue(bool((delta[:4] == 0).all()))
        self.assertTrue(bool((delta[7:, 2] == 0).all()))
        self.assertTrue(bool((delta[4:, 4] == 0).all()))
        self.assertGreater(float(delta[4:, :2].norm()), .01)
        active[4:] = False
        self.assertTrue(bool((self.delta(wrapped, active=active) == 0).all()))

    def test_implicit_condition_matches_explicit_original_input(self):
        wrapped = self.wrapper(activate=True)
        implicit = wrapped(self.x, self.coords, self.active)
        explicit = wrapped(self.x, self.coords, self.active, condition=self.x)
        self.assertTrue(torch.equal(implicit, explicit))

    def test_two_updates_reach_all_new_groups_and_initial_and_every_z(self):
        wrapped = self.wrapper()
        wrapped.base.requires_grad_(False)
        model = nn.Module()
        model.decoder_backbone = wrapped
        optimizer = torch.optim.SGD(wrapped.branch_parameters(), lr=.2)
        target = torch.randn(9, 5, 64, dtype=torch.float64)
        initial_parameters = {name: p.detach().clone() for name, p in wrapped.named_parameters()
                              if not name.startswith('base.')}
        for update in range(2):
            source = self.condition.clone().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            delta = self.delta(wrapped, condition=source)
            (delta[4:] - target).square().mean().backward()
            report = spatial_action_gradients(model)
            self.assertEqual(set(report), set(initial_parameters))
            self.assertTrue(all(row['has_grad'] and row['finite'] for row in report.values()))
            self.assertGreater(report['Wo.weight']['norm'], 0.)
            if update == 0:
                self.assertTrue(all(row['norm'] == 0 for name, row in report.items() if name != 'Wo.weight'))
                self.assertTrue(bool((source.grad == 0).all()))
            else:
                self.assertTrue(all(row['norm'] > 0 for row in report.values()), report)
                self.assertGreater(float(source.grad[3].norm()), 0.)
                self.assertTrue(bool((source.grad[4:].flatten(1).norm(dim=1) > 0).all()))
                self.assertTrue(bool((source.grad[:3] == 0).all()))
                self.assertTrue(bool(torch.isfinite(source.grad).all()))
            optimizer.step()
        for name, parameter in wrapped.named_parameters():
            if name in initial_parameters:
                self.assertFalse(torch.equal(parameter, initial_parameters[name]), name)
        self.assertGreater(spatial_action_grad_norm(model), 0.)
        self.assertTrue(all(p.grad is None for p in wrapped.base.parameters()))

    def test_disabled_skips_even_poisoned_branch_and_optimizer_cannot_update_it(self):
        wrapped = self.wrapper(enabled=False)
        with torch.no_grad():
            wrapped.Wo.weight.fill_(float('nan'))
        frozen = {name: p.detach().clone() for name, p in wrapped.named_parameters()
                  if not name.startswith('base.')}
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.1)
        expected = wrapped.base(self.x, self.coords, self.active, self.condition)
        actual = wrapped(self.x, self.coords, self.active, self.condition)
        self.assertTrue(torch.equal(expected, actual))
        actual.square().mean().backward()
        optimizer.step()
        for name, parameter in wrapped.named_parameters():
            if name in frozen:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0, equal_nan=True)

    def test_attach_names_parameters_modes_replay_and_double_wrap_rejection(self):
        self.base.train()
        self.base.unused_mode_probe.eval()
        self.base.linear.weight.requires_grad_(False)
        model = nn.Module()
        model.decoder_backbone = self.base
        added = attach_spatial_action(model, True)
        wrapped = model.decoder_backbone
        self.assertEqual({id(p) for p in added}, {id(p) for p in wrapped.branch_parameters()})
        self.assertEqual(sum(p.numel() for p in added), 15040)
        self.assertFalse(wrapped.base.linear.weight.requires_grad)
        self.assertFalse(wrapped.base.unused_mode_probe.training)
        self.assertTrue(wrapped.training)
        self.assertIn('decoder_backbone.base.linear.weight', model.state_dict())
        self.assertIn('decoder_backbone.Wo.weight', model.state_dict())
        self.assertFalse(any(name.startswith('P.') for name, _ in wrapped.named_parameters()))
        with torch.no_grad():
            wrapped.Wo.weight.copy_(torch.randn_like(wrapped.Wo.weight) * .1)
        restored = nn.Module()
        restored.decoder_backbone = ToyBackbone()
        attach_spatial_action(restored, True)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(wrapped(self.x, self.coords, self.active, self.condition),
                                    restored.decoder_backbone(self.x, self.coords, self.active, self.condition)))
        with self.assertRaises(ValueError):
            attach_spatial_action(model, True)
        old_local = ToyBackbone()
        old_local._gs_local_skip_wrapped = True
        with self.assertRaises(ValueError):
            SpatialActionBackbone(old_local, True)
        disabled = nn.Module()
        disabled.decoder_backbone = ToyBackbone()
        self.assertEqual(attach_spatial_action(disabled, False), [])
        self.assertEqual(set(disabled.state_dict()), set(model.state_dict()))

    def test_gradient_report_distinguishes_missing_nonfinite_and_finite(self):
        model = nn.Module()
        model.decoder_backbone = self.wrapper()
        report = spatial_action_gradients(model)
        self.assertTrue(all(not row['has_grad'] and row['finite'] is None and row['norm'] == 0
                            for row in report.values()))
        model.decoder_backbone.Wo.weight.grad = torch.full_like(model.decoder_backbone.Wo.weight, float('inf'))
        report = spatial_action_gradients(model)
        self.assertTrue(report['Wo.weight']['has_grad'])
        self.assertFalse(report['Wo.weight']['finite'])
        self.assertIsNone(report['Wo.weight']['norm'])
        with self.assertRaises(RuntimeError):
            spatial_action_grad_norm(model)

    def test_shape_rejection_and_input_preservation(self):
        wrapped = self.wrapper(activate=True)
        saved = [v.clone() for v in (self.x, self.coords, self.active, self.condition)]
        wrapped(self.x, self.coords, self.active, self.condition)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(saved, (self.x, self.coords, self.active, self.condition))))
        cases = [(self.x[:12], self.coords, self.active[:12], self.condition[:12]),
                 (self.x, self.coords[:, :2], self.active, self.condition),
                 (self.x, self.coords, self.active.float(), self.condition),
                 (self.x, self.coords, self.active, self.condition.float())]
        for args in cases:
            with self.assertRaises(ValueError):
                wrapped(*args)


if __name__ == '__main__':
    unittest.main()
