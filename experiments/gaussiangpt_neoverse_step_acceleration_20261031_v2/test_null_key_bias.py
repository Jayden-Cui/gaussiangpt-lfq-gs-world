"""Small CPU algebra/synthetic-name tests; not a real-QFormer dynamic witness."""

import copy
import unittest

import torch
from torch import nn

from null_key_bias import (PARAMETER_NAMES, POLICY, freeze_null_key_bias,
                           assert_null_key_bias_unchanged)


class AttentionNames(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(64, 32)


class LayerNames(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_token_to_image = AttentionNames()


class PosteriorNames(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LayerNames(), LayerNames()])
        self.final_attn_token_to_image = AttentionNames()


class NamedFixture(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.posteriors = nn.ModuleList([PosteriorNames() for _ in range(9)])
        self.other = nn.Parameter(torch.tensor(.4))
        self.already_frozen = nn.Parameter(torch.tensor(.2), requires_grad=False)
        if constant:
            self.posteriors.requires_grad_(False)


class NullKeyBiasTests(unittest.TestCase):
    def test_masked_and_unmasked_softmax_common_bias_invariance(self):
        torch.manual_seed(20261031)
        q = torch.randn(2, 4, 3, 8, dtype=torch.float64)
        k = torch.randn(2, 4, 5, 8, dtype=torch.float64)
        v = torch.randn(2, 4, 5, 6, dtype=torch.float64)
        bias = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        change = torch.randn_like(bias)*2
        valid = torch.tensor([[True,False,True,True,False],[False,True,False,False,False]])
        for mask in (None, valid):
            def attend(value):
                scores = q @ (k+value[None,:,None]).transpose(-1,-2) / (8**.5)
                if mask is not None:
                    scores = scores.masked_fill(~mask[:,None,None], float('-inf'))
                return scores.softmax(-1) @ v
            output = attend(bias)
            torch.testing.assert_close(output, attend(bias+change), rtol=1e-12, atol=1e-12)
            gradient, = torch.autograd.grad(output.square().sum(), bias)
            self.assertLess(float(gradient.abs().max()), 1e-12)

    def test_preserve_partial_values_groups_and_existing_moments(self):
        model = NamedFixture()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        # Synthetic direct loss deliberately populates existing moments. It is
        # not the QFormer objective and is never presented as a training record.
        sum(parameter.square().mean() for parameter in model.parameters() if parameter.requires_grad).backward()
        optimizer.step()
        original_groups = [[id(p) for p in group['params']] for group in optimizer.param_groups]
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        guard = freeze_null_key_bias(model, optimizer)
        self.assertEqual(guard.in_optimizer_count, 27)
        self.assertEqual(original_groups, [[id(p) for p in group['params']] for group in optimizer.param_groups])
        after_freeze = optimizer.state_dict()
        self.assertEqual(saved_optimizer['param_groups'], after_freeze['param_groups'])
        for identity, state in saved_optimizer['state'].items():
            self.assertEqual(set(state), set(after_freeze['state'][identity]))
            for key, value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, after_freeze['state'][identity][key], rtol=0, atol=0)
                else:
                    self.assertEqual(value, after_freeze['state'][identity][key])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            if name in PARAMETER_NAMES:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
        # Non-policy parameters still train; selected nonzero moments must not
        # drift their parameters or their state while their grad stays None.
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().mean() for parameter in model.parameters() if parameter.requires_grad).backward()
        optimizer.step()
        record = assert_null_key_bias_unchanged(model, optimizer, guard)
        self.assertEqual(record['policy'], POLICY)
        self.assertEqual(record['count'], 27)
        self.assertEqual(record['scalar_count'], 864)
        self.assertTrue(record['unchanged'])
        self.assertNotEqual(float(model.other.detach()), float(before['other']))
        self.assertEqual(len(saved_optimizer['state']), len(optimizer.state_dict()['state']))
        # Loading weights does not restore requires_grad: explicitly reapply.
        again = freeze_null_key_bias(model, optimizer)
        self.assertEqual(assert_null_key_bias_unchanged(model,optimizer,again), record)
        optimizer.state[model.get_parameter(PARAMETER_NAMES[0])]['exp_avg'].add_(1)
        with self.assertRaisesRegex(RuntimeError, 'optimizer state changed'):
            assert_null_key_bias_unchanged(model,optimizer,again)

    def test_constant_absent_state_and_policy_violation(self):
        model = NamedFixture(constant=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        guard = freeze_null_key_bias(model, optimizer)
        self.assertEqual(guard.in_optimizer_count, 0)
        before_state_count = len(optimizer.state)
        record = assert_null_key_bias_unchanged(model, optimizer, guard)
        self.assertEqual(len(optimizer.state), before_state_count)
        self.assertEqual(record['in_optimizer_count'], 0)
        selected = model.get_parameter(PARAMETER_NAMES[0])
        selected.grad = torch.zeros_like(selected)
        with self.assertRaisesRegex(RuntimeError, 'received a gradient'):
            assert_null_key_bias_unchanged(model,optimizer,guard)
        selected.grad = None
        with torch.no_grad():
            selected.add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'value changed'):
            assert_null_key_bias_unchanged(model,optimizer,guard)

    def test_wrong_geometry_is_rejected_before_any_freezing(self):
        model = NamedFixture()
        model.posteriors[0].layers[0].cross_attn_token_to_image.k_proj = nn.Linear(64, 16)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        with self.assertRaisesRegex(RuntimeError, 'geometry'):
            freeze_null_key_bias(model,optimizer)
        self.assertTrue(all(model.get_parameter(name).requires_grad for name in PARAMETER_NAMES))


if __name__ == '__main__':
    unittest.main(verbosity=2)
