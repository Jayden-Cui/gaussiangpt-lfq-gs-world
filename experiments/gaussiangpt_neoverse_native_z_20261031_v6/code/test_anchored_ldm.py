"""Small CPU tests using the real pinned QFormer and installed LFQ package.

No weights, GPU, renderer, optimization step, or dependency installation.
Run: python -m unittest -v test_anchored_ldm
"""

import copy
import unittest
from unittest.mock import patch

import torch

from anchored_ldm import AnchoredLDM, make_condition


def window(seed=71, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    return [dict(coordinates=coordinates.clone(),
                 features=(torch.randn(4, 64, generator=generator) + .03 * h)
                 .requires_grad_(requires_grad),
                 pts=h / 29.97, frame_index=h)
            for h in range(10)]


class AnchoredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        cls.model = AnchoredLDM(checkpoint_blocks=False).eval()
        cls.windows = [window(71), window(81)]
        cls.model.initialize_output_units(torch.cat([s['features'] for w in cls.windows for s in w]))

    def model_with_nonzero_head(self, mode='posterior'):
        if mode == 'posterior':
            model = copy.deepcopy(self.model)
        else:
            model = AnchoredLDM(code_mode=mode, checkpoint_blocks=False).eval()
            model.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            model.features.weight.copy_(torch.eye(64) * .2)
            model.features.bias.zero_()
        return model

    def test_zero_head_exactly_recovers_all_initial_rows_ten_times(self):
        conditions = [make_condition(w[0]) for w in self.windows]
        with patch('torch.cdist', side_effect=AssertionError('Nearest search is forbidden')):
            condition = make_condition(self.windows[0][0])
        self.assertIs(condition.coordinates, self.windows[0][0]['coordinates'])
        with torch.no_grad():
            encoded = self.model.encode_windows(self.windows)
            predictions = self.model.decode_latents(
                encoded['z_quantized'], conditions, initial_contexts=encoded['initial_contexts'])
        self.assertEqual(len({id(p) for p in self.model.posteriors}), 9)
        self.assertFalse(hasattr(self.model, 'occupancy'))
        for b, sequence in enumerate(predictions):
            self.assertEqual(len(sequence), 10)
            for prediction in sequence:
                self.assertEqual(set(prediction), {'coordinates', 'features'})
                self.assertTrue(torch.equal(prediction['coordinates'], self.windows[b][0]['coordinates']))
                self.assertTrue(torch.equal(prediction['features'], self.windows[b][0]['features']))

    def test_nonzero_head_uses_real_valid_lfq_ids_and_bounds_residual(self):
        model = self.model_with_nonzero_head()
        condition = make_condition(self.windows[0][0])
        with torch.no_grad():
            low = model.decode_ids(torch.zeros(1, 9, dtype=torch.long), [condition])
            high = model.decode_ids(torch.full((1, 9), 65535, dtype=torch.long), [condition])
        differences = [float((a['features'] - b['features']).abs().max())
                       for a, b in zip(low[0], high[0])]
        self.assertGreater(max(differences), 1e-6)
        bound = model.output_std * model.residual_scale
        for prediction in low[0] + high[0]:
            residual = (prediction['features'] - condition.state['features']).abs()
            self.assertTrue(bool((residual <= bound + 1e-6).all()))

    def test_shared_initial_condition_has_no_true_future_dependence(self):
        changed = copy.deepcopy(self.windows[0])
        for h in range(1, 10):
            changed[h]['coordinates'][0] = torch.tensor([100 + h, 12, 5])
            changed[h]['features'] = changed[h]['features'] * 30 + 50
        with torch.no_grad():
            original = self.model.encode_windows([self.windows[0]])
            altered = self.model.encode_windows([changed])
            alone = self.model.encode_initial(self.windows[0][0])
        torch.testing.assert_close(original['initial_contexts'][0].values, alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(altered['initial_contexts'][0].values, alone, atol=1e-5, rtol=1e-5)
        first, second = make_condition(self.windows[0][0]), make_condition(changed[0])
        self.assertTrue(torch.equal(first.coordinates, second.coordinates))
        self.assertTrue(torch.equal(first.weights, second.weights))

    def test_constant_mode_skips_posteriors_and_true_future_and_keeps_inverse_trainable(self):
        model = self.model_with_nonzero_head('constant')
        changed = copy.deepcopy(self.windows[0])
        for h in range(1, 10):
            # Constant control must not even inspect future data.
            changed[h] = {'deliberately_unread_future': object()}
        with patch.object(model.posteriors[0], 'forward', side_effect=AssertionError('Posterior used')):
            first = model.encode_windows([self.windows[0]])
            second = model.encode_windows([changed])
        self.assertIsNone(first['z_prequant'])
        self.assertFalse(first['posterior_used'])
        self.assertFalse(first['aux_loss'].requires_grad)
        self.assertEqual(float(first['aux_loss']), 0.)
        self.assertTrue(torch.equal(first['indices'], torch.zeros(1, 9, dtype=torch.long)))
        self.assertTrue(torch.equal(first['z_quantized'], second['z_quantized']))
        a = model.decode_latents(first['z_quantized'], [make_condition(self.windows[0][0])],
                                 initial_contexts=first['initial_contexts'])
        b = model.decode_latents(second['z_quantized'], [make_condition(changed[0])],
                                 initial_contexts=second['initial_contexts'])
        for x, y in zip(a[0], b[0]):
            torch.testing.assert_close(x['features'], y['features'], atol=0, rtol=0)
        loss = sum((p['features'] * torch.linspace(-1, 1, 64)).sum() for p in a[0])
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum())
                               for p in model.quantizer.project_out.parameters() if p.grad is not None), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.posteriors.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.quantizer.project_in.parameters()))

    def test_zero_head_first_backward_is_head_only_for_motion_branch(self):
        model = copy.deepcopy(self.model)
        encoded = model.encode_windows(self.windows[:1])
        predictions = model.decode_latents(encoded['z_quantized'],
            [make_condition(self.windows[0][0])], initial_contexts=encoded['initial_contexts'])
        # Exclude LFQ auxiliary loss deliberately: inspect the RGB/reconstruction
        # branch's expected initial gradient, not entropy's separate gradient.
        loss = sum((p['features'] - self.windows[0][h]['features']).square().mean()
                   for h, p in enumerate(predictions[0]))
        loss.backward()
        self.assertGreater(float(model.features.weight.grad.abs().sum()), 0)
        for module in (model.posteriors, model.encoder_backbone, model.decoder_backbone):
            self.assertEqual(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_four_and_twenty_block_schedules_and_fixed_calibration(self):
        # TRAIN frame0-scale audits found h9 p99 normalized feature changes
        # around 2.20..2.39; 3.0 is a predeclared bound, not tuned on this test.
        self.assertEqual(self.model.residual_scale, 3.)
        self.assertEqual(self.model.inv_temperature, 1.)
        self.assertEqual(len(self.model.encoder_backbone.blocks), 4)
        self.assertEqual(self.model.encoder_backbone.scales, (1, 2, 4, 8))
        self.assertEqual(self.model.decoder_backbone.scales, (8, 4, 2, 1))
        self.assertTrue(all(b.condition_norm is not None for b in self.model.decoder_backbone.blocks))
        full = AnchoredLDM(blocks=20, checkpoint_blocks=False)
        self.assertEqual(len(full.encoder_backbone.blocks), 20)
        self.assertEqual(len(full.decoder_backbone.blocks), 20)
        with self.assertRaisesRegex(RuntimeError, 'already bound'):
            self.model.initialize_output_units(torch.randn(8, 64))
        with self.assertRaisesRegex(ValueError, '4 or 20'):
            AnchoredLDM(blocks=5)

    def test_entropy_temperature_keeps_hard_codes_but_changes_auxiliary_gradient(self):
        model = copy.deepcopy(self.model).train()
        z = torch.randn(4, 1, 64, requires_grad=True) * .07
        low = model.quantizer(z, inv_temperature=1.)
        high = model.quantizer(z, inv_temperature=100.)
        self.assertTrue(torch.equal(low[1], high[1]))
        torch.testing.assert_close(low[0], high[0], atol=0, rtol=0)
        gradient_low = torch.autograd.grad(low[2], z, retain_graph=True)[0]
        gradient_high = torch.autograd.grad(high[2], z)[0]
        self.assertTrue(bool(torch.isfinite(gradient_low).all()))
        self.assertGreater(float(gradient_low.abs().sum()), 0)
        self.assertFalse(torch.allclose(gradient_low, gradient_high, atol=1e-7, rtol=1e-5))

    def test_changed_or_expanded_initial_condition_is_rejected(self):
        condition = make_condition(self.windows[0][0])
        condition.coordinates = torch.cat([condition.coordinates, torch.tensor([[9, 9, 9]])])
        with self.assertRaisesRegex(ValueError, 'every initial coordinate'):
            self.model.decode_latents(torch.randn(1, 9, 64), [condition])


if __name__ == '__main__':
    unittest.main(verbosity=2)
