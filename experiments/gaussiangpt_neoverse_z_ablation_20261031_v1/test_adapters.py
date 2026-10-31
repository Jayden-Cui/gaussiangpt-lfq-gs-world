"""Real pinned-model CPU tests; no mocks of LFQ/backbone and no optimizer steps.

python -B -m unittest -v test_adapters
Optional Z_ADAPTER_SOURCE_PARENT points to the existing experiments/outputs dir.
"""
from __future__ import annotations

import copy
import io
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import torch
from action_bottleneck import ActionBottleneck, soft_codes
from model_adapter import load_pinned, attach_adapters, clone_pair


def window(seed=71, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.tensor([[2, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [4, 0, 0], [8, 0, 0]])
    return [dict(coordinates=coordinates.clone(),
                 features=(torch.randn(6, 64, generator=generator)+.03*h).requires_grad_(requires_grad),
                 pts=h/29.97, frame_index=h) for h in range(10)]


def nonzero_head(model):
    with torch.no_grad():
        model.features.weight.copy_(torch.eye(64)*.2)
        model.features.bias.fill_(.015)
        model.decoder_backbone.Wo.weight.copy_(torch.eye(64)*.07)
    return model


def features(predictions):
    return torch.stack([row['features'] for row in predictions[0]])


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261031)
        parent = os.environ.get('Z_ADAPTER_SOURCE_PARENT', str(Path(__file__).resolve().parent.parent))
        cls.api = load_pinned(parent)
        base = cls.api.AnchoredLDM(checkpoint_blocks=False).eval()
        base.initialize_output_units(window()[0]['features'])
        cls.api.attach_spatial_action(base, True)
        cls.original = nonzero_head(base)
        cls.common = copy.deepcopy(cls.original)
        cls.original_parameters = {k: tuple(p.shape) for k, p in cls.common.named_parameters()}
        cls.receipt = attach_adapters(cls.common, s_ref=1.)

    def model(self, mode='continuous16'):
        model = copy.deepcopy(self.common)
        model.action_bottleneck.mode = mode
        self.assertIs(model.encode_windows.__self__, model)
        return model

    def test_same_initial_parameters_and_source_apis_are_preserved(self):
        self.assertEqual(len(self.receipt['replacements']), 12)
        self.assertEqual({k: tuple(p.shape) for k, p in self.common.named_parameters()}, self.original_parameters)
        pair = clone_pair(self.common)
        left, right = pair.values()
        self.assertEqual(set(left.state_dict()), set(right.state_dict()))
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)
        for a, b in zip(left.parameters(), right.parameters()):
            self.assertIsNot(a, b)
            self.assertEqual(a.requires_grad, b.requires_grad)
        # The unadapted historical model has not been altered by injection into its clone.
        self.assertTrue(self.original.decoder_backbone.base.blocks[0].temporal.norm1.across_time)
        with self.assertRaisesRegex(ValueError, 'unadapted'):
            attach_adapters(self.common, s_ref=1.)

    def test_lfq_scale_one_matches_real_library_forward_backward_and_ids(self):
        for training in (True, False):
            q1 = copy.deepcopy(self.common.quantizer).train(training)
            q2 = copy.deepcopy(q1)
            bottle = ActionBottleneck('lfq16', s_ref=1.).train(training)
            generator = torch.Generator().manual_seed(190)
            a = torch.randn(4, 9, 64, generator=generator, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            reference = [q1(a[:, h:h+1], inv_temperature=1.) for h in range(9)]
            expected = torch.cat([x.quantized for x in reference], 1)
            expected_aux = torch.stack([x.entropy_aux_loss for x in reference]).mean()
            actual = bottle(b, q2)
            torch.testing.assert_close(expected, actual['action_latents'], atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(expected_aux, actual['aux_loss'], atol=0, rtol=0)
            self.assertTrue(torch.equal(torch.cat([x.indices for x in reference], 1), actual['indices']))
            probe = torch.randn(expected.shape, generator=generator)
            lhs = (expected*probe).mean()+.1*expected_aux
            rhs = (actual['action_latents']*probe).mean()+.1*actual['aux_loss']
            lhs.backward(); rhs.backward()
            if training:
                self.assertGreater(float(a.grad.abs().sum()), 0.)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)
            else:
                self.assertIsNone(a.grad)
                self.assertIsNone(b.grad)
            for (name1, p1), (name2, p2) in zip(q1.named_parameters(), q2.named_parameters()):
                self.assertEqual(name1, name2)
                if p1.grad is None:
                    self.assertIsNone(p2.grad)
                else:
                    torch.testing.assert_close(p1.grad, p2.grad, atol=2e-7, rtol=2e-5)

    def test_scale_is_global_once_bound_and_auxiliary_is_shared(self):
        q = copy.deepcopy(self.common.quantizer).train()
        generator = torch.Generator().manual_seed(193)
        batches = [torch.randn(b, 9, 64, generator=generator) for b in (2, 3)]
        unbound = ActionBottleneck('lfq16')
        with self.assertRaisesRegex(RuntimeError, 'Bind'):
            unbound(batches[0], q)
        evidence = unbound.calibrate(iter(batches), q)
        all_u = torch.cat([soft_codes(z, q).detach().double().flatten() for z in batches])
        self.assertAlmostEqual(evidence['s_ref'], float(all_u.square().mean().sqrt()), places=6)
        self.assertEqual(evidence['entries'], 5*9*16)
        with self.assertRaisesRegex(RuntimeError, 'already bound'):
            unbound.calibrate(iter(batches), q)
        continuous = copy.deepcopy(unbound)
        continuous.mode = 'continuous16'
        z1 = batches[0].clone().requires_grad_()
        z2 = batches[0].clone().requires_grad_()
        hard, soft = unbound(z1, q), continuous(z2, q)
        torch.testing.assert_close(hard['aux_loss'], soft['aux_loss'], atol=0, rtol=0)
        grad1 = torch.autograd.grad(hard['aux_loss'], z1)[0]
        grad2 = torch.autograd.grad(soft['aux_loss'], z2)[0]
        self.assertGreater(float(grad1.abs().sum()), 0.)
        torch.testing.assert_close(grad1, grad2, atol=0, rtol=0)
        saved = unbound.s_ref.clone()
        unbound(batches[0]*13, q)
        torch.testing.assert_close(saved, unbound.s_ref, atol=0, rtol=0)
        self.assertIsNone(soft['indices'])
        self.assertFalse(torch.equal(hard['action_latents'], soft['action_latents']))

    def test_posterior_prefix_is_invariant_but_can_read_observed_future(self):
        model = self.model()
        original = window(71)
        changed = copy.deepcopy(original)
        for h in range(4, 10):
            changed[h]['coordinates'] += torch.tensor([30+h, 0, 0])
            changed[h]['features'] = changed[h]['features']*20+15
        with torch.no_grad():
            first = model.encode_windows([original])
            second = model.encode_windows([changed])
            initial = model.encode_initial(original[0])
        torch.testing.assert_close(first['z_prequant'][:, :3], second['z_prequant'][:, :3], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(first['initial_contexts'][0].values, initial, atol=0, rtol=0)
        torch.testing.assert_close(second['initial_contexts'][0].values, initial, atol=0, rtol=0)
        self.assertGreater(float((first['z_prequant'][:, 3:]-second['z_prequant'][:, 3:]).abs().max()), 1e-5)
        differentiable = window(77, requires_grad=True)
        z = model.posterior_vectors([differentiable])
        probe = torch.linspace(-1, 1, 64)
        observed, later = torch.autograd.grad((z[:, 2]*probe).sum(),
            [differentiable[3]['features'], differentiable[7]['features']], allow_unused=True)
        self.assertGreater(float(observed.abs().sum()), 1e-7)
        self.assertTrue(later is None or torch.count_nonzero(later) == 0)

    def test_nonzero_decoder_prefix_values_and_gradients_for_later_actions(self):
        model = self.model()
        initial = window(88)[0]
        condition = [self.api.make_condition(initial)]
        codes = torch.randn(1, 9, 16, generator=torch.Generator().manual_seed(118), requires_grad=True)
        decoded = features(model.decode_action_codes(codes, condition))
        probe = torch.linspace(-.9, .7, decoded.numel()).reshape_as(decoded)
        self.assertGreater(float((decoded-initial['features']).detach().abs().max()), 1e-4)
        for h in (0, 1, 4, 8):
            gradient = torch.autograd.grad((decoded[:h+1]*probe[:h+1]).sum(), codes, retain_graph=True)[0]
            self.assertEqual(int(torch.count_nonzero(gradient[:, h:])), 0)
            if h:
                self.assertGreater(float(gradient[:, h-1].abs().sum()), 1e-6)
            changed = codes.detach().clone()
            changed[:, h:] = changed[:, h:]*-4+2.1
            with torch.no_grad():
                altered = features(model.decode_action_codes(changed, condition))
            torch.testing.assert_close(decoded[:h+1], altered[:h+1], atol=1e-6, rtol=1e-6)
            self.assertGreater(float((decoded[h+1:]-altered[h+1:]).detach().abs().max()), 1e-5)
        # Sensitivity witness: the same test input exposes the OLD full-window path.
        with torch.no_grad():
            old_a = features(self.original.decode_latents(self.original.quantizer.project_out(codes.detach()), condition))
            old_b = features(self.original.decode_latents(self.original.quantizer.project_out(-4*codes.detach()+2.1), condition))
        self.assertGreater(float((old_a[0]-old_b[0]).abs().max()), 1e-5)

    def test_real_encode_decode_keeps_every_initial_row_and_has_gradient_to_each_action(self):
        model = self.model().train()
        windows = [window(81), window(85)]
        encoded = model.encode_windows(windows)
        self.assertEqual(encoded['action_codes'].shape, (2, 9, 16))
        self.assertEqual(encoded['action_latents'].shape, (2, 9, 64))
        encoded['action_latents'].retain_grad()
        decoded = model.decode_latents(encoded['action_latents'], [self.api.make_condition(w[0]) for w in windows],
                                       initial_contexts=encoded['initial_contexts'])
        loss = sum((prediction['features']*torch.linspace(-1, 1, 64)).sum()
                   for sample in decoded for prediction in sample)
        loss.backward()
        self.assertTrue(bool((encoded['action_latents'].grad.norm(dim=2) > 0).all()))
        for b, row in enumerate(decoded):
            self.assertEqual(len(row), 10)
            self.assertTrue(all(torch.equal(p['coordinates'], windows[b][0]['coordinates']) for p in row))
        self.assertGreater(float(model.quantizer.project_in.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.queries.weight.grad.abs().sum()), 0.)

    def test_continuous_float_serialization_and_ids_cannot_replace_codes(self):
        for mode in ('continuous16', 'lfq16'):
            model = self.model(mode)
            with torch.no_grad():
                encoded = model.encode_windows([window(98)])
            packet = model.action_bottleneck.packet(encoded)
            buffer = io.BytesIO()
            torch.save(packet, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, map_location='cpu', weights_only=True)
            self.assertTrue(torch.equal(packet['codes'], loaded['codes']))
            replay = model.action_bottleneck.packet_to_latents(loaded, model.quantizer)
            torch.testing.assert_close(replay, encoded['action_latents'], atol=0, rtol=0)
            if mode == 'continuous16':
                self.assertFalse(bool(((loaded['codes'] == -1) | (loaded['codes'] == 1)).all()))
                with self.assertRaisesRegex(ValueError, 'actual float'):
                    model.decode_ids(loaded['diagnostic_ids'], [self.api.make_condition(window(98)[0])])
                loaded['indices'] = loaded['diagnostic_ids']
                with self.assertRaisesRegex(ValueError, 'cannot be represented'):
                    model.action_bottleneck.packet_to_latents(loaded, model.quantizer)
            else:
                loaded['indices'][0, 0] ^= 1
                with self.assertRaisesRegex(ValueError, 'disagree'):
                    model.action_bottleneck.packet_to_latents(loaded, model.quantizer)

    def test_model_scale_calibration_preserves_modes_and_rejects_empty_data(self):
        common = copy.deepcopy(self.original).train()
        common.encoder_backbone.eval()  # Deliberately mixed mode must survive.
        attach_adapters(common)
        before = [m.training for m in common.modules()]
        with self.assertRaisesRegex(ValueError, 'Empty'):
            common.calibrate_actions([])
        self.assertFalse(bool(common.action_bottleneck.scale_bound))
        receipt = common.calibrate_actions([[window(100)], [window(101)]])
        self.assertEqual(receipt['entries'], 2*9*16)
        self.assertEqual(before, [m.training for m in common.modules()])
        clones = clone_pair(common)
        self.assertEqual(float(clones['lfq16'].action_bottleneck.s_ref), float(clones['continuous16'].action_bottleneck.s_ref))

    def test_state_dict_round_trip_requires_mode_metadata_and_injected_structure(self):
        source = self.model('continuous16')
        buffer = io.BytesIO()
        torch.save(dict(action_mode=source.action_bottleneck.mode, model=source.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=True)
        restored = copy.deepcopy(self.original)
        before = dict(restored.named_parameters())
        attach_adapters(restored, mode=saved['action_mode'])
        self.assertTrue(all(dict(restored.named_parameters())[name] is value for name, value in before.items()))
        restored.load_state_dict(saved['model'], strict=True)
        with torch.no_grad():
            left = source.encode_windows([window(109)])
            right = restored.encode_windows([window(109)])
        torch.testing.assert_close(left['action_codes'], right['action_codes'], atol=0, rtol=0)
        self.assertTrue(bool(restored.action_bottleneck.scale_bound))
        self.assertEqual(restored.action_bottleneck.mode, 'continuous16')
        self.assertNotIn('action_mode', restored.state_dict())


if __name__ == '__main__':
    unittest.main(verbosity=2)
