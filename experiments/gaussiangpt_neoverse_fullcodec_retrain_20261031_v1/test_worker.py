"""Small actual CPU checks: state roundtrip, coverage gates, and resource rejection.

Never instantiate CUDA, MinkowskiEngine, native renderer or a pretend GPU.
"""
import io
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import worker as w


class WorkerTests(unittest.TestCase):
    def test_actual_protocol_and_semantic_drift_rejection(self):
        protocol = w.read(Path(__file__).with_name('protocol.json'))
        w.validate_protocol(protocol)
        for keys, value in ((('train', 'learning_rate'), .001),
                            (('train', 'force_target_epochs'), [True, True]),
                            (('resources', 'gpu_physical_id'), 0),
                            (('model', 'state_quantization'), True)):
            changed = deepcopy(protocol)
            changed[keys[0]][keys[1]] = value
            with self.assertRaisesRegex(RuntimeError, 'Protocol drift'):
                w.validate_protocol(changed)

    def test_weights_only_rng_roundtrip_replays_real_cpu_generators(self):
        original = w.capture_rng(include_cuda=False)
        try:
            random.seed(71); np.random.seed(72); torch.manual_seed(73)
            state = w.capture_rng(include_cuda=False)
            expected = (random.random(), np.random.rand(5), torch.rand(5))
            stream = io.BytesIO()
            torch.save({'rng': state}, stream)
            stream.seek(0)
            loaded = torch.load(stream, map_location='cpu', weights_only=True)
            self.assertFalse(loaded['rng']['cuda_captured'])
            self.assertEqual(loaded['rng']['torch_cuda'], [])
            w.restore_rng(loaded['rng'], require_cuda=False)
            self.assertEqual(random.random(), expected[0])
            np.testing.assert_array_equal(np.random.rand(5), expected[1])
            self.assertTrue(torch.equal(torch.rand(5), expected[2]))
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                w.restore_rng(loaded['rng'], require_cuda=True)
        finally:
            w.restore_rng(original, require_cuda=False)

    def test_complete_epoch_cursor_and_missing_or_foreign_frame_rejection(self):
        expected = {(f'clip_{c}', f) for c in range(50) for f in range(177)}
        first = w.checkpoint_cursor(2213, [set(expected), set()], expected, w.PLAN_SHAS)
        self.assertEqual((first['epoch_index'], first['next_batch_index'], first['next_global_step']), (1, 0, 2214))
        self.assertIs(first['next_force_target'], False)
        final = w.checkpoint_cursor(4426, [set(expected), set(expected)], expected, w.PLAN_SHAS)
        self.assertEqual(final['total_frame_exposures'], 17700)
        self.assertTrue(final['training_complete'])
        self.assertIsNone(final['next_global_step'])
        invalid = set(expected); invalid.pop(); invalid.add(('not_train', 0))
        for step, seen, plans in ((2213, [invalid, set()], w.PLAN_SHAS),
                                   (4426, [set(expected), invalid], w.PLAN_SHAS),
                                   (3940, [set(expected), set()], w.PLAN_SHAS),
                                   (2213, [set(expected), set()], list(reversed(w.PLAN_SHAS)))):
            with self.assertRaises(RuntimeError):
                w.checkpoint_cursor(step, seen, expected, plans)

    def test_coverage_rejects_repeat_before_count_can_mask_missing_frame(self):
        expected = {('a', i) for i in range(6)}
        refs = [SimpleNamespace(clip='a', frame=i) for i in range(6)]
        seen = set()
        w.add_coverage(seen, refs[:4], expected)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            w.add_coverage(seen, refs[2:4], expected)
        w.add_coverage(seen, refs[4:], expected)
        self.assertEqual(seen, expected)

    def test_resource_gates_fail_closed_without_any_cuda_calls(self):
        w.check_admission(1, 16 * 1024, 24 * 1024, 8 * w.GIB, '1')
        for args in ((0, 24*1024, 24*1024, 8*w.GIB, '0'),
                     (1, 16*1024-1, 24*1024, 8*w.GIB, '1'),
                     (1, 16*1024, 24*1024, 8*w.GIB-1, '1'),
                     (1, float('nan'), 24*1024, 8*w.GIB, '1')):
            with self.assertRaises(RuntimeError): w.check_admission(*args)
        w.check_limits(7200, 10*w.GIB, 12*w.GIB, 12*w.GIB)
        with self.assertRaises(TimeoutError): w.check_limits(7200.01, 0, 0, 0)
        for readings in ((0, 12*w.GIB+1, 0, 0), (0, 0, 0, 12*w.GIB+1), (0, 0, float('nan'), 0)):
            with self.assertRaises(RuntimeError): w.check_limits(*readings)

    def test_adam_state_integrity_checks_real_cpu_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=0.)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            w.assert_finite_state(model, optimizer)
        model(torch.ones(1, 2)).square().sum().backward(); optimizer.step()
        w.assert_finite_state(model, optimizer)
        optimizer.state[model.weight]['exp_avg'].fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            w.assert_finite_state(model, optimizer)


if __name__ == '__main__':
    unittest.main()
