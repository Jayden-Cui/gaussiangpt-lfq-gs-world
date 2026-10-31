"""Actual small CPU tensors; no codec, LFQ dependency, renderer or CUDA witness."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
import runner_core as core


def source():
    return {name: char*64 for name, char in zip(core.SOURCE_KEYS, 'abcd')}


def actual_codes():
    # Every start/h/channel is distinct, so dropped/reordered horizons are visible.
    return torch.arange(168*9*16, dtype=torch.float32).reshape(168, 9, 16)/1003. - 7.


def packet():
    return core.make_action_packet(actual_codes(), mode='continuous16', starts=list(range(168)),
                                   s_ref=.271, source=source())


class RunnerCoreTests(unittest.TestCase):
    def test_exact_candidate_plan_is_paired_and_not_frozen(self):
        plan = core.candidate_plan()
        self.assertFalse(plan['budget_frozen'])
        self.assertEqual(plan['status'], 'candidate')
        self.assertEqual(core.plan_hash(plan['batches']), core.CANDIDATE_PLAN_SHA)
        self.assertEqual(len(plan['batches']), 800)
        self.assertTrue(all(len(batch) == len(set(batch)) == 4 for batch in plan['batches']))
        for cycle in range(19):
            values = [s for batch in plan['batches'][cycle*42:(cycle+1)*42] for s in batch]
            self.assertEqual(sorted(values), list(range(168)))
        self.assertEqual(sum(v == 19 for v in plan['exposures_per_start'].values()), 160)
        self.assertEqual(sum(v == 20 for v in plan['exposures_per_start'].values()), 8)
        self.assertEqual(core.candidate_plan()['batches'], plan['batches'])

    def test_continuous_file_roundtrip_is_bit_exact_and_metadata_bound(self):
        original = packet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'actions.pt'
            receipt = core.write_action_packet(path, original)
            args = dict(expected_file_sha256=receipt['file_sha256'], expected_mode='continuous16',
                        expected_source=source(), expected_s_ref=.271, expected_starts=list(range(168)))
            loaded = core.read_action_packet(path, **args)
            self.assertTrue(torch.equal(loaded['codes'], actual_codes()))
            self.assertEqual(loaded['codes'].dtype, torch.float32)
            self.assertFalse(loaded['codec_admission_verified'])
            with self.assertRaises(FileExistsError): core.write_action_packet(path, original)
            with self.assertRaisesRegex(ValueError, 'mode mismatch'):
                core.read_action_packet(path, **{**args, 'expected_mode': 'lfq16'})
            wrong = {**source(), 'codec_binding_sha256': 'e'*64}
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                core.read_action_packet(path, **{**args, 'expected_source': wrong})
            with self.assertRaisesRegex(ValueError, 'scale mismatch'):
                core.read_action_packet(path, **{**args, 'expected_s_ref': .272})
            with self.assertRaisesRegex(ValueError, 'file SHA256'):
                core.read_action_packet(path, **{**args, 'expected_file_sha256': '0'*64})
        altered = deepcopy(original); altered['codes'][0, 0, 0] += .001
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            core.validate_action_packet(altered, expected_mode='continuous16', expected_source=source(), expected_s_ref=.271)

    def test_lfq_actual_signs_and_continuous_shape_are_checked_without_quantizer(self):
        signs = torch.where(actual_codes() > 0, 1., -1.)
        valid = core.make_action_packet(signs, mode='lfq16', starts=list(range(168)), s_ref=.271, source=source())
        core.validate_action_packet(valid, expected_mode='lfq16', expected_source=source(), expected_s_ref=.271)
        for codes, mode in ((actual_codes(), 'lfq16'), (actual_codes().long(), 'continuous16'),
                            (actual_codes()[:, :8], 'continuous16')):
            with self.assertRaises(ValueError):
                core.make_action_packet(codes, mode=mode, starts=list(range(168)), s_ref=.271, source=source())
        bad = actual_codes(); bad[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            core.make_action_packet(bad, mode='continuous16', starts=list(range(168)), s_ref=.271, source=source())

    def test_controls_move_whole_bundle_and_preserve_recipient_frame_identity(self):
        original = packet(); recipients = [0, 35, 123, 167]
        options = dict(expected_mode='continuous16', expected_source=source(), expected_s_ref=.271)
        for control in core.CODE_CONTROLS:
            result = core.select_control_codes(original, control, recipients, **options)
            for i, row in enumerate(result['mapping']):
                self.assertEqual(row['recipient_frames'], list(range(recipients[i], recipients[i]+10)))
                if control.startswith('shuffle'):
                    self.assertNotEqual(row['donor_start'], recipients[i])
                expected = actual_codes()[row['donor_start']]
                if control == 'fixed_single': expected = actual_codes()[0, 0].expand(9, 16)
                self.assertTrue(torch.equal(result['codes'][i], expected))
                self.assertEqual(row['donor_horizons'], [1]*9 if control == 'fixed_single' else list(range(1, 10)))
        # Stored row order is explicit, not implicitly equal to the start index.
        reverse = core.make_action_packet(actual_codes().flip(0), mode='continuous16',
            starts=list(reversed(range(168))), s_ref=.271, source=source())
        got = core.select_control_codes(reverse, 'shuffle84', [167], **options)['codes'][0]
        self.assertTrue(torch.equal(got, actual_codes()[83]))
        partial = core.make_action_packet(actual_codes()[5:6], mode='continuous16', starts=[5], s_ref=.271, source=source())
        with self.assertRaisesRegex(ValueError, 'donor absent'):
            core.select_control_codes(partial, 'fixed_bundle', [5], **options)

    def test_blinding_removes_future_geometry_and_preserves_actual_pts(self):
        states = [dict(coordinates=torch.tensor([[h, 0, 0], [h, 1, 0]], dtype=torch.int32),
                       features=torch.full((2, 64), float(h)), pts=1.2 + .033*h + .00001*h*h,
                       source_pts=100. + .033*h, frame_index=35+h, clip='clip', stride=4, grid=.008,
                       metadata={'future_geometry': torch.full((3, 3), 900+h)}) for h in range(10)]
        blinded = core.blind_window(states)
        for h, value in enumerate(blinded):
            self.assertTrue(torch.equal(value['coordinates'], states[0]['coordinates']))
            self.assertTrue(torch.equal(value['features'], states[0]['features']))
            self.assertEqual(value['pts'], states[h]['pts'])
            self.assertEqual(value['source_pts'], states[h]['source_pts'])
            self.assertEqual(value['frame_index'], 35+h)
            self.assertTrue(value['blinded_posterior'])
            self.assertNotIn('metadata', value)
        blinded[1]['features'].add_(7)
        self.assertTrue(torch.equal(states[0]['features'], torch.zeros(2, 64)))
        self.assertTrue(torch.equal(blinded[2]['features'], torch.zeros(2, 64)))
        invalid = deepcopy(states); invalid[2]['pts'] = invalid[1]['pts']
        with self.assertRaisesRegex(ValueError, 'strictly increasing'): core.blind_window(invalid)
        with self.assertRaisesRegex(ValueError, 'T10'): core.blind_window(states[:-1])

    def test_evaluation_coverage_keeps_all_nine_horizons_and_overlaps(self):
        coverage = core.evaluation_coverage()
        self.assertEqual(coverage['future_exposures'], 1512)
        self.assertEqual(coverage['physical_future_frame_count'], 176)
        self.assertEqual(coverage['exposures_per_horizon'], {h: 168 for h in range(1, 10)})
        self.assertEqual(coverage['initial_exposures'], 168)
        self.assertEqual(coverage['all_physical_frames'], list(range(177)))
        counts = coverage['exposures_per_physical_frame']
        self.assertEqual((counts[1], counts[9], counts[168], counts[176]), (1, 9, 9, 1))
        self.assertEqual(sum(counts.values()), 1512)
        with self.assertRaisesRegex(ValueError, 'All 168'): core.evaluation_coverage(list(range(167)))
        with self.assertRaisesRegex(ValueError, 'Duplicate'): core.evaluation_coverage([0]*168)


if __name__ == '__main__':
    unittest.main()
