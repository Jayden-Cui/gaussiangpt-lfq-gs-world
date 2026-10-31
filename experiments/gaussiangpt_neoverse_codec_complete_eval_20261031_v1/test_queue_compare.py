"""Small CPU fixtures for identity, independent failure and paired statistics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch
import compare as c
import pipeline as q
import worker as w


def metrics(value):
    return {group: {metric: float(value) for metric in c.METRICS} for group in c.GROUPS}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/trainer')
        self.cfg = dict(q.QUEUE)
        self.identity = dict(pid=self.cfg['training_pid'], start_ticks=str(self.cfg['training_start_ticks']),
                             argv=['python', str(self.root / 'worker.py'), '--root', str(self.root)])

    def test_identity_never_admits_reused_pid_or_different_command(self):
        self.assertTrue(q.check_identity(self.identity, self.cfg, self.root))
        self.assertFalse(q.check_identity(None, self.cfg, self.root))
        for change in ({'pid': 1}, {'start_ticks': '40929789'}, {'argv': ['python', 'other.py']},
                       {'argv': ['python', str(self.root / 'worker.py'), '--root', '/other']}):
            with self.assertRaises(RuntimeError):
                q.check_identity({**self.identity, **change}, self.cfg, self.root)

    def test_wait_needs_both_successful_receipts_and_process_exit(self):
        snapshots = iter([self.identity, None])
        sleeps = []
        output = q.wait_for_training(self.root, self.cfg, 9000, lambda _: next(snapshots), threading.Event(),
            probe=lambda _: {'status': 'eligible'}, clock=lambda: 0, sleep=sleeps.append)
        self.assertEqual(sleeps, [5.])
        self.assertTrue(output['external_process_exited'])
        self.assertFalse(output['unrelated_processes_signalled'])
        with self.assertRaisesRegex(RuntimeError, 'without successful'):
            q.wait_for_training(self.root, self.cfg, 9000, lambda _: None, threading.Event(),
                probe=lambda _: None, clock=lambda: 0, sleep=sleeps.append)

    def test_wait_timeout_stop_and_failed_training_do_not_signal(self):
        clock_values = iter([0, 8500])
        with self.assertRaisesRegex(RuntimeError, 'wait expired'):
            q.wait_for_training(self.root, self.cfg, 9000, lambda _: self.identity, threading.Event(),
                clock=lambda: next(clock_values), sleep=lambda _: self.fail('unexpected sleep'))
        stop = threading.Event(); stop.set()
        with self.assertRaisesRegex(RuntimeError, 'Queue stopped'):
            q.wait_for_training(self.root, self.cfg, 9000, lambda _: self.identity, stop,
                clock=lambda: 0, sleep=lambda _: self.fail('unexpected sleep'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FAILED.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                q.completion_probe(root)

    def test_completion_probe_rejects_mismatched_receipts_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root / 'arms' / 'full_codec'; folder.mkdir(parents=True)
            (root / 'DONE.json').write_text('{}')
            self.assertIsNone(q.completion_probe(root))
            (folder / 'TRAIN_DONE.json').write_text('{}')
            (folder / 'step_4426.pt').write_bytes(b'checkpoint')
            old = dict(code_manifest_sha256=w.OLD_MANIFEST)
            with patch.object(w, 'full_receipts', return_value=({}, {'different': True}, {})), \
                 patch.object(w, 'read', return_value={'source': old}):
                with self.assertRaisesRegex(RuntimeError, 'must match'):
                    q.completion_probe(root)
            with patch.object(w, 'full_receipts', return_value=({}, {}, {})), \
                 patch.object(w, 'read', return_value={'source': old}), \
                 patch.object(w, 'validate_full_done', return_value='0' * 64):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint differs'):
                    q.completion_probe(root)

    def test_partial_or_failed_is_never_full_compare(self):
        complete = [dict(arm=arm, status='complete') for arm in w.ARMS]
        self.assertEqual(q.final_status(complete, True), 'complete')
        self.assertEqual(q.final_status(complete, False), 'partial')
        self.assertEqual(q.final_status(complete[:2], True), 'partial')
        self.assertEqual(q.final_status([complete[0], dict(arm='decoder_only', status='failed'),
                                       dict(arm='full_codec', status='failed')], True), 'partial')
        self.assertEqual(q.final_status([dict(arm=arm, status='failed') for arm in w.ARMS], False), 'failed')

    def test_resource_admission_checks_only_assigned_gpu_and_full_disk(self):
        self.assertEqual(q.check_resources('1, 16384', 100 * 2**30, 1, self.cfg)['gpu'], 1)
        for text, disk in [('1, 16383', 100 * 2**30), ('0, 20000', 100 * 2**30),
                           ('1, 20000', 100 * 2**30 - 1), ('1, 20000\n0, 20000', 100 * 2**30)]:
            with self.assertRaises(RuntimeError):
                q.check_resources(text, disk, 1, self.cfg)


class CompareTests(unittest.TestCase):
    def test_artifact_path_rejects_escape_without_reading_external_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(c.artifact_path(folder, 'states/clip/000.pt'), folder / 'states/clip/000.pt')
            with self.assertRaisesRegex(RuntimeError, 'escapes'):
                c.artifact_path(folder, '../outside.pt')

    def test_small_frame_set_rejects_partial_duplicate_and_boolean_identity(self):
        keys = {('a', 0), ('a', 1)}
        rows = [dict(clip='a', frame=0), dict(clip='a', frame=1)]
        self.assertEqual(set(c.index_frames(rows, keys)), keys)
        for invalid in (rows[:1], [rows[0], rows[0]], [rows[0], dict(clip='a', frame=True)]):
            with self.assertRaises(RuntimeError):
                c.index_frames(invalid, keys)

    def test_source_groups_are_equal_weight_and_exposed_eval_stays_separate(self):
        by_clip = dict(a=metrics(1), b=metrics(3), c=metrics(10), exposed=metrics(90))
        result = c.aggregates(by_clip, dict(a='first', b='first', c='second'), 'exposed')
        self.assertEqual(result['train_clip_equal']['codec_real']['mae'], 14 / 3)
        self.assertEqual(result['train_source_group_equal']['codec_real']['mae'], 6)
        self.assertEqual(result['exposed_eval_only']['codec_real']['mae'], 90)
        self.assertEqual(result['group_clip_counts'], {'first': 2, 'second': 1})

    def test_pairing_rejects_pts_roles_groups_or_missing_coverage(self):
        keys = {('clip', 0)}
        a = {('clip', 0): dict(split='TRAIN', source_group='source', pts=0.)}
        observed = []
        c.validate_pairing(a, deepcopy(a), keys, lambda x, y: observed.append((x, y)))
        self.assertEqual(len(observed), 1)
        for field, value in [('pts', 1.), ('split', 'EVAL-only'), ('source_group', 'other')]:
            b = deepcopy(a); b['clip', 0][field] = value
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                c.validate_pairing(a, b, keys, lambda *_: None)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            c.validate_pairing(a, {}, keys, lambda *_: None)

    def test_descriptive_comparisons_have_consistent_sign_and_no_ci_or_gate(self):
        evaluations = {}
        for arm, value in [('baseline', 10), ('decoder_only', 7), ('full_codec', 5)]:
            per_clip = dict(train=metrics(value), exposed=metrics(value + 1))
            evaluations[arm] = dict(per_clip=per_clip, aggregates=c.aggregates(per_clip, {'train': 'group'}, 'exposed'))
        result = c.descriptive_differences(evaluations)
        self.assertEqual(result['full_codec_vs_decoder_only']['aggregate_reference_minus_candidate']['train_clip_equal']['codec_real']['mae'], 2)
        self.assertFalse(result['full_codec_vs_baseline']['confidence_intervals_computed'])
        self.assertFalse(result['full_codec_vs_baseline']['automatic_quality_gate'])

    def test_replay_failure_is_data_and_false_pass_claim_is_rejected(self):
        def fields(actual, expected, context):
            for key, value in expected.items():
                w.require(type(actual.get(key)) is type(value) and actual[key] == value, context + '.' + key)
        replay = {'real_view': dict(mae=2e-5, psnr=70., workspace_roi_mae=0., workspace_roi_psnr=100.),
                  'nearby_view': dict(mae=0., psnr=100., workspace_roi_mae=0., workspace_roi_psnr=100.)}
        row = dict(**w.replay_status(replay), decoded_gaussians=10, replay_decoded_gaussians=11, decoded_count_difference=1)
        self.assertFalse(c.validate_replay_fields(row, replay, SimpleNamespace(fields=fields))['replay_passed'])
        row['replay_passed'] = True
        with self.assertRaisesRegex(RuntimeError, 'replay_passed'):
            c.validate_replay_fields(row, replay, SimpleNamespace(fields=fields))


if __name__ == '__main__':
    unittest.main()
