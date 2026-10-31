"""Pure CPU receipt/metric checks; no Torch/ME imports and no fake CUDA witness."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import worker as w


def metrics(mae=0., psnr=120.):
    return dict(mae=mae, psnr=psnr, workspace_roi_mae=mae, workspace_roi_psnr=psnr)


class WorkerTests(unittest.TestCase):
    def test_unpinned_inference_result_blocks_runtime_protocol_admission(self):
        p = w.read(Path(__file__).resolve().parent / 'protocol.json')
        p['inference']['diagnostic_result_sha256'] = None
        with self.assertRaisesRegex(RuntimeError, 'RESULT SHA'):
            w.validate_protocol(p)

    def test_original_threshold_is_inclusive_and_threshold_failure_is_recorded(self):
        replay = dict(real_view=metrics(1e-5, 60.), nearby_view=metrics())
        self.assertTrue(w.replay_status(replay)['replay_passed'])
        replay['real_view']['mae'] = 1.0000001e-5
        status = w.replay_status(replay)
        self.assertFalse(status['replay_passed'])
        self.assertEqual(status['failed_views'], ['real_view'])
        replay['nearby_view']['psnr'] = 59.9999
        self.assertEqual(w.replay_status(replay)['failed_views'], ['nearby_view', 'real_view'])

    def test_nan_or_malformed_metrics_remain_fatal(self):
        for key in w.METRICS:
            replay = dict(real_view=metrics(), nearby_view=metrics())
            replay['real_view'][key] = float('nan')
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                w.replay_status(replay)
        for replay in ({}, {'real_view': metrics()}, {'real_view': {}, 'nearby_view': metrics()}):
            with self.assertRaises(RuntimeError): w.replay_status(replay)

    def test_complete_quality_collection_never_hides_one_replay_failure(self):
        expected = {(f'clip_{c}', f) for c in range(51) for f in range(177)}
        rows = [dict(clip=c, frame=f, replay=dict(real_view=metrics(), nearby_view=metrics()), replay_passed=True)
                for c, f in sorted(expected)]
        rows[114]['replay']['real_view'] = metrics(1.2637862710107584e-5, 72.6033706665039)
        rows[114]['replay_passed'] = False
        summary = w.final_replay_summary(rows, expected)
        self.assertFalse(summary['replay_passed_all'])
        self.assertFalse(summary['codec_accepted'])
        self.assertEqual(summary['replay_failed_frames'], 1)
        self.assertEqual(summary['replay_acceptance']['status'], 'failed')
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            w.final_replay_summary(rows[:-1], expected)
        rows[114]['replay_passed'] = True
        with self.assertRaisesRegex(RuntimeError, 'flag differs'):
            w.final_replay_summary(rows, expected)

    def test_full_done_checkpoint_source_and_coverage_gates(self):
        new = dict(run='retrain', code_manifest_sha256=w.FULL_MANIFEST)
        old = dict(code_manifest_sha256=w.OLD_MANIFEST, epoch_plan_sha256=['one', 'two'])
        source = dict(new_source=new, old_source=old)
        sampling = dict(training_complete=True, completed_updates=4426, epoch_frames_seen=[8850, 8850],
                        total_frame_exposures=17700, epoch_index=2, next_global_step=None,
                        next_batch_index=0, epoch_plan_sha256=['one', 'two'])
        done = dict(status='complete', training_complete=True, steps=4426, arm='full_codec',
                    additional_codec_updates=4426, source=source, sampling=sampling,
                    checkpoint_sha256='a'*64, initial_checkpoint_sha256='b'*64)
        self.assertEqual(w.validate_full_done(done, deepcopy(done), new, old, has_failure=False), 'a'*64)
        for field, value in (('steps', 2213), ('training_complete', False), ('source', {}),
                             ('sampling', {**sampling, 'epoch_frames_seen': [8850, 6908]})):
            invalid = deepcopy(done); invalid[field] = value
            with self.assertRaises(RuntimeError):
                w.validate_full_done(invalid, deepcopy(invalid), new, old, has_failure=False)
        with self.assertRaisesRegex(RuntimeError, 'Failed'):
            w.validate_full_done(done, done, new, old, has_failure=True)
        saved = dict(schema='fullcodec_retrain_checkpoint_v1', arm='full_codec', step=4426,
                     failed=False, eligible_for_final_evaluation=True, requires_successful_done_receipt=True,
                     source=source, sampling=sampling, initial_checkpoint_sha256='b'*64)
        w.validate_full_checkpoint(saved, done, 'a'*64)
        for changes in ({'failed': True}, {'step': 2213}, {'eligible_for_final_evaluation': False}, {'source': {}}):
            with self.assertRaises(RuntimeError):
                w.validate_full_checkpoint({**saved, **changes}, done, 'a'*64)

    def test_missing_training_done_fails_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'not complete'):
                w.full_receipts(Path(tmp))


if __name__ == '__main__':
    unittest.main()
