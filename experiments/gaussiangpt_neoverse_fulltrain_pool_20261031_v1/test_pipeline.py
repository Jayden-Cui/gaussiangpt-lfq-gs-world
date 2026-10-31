"""CPU-only queue safety/coverage checks; never launch GPU or network work."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline as q

HERE = Path(__file__).resolve().parent


def config():
    return dict(schema='gs_fulltrain_pool_v1', queue=copy.deepcopy(q.QUEUE), resources=copy.deepcopy(q.RESOURCES),
                dependency=copy.deepcopy(q.DEPENDENCY), process_helper=dict(path=q.HELPER_PATH, sha256=q.HELPER_SHA),
                automatic_retry=False, training_executed=False, goal_complete=False)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helper = SimpleNamespace(STOP_REQUESTED=threading.Event(), ACTIVE_CHILDREN={},
                                      run_child=Mock(), terminate_queue=Mock())
        self.helper.run_child.return_value = dict(pid=901, exit_code=0, timed_out=False, elapsed_seconds=.01)
        def stop(sig, frame):
            self.helper.STOP_REQUESTED.set()
            raise SystemExit(128 + sig)
        self.helper.terminate_queue.side_effect = stop

    def write(self, relative, value):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        q.write_new(target, value)
        return target

    def test_pool_assignments_cover_all_48_once_and_preserve_declared_membership(self):
        pool = q.bindings.load_pool(HERE)
        lanes = q.assignments(pool)
        self.assertEqual([len(lanes[i]) for i in (0, 1)], [24, 24])
        self.assertEqual(set(lanes[0] + lanes[1]), set(q.bindings.EXPECTED_CLIPS) - q.bindings.REUSE)
        reversed_pool = copy.deepcopy(pool)
        reversed_pool['clips'].reverse()
        self.assertEqual(q.assignments(reversed_pool), lanes)
        self.assertNotIn('buildingblock_tower_14_0', lanes[0] + lanes[1])

    def test_invalid_pool_sampling_or_role_is_not_scheduled(self):
        for key, value in [('current_role', 'DEV'), ('requires_new_full177_export', False)]:
            pool = q.bindings.load_pool(HERE)
            pool['clips'][0][key] = value
            with self.assertRaises(RuntimeError):
                q.assignments(pool)

    def test_fixed_budgets_runtime_and_dependency_cannot_be_swapped(self):
        q.protocol_contract(config())
        for section, key, value in [('queue', 'teacher_seconds', 2400), ('resources', 'minimum_free_gib', 24),
                                    ('dependency', 'pid', 9), ('dependency', 'done_sha256', 'f'*64),
                                    ('process_helper', 'path', '/tmp/other.py')]:
            cfg = config()
            cfg[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(RuntimeError):
                q.protocol_contract(cfg)

    def test_no_retry_training_or_success_relabel(self):
        for key in ('automatic_retry', 'training_executed', 'goal_complete'):
            cfg = config()
            cfg[key] = True
            with self.assertRaises(RuntimeError):
                q.protocol_contract(cfg)

    def test_local_verify_uses_pinned_sibling_and_ignores_generated_md(self):
        self.write('protocol.json', config())
        self.write('CODE_MANIFEST.json', {})
        self.write('INPUT_POOL.json', {})
        (self.root / 'SUMMARY.md').write_text('generated output', encoding='utf-8')
        actual_sha = q.sha
        def digest(path):
            return q.HELPER_SHA if Path(path) == q.helper_file(self.root) else actual_sha(path)
        with patch.object(q.bindings, 'verify_manifest', return_value=dict.fromkeys(q.REQUIRED, 'a'*64)), \
             patch.object(q, 'sha', side_effect=digest):
            cfg, proof = q.verify(self.root, require_cpu=False)
            self.assertNotIn('cpu_validation_sha256', proof)
            self.assertEqual(cfg, config())
            self.assertEqual(proof['process_helper_sha256'], q.HELPER_SHA)
            self.write('CPU_VALIDATION.json', dict(status='passed', gpu_executed=False,
                code_manifest_sha256=proof['code_manifest_sha256'], protocol_sha256=proof['protocol_sha256'],
                actual_tests_executed=0))
            with self.assertRaisesRegex(RuntimeError, 'witness'):
                q.verify(self.root)

    def test_manifest_requires_documented_execution_and_every_queue_source(self):
        for omitted in ('pipeline.py', 'admission.py', 'validate.py', 'EXECUTION.md'):
            manifest = dict.fromkeys(q.REQUIRED - {omitted}, 'a'*64)
            with patch.object(q.bindings, 'verify_manifest', return_value=manifest):
                with self.assertRaisesRegex(RuntimeError, 'Manifest'):
                    q.verify(self.root, require_cpu=False)

    def test_exact_process_identity_and_script_position(self):
        dep = config()['dependency']
        snap = dict(pid=dep['pid'], start_ticks=dep['start_ticks'],
                    argv=[q.RUNTIME, '-B', '-u', dep['root']+'/pipeline.py', '--root', dep['root']])
        self.assertEqual(q.classify_parent(snap, dep), 'alive')
        self.assertEqual(q.classify_parent(None, dep), 'exited')
        for field, value in [('pid', 42), ('start_ticks', '999'),
                             ('argv', [q.RUNTIME, '-c', 'print(1)', dep['root']+'/pipeline.py', '--root', dep['root']])]:
            bad = dict(snap, **{field:value})
            with self.assertRaises(RuntimeError):
                q.classify_parent(bad, dep)
        worker = dict(argv=[q.RUNTIME, '-B', dep['root']+'/worker.py', '--root', dep['root'], '--mode', 'eval'])
        self.assertTrue(q.dependency_command(worker, dep))
        worker['argv'][2] = '/shared/worker.py'
        self.assertFalse(q.dependency_command(worker, dep))

    def test_dependency_terminal_requires_exact_done_no_failed_marker(self):
        dep = config()['dependency']
        dep['root'] = str(self.root)
        self.write('CODE_MANIFEST.json', {'a':'b'})
        self.write('SUMMARY.json', {'status':'complete'})
        dep['manifest_sha256'] = q.sha(self.root/'CODE_MANIFEST.json')
        dep['summary_sha256'] = q.sha(self.root/'SUMMARY.json')
        self.write('queue/DONE.json', dict(status='complete', admission=dict(code_manifest_sha256=dep['manifest_sha256']),
                                         summary_sha256=dep['summary_sha256']))
        dep['done_sha256'] = q.sha(self.root/'queue/DONE.json')
        self.assertTrue(q.read_terminal(dep)['resource_dependency_only'])
        (self.root/'SUMMARY.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'bytes changed'):
            q.read_terminal(dep)
        self.write('queue/FAILED.json', {'status':'failed'})
        with self.assertRaisesRegex(RuntimeError, 'FAILED'):
            q.read_terminal(dep)

    def test_gpu_admission_is_single_query_and_rejects_small_or_missing_gpu(self):
        for text in ('0, 81920\n1, 81920\n', '0, 81920\n1, 24576\n', '0, 81920\n'):
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(stdout=text)) as run:
                    if text == '0, 81920\n1, 81920\n':
                        q.gpu_admission(Path(tmp))
                    else:
                        with self.assertRaises(RuntimeError):
                            q.gpu_admission(Path(tmp))
                    run.assert_called_once()
                    self.assertEqual(run.call_args.kwargs['timeout'], 15)

    def test_disk_reserve_refuses_without_reducing_requested_data(self):
        with patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=63*2**30)):
            with self.assertRaisesRegex(RuntimeError, 'no retry or reduction'):
                q.disk_admission(self.root, 64)

    def test_job_environment_cpu_and_gpu_are_isolated_and_deadline_bounded(self):
        outcomes = []
        with patch.object(q.time, 'monotonic', return_value=100):
            row = q.run_job('teacher', ['python','export.py'], 1, 1200, self.root, self.helper, outcomes, 130)
        args = self.helper.run_child.call_args.args
        self.assertEqual(args[1]['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(args[-1], 30)
        self.assertEqual(row['maximum_process_seconds'], 1200)
        self.assertEqual(q.environment(None)['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(q.environment(0)['OMP_NUM_THREADS'], '6')
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(q.read(self.root/'teacher_PROCESS.json')['status'], 'complete')

    def test_failed_or_timedout_job_stops_siblings_and_never_retries(self):
        for response in (dict(exit_code=1, timed_out=False), dict(exit_code=0, timed_out=True)):
            with tempfile.TemporaryDirectory() as tmp:
                self.helper.STOP_REQUESTED.clear()
                self.helper.run_child.reset_mock()
                self.helper.run_child.return_value=response
                outcomes=[]
                with self.assertRaises(RuntimeError):
                    q.run_job('failed', ['python'], 0, 12, Path(tmp), self.helper, outcomes)
                self.helper.run_child.assert_called_once()
                self.assertTrue(self.helper.STOP_REQUESTED.is_set())
                self.assertEqual(outcomes[0]['status'], 'failed')
                with self.assertRaises(RuntimeError):
                    q.run_job('next', ['python'], 1, 12, Path(tmp), self.helper, outcomes)
                self.helper.run_child.assert_called_once()

    def test_export_deadline_prevents_any_new_child(self):
        with patch.object(q.time, 'monotonic', return_value=101):
            with self.assertRaises(TimeoutError):
                q.run_job('expired', ['python'], 0, 1200, self.root, self.helper, [], 100)
        self.helper.run_child.assert_not_called()

    def test_bound_helper_stops_only_its_owned_groups(self):
        helper = q.load_helper(HERE)
        helper.ACTIVE_CHILDREN = {1001:SimpleNamespace(pid=1001), 1002:SimpleNamespace(pid=1002)}
        with patch.object(helper, 'signal_group') as send, patch.object(helper.signal, 'SIGKILL', 9, create=True):
            q.stop_owned(helper)
        self.assertEqual({call.args[0] for call in send.call_args_list}, {1001,1002})
        self.assertNotIn(q.DEPENDENCY['pid'], {call.args[0] for call in send.call_args_list})
        self.assertTrue(helper.STOP_REQUESTED.is_set())

    def test_parallel_failure_cancels_other_lane_before_next_clip(self):
        started = threading.Event()
        stopped = threading.Event()
        def lane(root, output, gpu, clips, helper, outcomes, deadline, before):
            if gpu == 1:
                started.set()
                if not helper.STOP_REQUESTED.wait(5):
                    raise AssertionError('sibling was not stopped')
                stopped.set()
                return
            if not started.wait(5):
                raise AssertionError('sibling did not start')
            raise RuntimeError('first clip failed')
        with patch.object(q, 'run_lane', side_effect=lane):
            with self.assertRaisesRegex(RuntimeError, 'first clip'):
                q.execute_exports(self.root, self.root, {0:['a'],1:['b']}, self.helper, [], lambda:None)
        self.assertTrue(stopped.is_set())

    def test_existing_clip_attempt_refused_without_launch(self):
        clip = 'buildingblock_tower_0_0'
        (self.root/'clips'/clip).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'Existing clip'):
            q.run_lane(self.root, self.root, 0, [clip], self.helper, [], 1e20, lambda:None)
        self.helper.run_child.assert_not_called()

    def test_full_teacher_recipe_does_not_sample_or_reduce_resolution(self):
        for clip in ('buildingblock_tower_0_0', 'buildingblock_tower_21_528'):
            argv = q.prepare_clip.teacher_command(HERE, clip)
            for key, value in [('--num-frames','177'),('--frame-stride','1'),('--height','252'),('--width','448'),
                               ('--save-gs','177'),('--start-frame','0'),('--device','cuda:0')]:
                self.assertEqual(argv[argv.index(key)+1], value)
            self.assertEqual(argv[0], q.prepare_clip.RUNTIME)
        with self.assertRaises(RuntimeError):
            q.prepare_clip.teacher_command(HERE, 'buildingblock_tower_10_176')

    def test_complete_receipt_rejects_duplicate_clip_wrong_hash_and_global_unique_claim(self):
        proof = dict(code_manifest_sha256='a'*64, input_pool_sha256='b'*64)
        value = dict(status='complete', goal_complete=False, training_executed=False, **proof,
                     clips=[dict(clip_id=c) for c in sorted(q.bindings.EXPECTED_CLIPS)],
                     actual_clip_internal_frames_total=8850, all_clips_complete=True,
                     cross_clip_physical_frame_unique_count=None)
        self.write('POOL_COMPLETE.json', value)
        q.receipt_contract(self.root, 'POOL_COMPLETE.json', 50, proof)
        for mode in ('duplicate','hash','unique','frames'):
            bad = copy.deepcopy(value)
            if mode == 'duplicate':bad['clips'][-1] = bad['clips'][0]
            if mode == 'hash':bad['code_manifest_sha256'] = 'c'*64
            if mode == 'unique':bad['cross_clip_physical_frame_unique_count'] = 8850
            if mode == 'frames':bad['actual_clip_internal_frames_total'] = 1
            (self.root/'POOL_COMPLETE.json').write_text(json.dumps(bad), encoding='utf-8')
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                q.receipt_contract(self.root, 'POOL_COMPLETE.json', 50, proof)

    def test_receipt_creation_never_overwrites_a_prior_attempt(self):
        self.write('RECEIPT.json', {'first':1})
        with self.assertRaises(FileExistsError):
            q.write_new(self.root/'RECEIPT.json', {'second':2})
        self.assertEqual(q.read(self.root/'RECEIPT.json'), {'first':1})


if __name__ == '__main__':
    unittest.main()
