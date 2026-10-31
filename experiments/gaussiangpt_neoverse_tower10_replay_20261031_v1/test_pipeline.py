"""Pure CPU mocks: no SSH, GPU query/import, training or old-run mutation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
import pipeline as queue

ROOT = Path(__file__).resolve().parent


class SimulatedStop:
    def __init__(self):
        self.stopped = False
        self.now = 0.0
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        return self.stopped


class QueueContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = queue.read(ROOT / 'protocol.json')
        self.binding = queue.read(ROOT / 'input_binding.json')
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'fresh'
        self.root.mkdir()
        self.output = self.root / 'queue'
        self.output.mkdir()
        self.stop = SimulatedStop()
        self.helper = SimpleNamespace(STOP_REQUESTED=self.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def snapshot(self, pid=None, ticks=None, script='pipeline.py', root=None):
        parent = self.cfg['parent']
        source = root or parent['root']
        return dict(pid=parent['pid'] if pid is None else pid, stable=True,
                    start_ticks=parent['start_ticks'] if ticks is None else ticks,
                    argv=['/env/bin/python', '-B', '-u', source + '/' + script, '--root', source])

    def test_exact_parent_identity_and_reused_pid_fail_closed(self):
        parent = self.cfg['parent']
        self.assertEqual(queue.classify_parent(None, parent), 'exited')
        self.assertEqual(queue.classify_parent(self.snapshot(), parent), 'alive')
        bad = [self.snapshot(pid=parent['pid'] + 1), self.snapshot(ticks='999'),
               self.snapshot(root='/other'), self.snapshot(script='worker.py'),
               dict(self.snapshot(), stable=False), dict(self.snapshot(), argv=['python', '-c', 'pipeline.py'])]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                queue.classify_parent(value, parent)

    def test_same_root_workers_only(self):
        for name in ('worker.py', 'compare.py'):
            self.assertTrue(queue.worker_match(self.snapshot(script=name), self.cfg['parent']))
            self.assertFalse(queue.worker_match(self.snapshot(script=name, root='/another'), self.cfg['parent']))
        self.assertFalse(queue.worker_match(self.snapshot(), self.cfg['parent']))
        bad = self.snapshot(script='worker.py')
        bad['argv'] += ['--root', self.cfg['parent']['root']]
        self.assertFalse(queue.worker_match(bad, self.cfg['parent']))

    def test_terminals_missing_conflicting_invalid_fail_and_v8_failure_is_allowed(self):
        done, failed = {'status': 'complete'}, {'status': 'failed'}
        self.assertEqual(queue.terminal_decision(done, None)['parent_terminal'], 'DONE')
        self.assertEqual(queue.terminal_decision(None, failed)['parent_terminal'], 'FAILED')
        for values in ((None, None), (done, failed), (failed, None), (None, done), ([], None)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                queue.terminal_decision(*values)

    def test_wait_includes_workers_after_parent_exit(self):
        observations = [dict(parent_state='alive', workers=[], all_exited=False),
                        dict(parent_state='exited', workers=[self.snapshot(script='worker.py')], all_exited=False),
                        dict(parent_state='exited', workers=[], all_exited=True)]
        terminal = dict(decision=queue.terminal_decision(None, {'status': 'failed'}), hashes={})
        with patch.object(queue, 'observe', side_effect=observations), patch.object(queue, 'read_terminal', return_value=terminal) as reader, \
             patch.object(queue.time, 'monotonic', side_effect=lambda: self.stop.now):
            self.assertEqual(queue.wait_for_parent(self.output, self.cfg, self.helper), terminal)
        self.assertEqual(self.stop.waits, [60, 60])
        reader.assert_called_once()

    def test_missing_terminal_after_exit_is_not_success_or_another_wait(self):
        with patch.object(queue, 'observe', return_value=dict(parent_state='exited', workers=[], all_exited=True)), \
             patch.object(queue, 'read_terminal', side_effect=ValueError('missing receipt')):
            with self.assertRaisesRegex(ValueError, 'missing receipt'):
                queue.wait_for_parent(self.output, self.cfg, self.helper)
        self.assertEqual(self.stop.waits, [])

    def test_wait_timeout_is10800_and_uses60_second_interval(self):
        with patch.object(queue, 'observe', return_value=dict(parent_state='alive', workers=[], all_exited=False)), \
             patch.object(queue.time, 'monotonic', side_effect=lambda: self.stop.now), \
             patch.object(queue, 'write'):
            with self.assertRaises(TimeoutError):
                queue.wait_for_parent(self.output, self.cfg, self.helper)
        self.assertEqual(sum(self.stop.waits), 10800)
        self.assertEqual(set(self.stop.waits), {60})

    def test_unreadable_live_pid_and_identity_race_not_assumed_exited(self):
        fake = SimpleNamespace(process_snapshot=Mock(return_value=None))
        with patch.object(Path, 'read_text', return_value='1 (python) S'):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                queue.stable_snapshot(fake, 1)
        fake.process_snapshot = Mock(side_effect=[self.snapshot(), self.snapshot(ticks='changed')])
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            queue.stable_snapshot(fake, 1)
        fake.process_snapshot = Mock(return_value=None)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertIsNone(queue.stable_snapshot(fake, 1))

    def test_observe_rejects_replacement_parent_and_preserves_workers(self):
        parent = self.cfg['parent']
        replacement = self.snapshot(pid=9)
        self.helper.process_snapshot = Mock(return_value=replacement)
        with patch.object(queue, 'sha', return_value=parent['manifest_sha256']), \
             patch.object(Path, 'iterdir', return_value=[Path('/proc/9')]), \
             patch.object(queue, 'stable_snapshot', side_effect=[None, replacement]):
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                queue.observe(self.helper, parent)
        worker = self.snapshot(pid=9, script='worker.py')
        self.helper.process_snapshot = Mock(return_value=worker)
        with patch.object(queue, 'sha', return_value=parent['manifest_sha256']), \
             patch.object(Path, 'iterdir', return_value=[Path('/proc/9')]), \
             patch.object(queue, 'stable_snapshot', side_effect=[None, worker]):
            result = queue.observe(self.helper, parent)
        self.assertFalse(result['all_exited'])
        self.assertEqual(result['workers'], [worker])

    def test_protocol_and_binding_parameters_are_fixed(self):
        queue.protocol_contract(self.cfg)
        queue.binding_contract(self.binding, self.cfg)
        for key, value in {'stride': 11, 'frames': 17, 'gpu': 1, 'training_performed': True,
                           'automatic_retry': True, 'teacher_process_seconds': 2400}.items():
            cfg = dict(self.cfg, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.protocol_contract(cfg)
        for key, value in {'use_motion': True, 'frame_stride': 11, 'frames': 17,
                           'source_video_path': '/another.mp4', 'renderer_sha256': '0' * 64}.items():
            binding = copy.deepcopy(self.binding)
            binding['expected_teacher'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.binding_contract(binding, self.cfg)
        duplicate = copy.deepcopy(self.binding)
        duplicate['files'].append(duplicate['files'][0])
        with self.assertRaises(ValueError):
            queue.binding_contract(duplicate, self.cfg)

    def test_commands_match_original_export_cli_and_prepare_wrapper(self):
        jobs = queue.commands(self.root, self.binding)
        self.assertEqual([name for name, _ in jobs], ['teacher_export', 'frozen_codec_preparation'])
        first, second = jobs[0][1], jobs[1][1]
        self.assertEqual(first[:4], [str(queue.RUNTIME), '-B', '-u', str(queue.RENDERER)])
        expected = {'--repo': str(queue.REPO), '--checkpoint': str(queue.CHECKPOINT),
                    '--input': self.binding['expected_teacher']['source_video_path'],
                    '--output': str(self.root / 'teacher' / queue.CLIP), '--num-frames': '177',
                    '--start-frame': '0', '--frame-stride': '1', '--fps': '29.97', '--height': '252', '--width': '448',
                    '--save-gs': '177', '--novel-view-shift': '0.03', '--device': 'cuda:0', '--dtype': 'bfloat16'}
        for flag, value in expected.items():
            self.assertEqual(first[first.index(flag) + 1], value)
        self.assertEqual(second, [str(queue.GS_PYTHON), '-B', '-u', str(self.root / 'prepare.py'), '--root', str(self.root)])
        for forbidden in ('--use-motion', '--overwrite', '--train', '--resume', '--skip-checkpoint-hash'):
            self.assertNotIn(forbidden, first + second)

    def test_first_child_failure_or_timeout_stops_without_second_child_or_retry(self):
        for result in (dict(exit_code=1, timed_out=False), dict(exit_code=0, timed_out=True)):
            self.helper.run_child = Mock(return_value=result)
            self.helper.terminate_queue = Mock(side_effect=SystemExit(143))
            outcomes = []
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                queue.execute_jobs(self.root, self.output, self.cfg, self.binding, self.helper, outcomes, lambda: None)
            self.helper.run_child.assert_called_once()
            self.helper.terminate_queue.assert_called_once()
            self.assertEqual(len(outcomes), 1)

    def test_two_sequential_tasks_use_fixed_budget_and_gpu_environment(self):
        self.helper.run_child = Mock(return_value=dict(exit_code=0, timed_out=False))
        outcomes, before_launch = [], Mock()
        queue.execute_jobs(self.root, self.output, self.cfg, self.binding, self.helper, outcomes, before_launch)
        self.assertEqual([call.args[-1] for call in self.helper.run_child.call_args_list], [1200, 1230])
        self.assertEqual(before_launch.call_count, 2)
        for call in self.helper.run_child.call_args_list:
            env = call.args[1]
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(env['OMP_NUM_THREADS'], '6')
        self.assertEqual(len(outcomes), 2)

    def test_existing_teacher_prevents_any_new_child(self):
        (self.root / 'teacher').mkdir()
        self.helper.run_child = Mock()
        with self.assertRaises(FileExistsError):
            queue.execute_jobs(self.root, self.output, self.cfg, self.binding, self.helper, [], lambda: None)
        self.helper.run_child.assert_not_called()

    def test_gpu_admission_mock_needs60gib_and_never_polls(self):
        for memory, passes in (('61440\n', True), ('61439\n', False), ('61440\n61440\n', False)):
            with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(stdout=memory)) as run:
                if passes:
                    queue.gpu_admission(self.output)
                else:
                    with self.assertRaises((RuntimeError, ValueError)):
                        queue.gpu_admission(self.output)
                run.assert_called_once()
                self.assertIn('--id=0', run.call_args.args[0])

    def make_manifest_fixture(self):
        for relative in queue.REQUIRED:
            path = self.root / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            source = ROOT / relative
            path.write_bytes(source.read_bytes() if source.is_file() else b'# local fixture\n')
        manifest = {relative: queue.sha(self.root / relative) for relative in queue.REQUIRED}
        queue.write(self.root / 'CODE_MANIFEST.json', manifest)

    def test_cpu_verify_opens_no_remote_paths_and_validation_binds_exact_manifest(self):
        self.make_manifest_fixture()
        real_sha = queue.sha

        def local_sha(path):
            self.assertFalse(str(path).startswith('/data/'))
            return real_sha(path)

        with patch.object(queue, 'sha', side_effect=local_sha):
            cfg, admission = queue.verify(self.root, cpu=False)
        queue.write(self.root / 'CPU_VALIDATION.json', dict(schema='cpu_validation_v1', status='passed',
                    gpu_executed=False, **admission))
        queue.verify(self.root)
        receipt = queue.read(self.root / 'CPU_VALIDATION.json')
        receipt['code_manifest_sha256'] = '0' * 64
        queue.write(self.root / 'CPU_VALIDATION.json', receipt)
        with self.assertRaisesRegex(ValueError, 'CPU validation'):
            queue.verify(self.root)

    def test_mutated_manifest_file_is_rejected(self):
        self.make_manifest_fixture()
        (self.root / 'prepare.py').write_text('# mutated', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Immutable source'):
            queue.verify(self.root, cpu=False)

    def test_complete_summary_preserves_train_identity_and_never_scientific_success(self):
        summary = dict(status='complete', preparation_contract_passed=True, frames=177, adjacent_pairs=176,
            clip=queue.CLIP, source_points_per_frame=112896, all_input_points_aggregated=True, codec_frozen=True,
            training_performed=False, scientific_transfer_passed=None, goal_complete=False,
            input_binding_sha256='a' * 64, source_role='TRAIN', source_group='buildingblock_tower_10',
            official_independent_test_split=False, codec_state_sha256_before_and_after='b' * 64)
        queue.prepared_contract(summary, 'a' * 64)
        for key, value in {'frames': 17, 'training_performed': True, 'source_role': 'TEST',
                           'source_points_per_frame': 1000, 'goal_complete': True,
                           'scientific_transfer_passed': True, 'codec_state_sha256_before_and_after': ''}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                queue.prepared_contract(dict(summary, **{key: value}), 'a' * 64)

    def test_main_failure_writes_failed_status_and_no_done(self):
        # Move out of the fixture's pre-created output: main must create it exclusively.
        self.output.rmdir()
        helper = SimpleNamespace(terminate_queue=Mock(side_effect=SystemExit(143)))
        fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with patch.object(queue, 'BASE', self.root.parent), patch.object(queue, 'RUN', self.root.name), \
             patch.object(queue.signal, 'SIGALRM', 14, create=True), patch.object(queue.signal, 'alarm', create=True), \
             patch.object(queue.signal, 'signal'), patch.object(queue, 'verify', return_value=(self.cfg, {})), \
             patch.object(queue.importlib.util, 'spec_from_file_location', return_value=fake_spec), \
             patch.object(queue.importlib.util, 'module_from_spec', return_value=helper), \
             patch.object(queue, 'wait_for_parent', side_effect=TimeoutError('bounded wait expired')):
            with self.assertRaises(TimeoutError):
                queue.main(self.root)
        self.assertEqual(queue.read(self.output / 'STATUS.json')['status'], 'failed')
        self.assertEqual(queue.read(self.output / 'FAILED.json')['automatic_retry'], False)
        self.assertFalse((self.output / 'DONE.json').exists())
        helper.terminate_queue.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
