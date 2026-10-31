"""CPU queue contracts; no SSH, CUDA imports, nvidia-smi, or real child processes."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import signal
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline as q

ROOT = Path(__file__).resolve().parent


class SimulatedStop:
    def __init__(self):
        self.now, self.waits, self.stopped = 0.0, [], False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        return self.stopped


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = q.read(ROOT / 'protocol.json')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'new'
        self.root.mkdir()
        self.output = self.root / 'queue'
        self.output.mkdir()
        self.stop = SimulatedStop()

    def helper(self, child=None, event=None):
        event = event or self.stop
        def terminate(signum, frame):
            event.set()
            raise SystemExit(128 + signum)
        return SimpleNamespace(STOP_REQUESTED=event, run_child=child or Mock(),
                               terminate_queue=Mock(side_effect=terminate))

    def snapshot(self, pid=None, ticks=None, script='pipeline.py', root=None):
        dep = self.cfg['dependency']
        source = root or dep['root']
        return dict(pid=dep['pid'] if pid is None else pid, stable=True,
                    start_ticks=dep['start_ticks'] if ticks is None else ticks,
                    argv=['/env/bin/python', '-B', '-u', source + '/' + script, '--root', source])

    def test_exact_dependency_identity_rejects_pid_reuse_and_argv_drift(self):
        dep = self.cfg['dependency']
        self.assertEqual(q.classify_parent(None, dep), 'exited')
        self.assertEqual(q.classify_parent(self.snapshot(), dep), 'alive')
        bad = [self.snapshot(pid=1), self.snapshot(ticks='new'), self.snapshot(root='/another'),
               self.snapshot(script='prepare.py'), dict(self.snapshot(), stable=False),
               dict(self.snapshot(), argv=['python', '-c', 'pipeline.py'])]
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                q.classify_parent(candidate, dep)

    def test_dependency_worker_match_requires_actual_output_or_root(self):
        dep = self.cfg['dependency']
        self.assertTrue(q.dependency_worker(self.snapshot(script='prepare.py'), dep))
        self.assertFalse(q.dependency_worker(self.snapshot(script='prepare.py', root='/elsewhere'), dep))
        teacher = dict(self.snapshot(), argv=[q.RUNTIME, '-B', '-u',
            str(q.BASE / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py'),
            '--output', dep['root'] + '/teacher/buildingblock_tower_10_176'])
        self.assertTrue(q.dependency_worker(teacher, dep))
        teacher['argv'][-1] += '_other'
        self.assertFalse(q.dependency_worker(teacher, dep))

    def test_live_unreadable_or_racing_pid_is_not_exit(self):
        helper = SimpleNamespace(process_snapshot=Mock(return_value=None))
        with patch.object(Path, 'read_text', return_value='1 (python) S'):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                q.stable_snapshot(helper, 1)
        helper.process_snapshot = Mock(side_effect=[self.snapshot(), self.snapshot(ticks='changed')])
        with self.assertRaisesRegex(RuntimeError, 'raced'):
            q.stable_snapshot(helper, 1)

    def test_exit_still_waits_for_dependency_workers(self):
        helper = self.helper()
        observations = [dict(parent_state='alive', workers=[], all_exited=False),
                        dict(parent_state='exited', workers=[self.snapshot(script='prepare.py')], all_exited=False),
                        dict(parent_state='exited', workers=[], all_exited=True)]
        terminal = {'parent_terminal': 'DONE', 'hashes': {}}
        with patch.object(q, 'observe', side_effect=observations), \
             patch.object(q, 'read_terminal', return_value=terminal) as read_terminal, \
             patch.object(q.time, 'monotonic', side_effect=lambda: self.stop.now):
            self.assertEqual(q.wait_for_parent(self.output, self.cfg, helper), terminal)
        self.assertEqual(self.stop.waits, [60, 60])
        read_terminal.assert_called_once()
        helper.terminate_queue.assert_not_called()

    def test_missing_or_failed_terminal_rejected_after_exit_without_retry(self):
        dependency = dict(self.cfg['dependency'], root=str(self.root))
        with self.assertRaisesRegex(ValueError, 'no DONE'):
            q.read_terminal(dependency)
        q.write(self.output / 'FAILED.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'failure receipt'):
            q.read_terminal(dependency)
        helper = self.helper()
        with patch.object(q, 'observe', return_value={'all_exited': True}), \
             patch.object(q, 'read_terminal', side_effect=ValueError('Missing terminal')):
            with self.assertRaisesRegex(ValueError, 'Missing terminal'):
                q.wait_for_parent(self.output, self.cfg, helper)
        self.assertEqual(self.stop.waits, [])
        helper.terminate_queue.assert_not_called()

    def test_wait_timeout_never_stops_or_restarts_dependency(self):
        helper = self.helper()
        with patch.object(q, 'observe', return_value={'all_exited': False}), \
             patch.object(q.time, 'monotonic', side_effect=lambda: self.stop.now), patch.object(q, 'write'):
            with self.assertRaisesRegex(TimeoutError, '15000'):
                q.wait_for_parent(self.output, self.cfg, helper)
        self.assertEqual(sum(self.stop.waits), 15000)
        self.assertEqual(set(self.stop.waits), {60})
        helper.terminate_queue.assert_not_called()
        helper.run_child.assert_not_called()

    def test_fixed_commands_and_each_arm_gpu_assignment(self):
        expected = {'source_admission': ([None], [300]), 'preflight': ([0], [930]),
                    'baseline': ([0], [1830]), 'train': ([0, 1], [1830, 1830]),
                    'eval': ([0, 1], [1830, 1830]), 'compare': ([None], [300])}
        for stage, (gpus, budgets) in expected.items():
            tasks = q.jobs(self.root, stage)
            self.assertEqual([task[1] for task in tasks], gpus)
            self.assertEqual([task[3] for task in tasks], budgets)
            for task in tasks:
                command = task[2]
                self.assertEqual(command[:3], [q.RUNTIME, '-B', '-u'])
                self.assertEqual(command[command.index('--root') + 1], str(self.root))
                self.assertFalse(any(flag in command for flag in ('--resume', '--retry', '--overwrite')))
        self.assertNotIn('--arm', q.jobs(self.root, 'preflight')[0][2])
        self.assertNotIn('--arm', q.jobs(self.root, 'baseline')[0][2])
        for stage in ('train', 'eval'):
            self.assertEqual([job[2][-1] for job in q.jobs(self.root, stage)], list(q.ARMS))

    def test_failed_or_timed_out_child_stops_without_retry(self):
        for result in ({'exit_code': 1, 'timed_out': False}, {'exit_code': 0, 'timed_out': True}):
            with self.subTest(result=result):
                self.stop = SimulatedStop()
                helper = self.helper(Mock(return_value=result))
                outcomes = []
                with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                    q.execute_stage(self.root, self.output, 'preflight', helper, outcomes, Mock())
                helper.run_child.assert_called_once()
                helper.terminate_queue.assert_called_once()
                self.assertEqual(outcomes[0]['exit_code'], result['exit_code'])
                self.assertEqual(outcomes[0]['timed_out'], result['timed_out'])
                self.assertEqual(outcomes[0]['status'], 'failed')

    def test_parallel_first_failure_immediately_stops_sibling_and_keeps_outcomes(self):
        event, barrier = threading.Event(), threading.Barrier(2)
        def child(command, env, log, timeout):
            barrier.wait(timeout=3)
            if command[-1] == 'decoder_only':
                return dict(exit_code=1, timed_out=False)
            if not event.wait(timeout=3):
                raise AssertionError('Sibling was not stopped immediately')
            return dict(exit_code=-15, timed_out=False)
        helper = self.helper(Mock(side_effect=child), event)
        outcomes = []
        with self.assertRaises(RuntimeError):
            q.execute_stage(self.root, self.output, 'train', helper, outcomes, Mock())
        self.assertTrue(event.is_set())
        self.assertEqual(helper.run_child.call_count, 2)
        self.assertEqual({row['name'] for row in outcomes}, {'train_decoder_only', 'train_full_codec'})
        self.assertTrue(all(row['status'] == 'failed' for row in outcomes))
        self.assertEqual({call.args[1]['CUDA_VISIBLE_DEVICES'] for call in helper.run_child.call_args_list}, {'0', '1'})

    def test_child_exception_requests_stop_and_persists_error(self):
        helper = self.helper(Mock(side_effect=RuntimeError('worker launch error')))
        outcomes = []
        with self.assertRaisesRegex(RuntimeError, 'worker launch error'):
            q.run_job(q.jobs(self.root, 'compare')[0], self.output, helper, outcomes)
        helper.terminate_queue.assert_called_once()
        self.assertIn('worker launch error', outcomes[0]['error'])
        self.assertEqual(helper.run_child.call_args.args[1]['CUDA_VISIBLE_DEVICES'], '')

    def test_preflight_requires_both_discarded_updates_and_same_initial_source(self):
        evidence = {'binding_sha256': 'a' * 64}
        arm = dict(passed=True, discarded_updates=2, production_updates=0, initial_model_sha256='b' * 64)
        gate = dict(status='passed', source=evidence, arms={name: dict(arm) for name in q.ARMS},
                    initial_arms_identical=True, goal_complete=False)
        q.write(self.root / 'PREFLIGHT.json', gate)
        q.preflight_contract(self.root, evidence)
        for key, value in [('discarded_updates', 1), ('production_updates', 1),
                            ('initial_model_sha256', 'c' * 64), ('passed', False)]:
            altered = deepcopy(gate)
            altered['arms']['full_codec'][key] = value
            q.write(self.root / 'PREFLIGHT.json', altered)
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.preflight_contract(self.root, evidence)

    def make_manifest_fixture(self):
        for relative in q.REQUIRED:
            path = self.root / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            source = ROOT / relative
            path.write_bytes(source.read_bytes() if source.is_file() else b'# fixture\n')
        q.write(self.root / 'CODE_MANIFEST.json', {name: q.sha(self.root / name) for name in q.REQUIRED})

    def test_local_verify_does_not_access_remote_data_and_proof_is_exact(self):
        self.make_manifest_fixture()
        real_sha = q.sha
        def local_sha(path):
            self.assertTrue(Path(path).resolve().is_relative_to(Path(self.temp.name)))
            return real_sha(path)
        with patch.object(q, 'sha', side_effect=local_sha):
            _, admission = q.verify(self.root, cpu=False)
        q.write(self.root / 'CPU_VALIDATION.json', dict(status='passed', gpu_executed=False, **admission))
        q.verify(self.root)
        proof = q.read(self.root / 'CPU_VALIDATION.json')
        proof['protocol_sha256'] = '0' * 64
        q.write(self.root / 'CPU_VALIDATION.json', proof)
        with self.assertRaisesRegex(ValueError, 'CPU_VALIDATION'):
            q.verify(self.root)
        self.assertFalse((self.root / 'DATA_BINDING.json').exists())

    def test_manifest_mutation_is_rejected_before_remote_access(self):
        self.make_manifest_fixture()
        (self.root / 'worker.py').write_bytes(b'# changed')
        with self.assertRaisesRegex(ValueError, 'Immutable source mismatch'):
            q.verify(self.root, cpu=False)

    def test_two_gpu_admission_is_one_mocked_query_with55gib_each(self):
        for stdout, passes in [('0, 56320\n1, 56320\n', True), ('0, 56320\n1, 56319\n', False),
                                ('0, 56320\n', False), ('0, 56320\n0, 56320\n', False)]:
            with self.subTest(stdout=stdout), patch.object(q.subprocess, 'run', return_value=SimpleNamespace(stdout=stdout)) as run:
                if passes:
                    q.gpu_admission(self.output)
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        q.gpu_admission(self.output)
                run.assert_called_once()

    def test_fixed_queue_limits_and_no_retry_contract(self):
        q.protocol_contract(self.cfg)
        for key, value in [('outer_seconds', 50000), ('wait_seconds', 15001), ('train_process_seconds', 2000)]:
            cfg = deepcopy(self.cfg)
            cfg['queue'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.protocol_contract(cfg)

    def test_main_failure_has_consistent_failed_status_and_no_done(self):
        self.output.rmdir()
        helper = self.helper()
        with patch.object(q, 'BASE', self.root.parent), patch.object(q, 'RUN', self.root.name), \
             patch.object(q.signal, 'SIGALRM', 14, create=True), patch.object(q.signal, 'alarm', create=True), \
             patch.object(q.signal, 'signal'), patch.object(q, 'verify', return_value=(self.cfg, {})), \
             patch.object(q, 'load_helper', return_value=helper), \
             patch.object(q, 'wait_for_parent', side_effect=TimeoutError('dependency wait expired')):
            with self.assertRaisesRegex(TimeoutError, 'dependency wait expired'):
                q.main(self.root)
        self.assertEqual(q.read(self.output / 'STATUS.json')['status'], 'failed')
        failed = q.read(self.output / 'FAILED.json')
        self.assertEqual(failed['status'], 'failed')
        self.assertFalse(failed['old_processes_signalled'])
        self.assertFalse(failed['automatic_retry'])
        self.assertFalse((self.output / 'DONE.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
