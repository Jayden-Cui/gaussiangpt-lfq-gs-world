"""Pure-CPU in-memory queue contract tests and optional fresh local document witness."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
import queue_prepare as queue


class QueueContracts(unittest.TestCase):
    def snapshot(self, pid=8123, ticks='9911', script=None, root=None):
        return dict(pid=pid, start_ticks=ticks, stable=True,
                    argv=['/env/bin/python', '-B', '-u', str(script or queue.V7 / 'postfit_code/queue_postfit.py'),
                          '--root', str(root or queue.V7)])

    def test_parent_identity_and_pid_reuse(self):
        script = queue.V7 / 'postfit_code/queue_postfit.py'
        classify = lambda value: queue.classify_parent(value, 8123, '9911', script)
        self.assertEqual(classify(None), 'exited')
        self.assertEqual(classify(self.snapshot()), 'alive')
        self.assertEqual(classify(self.snapshot(ticks='9912')), 'replacement')
        self.assertEqual(classify(self.snapshot(ticks='9912', script=Path('/elsewhere/other.py'))), 'pid_reused_unrelated')
        self.assertEqual(classify(self.snapshot(root=Path('/other/root'))), 'identity_mismatch')
        self.assertEqual(classify(dict(self.snapshot(), stable=False)), 'identity_mismatch')
        self.assertEqual(classify(dict(self.snapshot(), argv=['python', '-c', 'mentions queue_postfit.py'])), 'identity_mismatch')
        main = self.snapshot(queue.MAIN_PID, queue.MAIN_TICKS, queue.V7 / 'pipeline.py')
        self.assertEqual(queue.classify_parent(main, queue.MAIN_PID, queue.MAIN_TICKS, queue.V7 / 'pipeline.py'), 'alive')

    def test_exact_workers(self):
        for script in ('runner.py', 'compare.py', 'postfit_code/render_six_panel.py', 'postfit_code/nearby_view_probe.py'):
            self.assertTrue(queue.worker_match(self.snapshot(script=queue.V7 / script)))
            self.assertFalse(queue.worker_match(self.snapshot(script=queue.V7 / script, root=Path('/another/run'))))
        self.assertFalse(queue.worker_match(self.snapshot(script=Path('/other/runner.py'))))
        self.assertFalse(queue.worker_match(dict(self.snapshot(script=queue.V7 / 'runner.py'),
            argv=['python', str(queue.V7 / 'runner.py'), '--root', str(queue.V7), '--root', str(queue.V7)])))

    def test_terminal_choices(self):
        done, failed = {'status': 'complete'}, {'status': 'failed'}
        self.assertEqual(queue.terminal_decision(done, None, done, None)['postfit_terminal'], 'DONE')
        self.assertEqual(queue.terminal_decision(done, None, None, failed)['postfit_terminal'], 'FAILED')
        for values in ((None, None, done, None), (done, failed, done, None),
                       (done, None, None, None), (done, None, done, failed), (done, None, failed, None)):
            with self.assertRaises(ValueError):
                queue.terminal_decision(*values)

    def test_only_matched_processes_require_strict_reread(self):
        self.assertFalse(queue.relevant_process(None))
        self.assertFalse(queue.relevant_process(dict(self.snapshot(script=Path('/other/worker.py')), stable=False)))
        self.assertFalse(queue.relevant_process(dict(self.snapshot(), argv=[])))
        self.assertFalse(queue.relevant_process(self.snapshot(script=queue.V7 / 'runner.py', root=Path('/other/run'))))
        self.assertTrue(queue.relevant_process(dict(self.snapshot(script=queue.V7 / 'runner.py'), stable=False)))
        self.assertTrue(queue.relevant_process(self.snapshot(script=queue.V7 / 'pipeline.py')))
        self.assertTrue(queue.relevant_process(self.snapshot()))

    def test_fixed_commands_and_no_training(self):
        root = queue.BASE / queue.RUN
        jobs = queue.commands(root)
        self.assertEqual([name for name, _ in jobs], ['teacher_export', 'frozen_codec_preparation'])
        first, second = jobs[0][1], jobs[1][1]
        self.assertEqual(first[:4], [str(queue.RUNTIME), '-B', '-u', str(queue.RENDERER)])
        expected = {'--input': str(queue.VIDEO), '--output': str(root / 'teacher/native448'), '--num-frames': '177',
                    '--start-frame': '0', '--frame-stride': '1', '--fps': '29.97', '--height': '252', '--width': '448',
                    '--save-gs': '177', '--novel-view-shift': '0.03', '--device': 'cuda:0', '--dtype': 'bfloat16'}
        for flag, value in expected.items():
            self.assertEqual(first[first.index(flag) + 1], value)
        for forbidden in ('--use-motion', '--overwrite', '--skip-checkpoint-hash', 'run.py', '--train', '--resume'):
            self.assertNotIn(forbidden, first + second)
        self.assertEqual(second[:4], [str(queue.GS_PYTHON), '-B', '-u', str(root / 'prepare_frozen_codec.py')])
        self.assertEqual(second[-2:], ['--max-seconds', '1200'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueContracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    for name in ('queue_prepare.py', 'test_queue_prepare.py', 'prepare_frozen_codec.py'):
        ast.parse((root / name).read_text(encoding='utf-8'), filename=name)
    queue.protocol_contract(queue.read(root / 'PREPARATION_PROTOCOL.json'))
    help_result = subprocess.run([sys.executable, '-B', str(root / 'prepare_frozen_codec.py'), '--help'],
                                 capture_output=True, text=True, timeout=20, check=True)
    if '--codec-root' not in help_result.stdout or '--max-seconds' not in help_result.stdout:
        raise ValueError('Frozen preparation help contract missing')
    if args.write_receipt:
        admission = queue.local_manifest(root)  # Never opens remote paths listed in input_binding.files.
        target = root / 'CPU_VALIDATION.json'
        if target.exists():
            raise FileExistsError('Refusing to replace an existing fresh witness')
        receipt = dict(status='passed', gpu_executed=False, cpu_only=True, goal_complete=False,
                       tests_run=result.testsRun, checks=['in_memory_contracts', 'AST', 'prepare_help', 'local_manifest'], **admission)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2)
        print('CPU witness written: ' + str(target))
    else:
        print('Pure CPU contracts, AST and preparation --help passed; no receipt written')


if __name__ == '__main__':
    main()
