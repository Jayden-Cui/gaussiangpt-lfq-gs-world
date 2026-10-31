"""Bounded CPU-only queue regressions; real helper, simulated OS processes."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pipeline


class FakeProcess:
    def __init__(self,pid,wait):
        self.pid=pid
        self._wait=wait
        self.returncode=None
        self.killed=threading.Event()

    def wait(self,timeout):
        if self.returncode is None:self.returncode=self._wait(self)
        return self.returncode

    def poll(self):return self.returncode


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.helper=pipeline.load('_pipeline_test_owned_groups',Path(pipeline.__file__).parent/pipeline.HELPER)
        self.cfg={'queue':{'gpu_workers':{'0':['base_z','local_constant'],'1':['base_constant','local_z']},
                           'train_process_seconds':1900,'eval_process_seconds':2500,'comparison_seconds':180}}
        self.calls=[]
        self.processes={}
        self.signals=[]
        self.sibling_ready=threading.Event()

    def bounded_wait(self,event,reason):
        if not event.wait(3):raise AssertionError(reason)

    def run_queue(self,scenario,failing_arm='base_constant'):
        def fake_popen(command,**kwargs):
            self.assertTrue(kwargs['start_new_session'])
            arm=command[command.index('--arm')+1] if '--arm' in command else None
            mode=command[command.index('--mode')+1] if '--mode' in command else 'compare'
            self.calls.append((arm,mode))
            pid=10000+len(self.calls)
            if scenario=='launch_race' and arm!=failing_arm:
                self.sibling_ready.set()
                self.bounded_wait(self.helper.STOP_REQUESTED,'Failure did not stop the queue during Popen')

            def wait(process):
                if scenario=='success':return 0
                if arm==failing_arm:
                    self.bounded_wait(self.sibling_ready,'Sibling never entered its first process')
                    return 0 if scenario=='bad_train_result' else 7
                self.sibling_ready.set()
                self.bounded_wait(process.killed,'Failed arm did not terminate its own running sibling group')
                return -9

            process=FakeProcess(pid,wait)
            self.processes[pid]=process
            return process

        def signal_group(pid,sig):
            self.assertIn(pid,self.processes,'Signalled a process outside this queue')
            self.signals.append((pid,sig))
            if sig==9:
                self.processes[pid].returncode=-9
                self.processes[pid].killed.set()

        def check_train(root,arm):
            if scenario=='bad_train_result' and arm==failing_arm:
                raise RuntimeError('actual training stopped before600')

        with patch.object(pipeline,'verify',return_value=(self.cfg,{})), \
             patch.object(pipeline,'load',return_value=self.helper), \
             patch.object(pipeline,'check_train_result',side_effect=check_train), \
             patch.object(pipeline.signal,'signal'), \
             patch.object(pipeline.signal,'alarm',create=True), \
             patch.object(pipeline.signal,'SIGALRM',14,create=True), \
             patch.object(pipeline.signal,'SIGKILL',9,create=True), \
             patch.object(self.helper.subprocess,'Popen',side_effect=fake_popen), \
             patch.object(self.helper,'signal_group',side_effect=signal_group):
            if scenario=='success':
                pipeline.write(self.root/'SUMMARY.json',dict(schema='gs_local_condition_comparison_v1',
                    engineering_status='complete',comparison_eligible=True,selected_step=600,goal_complete=False))
                pipeline.main(self.root)
            else:
                with self.assertRaises(RuntimeError) as caught:pipeline.main(self.root)
                if scenario=='bad_train_result':self.assertIn('before600',str(caught.exception))

    def assert_failed_without_later_arms(self):
        self.assertEqual(sorted(self.calls),[('base_constant','train'),('base_z','train')])
        self.assertTrue(self.helper.STOP_REQUESTED.is_set())
        self.assertEqual(self.helper.ACTIVE_CHILDREN,{})
        self.assertTrue(self.signals)
        status=pipeline.read(self.root/'queue/STATUS.json')
        failure=pipeline.read(self.root/'queue/FAILED.json')
        self.assertEqual(status['status'],'failed')
        self.assertEqual(failure['status'],'failed')
        self.assertEqual(status['error'],failure['error'])
        self.assertFalse((self.root/'queue/DONE.json').exists())
        for arm in ('base_z','base_constant'):
            self.assertTrue((self.root/'queue'/f'{arm}_train_PROCESS.json').exists())

    def test_failure_in_second_future_stops_running_first_future(self):
        self.run_queue('child_failed')
        self.assert_failed_without_later_arms()

    def test_incomplete_training_exception_stops_other_gpu(self):
        self.run_queue('bad_train_result')
        self.assert_failed_without_later_arms()

    def test_stop_during_popen_kills_late_registered_child(self):
        self.run_queue('launch_race',failing_arm='base_z')
        self.assert_failed_without_later_arms()

    def test_admission_error_writes_terminal_status(self):
        with patch.object(pipeline,'verify',side_effect=ValueError('source changed')), \
             patch.object(pipeline.signal,'signal'), \
             patch.object(pipeline.signal,'alarm',create=True), \
             patch.object(pipeline.signal,'SIGALRM',14,create=True):
            with self.assertRaisesRegex(ValueError,'source changed'):pipeline.main(self.root)
        status=pipeline.read(self.root/'queue/STATUS.json')
        failure=pipeline.read(self.root/'queue/FAILED.json')
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['error'],failure['error'])

    def test_success_completes_same_four_arms_and_comparison(self):
        self.run_queue('success')
        self.assertEqual(len(self.calls),9)
        self.assertEqual(self.calls[-1],(None,'compare'))
        for mode in ('train','eval'):
            self.assertEqual(sorted(arm for arm,m in self.calls if m==mode),sorted(pipeline.ARMS))
        self.assertFalse(self.helper.STOP_REQUESTED.is_set())
        self.assertEqual(self.signals,[])
        self.assertEqual(self.helper.ACTIVE_CHILDREN,{})
        self.assertEqual(pipeline.read(self.root/'queue/STATUS.json')['status'],'complete')
        self.assertEqual(pipeline.read(self.root/'queue/DONE.json')['selected_step'],600)
        self.assertFalse((self.root/'queue/FAILED.json').exists())


if __name__=='__main__':unittest.main()
