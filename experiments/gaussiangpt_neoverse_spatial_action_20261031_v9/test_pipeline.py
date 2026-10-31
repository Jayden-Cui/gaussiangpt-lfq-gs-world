"""CPU orchestration gates; no real subprocess training, SSH or GPU queries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipeline as q
import worker

ROOT=Path(__file__).resolve().parent


class FakeStop:
    def __init__(self):self.stopped=False;self.now=0.;self.waits=[]
    def is_set(self):return self.stopped
    def set(self):self.stopped=True
    def wait(self,seconds):self.now+=seconds;self.waits.append(seconds);return self.stopped


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run';self.root.mkdir()
        self.output=self.root/'queue';self.output.mkdir()
        self.cfg=q.read(ROOT/'protocol.json');self.stop=FakeStop()

    def helper(self,child=None,event=None):
        event=event or self.stop
        def terminate(signum,frame):event.set();raise SystemExit(128+signum)
        return SimpleNamespace(STOP_REQUESTED=event,run_child=child or Mock(),terminate_queue=Mock(side_effect=terminate))

    def snapshot(self,**changes):
        dep=self.cfg['dependency'];result=dict(pid=dep['pid'],start_ticks=dep['start_ticks'],stable=True,
            argv=[q.RUNTIME,'-B','-u',dep['root']+'/pipeline.py','--root',dep['root']])
        result.update(changes);return result

    def test_dependency_exact_identity_and_pid_reuse(self):
        dependency=self.cfg['dependency']
        self.assertEqual(q.classify_parent(self.snapshot(),dependency),'alive')
        self.assertEqual(q.classify_parent(None,dependency),'exited')
        for bad in (self.snapshot(pid=1),self.snapshot(start_ticks='changed'),self.snapshot(stable=False),
                    self.snapshot(argv=[q.RUNTIME,'-c','pass'])):
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):q.classify_parent(bad,dependency)

    def test_resource_dependency_allows_unique_failed_but_not_missing_or_conflicting(self):
        dep=self.cfg['dependency'];admission={'code_manifest_sha256':dep['manifest_sha256']}
        done={'status':'complete','admission':admission};failed={'status':'failed','admission':admission}
        self.assertEqual(q.terminal_contract(done,None,dep)['parent_terminal'],'DONE')
        self.assertEqual(q.terminal_contract(None,failed,dep)['parent_terminal'],'FAILED')
        for a,b in ((None,None),(done,failed),(failed,None),(None,done)):
            with self.assertRaises(ValueError):q.terminal_contract(a,b,dep)
        bad=deepcopy(failed);bad['admission']['code_manifest_sha256']='0'*64
        with self.assertRaises(ValueError):q.terminal_contract(None,bad,dep)

    def test_same_root_workers_keep_dependency_waiting(self):
        helper=self.helper();dep=self.cfg['dependency'];worker=self.snapshot()
        worker['argv']=[q.RUNTIME,dep['root']+'/worker.py','--root',dep['root'],'--mode','train','--arm','full_codec']
        self.assertTrue(q.dependency_worker(worker,dep))
        observations=[dict(all_exited=False,workers=[worker]),dict(all_exited=True,workers=[])]
        terminal={'decision':{'parent_terminal':'FAILED'},'hashes':{}}
        with patch.object(q,'observe',side_effect=observations),patch.object(q,'read_terminal',return_value=terminal),\
             patch.object(q.time,'monotonic',side_effect=lambda:self.stop.now):
            self.assertEqual(q.wait_for_parent(self.output,self.cfg,helper),terminal)
        self.assertEqual(self.stop.waits,[60]);helper.terminate_queue.assert_not_called()

    def test_dependency_missing_terminal_and_wait_timeout_never_restart_or_kill_parent(self):
        helper=self.helper()
        with patch.object(q,'observe',return_value={'all_exited':True}),patch.object(q,'read_terminal',side_effect=ValueError('missing')):
            with self.assertRaisesRegex(ValueError,'missing'):q.wait_for_parent(self.output,self.cfg,helper)
        with patch.object(q,'observe',return_value={'all_exited':False}),patch.object(q,'write'),\
             patch.object(q.time,'monotonic',side_effect=lambda:self.stop.now):
            with self.assertRaisesRegex(TimeoutError,'7200'):q.wait_for_parent(self.output,self.cfg,helper)
        self.assertEqual(sum(self.stop.waits),7200);helper.run_child.assert_not_called();helper.terminate_queue.assert_not_called()

    def test_first_parallel_failure_stops_owned_sibling_and_preserves_results(self):
        event=threading.Event();barrier=threading.Barrier(2)
        def child(command,env,log,timeout):
            barrier.wait(timeout=3)
            if command[-1]=='spatial_z':return dict(exit_code=1,timed_out=False,pid=1)
            if not event.wait(3):raise AssertionError('Sibling did not receive shared stop')
            return dict(exit_code=-15,timed_out=False,pid=2)
        helper=self.helper(Mock(side_effect=child),event);outcomes=[]
        with self.assertRaises(RuntimeError):q.execute_stage(self.root,self.output,'train',helper,outcomes,Mock())
        self.assertTrue(event.is_set());self.assertEqual(helper.run_child.call_count,2)
        self.assertEqual({r['name'] for r in outcomes},{'train_spatial_z','train_spatial_constant'})
        self.assertEqual({r['exit_code'] for r in outcomes},{1,-15})

    def test_timeout_does_not_retry_or_launch_next_stage(self):
        helper=self.helper(Mock(return_value={'exit_code':0,'timed_out':True}));outcomes=[]
        with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
            q.execute_stage(self.root,self.output,'preflight',helper,outcomes,Mock())
        helper.run_child.assert_called_once();helper.terminate_queue.assert_called_once()
        self.assertTrue(outcomes[0]['timed_out'])

    def test_fixed_commands_only_two_new_arms_and_budgets(self):
        q.protocol_contract(self.cfg)
        self.assertNotIn('--arm',q.jobs(self.root,'preflight')[0][2])
        self.assertEqual(q.jobs(self.root,'preflight')[0][1::2],(0,630))
        for stage,timeout in [('train',1900),('eval',2500)]:
            tasks=q.jobs(self.root,stage)
            self.assertEqual([t[1] for t in tasks],[0,1])
            self.assertEqual([t[2][-1] for t in tasks],['spatial_z','spatial_constant'])
            self.assertEqual([t[3] for t in tasks],[timeout,timeout])
        self.assertEqual(q.jobs(self.root,'compare')[0][3],300)

    def make_manifest(self):
        files=q.REQUIRED|set(q.external_helpers())
        for relative in files:
            destination=self.root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            origin=ROOT/relative;destination.write_bytes(origin.read_bytes() if origin.is_file() else b'# CPU fixture\n')
        q.write(self.root/'CODE_MANIFEST.json',{p:q.sha(self.root/p) for p in files})

    def test_compare_generated_summary_md_remains_output_not_source(self):
        self.make_manifest();_,before=q.verify(self.root,cpu=False)
        (self.root/'SUMMARY.md').write_text('# Generated complete comparison\n',encoding='utf-8')
        _,after=q.verify(self.root,cpu=False)
        self.assertEqual(before,after)
        self.assertNotIn('SUMMARY.md',q.read(self.root/'CODE_MANIFEST.json'))
        q.write(self.root/'CPU_VALIDATION.json',dict(status='passed',gpu_executed=False,**before))
        q.verify(self.root)

    def test_cpu_verify_only_local_hashes_and_exact_proof(self):
        self.make_manifest();real_sha=q.sha
        def local_sha(path):
            self.assertTrue(Path(path).resolve().is_relative_to(Path(self.temp.name)))
            return real_sha(path)
        with patch.object(q,'sha',side_effect=local_sha):_,admission=q.verify(self.root,cpu=False)
        q.write(self.root/'CPU_VALIDATION.json',dict(status='passed',gpu_executed=False,**admission))
        _,bound=q.verify(self.root);self.assertIn('cpu_validation_sha256',bound)
        proof=q.read(self.root/'CPU_VALIDATION.json');proof['gpu_executed']=True
        q.write(self.root/'CPU_VALIDATION.json',proof)
        with self.assertRaisesRegex(ValueError,'CPU_VALIDATION'):q.verify(self.root)

    def train_fixture(self):
        arm='spatial_z';folder=self.root/'arms'/arm;folder.mkdir(parents=True)
        starts=[[0,42,84,126] for _ in range(600)];admission={'code_manifest_sha256':'m','protocol_sha256':'p'}
        source=dict(schema='gs_spatial_action_source_v1',arm=arm,source_arm='z_delta',source_steps=400,
            source_checkpoint_sha256='a'*64,source_manifest_sha256=self.cfg['source_manifest_sha256'],
            code_manifest_sha256='m',protocol_sha256='p',reference_baselines=self.cfg['reference_baselines'],
            loaded_model_sha256='b'*64,loaded_codec_sha256='c'*64,original_optimizer_state_sha256='d'*64,
            original_rng_state_sha256='e'*64,original_config={'seed':20261031},fixed_null_key_bias={'count':27},
            output_std_sha256='f'*64,full_plan_sha256=worker.digest(starts),continuation_plan_sha256=worker.digest(starts[400:]),
            spatial_action_enabled=True,optimizer_rng_restored_from_source=True,goal_complete=False)
        plan=dict(schema='gs_spatial_action_plan_v1',source_steps=400,intended_total_steps=600,starts=starts,
                  continued_batch_indices=list(range(400,600)),full_plan_sha256=source['full_plan_sha256'],
                  continuation_plan_sha256=source['continuation_plan_sha256'])
        q.write(folder/'SOURCE.json',source);q.write(folder/'PLAN.json',plan)
        q.write(self.root/'PREFLIGHT.json',{'status':'passed'})
        rows=[dict(step=step,plan_index=step-1,added_steps=step-400,starts=starts[step-1]) for step in range(401,601)]
        (folder/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        for step in (500,600):(folder/f'step_{step:04d}.pt').write_bytes(b'CPU checkpoint gate fixture')
        done=dict(source,schema='gs_spatial_action_train_v1',status='complete',completed_steps=600,steps=600,additional_steps=200,
                  original_config_preserved=True,encoder_unchanged=True,last=rows[-1],
                  source_sha256=q.sha(folder/'SOURCE.json'),plan_sha256=q.sha(folder/'PLAN.json'),
                  preflight_sha256=q.sha(self.root/'PREFLIGHT.json'),train_jsonl_sha256=q.sha(folder/'train.jsonl'),
                  checkpoint_sha256={str(step):q.sha(folder/f'step_{step:04d}.pt') for step in (500,600)})
        q.write(folder/'TRAIN_DONE.json',done)
        return folder,admission,{'arms':{arm:source}},done

    def test_partial600_or_incomplete200_trace_cannot_enter_evaluation(self):
        folder,admission,preflight,done=self.train_fixture()
        q.train_contract(self.root,'spatial_z',self.cfg,admission,preflight,{})
        for key,value in [('completed_steps',599),('partial',True),('additional_steps',199),('actual_step',599)]:
            bad=dict(done,**{key:value});q.write(folder/'TRAIN_DONE.json',bad)
            with self.subTest(key=key),self.assertRaises(ValueError):
                q.train_contract(self.root,'spatial_z',self.cfg,admission,preflight,{})
        q.write(folder/'TRAIN_DONE.json',done)
        lines=(folder/'train.jsonl').read_text().splitlines();(folder/'train.jsonl').write_text('\n'.join(lines[1:]))
        with self.assertRaisesRegex(ValueError,'Partial trace'):
            q.train_contract(self.root,'spatial_z',self.cfg,admission,preflight,{})

    def test_gpu_admission_one_mock_query_and_engineering_not_scientific_gate(self):
        with patch.object(q.subprocess,'run',return_value=SimpleNamespace(stdout='0, 56320\n1, 56320\n')) as query:
            q.gpu_admission(self.output);query.assert_called_once()
        summary=dict(schema='gs_spatial_action_comparison_v1',engineering_status='complete',comparison_eligible=True,
            comparison_eligibility={'eligible':True},selected_step=600,goal_complete=False,new_arms=list(q.ARMS),
            reference_arms=['base_z','base_constant'],reference_root=self.cfg['reference_baselines']['root'],
            comparison_source_sha256=q.STATISTICS_SHA,reference_validator_sha256=q.REFERENCE_COMPARE_SHA,
            scientific_status='spatial_action_screen_not_passed')
        q.write(self.root/'SUMMARY.json',summary);q.comparison_contract(self.root,self.cfg)
        summary['comparison_eligible']=False;q.write(self.root/'SUMMARY.json',summary)
        with self.assertRaises(ValueError):q.comparison_contract(self.root,self.cfg)


if __name__=='__main__':unittest.main(verbosity=2)
