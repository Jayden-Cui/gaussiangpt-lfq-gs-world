"""CPU-only bounded queue and receipt integrity tests; fixtures are synthetic."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import pipeline as q


PROTOCOL = q.read(Path(__file__).parent / 'protocol.json')
PROOF = dict(code_manifest_sha256='1'*64, protocol_sha256='2'*64)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def common():
    return dict(PROOF, goal_complete=False, gpu_peak_gib=4., gpu_reserved_peak_gib=5.,
                memory_admission=dict(allocator_cap_gib=16, free_gib=21.,
                    measurement_scope='PyTorch allocated/reserved; not NVML whole-process peak'))


def observation(start, factor=1.):
    rows=[]
    for h in range(10):
        v=.1 if h == 0 else .1*factor
        rows.append(dict(horizon=h, frame=start+h, gs_count=10,
                         metrics=dict(mae=v, hand_activity_mae=v, motion_mae=v, psnr=20.)))
    return q.worker.summarize(rows)


def fixture(root):
    source = root/'source.bin'; source.write_bytes(b'fixture-source')
    # 354 distinct paths with tiny files; no fabricated real data claimed.
    sources=[]
    for i in range(354):
        p=root/'sources'/f'{i:03d}.bin';p.parent.mkdir(exist_ok=True);p.write_bytes(b'x')
        sources.append(dict(path=str(p),bytes=1,sha256=q.sha(p)))
    pre=dict(common(),schema='gs_displacement_preflight_v1',status='passed',gpu_witness=True,
             source_checkpoint_sha256=PROTOCOL['source_hashes']['arms/spatial_z/step_0600.pt'],
             source_frames=177,source_files=sources,model_updates=0,oracle_updates=0,
             frozen_hashes=dict(model='3'*64,codec='4'*64),windows=[])
    for start in q.worker.STARTS:
        gradients={b:dict(forward_frames=10,backward_future_frames=9,h0_frozen=True,optimizer_updates=0,
                         gradient_norms=[1.]*(9 if b=='xyz_only' else 1)) for b in q.worker.BRANCHES}
        pre['windows'].append(dict(start=start,latent_rows=4,baseline=observation(start),
                                   actual_native_gradients=gradients))
    write(root/'PREFLIGHT.json',pre);pre_sha=q.sha(root/'PREFLIGHT.json')
    runs={}
    for start in q.worker.STARTS:
        out=root/'windows'/f'start_{start:04d}';out.mkdir(parents=True)
        names={f'{label}/h{h:02d}{suffix}' for label in q.LABELS for h in range(10)
               for suffix in ('.pt','_reference_rgb.pt','_view0.png','_view1.png')}
        names|={'baseline_state.pt','bounded_features_state.pt'}
        names|={f'{b}_variables.pt' for b in q.worker.BRANCHES}
        for name in names:
            p=out/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
        for b in q.worker.BRANCHES:
            (out/f'{b}_trace.jsonl').write_text('\n'.join(json.dumps(dict(step=i,loss_future_before_update=.1))
                                                         for i in range(1,101)),encoding='utf-8')
        branches={}
        for b in q.worker.BRANCHES:
            # One window passes, one intentionally fails; both are valid diagnostic results.
            factor=.7 if start==35 else 1.
            obs={str(step):observation(start,1. if step==0 else factor) for step in q.worker.CHECKPOINTS}
            branches[b]=dict(steps=100,parameters=30,observations=obs,legal_action_learning=False,
                             numeric_gate=q.worker.numeric_gate(obs['100'],observation(start)),
                             audit=[] if b=='xyz_only' else dict(max_abs_residual=.9))
        run=dict(common(),schema='gs_displacement_window_v1',status='complete_pending_independent_replay',
                 start=start,frames=list(range(start,start+10)),source_checkpoint_sha256=pre['source_checkpoint_sha256'],
                 source_files=sources,frozen_hashes=pre['frozen_hashes'],preflight_sha256=pre_sha,
                 baseline=observation(start),teacher_state_reference=observation(start,.5),branches=branches,
                 model_updates=0,oracle_updates=200,all_initial_rows=True,resolution_hw=[252,448],
                 artifact_sha256={p.relative_to(out).as_posix():q.sha(p) for p in out.rglob('*') if p.is_file()})
        write(out/'RUN_DONE.json',run);runs[start]=run
        records=[dict(label=label,horizon=h,frame=start+h,gs_count=10,max_abs=0.,mae=0.,
                      packet_sha256=run['artifact_sha256'][f'{label}/h{h:02d}.pt']) for label in q.LABELS for h in range(10)]
        replay=dict(common(),schema='gs_displacement_replay_v1',status='passed',start=start,
                    run_done_sha256=q.sha(out/'RUN_DONE.json'),records=records,model_loaded=False,
                    optimizer_loaded=False,z_required=False,displacement_required=False,views_per_frame=2)
        write(root/'replays'/f'start_{start:04d}'/'REPLAY_DONE.json',replay)
    return pre,pre_sha,runs


class SchedulingTests(unittest.TestCase):
    def test_commands_fixed_windows_and_separate_replay_process(self):
        root=Path('new-probe')
        self.assertNotIn('--start',q.command(root,'preflight'))
        for mode in ('run','replay'):
            for start in (35,123):
                argv=q.command(root,mode,start)
                self.assertEqual(argv[-4:],['--mode',mode,'--start',str(start)])
        for mode,start in [('preflight',35),('run',None),('run',44),('train',35)]:
            with self.subTest(mode=mode,start=start),self.assertRaises(RuntimeError):q.command(root,mode,start)
        self.assertEqual(q.LIMITS,dict(preflight=210,run=1830,replay=330,outer=3000))

    def test_real_memory_query_once_and_no_retry_when_low(self):
        with tempfile.TemporaryDirectory() as temp:
            result=SimpleNamespace(stdout='0, 22000\n1, 19000\n')
            with mock.patch.object(q.subprocess,'run',return_value=result) as run:
                with self.assertRaisesRegex(RuntimeError,'Less than 20'):q.gpu_admission(Path(temp))
                run.assert_called_once()
            self.assertEqual(q.read(Path(temp)/'GPU_ADMISSION.json')['queries'],1)

    def test_duplicate_or_missing_gpu_rejected(self):
        for text in ['0, 22000\n0, 23000\n','0, 22000\n','nonsense']:
            with tempfile.TemporaryDirectory() as temp,mock.patch.object(q.subprocess,'run',return_value=SimpleNamespace(stdout=text)):
                with self.assertRaises(RuntimeError):q.gpu_admission(Path(temp))

    def test_reserved_peak_has_its_own_guard(self):
        value=common();value['gpu_reserved_peak_gib']=16.01
        with self.assertRaisesRegex(RuntimeError,'16 GiB'):q.common_contract(value,PROOF)
        value=common();value['memory_admission']['free_gib']=19.
        with self.assertRaisesRegex(RuntimeError,'20 GiB'):q.common_contract(value,PROOF)

    def test_failure_stops_only_owned_helper_no_second_attempt(self):
        helper=SimpleNamespace(STOP_REQUESTED=threading.Event(),
                               run_child=mock.Mock(return_value=dict(exit_code=7,timed_out=False)),
                               terminate_queue=mock.Mock(side_effect=SystemExit(143)))
        with tempfile.TemporaryDirectory() as temp:
            rows=[]
            with self.assertRaisesRegex(RuntimeError,'failed or timed out'):
                q.run_job(Path(temp),Path(temp),'run',35,0,helper,rows,time.monotonic()+10)
            self.assertEqual(helper.run_child.call_count,1);helper.terminate_queue.assert_called_once()
            self.assertEqual(rows[0]['status'],'failed')
            self.assertEqual(q.read(Path(temp)/'run_0035_PROCESS.json')['status'],'failed')

    def test_expired_overall_budget_never_spawns_worker(self):
        helper=SimpleNamespace(STOP_REQUESTED=threading.Event(),run_child=mock.Mock(),
                               terminate_queue=mock.Mock(side_effect=SystemExit(143)))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'budget expired'):
                q.run_job(Path(temp),Path(temp),'run',35,0,helper,[],time.monotonic()-1)
            helper.run_child.assert_not_called()

    def test_existing_attempt_not_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);q.assert_fresh(root)
            for name in ['queue','windows','replays','PREFLIGHT.json','SUMMARY.json','SUMMARY.md']:
                p=root/name;p.write_bytes(b'old')
                with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'Existing probe'):q.assert_fresh(root)
                p.unlink()


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.pre,cls.pre_sha,cls.runs=fixture(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def window(self):return q.window_contract(self.root,35,PROTOCOL,PROOF,self.pre,self.pre_sha)

    def test_complete_end_to_end_cpu_receipts_and_scientific_failure_is_result(self):
        pre,digest=q.preflight_contract(self.root,PROTOCOL,PROOF)
        self.assertEqual(digest,self.pre_sha)
        summary=q.build_summary(self.root,PROTOCOL,PROOF,pre,digest)
        self.assertEqual(summary['numeric_gates_passed'],2)
        self.assertEqual(summary['numeric_gates_total'],4)
        self.assertFalse(summary['learned_action']);self.assertFalse(summary['goal_complete'])
        self.assertEqual(summary['oracle_updates'],400);self.assertEqual(summary['model_updates'],0)
        self.assertEqual(summary['visual_acceptance'],'pending')
        self.assertIn('没有训练或证明学会动作 Z',q.summary_markdown(summary))

    def test_preflight_cannot_omit_gradient_or_source_frame(self):
        value=copy.deepcopy(self.pre);value['windows'][0]['actual_native_gradients']['xyz_only']['gradient_norms']=[1.]
        with mock.patch.object(q,'read',return_value=value),self.assertRaisesRegex(RuntimeError,'every oracle variable'):
            q.preflight_contract(self.root,PROTOCOL,PROOF)
        value=copy.deepcopy(self.pre);value['source_files'].pop()
        with mock.patch.object(q,'read',return_value=value),self.assertRaisesRegex(RuntimeError,'354'):
            q.preflight_contract(self.root,PROTOCOL,PROOF)

    def test_true_complete_artifact_hashes_required(self):
        p=self.root/'windows/start_0035/xyz_only/h09.pt';old=p.read_bytes()
        try:
            p.write_bytes(b'contaminated')
            with self.assertRaisesRegex(RuntimeError,'Artifact hash'):self.window()
        finally:p.write_bytes(old)

    def test_numeric_gate_cannot_disagree_with_observations(self):
        changed=copy.deepcopy(self.runs[35]);changed['branches']['xyz_only']['numeric_gate']['future_full_nonworse']=False
        original=q.read
        def reader(path):return changed if Path(path).name=='RUN_DONE.json' else original(path)
        with mock.patch.object(q,'read',side_effect=reader),self.assertRaisesRegex(RuntimeError,'Numeric gate'):
            self.window()

    def test_frozen_h0_is_checked_separately(self):
        changed=copy.deepcopy(self.runs[35]);changed['branches']['xyz_only']['observations']['100']['per_horizon'][0]['metrics']['mae']=.2
        original=q.read
        with mock.patch.object(q,'read',side_effect=lambda p:changed if Path(p).name=='RUN_DONE.json' else original(p)),\
             self.assertRaisesRegex(RuntimeError,'Frozen h0'):
            self.window()

    def test_replay_rejects_duplicates_and_hidden_runtime_dependency(self):
        path=self.root/'replays/start_0035/REPLAY_DONE.json';original=q.read(path)
        for mutation in ('duplicate','hidden','tolerance'):
            value=copy.deepcopy(original)
            if mutation=='duplicate':value['records'][1]=value['records'][0]
            if mutation=='hidden':value['z_required']=True
            if mutation=='tolerance':value['records'][0]['max_abs']=2e-6
            with mock.patch.object(q,'read',return_value=value),self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                q.replay_contract(self.root,35,PROOF,self.runs[35],original['run_done_sha256'])

    def test_source_file_change_after_all_workers_rejected(self):
        p=Path(self.pre['source_files'][0]['path']);old=p.read_bytes()
        try:
            p.write_bytes(b'y')
            with self.assertRaisesRegex(RuntimeError,'Original data changed'):
                q.build_summary(self.root,PROTOCOL,PROOF,self.pre,self.pre_sha)
        finally:p.write_bytes(old)


if __name__=='__main__':unittest.main()
