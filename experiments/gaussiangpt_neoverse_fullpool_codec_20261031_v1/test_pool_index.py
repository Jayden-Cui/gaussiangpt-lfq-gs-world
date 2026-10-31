"""Synthetic CPU fixtures only; these are not real pool or tensor readiness evidence."""
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pool_index as p


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')


class Fixture:
    def __init__(self, parent):
        self.root = parent/p.POOL_RUN
        self.root.mkdir()
        real = Path(__file__).resolve().parent.parent/p.POOL_RUN
        manifest = json.loads((real/'CODE_MANIFEST.json').read_text())
        for name in ['CODE_MANIFEST.json', *manifest]:
            destination = self.root/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(real/name, destination)
        legacy = parent/p.LEGACY_RUN/'sources.py'
        legacy.parent.mkdir()
        shutil.copyfile(real.parent/p.LEGACY_RUN/'sources.py', legacy)
        self.pool = json.loads((self.root/'INPUT_POOL.json').read_text())
        self.records = {}
        for name in ['CODE_MANIFEST.json', *manifest]: self.add(self.root/name)
        for item in self.pool['clips']:
            self.records[item['video_path']] = dict(path=item['video_path'], bytes=item['video_bytes'], sha256=item['video_sha256'])
        static = set(self.records)
        self.normal = dict(center=[0.,0.,0.], scale=.5, source_frame=0, fixed_for_all_frames=True,
                           calibration_only_alpha_threshold=.01, uses_future_frames=False,
                           changes_codec_output_std=False)
        self.mixed_scale_clip = 'buildingblock_tower_18_352'
        self.clips = []
        reuse_paths = set()
        for item in self.pool['clips']:
            clip = item['clip_id']; previous = set(self.records)
            normal = dict(self.normal, scale=.25 if clip == self.mixed_scale_clip else .5)
            directory, teacher, cache, rows_path = p.layout(self.root, clip)
            cache.mkdir(parents=True, exist_ok=True)
            rows = []; output = []; natives = []
            for frame in range(177):
                path = cache/f'{frame:03d}.pt'
                path.write_bytes(b'NOT_A_TENSOR_SYNTHETIC_CACHE:'+clip.encode()+str(frame).encode())
                record = self.add(path); output.append(record)
                native_path = teacher/'gaussians'/f'native_group_{frame:03d}.npz'
                native = dict(path=str(native_path), bytes=17, sha256=sha(f'fake-native:{clip}:{frame}'.encode()))
                self.records[p.key(native_path)] = native; natives.append(native)
                rows.append(dict(clip=clip, frame=frame, source_count=p.POINTS, pts_seconds=frame/29.97,
                                 cache_sha256=record['sha256'], cache_bytes=record['bytes'], native_gs_sha256=native['sha256']))
            mapping = dict(clip=clip, source_sha256=item['video_sha256'], source_frame_indices=list(range(177)),
                           model_timestamps=list(range(177)), source_pts_seconds=[f/29.97 for f in range(177)],
                           adjacent_pairs=[[i,i+1] for i in range(176)])
            for path, value in [(directory/'normalization.json', normal),
                                (directory/'mappings'/f'{clip}.json', mapping),
                                (rows_path, {'rows' if clip == 'buildingblock_tower_17_176' else 'frames':rows})]:
                save(path, value); output.append(self.add(path))
            accepted = dict(clip_id=clip,status='complete',source_group=item['source_group'],
                original_role=item['original_role'],current_role='TRAIN',source_video_sha256=item['video_sha256'],
                actual_frames=177,actual_adjacent_pairs=176,actual_source_points_per_frame=p.POINTS,
                actual_cache_files=177,actual_native_gs_files=177,paths=dict(teacher=str(teacher),cache=str(cache)),
                state_encodings_reused_for_new_codec=False)
            if clip in p.REUSE:
                accepted.update(origin='admitted_existing_full177_sequence', normalization=normal,
                                role_basis=item['role_basis'], native_arrays_and_cache_tensors_checked=True)
                reference = dict(schema='gs_fulltrain_reuse_reference_v1',clip_id=clip,
                    status='reference_only_requires_admission',current_role='TRAIN',original_role=item['original_role'],
                    source_group=item['source_group'],paths_requiring_admission=accepted['paths'])
                path = self.root/'clips'/clip/'reuse_reference.json';save(path,reference);self.add(path)
                reuse_paths.update(set(self.records)-previous)
            else:
                admitted = {}
                for name in ('input_binding.json','SUMMARY.json','HELPER_SUMMARY.json'):
                    path=directory/name;save(path,dict(status='complete'));admitted[name]=self.add(path)
                inputs=dict(schema='gs_transfer_inputs_v1',teacher_hashes_computed_after_export=True,
                            binding=admitted['input_binding.json'],teacher_and_source_files=natives+[self.records[item['video_path']]])
                path=directory/'inputs_hashes.json';save(path,inputs);output.append(self.add(path))
                receipt=dict(schema='gs_fulltrain_clip_preparation_receipt_v1',status='complete',clip=clip,
                    source_group=item['source_group'],original_role=item['original_role'],current_role='TRAIN',
                    data_origin='new_full177_export',training_performed=False,goal_complete=False,
                    helper_sha256=p.PREPARE_HELPER_SHA,code_manifest_sha256=p.MANIFEST_SHA,input_pool_sha256=p.INPUT_POOL_SHA,
                    actual_frames=177,actual_cache_files=177,actual_native_gs_files=177,
                    actual_source_points_per_frame=p.POINTS,actual_adjacent_pairs=176,output_files=output,
                    input_binding_sha256=admitted['input_binding.json']['sha256'],summary_sha256=admitted['SUMMARY.json']['sha256'],
                    helper_summary_sha256=admitted['HELPER_SUMMARY.json']['sha256'])
                path=directory/'PREPARATION_RECEIPT.json';save(path,receipt);record=self.add(path)
                accepted.update(origin='new_full177_export',preparation_receipt=record,actual_state_files=177)
                save(self.root/'queue'/f'{clip}_COMPLETE.json', dict(clip=clip,status='complete',goal_complete=False,
                                                                  preparation_receipt_sha256=record['sha256']))
            self.clips.append(accepted)
        base=dict(status='complete',code_manifest_sha256=p.MANIFEST_SHA,input_pool_sha256=p.INPUT_POOL_SHA,
                  training_executed=False,training_performed=False,goal_complete=False,
                  official_independent_holdout=False,cross_clip_physical_frame_unique_count=None)
        reuse=dict(base,schema='gs_fulltrain_reuse_admission_v1',actual_clip_internal_frames_total=354,
                   clips=[c for c in self.clips if c['clip_id'] in p.REUSE],
                   files=[self.records[name] for name in sorted(static|reuse_paths)])
        save(self.root/'REUSE_ADMISSION.json',reuse);reuse_record=self.add(self.root/'REUSE_ADMISSION.json')
        self.complete=dict(base,schema='gs_fulltrain_pool_complete_v1',all_clips_complete=True,
            planned_clip_internal_frames_total=8850,actual_clip_internal_frames_total=8850,actual_adjacent_pairs_total=8800,
            fresh_clips=48,reused_clips=2,source_groups=19,native_width=448,native_height=252,gs_per_frame=p.POINTS,
            frame_stride=1,state_encodings_reused_for_new_codec=False,clips=self.clips,
            files=list(self.records.values()),reuse_admission=reuse_record)
        save(self.root/'POOL_COMPLETE.json',self.complete)
        cpu=dict(status='passed',gpu_executed=False,actual_tests_executed=1,code_manifest_sha256=p.MANIFEST_SHA,
                 protocol_sha256=manifest['protocol.json'])
        save(self.root/'CPU_VALIDATION.json',cpu)
        proof=dict(code_manifest_sha256=p.MANIFEST_SHA,input_pool_sha256=p.INPUT_POOL_SHA,protocol_sha256=manifest['protocol.json'],
                   process_helper_sha256=p.PROCESS_HELPER_SHA,cpu_validation_sha256=sha((self.root/'CPU_VALIDATION.json').read_bytes()))
        names={'reuse_admission','final_admission'}|{c+'_'+s for c in p.EXPECTED if c not in p.REUSE for s in ('binding','teacher','prepare')}
        self.done=dict(status='complete',admission=proof,pool_complete_sha256=sha((self.root/'POOL_COMPLETE.json').read_bytes()),
            reuse_admission_sha256=reuse_record['sha256'],clips=50,fresh_exports=48,reused_exports=2,clip_internal_frames=8850,
            cross_clip_physical_frame_unique_count=None,automatic_retry=False,training_executed=False,goal_complete=False,
            outcomes=[dict(name=n,status='complete',exit_code=0,timed_out=False) for n in sorted(names)])
        save(self.root/'queue/DONE.json',self.done)

    def add(self,path):
        path=path.resolve();raw=path.read_bytes()
        record=dict(path=str(path),bytes=len(raw),sha256=sha(raw));self.records[p.key(path)]=record
        return record

    def pins(self):
        return patch.multiple(p,DONE_SHA=sha((self.root/'queue/DONE.json').read_bytes()),
                              COMPLETE_SHA=sha((self.root/'POOL_COMPLETE.json').read_bytes()))


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='gs_pool_synthetic_')
        cls.fixture=Fixture(Path(cls.temp.name).resolve())
        with cls.fixture.pins():
            cls.index=p.load_pool_index(cls.fixture.root)

    @classmethod
    def tearDownClass(cls):
        p._MODULES.clear()
        cls.temp.cleanup()

    def test_fixed_membership_roles_and_all_frames(self):
        self.assertEqual(len(self.index),8850)
        self.assertEqual(Counter(r.clip for r in self.index),Counter({c:177 for c in p.EXPECTED}))
        self.assertEqual(len({r.source_group for r in self.index}),19)
        self.assertEqual(sum(r.original_role=='DEV' for r in self.index),177)
        self.assertTrue(all(r.current_role=='TRAIN' for r in self.index))
        self.assertEqual(Counter(r.normalization['scale'] for r in self.index),Counter({.5:8673,.25:177}))
        self.assertTrue(all(r.normalization['scale']==.25 for r in self.index if r.clip==self.fixture.mixed_scale_clip))

    def test_full_B4_deterministic_round_robin_and_actual_tail(self):
        a=p.balanced_plan(self.index,42,4);b=p.balanced_plan(tuple(reversed(self.index)),42,4)
        self.assertEqual(a,b)
        self.assertEqual(len(a),2213);self.assertEqual(len(a[-1]),2)
        flat=[r for batch in a for r in batch]
        for start in range(0,8850,50): self.assertEqual(len({r.clip for r in flat[start:start+50]}),50)
        self.assertNotEqual(a,p.balanced_plan(self.index,43,4))
        summary=p.plan_summary(self.index,a)
        self.assertEqual(summary['frames'],8850);self.assertEqual(summary['source_group_exposure']['buildingblock_tower_18'],708)
        self.assertIsNone(summary['cross_clip_physical_frame_unique_count'])

    def test_other_batches_and_uneven_generic_round_robin(self):
        for batch,updates,tail in [(2,4425,2),(8,1107,2),(16,554,2),(50,177,50),(10000,1,8850)]:
            plan=p.balanced_plan(self.index,7,batch)
            self.assertEqual((len(plan),len(plan[-1])),(updates,tail))
        small=(*self.index[:3],*self.index[177:179])
        plan=p.balanced_plan(small,7,4)
        self.assertEqual([len(batch) for batch in plan],[4,1])
        self.assertTrue(p.assert_coverage(small,plan))

    def test_duplicate_missing_foreign_and_mutated_frames_rejected(self):
        small=self.index[:3]
        for batches in [((small[0],small[0],small[2]),),((small[0],small[1]),),
                        ((small[0],small[1],self.index[4]),),((replace(small[0],cache_sha256='0'*64),*small[1:]),)]:
            with self.assertRaises(ValueError):p.assert_coverage(small,batches)
        with self.assertRaises(ValueError):p.balanced_plan((small[0],small[0]),2,1)
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):p.balanced_plan(small,2,value)

    def test_missing_terminal_blocks_before_any_cache_read(self):
        path=self.fixture.root/'queue/DONE.json';raw=path.read_bytes();path.unlink()
        try:
            with self.assertRaises(ValueError):p.load_pool_index(self.fixture.root)
        finally:path.write_bytes(raw)

    def test_failed_pool_is_rejected_even_with_done(self):
        path=self.fixture.root/'queue/FAILED.json';path.write_text('{}')
        try:
            with self.fixture.pins(),self.assertRaises(ValueError):p.load_pool_index(self.fixture.root)
        finally:path.unlink()

    def test_terminal_sha_and_manifest_source_changes_rejected(self):
        with patch.object(p,'DONE_SHA','0'*64),self.assertRaises(ValueError):p.load_pool_index(self.fixture.root)
        path=self.fixture.root/'admission.py';raw=path.read_bytes();path.write_bytes(raw+b'\n')
        try:
            with self.fixture.pins(),self.assertRaises(ValueError):p.load_pool_index(self.fixture.root)
        finally:path.write_bytes(raw)

    def test_actual_same_size_cache_corruption_rejected(self):
        path=Path(self.index[0].cache_path);raw=path.read_bytes();path.write_bytes(b'x'*len(raw))
        try:
            with self.fixture.pins(),self.assertRaisesRegex(ValueError,'SHA'):p.load_pool_index(self.fixture.root)
        finally:path.write_bytes(raw)

    def test_duplicate_clip_and_missing_cache_record_rejected_after_valid_hash_chain(self):
        complete_path=self.fixture.root/'POOL_COMPLETE.json';done_path=self.fixture.root/'queue/DONE.json'
        original_complete=complete_path.read_bytes();original_done=done_path.read_bytes()
        for case in ('duplicate_clip','missing_cache'):
            try:
                complete=json.loads(original_complete);done=json.loads(original_done)
                if case=='duplicate_clip':complete['clips'][-1]=complete['clips'][0]
                else:complete['files']=[r for r in complete['files'] if r['path']!=self.index[0].cache_path]
                save(complete_path,complete);done['pool_complete_sha256']=sha(complete_path.read_bytes());save(done_path,done)
                with self.fixture.pins(),self.assertRaises(ValueError):p.load_pool_index(self.fixture.root)
            finally:complete_path.write_bytes(original_complete);done_path.write_bytes(original_done)

    def test_old_validator_parameterized_without_mutating_old_source(self):
        legacy=p.isolated(self.index[0].legacy_source_path,p.LEGACY_SHA,'sources')
        self.assertEqual(set(legacy.ALL_CLIPS),set(p.EXPECTED))
        self.assertEqual(sha(Path(self.index[0].legacy_source_path).read_bytes()),p.LEGACY_SHA)
        directory,_,_,rows_path=p.layout(self.fixture.root,self.index[0].clip)
        rows=json.loads(rows_path.read_text())['frames'];mapping=json.loads((directory/'mappings'/f'{self.index[0].clip}.json').read_text())
        legacy.validate_index(self.index[0].clip,rows,mapping,self.fixture.normal)
        bad_mapping=dict(mapping,source_pts_seconds=[i/5 for i in range(177)])
        with self.assertRaises(ValueError):legacy.validate_index(self.index[0].clip,rows,bad_mapping,self.fixture.normal)
        with self.assertRaises(ValueError):legacy.validate_index(self.index[0].clip,rows,mapping,dict(self.fixture.normal,uses_future_frames=True))

    def test_load_frame_checks_exact_same_bytes_before_torch_deserialization(self):
        import sys
        ref=self.index[0];legacy=p.isolated(ref.legacy_source_path,p.LEGACY_SHA,'sources')
        calls=[]
        class FakeTorch:
            @staticmethod
            def load(stream,**kwargs):
                calls.append((stream,kwargs));raise RuntimeError('STOP_BEFORE_SYNTHETIC_TENSOR_LOAD')
        with patch.dict(sys.modules,{'torch':FakeTorch}):
            with self.assertRaisesRegex(RuntimeError,'STOP_BEFORE'):p.load_frame(ref,device='cpu')
            self.assertIsInstance(calls[0][0],io.BytesIO)
            self.assertEqual(calls[0][0].getvalue(),Path(ref.cache_path).read_bytes())
            self.assertEqual(calls[0][1],dict(map_location='cpu',weights_only=True))
            calls.clear()
            with self.assertRaises(ValueError):p.load_frame(replace(ref,cache_sha256='0'*64))
            self.assertEqual(calls,[])


if __name__=='__main__':
    unittest.main()
