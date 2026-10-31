"""Focused CPU contracts, real bound bootstrap, and fail-closed CLI checks.

No checkpoint deserialization, GPU, SSH, remote paths or rewritten production
metadata. Synthetic in-memory receipts exercise cross-run contracts directly;
the unchanged v8 loader remains responsible for full reference-file validation.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import compare as c

HERE=Path(__file__).resolve().parent
STATISTICS=c.load_statistics(HERE.parent/'gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py')


def paired_receipts():
    plan=[[(4*i+j)%168 for j in range(4)] for i in range(600)]
    shared=dict(data_hash='a'*64,source_binding={'state_checkpoint_sha256':'b'*64},
        frozen_reference_sha256='c'*64,original_config={'fixture':'same original config'},
        full_plan_sha256=c.digest(plan),continuation_plan_sha256=c.digest(plan[400:]))
    arms={}
    for name in c.ALL_ARMS:
        constant=name.endswith('constant');mode='constant' if constant else 'z'
        source=dict(shared,source_arm='constant_delta' if constant else 'z_delta',source_steps=400,
            source_checkpoint_path=f'/fixture/source400/{mode}.pt',source_checkpoint_sha256=c.digest(mode),
            source_selection_sha256='d'*64,frozen_encoder_sha256='e'*64,output_std_sha256='f'*64,
            fixed_null_key_bias={'fixture':mode},loaded_model_sha256=c.digest(mode+'model'),
            loaded_codec_sha256=c.digest(mode+'codec'))
        if name.startswith('base_'):
            source.update(schema='gs_local_condition_source_v1',arm=name,local_skip_enabled=False)
        else:
            source.update(schema='gs_spatial_action_source_v1',arm=name,spatial_action_enabled=True,
                          initial_spatial_parameters_sha256='9'*64)
        arms[name]=dict(steps=600,source=source,_plan=copy.deepcopy(plan),_rows=[])
    return arms


def target_rows(name):
    value={'base_z':.20,'spatial_z':.10,'base_constant':.40,'spatial_constant':.30}[name]
    constant=name.endswith('constant'); rows=[]
    for start in range(168):
        for horizon in range(1,10):
            row=dict(start=start,horizon=horizon,end=start+horizon,motion_ids=[0 if constant else start]*9,
                blinded_motion_ids=[0 if constant else start]*9,blinded_horizon_id_changed=False,delta={})
            for control in c.CONTROLS:
                score=.01 if control in c.FROZEN else value+(0. if control=='correct' else .1)
                if constant and control in c.LIVE[:5]:score=value
                row[control]=dict(mae=score,hand_activity_mae=score,motion_mae=score,psnr=30.-score,hand_activity_psnr=30.-score)
                row['delta'][control]=None if control in c.FROZEN else {m:score for m in c.DELTA.values()}
            rows.append(row)
    return rows


def screen_arms():
    return {name:dict(_rows=(rows:=target_rows(name)),_delta=c.delta_rows(rows),
               initial_reconstruction={'metrics':{'correct':{'frame_uniform':{'mae':.1,'hand_activity_mae':.1}}}})
            for name in c.ALL_ARMS}


class CompareTests(unittest.TestCase):
    def test_protocol_and_reference_identity_cannot_drift(self):
        protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
        evidence=c.Evidence();c.validate_protocol(protocol,evidence);self.assertEqual([],evidence.issues)
        for section,key,value in [('reference_baselines','summary_sha256','0'*64),
                                  ('reference_baselines','step',599),('train','additional_steps',201),
                                  ('invariants','lfq_bits',32)]:
            with self.subTest(section=section,key=key):
                changed=copy.deepcopy(protocol);changed[section][key]=value
                evidence=c.Evidence();c.validate_protocol(changed,evidence)
                self.assertTrue(evidence.issues)

    def test_source400_drift_and_fake_reference_are_rejected(self):
        arms=paired_receipts();evidence=c.Evidence();c.validate_pairing(arms,evidence)
        self.assertEqual([],evidence.issues)
        for name,key,value in [('spatial_z','source_checkpoint_sha256','0'*64),
                               ('spatial_z','loaded_codec_sha256','0'*64),
                               ('spatial_constant','original_config',{'changed':'loss'}),
                               ('base_z','local_skip_enabled',True)]:
            with self.subTest(name=name,key=key):
                changed=copy.deepcopy(arms);changed[name]['source'][key]=value
                evidence=c.Evidence();c.validate_pairing(changed,evidence)
                self.assertTrue(evidence.issues)
        self.assertEqual('gs_local_condition_source_v1',arms['base_z']['source']['schema'])
        self.assertEqual('gs_spatial_action_source_v1',arms['spatial_z']['source']['schema'])

    def test_unfinished600_and_plan_offset_rejected(self):
        done=dict(schema='gs_spatial_action_train_v1',status='complete',completed_steps=600,steps=600,
            additional_steps=200,original_config_preserved=True,encoder_unchanged=True,
            actual_wall_budget_seconds=1800,goal_complete=False)
        evidence=c.Evidence();c.validate_training_receipt(done,evidence);self.assertEqual([],evidence.issues)
        done['completed_steps']=599;evidence=c.Evidence();c.validate_training_receipt(done,evidence)
        self.assertTrue(evidence.issues)
        arms=paired_receipts();arms['spatial_z']['_plan'][400]=arms['spatial_z']['_plan'][401]
        evidence=c.Evidence();c.validate_pairing(arms,evidence)
        self.assertTrue(any('计划错位' in issue for issue in evidence.issues))

    def test_actual_continuation_log_not_just_done_label(self):
        arms=paired_receipts();source=arms['spatial_z']['source'];plan=arms['spatial_z']['_plan']
        logs=[dict(step=i+1,added_steps=i-399,plan_index=i,starts=plan[i],
                   **{key:source[key] for key in ('frozen_encoder_sha256','output_std_sha256','fixed_null_key_bias')})
              for i in range(400,600)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in logs),encoding='utf-8')
            evidence=c.Evidence();self.assertEqual(logs[-1],c.validate_train_log(path,plan,source,evidence))
            self.assertEqual([],evidence.issues)
            logs[-1]=logs[-2];path.write_text('\n'.join(json.dumps(row) for row in logs),encoding='utf-8')
            evidence=c.Evidence();self.assertIsNone(c.validate_train_log(path,plan,source,evidence))
            self.assertTrue(evidence.issues)

    def test_reference_file_bytes_are_checked_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('CODE_MANIFEST.json','protocol.json','SUMMARY.json'):(root/name).write_text('{}',encoding='utf-8')
            binding=dict(code_manifest_sha256='a'*64,protocol_sha256='b'*64,summary_sha256='c'*64)
            evidence=c.Evidence()
            with self.assertRaisesRegex(ValueError,'参考身份'):
                c.load_reference(root,{'reference_baselines':binding},evidence)
            self.assertEqual(3,len(evidence.issues))
            self.assertTrue(all('绑定缺失或不一致' in issue for issue in evidence.issues))

    def test_control_definition_drift_and_missing_new_controls_rejected(self):
        evaluation=dict(delta_definition=c.DELTA_DEFINITION,adapted_held_definition=c.HELD_DEFINITION,
            uncomputed_delta_controls=list(c.FROZEN),constant_shuffle_render_reused=False,constant_blinded_render_reused=False)
        evidence=c.Evidence();c.validate_controls(evaluation,'spatial_z',evidence);self.assertEqual([],evidence.issues)
        evaluation['adapted_held_definition']='old frozen initial codec'
        evidence=c.Evidence();c.validate_controls(evaluation,'spatial_z',evidence);self.assertTrue(evidence.issues)
        rows=target_rows('spatial_z');rows[0]['delta']['held_initial']={'delta_mae':0.}
        evidence=c.Evidence();self.assertFalse(c.validate_rows(rows,evidence));self.assertTrue(evidence.issues)
        rows=target_rows('spatial_z');del rows[0]['blinded']
        evidence=c.Evidence();self.assertFalse(c.validate_rows(rows,evidence));self.assertTrue(evidence.issues)

    def test_original_physical_frame_bootstrap(self):
        rows=target_rows('base_z')
        for row in rows:row['correct']['mae']=row['end']**2
        reduced=STATISTICS.reduce_rows(rows,'correct')
        self.assertEqual(176,reduced['unique_target_frames'])
        self.assertAlmostEqual(sum(i*i for i in range(1,177))/176,reduced['frame_uniform']['mae'])
        self.assertNotAlmostEqual(reduced['exposure_mean']['mae'],reduced['frame_uniform']['mae'])
        result=STATISTICS.moving_block_bootstrap({i:.25 for i in range(1,177)})
        self.assertEqual((.25,.25,16,2000,20261031),(result['ci95_low'],result['ci95_high'],result['block_length'],result['replicates'],result['seed']))
        with self.assertRaises(ValueError):STATISTICS.moving_block_bootstrap({1:1.})

    def test_current_held_gate_and_separate_constant_reference(self):
        screens,comparisons,checks=c.paired_screens(screen_arms(),STATISTICS)
        self.assertTrue(all(checks.values()))
        self.assertEqual({'base','spatial'},set(screens))
        for screen in screens.values():
            self.assertTrue(screen['passed']);self.assertEqual(14,len(screen['requirements']))
            self.assertTrue(any('adapted_held' in row['comparison'] for row in screen['requirements']))
            self.assertFalse(any('frozen' in row['comparison'] for row in screen['requirements']))
        self.assertFalse(comparisons['spatial_z_vs_frozen_held_descriptive']['mae']['positive_interval'])
        self.assertIn('spatial_constant_vs_base_constant',comparisons)

    def test_horizon_regression_overrides_positive_pooled_interval(self):
        arms=screen_arms()
        for row in arms['spatial_z']['_rows']:
            if row['horizon']==9:row['correct']['mae']=.25
        screens,comparisons,checks=c.paired_screens(arms,STATISTICS)
        self.assertTrue(comparisons['spatial_z_vs_shuffle42']['mae']['positive_interval'])
        self.assertFalse(screens['spatial']['passed']);self.assertFalse(checks['matched_action_screen'])
        self.assertTrue(any(row['horizon']==9 for row in screens['spatial']['horizon_degradations']))

    def test_missing_artifacts_write_refusal_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(2,c.main(['--root',directory]))
            summary=json.loads((Path(directory)/'SUMMARY.json').read_text(encoding='utf-8'))
            self.assertEqual('gs_spatial_action_comparison_v1',summary['schema'])
            self.assertFalse(summary['comparison_eligible']);self.assertFalse(summary['goal_complete'])
            self.assertEqual('inconclusive',summary['scientific_status']);self.assertEqual({},summary['comparisons'])
            self.assertTrue((Path(directory)/'SUMMARY.md').is_file())


if __name__=='__main__':unittest.main()
