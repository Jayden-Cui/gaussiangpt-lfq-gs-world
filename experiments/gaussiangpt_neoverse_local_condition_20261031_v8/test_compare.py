"""CPU-only synthetic evidence tests. Fixture .pt files are inert test bytes.

Tests never call worker, torch, SSH, GPU or real experiment outputs. The sole
patched production identity is the synthetic ancestor manifest hash.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare as c


HERE = Path(__file__).resolve().parent
STATISTICS = c.load_statistics(HERE.parent/'gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def image_metrics(value):
    return dict(mae=value, hand_activity_mae=value, motion_mae=value, psnr=30-value, hand_activity_psnr=30-value)


def target_rows(arm):
    value = dict(base_z=.2, local_z=.1, base_constant=.4, local_constant=.3)[arm]
    constant = arm.endswith('constant'); rows = []
    for start in range(168):
        ids = [0]*9 if constant else [start]*9
        for h in range(1, 10):
            row = dict(start=start, horizon=h, end=start+h, motion_ids=ids, blinded_motion_ids=ids,
                       blinded_horizon_id_changed=False, delta={})
            for control in c.CONTROLS:
                # Frozen held is intentionally BETTER: it must not silently
                # replace the newly declared current-codec held screen.
                v = .01 if control in c.FROZEN else value + (.1 if control != 'correct' else 0)
                if constant and control in c.LIVE[:5]: v = value
                row[control] = image_metrics(v)
                row['delta'][control] = None if control in c.FROZEN else {m:v for m in c.DELTA.values()}
            rows.append(row)
    return rows


def fixture(root):
    """Construct the exact receipt structure emitted by worker using test bytes."""
    old = root/'ancestor'; old.mkdir()
    write(old/'protocol.json', {'test_fixture': True})
    write(old/'CODE_MANIFEST.json', {'protocol.json': c.sha(old/'protocol.json')})
    source_manifest = c.sha(old/'CODE_MANIFEST.json')
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(source_root=str(old), source_manifest_sha256=source_manifest,
                    comparison_source_relative=str(HERE.parent/'gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py'))
    write(root/'protocol.json', protocol)
    for name in ('worker.py', 'local_skip.py', 'compare.py'):
        (root/name).write_text('# inert CPU fixture\n', encoding='utf-8')
    write(root/'CODE_MANIFEST.json', {name:c.sha(root/name) for name in ('worker.py','local_skip.py','compare.py','protocol.json')})
    starts = [[(4*i+j)%168 for j in range(4)] for i in range(600)]
    selection = dict(status='selected', selected_step=400, checkpoints={},
        code_manifest_sha256=source_manifest, protocol_sha256=c.sha(old/'protocol.json'),
        plan_prefix_sha256=c.digest(starts[:400]))
    for parent in ('z_delta','constant_delta'):
        path = old/(parent+'.pt'); path.write_bytes(('test source '+parent).encode())
        selection['checkpoints'][parent] = dict(path=str(path), sha256=c.sha(path))
    write(old/'SELECTION.json', selection)
    ref = old/'frozen.jsonl'; ref.write_text('test frozen reference\n', encoding='utf-8')
    sources = {}
    for arm in c.ARMS:
        parent = protocol['arms'][arm]['source_arm']; selected = selection['checkpoints'][parent]
        sources[arm] = dict(arm=arm, source_arm=parent, source_steps=400,
            source_checkpoint_path=selected['path'], source_checkpoint_sha256=selected['sha256'],
            source_manifest_sha256=source_manifest, source_selection_sha256=c.sha(old/'SELECTION.json'),
            code_manifest_sha256=c.sha(root/'CODE_MANIFEST.json'), protocol_sha256=c.sha(root/'protocol.json'),
            data_hash='a'*64, frozen_reference_sha256=c.sha(ref), frozen_reference_path=str(ref),
            frozen_encoder_sha256='b'*64, output_std_sha256='c'*64,
            loaded_model_sha256=c.digest(parent+'model'), loaded_codec_sha256=c.digest(parent+'codec'),
            zero_initial_projection_sha256='d'*64, source_binding={'test':'source'}, original_config={'test':'config'},
            fixed_null_key_bias={'test':'unchanged'}, local_parameter_names=['projection.weight'],
            local_skip_enabled=arm.startswith('local_'), delta_weight=1., goal_complete=False,
            full_plan_sha256=c.digest(starts), continuation_plan_sha256=c.digest(starts[400:]))
    preflight = dict(schema='gs_local_condition_preflight_v1', status='passed', gpu_witness=True,
        parameters_discarded=True, trial_optimizer_updates=4, production_optimizer_updates=0,
        code_manifest_sha256=c.sha(root/'CODE_MANIFEST.json'), protocol_sha256=c.sha(root/'protocol.json'), arms={})
    for arm, source in sources.items():
        preflight['arms'][arm] = dict(source, optimizer_updates=1, frozen_encoder_unchanged=True,
            all_initial_points_retained=True, projection_gradient_norm=1. if arm.startswith('local_') else 0.,
            projection_changed=arm.startswith('local_'), native_groups_updated=dict(native_out=True,native_heads=True,native_multiplier=True))
    write(root/'PREFLIGHT.json', preflight)
    for arm, source in sources.items():
        train_dir = root/'arms'/arm; eval_dir = root/'evaluations'/arm
        train_dir.mkdir(parents=True); eval_dir.mkdir(parents=True)
        write(train_dir/'SOURCE.json', dict(schema='gs_local_condition_source_v1', **source))
        write(train_dir/'PLAN.json', dict(schema='gs_local_condition_plan_v1', starts=starts, source_steps=400,
            continued_batch_indices=list(range(400,600)), full_plan_sha256=c.digest(starts), continuation_plan_sha256=c.digest(starts[400:])))
        logs = [dict(step=i+1, added_steps=i-399, plan_index=i, starts=starts[i],
                     **{k:source[k] for k in ('frozen_encoder_sha256','output_std_sha256','fixed_null_key_bias')}) for i in range(400,600)]
        (train_dir/'train.jsonl').write_text('\n'.join(json.dumps(row) for row in logs), encoding='utf-8')
        (train_dir/'step_0600.pt').write_bytes(('inert checkpoint '+arm).encode())
        done = dict(schema='gs_local_condition_train_v1', status='complete', **source,
            completed_steps=600, steps=600, additional_steps=200, original_config_preserved=True,
            encoder_unchanged=True, actual_wall_budget_seconds=1800, last=logs[-1],
            source_sha256=c.sha(train_dir/'SOURCE.json'), plan_sha256=c.sha(train_dir/'PLAN.json'),
            train_jsonl_sha256=c.sha(train_dir/'train.jsonl'), preflight_sha256=c.sha(root/'PREFLIGHT.json'),
            checkpoint_sha256={'600':c.sha(train_dir/'step_0600.pt')})
        write(train_dir/'TRAIN_DONE.json', done)
        rows = target_rows(arm); delta = c.delta_rows(rows)
        (eval_dir/'targets.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        initial_rows = [dict(start=s, end=s, horizon=0, **{ctrl:dict(rows[s*9][ctrl]) for ctrl in c.LIVE},
                             held_initial=None, frozen_target=None) for s in range(168)]
        write(eval_dir/'initial_reconstruction.json', dict(schema='gs_local_condition_initial_reconstruction_v1', rows=initial_rows,
            unique_initial_frames=168, uncomputed_controls=list(c.FROZEN),
            metrics={ctrl:{m:STATISTICS.reduce_rows(initial_rows,ctrl)['frame_uniform'][m] for m in c.INITIAL_METRICS} for ctrl in c.LIVE}))
        (eval_dir/'actions.pt').write_bytes(b'inert action witness')
        write(eval_dir/'snapshots.json', {'test_fixture': True})
        evaluation = dict(schema='gs_local_condition_evaluation_v1', status='complete', **source,
            completed_steps=600, steps=600, selected_step=600, checkpoint_step=600, additional_steps=200,
            optimizer_updates=0, train=done, windows=168, real_future_exposures=1512, unique_future_frames=176,
            physical_frames_with_some_reconstruction=177, all_initial_latent_points_retained=True,
            all_ten_frames=True, full_native_resolution=[252,448], uncomputed_delta_controls=list(c.FROZEN),
            metrics={ctrl:c.stored_aggregate(STATISTICS.reduce_rows(rows,ctrl)) for ctrl in c.CONTROLS},
            delta_metrics={ctrl:c.stored_aggregate(STATISTICS.reduce_rows(delta,ctrl),True) if ctrl in c.LIVE else None for ctrl in c.CONTROLS})
        for key, path in dict(source_record_sha256=train_dir/'SOURCE.json',train_done_sha256=train_dir/'TRAIN_DONE.json',
            plan_sha256=train_dir/'PLAN.json',checkpoint_sha256=train_dir/'step_0600.pt',targets_sha256=eval_dir/'targets.jsonl',
            initial_reconstruction_sha256=eval_dir/'initial_reconstruction.json',actions_sha256=eval_dir/'actions.pt',snapshots_sha256=eval_dir/'snapshots.json').items():
            evaluation[key] = c.sha(path)
        write(eval_dir/'EVALUATION.json', evaluation)
    return source_manifest


class ComparisonTests(unittest.TestCase):
    def test_protocol_budget_cannot_be_relaxed(self):
        protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
        evidence=c.Evidence(); c.validate_protocol(protocol,evidence)
        self.assertEqual(evidence.issues,[])
        protocol['train']['additional_steps']=201
        c.validate_protocol(protocol,evidence)
        self.assertTrue(any('train' in x for x in evidence.issues))

    def test_physical_frames_and_block_statistics_are_original(self):
        rows = target_rows('base_z')
        for row in rows: row['correct']['mae'] = row['end']**2
        reduced = STATISTICS.reduce_rows(rows, 'correct')
        self.assertEqual(reduced['unique_target_frames'], 176)
        self.assertAlmostEqual(reduced['frame_uniform']['mae'], sum(i*i for i in range(1,177))/176)
        self.assertNotAlmostEqual(reduced['exposure_mean']['mae'], reduced['frame_uniform']['mae'])
        one = STATISTICS.moving_block_bootstrap({i:1. for i in range(1,177)})
        self.assertEqual((one['ci95_low'],one['ci95_high'],one['block_length'],one['replicates'],one['seed']), (1.,1.,16,2000,20261031))
        with self.assertRaises(ValueError): STATISTICS.moving_block_bootstrap({1:1.})

    def test_h0_and_delta_fail_closed(self):
        rows = target_rows('base_z'); rows[0]['delta']['held_initial'] = {'delta_mae':0}
        evidence = c.Evidence(); self.assertFalse(c.validate_rows(rows,evidence)); self.assertTrue(evidence.issues)
        evidence = c.Evidence(); self.assertIsNone(c.validate_initial({'rows':[]},STATISTICS,evidence))
        self.assertTrue(evidence.issues)

    def test_positive_aggregate_does_not_override_horizon_degradation(self):
        arms={}
        for name in c.ARMS:
            rows=target_rows(name)
            if name=='local_z':
                for row in rows:
                    if row['horizon']==9: row['correct']['mae']=.25
            arms[name]=dict(_rows=rows,_delta=c.delta_rows(rows),
                initial_reconstruction={'metrics': {'correct': {'frame_uniform': {'mae':.1,'hand_activity_mae':.1}}}})
        screens,comparisons,checks=c.paired_screens(arms,STATISTICS)
        self.assertTrue(comparisons['local_z_vs_shuffle42']['mae']['positive_interval'])
        self.assertFalse(screens['local']['passed'])
        self.assertTrue(any(x['horizon']==9 for x in screens['local']['horizon_degradations']))
        self.assertFalse(checks['matched_action_screen'])

    def test_complete_worker_contract_and_current_held_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ancestor_hash=fixture(root)
            with patch.object(c,'SOURCE_MANIFEST_SHA256',ancestor_hash): summary=c.build_summary(root)
            self.assertTrue(summary['comparison_eligible'], summary['issues'])
            self.assertEqual(summary['engineering_status'],'complete'); self.assertEqual(summary['selected_step'],600)
            self.assertFalse(summary['goal_complete']); self.assertTrue(summary['local_skip_screen_passed'])
            for screen in summary['z_contribution_screens'].values():
                self.assertTrue(screen['passed']); self.assertEqual(len(screen['requirements']),14)
                self.assertTrue(any('adapted_held' in r['comparison'] for r in screen['requirements']))
                self.assertFalse(any('frozen' in r['comparison'] for r in screen['requirements']))
            self.assertFalse(summary['comparisons']['local_z_vs_frozen_held_descriptive']['mae']['positive_interval'])
            self.assertIn('local_constant_vs_base_constant', summary['comparisons'])
            self.assertIn('h0',c.markdown(summary))

    def test_artifact_tamper_blocks_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ancestor_hash=fixture(root)
            (root/'evaluations/local_z/actions.pt').write_bytes(b'changed')
            with patch.object(c,'SOURCE_MANIFEST_SHA256',ancestor_hash): summary=c.build_summary(root)
            self.assertFalse(summary['comparison_eligible']); self.assertFalse(summary['goal_complete'])
            self.assertEqual(summary['comparisons'],{}); self.assertTrue(any('actions_sha256' in x for x in summary['issues']))

    def test_actual_update_evidence_and_failed_marker_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ancestor_hash=fixture(root); train=root/'arms/base_z'
            logs=(train/'train.jsonl').read_text().splitlines(); logs[-1]=logs[-2]
            (train/'train.jsonl').write_text('\n'.join(logs))
            write(root/'evaluations/local_constant/FAILED.json', {'status':'failed'})
            with patch.object(c,'SOURCE_MANIFEST_SHA256',ancestor_hash): summary=c.build_summary(root)
            self.assertFalse(summary['comparison_eligible']); self.assertEqual(summary['comparisons'],{})
            self.assertTrue(any('更新600' in x for x in summary['issues']))
            self.assertTrue(any('FAILED.json' in x for x in summary['issues']))

    def test_empty_root_still_produces_failure_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary=c.build_summary(Path(tmp))
            self.assertFalse(summary['comparison_eligible']); self.assertFalse(summary['goal_complete'])
            self.assertTrue(summary['issues']); json.dumps(summary,allow_nan=False)


if __name__ == '__main__': unittest.main()
