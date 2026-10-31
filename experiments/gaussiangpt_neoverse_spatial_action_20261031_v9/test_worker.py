"""CPU checks for spatial attachment/continuation gates; no native GS or GPU."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

import worker as w
from spatial_action import spatial_action_gradients


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64,64)

    def forward(self, x, coords, active, condition=None):
        return self.linear(x)


def toy():
    model = nn.Module()
    model.decoder_backbone = ToyBackbone()
    return model


def measured():
    return dict(rgb_quantized_grad_per_sample_h=[[1.]*9 for _ in range(4)],
                feature_grad_norm_per_sample_h=[[1.]*10 for _ in range(4)],
                initial_rgb_grad_norm_per_sample=[1.]*4, rgb_query_grad_per_h=[1.]*9, auxiliary=0.)


def synthetic_update(update, before='a', after='b'):
    gradients = {name: dict(requires_grad=True,has_grad=True,finite=True,norm=1.) for name in w.SPATIAL_SHAPES}
    record = w.spatial_update_evidence(gradients,dict.fromkeys(gradients,before),dict.fromkeys(gradients,after),update)
    record.update(native_groups_updated=dict.fromkeys(('native_out','native_heads','native_multiplier'),True),
                  optimizer_before={'sha256':before},optimizer_after={'sha256':after})
    return record


class SpatialWorkerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261031)
        self.root = Path(__file__).resolve().parent
        self.protocol = w.read(self.root/'protocol.json')

    def test_only_two_spatial_arms_and_exact_continuation(self):
        w.protocol_contract(self.protocol)
        for section,key,value in [('train','source_step',399),('train','additional_steps',199),
                                  ('preflight','updates_per_arm',1),('preflight','starts',[0]),
                                  ('reference_baselines','step',500)]:
            bad=deepcopy(self.protocol);bad[section][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):w.protocol_contract(bad)
        bad=deepcopy(self.protocol);bad['arms']['base_z']={'source_arm':'z_delta','spatial_action_enabled':False}
        with self.assertRaises(RuntimeError):w.protocol_contract(bad)
        bad=deepcopy(self.protocol);bad['arms']['spatial_z']['spatial_action_enabled']=False
        with self.assertRaises(RuntimeError):w.protocol_contract(bad)

    def test_append_preserves_old_optimizer_members_moments_order_and_rng(self):
        model=toy();native=nn.Linear(64,64)
        opt=torch.optim.AdamW([{'params':list(model.parameters()),'lr':.003},
                               {'params':list(native.parameters()),'lr':.002}],lr=.001,betas=(.8,.9),weight_decay=.03)
        loss=sum(p.square().sum() for group in opt.param_groups for p in group['params'])
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        before=w.optimizer_evidence(opt);members=[[id(p) for p in g['params']] for g in opt.param_groups]
        rng=torch.get_rng_state().clone()
        extra,evidence=w.append_spatial_parameters(model,opt,.003)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(w.optimizer_evidence(opt,2),before)
        self.assertEqual(members,[[id(p) for p in g['params']] for g in opt.param_groups[:-1]])
        self.assertEqual(len(opt.param_groups),3);self.assertEqual(len(extra),8)
        self.assertEqual(sum(p.numel() for _,p in extra),15040)
        self.assertEqual(evidence['original_groups_before'],evidence['original_groups_after'])
        self.assertEqual(opt.param_groups[-1]['betas'],(.8,.9))
        self.assertEqual(opt.param_groups[-1]['weight_decay'],.03)
        self.assertTrue(all(p not in opt.state for _,p in extra))

    def test_two_discarded_toy_b4_steps_expose_all_new_groups(self):
        model=toy();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        extra,_=w.append_spatial_parameters(model,opt,.001)
        inputs=[torch.randn(13,5,64) for _ in range(4)]
        targets=[torch.randn(9,5,64) for _ in range(4)]
        coords=torch.arange(15).reshape(5,3);active=torch.ones(13,5,dtype=torch.bool)
        records=[]
        for update in (1,2):
            before=w.spatial_parameter_hashes(extra);opt.zero_grad(set_to_none=True)
            for x,target in zip(inputs,targets):
                prediction=model.decoder_backbone(x,coords,active)
                ((prediction[4:]-target).square().mean()/4).backward()
            gradients=spatial_action_gradients(model);opt.step()
            record=w.spatial_update_evidence(gradients,before,w.spatial_parameter_hashes(extra),update)
            records.append(record)
        self.assertGreater(records[0]['groups']['Wo']['gradient_norm'],0)
        self.assertTrue(all(records[0]['groups'][g]['gradient_norm']==0 for g in w.SPATIAL_GROUPS if g!='Wo'))
        self.assertTrue(all(row['gradient_norm']>0 and row['changed'] for row in records[1]['groups'].values()))
        self.assertEqual(records[0]['parameter_sha256_after'],records[1]['parameter_sha256_before'])

    def test_disconnected_nonfinite_or_unchanged_second_step_is_rejected(self):
        record=synthetic_update(2)
        for name in w.SPATIAL_GROUPS:
            bad=deepcopy(record['gradients'])
            for parameter in bad:
                if parameter.split('.')[0]==name:bad[parameter]['norm']=0.
            with self.subTest(group=name),self.assertRaises(RuntimeError):
                w.spatial_update_evidence(bad,record['parameter_sha256_before'],record['parameter_sha256_after'],2)
        for key,value in [('has_grad',False),('finite',False),('norm',float('nan'))]:
            bad=deepcopy(record['gradients']);bad['Wf.weight'][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                w.spatial_update_evidence(bad,record['parameter_sha256_before'],record['parameter_sha256_after'],2)
        with self.assertRaises(RuntimeError):
            w.spatial_update_evidence(record['gradients'],record['parameter_sha256_before'],record['parameter_sha256_before'],2)

    def test_null_bias_guard_rebind_retains_all27_values_and_old_optimizer(self):
        path=self.root.parent/'gaussiangpt_neoverse_step_acceleration_20261031_v2/null_key_bias.py'
        guard=w.module_at('_spatial_worker_test_null_guard',path)
        model=toy()
        for name in guard.PARAMETER_NAMES:
            parent=model
            for part in name.split('.')[:-2]:
                if not hasattr(parent,part):parent.add_module(part,nn.Module())
                parent=getattr(parent,part)
            parent.add_module('k_proj',nn.Linear(64,32))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        fixed=guard.freeze_null_key_bias(model,optimizer)
        original=guard.assert_null_key_bias_unchanged(model,optimizer,fixed)
        w.append_spatial_parameters(model,optimizer,.001)
        rebound=guard.freeze_null_key_bias(model,optimizer)
        self.assertEqual(original,guard.assert_null_key_bias_unchanged(model,optimizer,rebound))
        named=dict(model.named_parameters())
        self.assertTrue(all(not named[name].requires_grad for name in guard.PARAMETER_NAMES))

    def gate_fixture(self,root):
        e={'root':root,'manifest_hash':'manifest','protocol_hash':'protocol','protocol':self.protocol,
           'context':{'arms':{parent:{'evaluation':{'checkpoint_sha256':parent}}
                              for parent in ('z_delta','constant_delta')}}}
        arm_results={}
        for arm in w.ARMS:
            arm_results[arm]=dict(source_checkpoint_sha256=self.protocol['arms'][arm]['source_arm'],
                optimizer_updates=2,frozen_encoder_unchanged=True,
                native_groups_updated=dict.fromkeys(('native_out','native_heads','native_multiplier'),True),
                parity=[dict(start=s,horizon=h,zero_Wo_original_function_exact=True) for s in (35,123) for h in range(10)],
                all_initial_points_retained=True,frames_per_window=10,starts=[0,42,84,126],
                optimizer_attachment=dict(original_groups_before={},original_groups_after={},
                    original_membership_and_order_preserved=True,added_state_initially_empty=True),
                measurements=[measured(),measured()],spatial_updates=[synthetic_update(1,'a','b'),synthetic_update(2,'b','c')])
        receipt=dict(schema='gs_spatial_action_preflight_v1',status='passed',gpu_witness=True,
            parameters_discarded=True,code_manifest_sha256='manifest',protocol_sha256='protocol',
            trial_optimizer_updates=4,production_optimizer_updates=0,arms=arm_results)
        (root/'preflight').mkdir()
        return e,receipt

    def save_gate(self,root,value):
        # Fixtures deliberately replace only their own temporary test receipts.
        import json
        content=json.dumps(value)
        (root/'PREFLIGHT.json').write_text(content,encoding='utf-8')
        (root/'preflight/RESULT.json').write_text(content,encoding='utf-8')

    def test_incomplete_preflight_or_missing_second_update_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e,receipt=self.gate_fixture(root);self.save_gate(root,receipt)
            w.witness(e)
            for key,value in [('trial_optimizer_updates',2),('production_optimizer_updates',1),('status','failed')]:
                bad=deepcopy(receipt);bad[key]=value;self.save_gate(root,bad)
                with self.subTest(key=key),self.assertRaises(RuntimeError):w.witness(e)
            bad=deepcopy(receipt);bad['arms']['spatial_z']['spatial_updates'].pop();self.save_gate(root,bad)
            with self.assertRaises(RuntimeError):w.witness(e)
            self.save_gate(root,receipt);(root/'preflight/FAILED.json').write_text('{}')
            with self.assertRaises(RuntimeError):w.witness(e)

    def test_any_preflight_artifact_forbids_in_place_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w.assert_fresh_preflight(root)
            for name in ('PREFLIGHT.json','PREFLIGHT.json.tmp','PREFLIGHT_spatial_z.json'):
                path=root/name;path.write_text('{}')
                with self.subTest(name=name),self.assertRaises(RuntimeError):w.assert_fresh_preflight(root)
                path.unlink()
            (root/'preflight').mkdir()
            with self.assertRaises(RuntimeError):w.assert_fresh_preflight(root)

    def test_fixed_known_failure_screens_and_preserved_original_gradient_gate(self):
        class Timing:
            @staticmethod
            def frame_mapping():
                return [dict(source_frame=f,start=0 if f==0 else min((f-1)//9*9,167),
                             horizon=f-(0 if f==0 else min((f-1)//9*9,167))) for f in range(177)]
        mapping=w.snapshot_mapping({'timing':Timing})
        self.assertEqual(mapping[44],(35,9));self.assertEqual(mapping[132],(123,9))
        w.check_gradients(measured(),True);w.check_gradients(measured(),False)
        bad=measured();bad['rgb_query_grad_per_h'][0]=0.
        with self.assertRaises(RuntimeError):w.check_gradients(bad,True)


if __name__=='__main__':unittest.main(verbosity=2)
