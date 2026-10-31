"""CPU-only protocol, coverage and null-guard integration checks; no runtime data."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import worker


class WorkerContracts(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent
        self.protocol = worker.read(self.root/'protocol.json')

    def test_fixed_protocol_and_rejections(self):
        worker.protocol_contract(self.protocol)
        for section,key,value in [('train','total_steps',599),('train','max_seconds',1801),
                                  ('preflight','starts',[0,42,84]),('resources','peak_limit_gib',49)]:
            bad=copy.deepcopy(self.protocol);bad[section][key]=value
            with self.assertRaises(RuntimeError):worker.protocol_contract(bad)
        bad=copy.deepcopy(self.protocol);bad['arms']['local_z']['source_arm']='constant_delta'
        with self.assertRaises(RuntimeError):worker.protocol_contract(bad)

    def test_manifest_allows_only_bounded_relative_paths(self):
        self.assertEqual(worker.manifest_path(self.root,'../sibling/file.py'),self.root.parent/'sibling/file.py')
        for name in ('../../escape.py',str(self.root/'absolute.py')):
            with self.assertRaises(RuntimeError):worker.manifest_path(self.root,name)

    def test_full_gradient_coverage_and_finite(self):
        row=dict(rgb_quantized_grad_per_sample_h=[[1.]*9 for _ in range(4)],
                 feature_grad_norm_per_sample_h=[[1.]*10 for _ in range(4)],
                 initial_rgb_grad_norm_per_sample=[1.]*4,rgb_query_grad_per_h=[1.]*9,auxiliary=0.)
        worker.check_gradients(row,True);worker.check_gradients(row,False)
        for key,value in [('rgb_quantized_grad_per_sample_h',[[1.]*8 for _ in range(4)]),
                          ('feature_grad_norm_per_sample_h',[[1.]*9 for _ in range(4)]),
                          ('rgb_query_grad_per_h',[1.]*8+[0.]),
                          ('initial_rgb_grad_norm_per_sample',[1.,1.,1.,float('nan')])]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(RuntimeError):worker.check_gradients(bad,True)

    def test_snapshot_known_failure_mapping_overrides_only_two(self):
        class Timing:
            @staticmethod
            def frame_mapping():
                return [dict(source_frame=f,start=0 if f==0 else min((f-1)//9*9,167),
                             horizon=f-(0 if f==0 else min((f-1)//9*9,167))) for f in range(177)]
        self.assertEqual(worker.snapshot_mapping({'timing':Timing}),
                         {0:(0,0),44:(35,9),88:(81,7),132:(123,9),176:(167,9)})

    def test_null_guard_rebinding_preserves_original_state(self):
        import torch
        from local_skip import attach_local_skip
        source=self.root.parent/'gaussiangpt_neoverse_step_acceleration_20261031_v2/null_key_bias.py'
        guard_module=worker.module_at('_worker_test_null_guard',source)
        model=torch.nn.Module()
        for name in guard_module.PARAMETER_NAMES:
            parent=model
            for part in name.split('.')[:-2]:
                if not hasattr(parent,part):parent.add_module(part,torch.nn.Module())
                parent=getattr(parent,part)
            parent.add_module('k_proj',torch.nn.Linear(64,32))
        model.decoder_backbone=torch.nn.Linear(64,64)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        guard=guard_module.freeze_null_key_bias(model,optimizer)
        original=guard_module.assert_null_key_bias_unchanged(model,optimizer,guard)
        members=[[id(p) for p in g['params']] for g in optimizer.param_groups]
        before=torch.get_rng_state().clone();added=attach_local_skip(model,True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        optimizer.add_param_group(dict(params=added,lr=.001))
        self.assertEqual(members,[[id(p) for p in g['params']] for g in optimizer.param_groups[:-1]])
        with self.assertRaises(RuntimeError):guard_module.assert_null_key_bias_unchanged(model,optimizer,guard)
        rebound=guard_module.freeze_null_key_bias(model,optimizer)
        self.assertEqual(original,guard_module.assert_null_key_bias_unchanged(model,optimizer,rebound))
        model.eval()
        self.assertEqual(original,guard_module.assert_null_key_bias_unchanged(model,optimizer,rebound))


if __name__=='__main__':unittest.main(verbosity=2)
