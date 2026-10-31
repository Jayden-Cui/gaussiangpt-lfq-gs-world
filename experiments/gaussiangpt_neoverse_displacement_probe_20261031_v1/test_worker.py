"""Numerical/serialization CPU checks; no remote calls, native codec or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import worker as w


class DisplacementProbeTests(unittest.TestCase):
    def make_gs(self, count=7):
        g=torch.zeros(count,14,dtype=torch.float32)
        g[:,:3]=torch.arange(count*3).reshape(count,3)/100
        g[:,3:6]=.4;g[:,6:9]=.01;g[:,9]=1;g[:,13]=.5
        return g

    def packet(self):
        gs=self.make_gs();sh=torch.arange(7*12,dtype=torch.float32).reshape(7,4,3)/100
        record=dict(cameras=torch.eye(4).repeat(2,1,1),intrinsics=torch.tensor([.8,.8,.5,.5]),near=.001,far=100.)
        record['cameras'][1,0,3]=.02
        return w.make_packet(gs,sh,record,35,9,44/29.97,dict(scale=.5),dict(checkpoint='test'))

    def test_import_is_cpu_only(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_shared_memory_admission_is_real_and_does_not_initialize_cuda_in_test(self):
        p=json.loads((Path(__file__).parent/'protocol.json').read_text())
        with mock.patch.object(torch.cuda,'set_device'),mock.patch.object(torch.cuda,'mem_get_info',return_value=(21*2**30,80*2**30)),\
             mock.patch.object(torch.cuda,'set_per_process_memory_fraction') as set_fraction,mock.patch.object(torch.cuda,'reset_peak_memory_stats'):
            admitted=w.admit_memory(p)
            set_fraction.assert_called_once_with(.2)
            self.assertEqual(admitted['free_gib'],21.)
            self.assertTrue(admitted['source_checkpoint_config_unchanged'])
        with mock.patch.object(torch.cuda,'set_device'),mock.patch.object(torch.cuda,'mem_get_info',return_value=(19*2**30,80*2**30)),\
             mock.patch.object(torch.cuda,'set_per_process_memory_fraction') as set_fraction:
            with self.assertRaises(RuntimeError):w.admit_memory(p)
            set_fraction.assert_not_called()

    def test_reserved_peak_alone_blocks_even_when_allocated_is_low(self):
        budget=object.__new__(w.Budget);budget.started=w.time.monotonic();budget.seconds=10
        with mock.patch.object(torch.cuda,'max_memory_allocated',return_value=2*2**30),\
             mock.patch.object(torch.cuda,'max_memory_reserved',return_value=17*2**30):
            with self.assertRaises(RuntimeError):budget.check()

    def test_xyz_gradient_reaches_positions_without_touching_attributes(self):
        gs=self.make_gs();original=gs.clone();d=torch.nn.Parameter(torch.zeros(7,3))
        moved=w.move_means(gs,d);torch.testing.assert_close(moved,gs,rtol=0,atol=0)
        target=gs[:,:3]+torch.tensor([.1,-.2,.3])
        (moved[:,:3]-target).square().sum().backward()
        torch.testing.assert_close(d.grad,2*(gs[:,:3]-target))
        self.assertTrue(torch.equal(gs,original));self.assertTrue(torch.equal(moved[:,3:],gs[:,3:]))
        d.data.fill_(12.)
        moved=w.move_means(gs,d)
        self.assertGreater(float(moved[:,0].detach().max()),12.)
        self.assertTrue(torch.equal(moved[:,3:],original[:,3:]))

    def test_xyz_rejects_dropped_rows(self):
        with self.assertRaises(RuntimeError):w.move_means(self.make_gs(),torch.zeros(6,3))

    def test_feature_units_preserve_zero_channels_and_h0(self):
        initial=torch.randn(5,64,dtype=torch.float64);unit=torch.rand(64,dtype=torch.float64);unit[3]=0
        residual=torch.randn(9,5,64,dtype=torch.float64).tanh();features=initial[None]+unit*residual
        recovered=w.normalized_residual(features,initial,unit)
        torch.testing.assert_close(initial[None]+unit*recovered,features,rtol=1e-12,atol=1e-12)
        self.assertTrue(bool((recovered[:,:,3]==0).all()))
        bad=features.clone();bad[2,1,3]+=1e-5
        with self.assertRaises(RuntimeError):w.normalized_residual(bad,initial,unit)
        with self.assertRaises(RuntimeError):w.normalized_residual(initial[None]+2*unit,initial,unit)

    def test_ten_frame_loss_matches_direct_v9_reduction_and_gradient(self):
        torch.manual_seed(19)
        prediction=torch.rand(9,6,7,3,dtype=torch.float64,requires_grad=True)
        replica=prediction.detach().clone().requires_grad_(True)
        fixed0=torch.rand(6,7,3,dtype=torch.float64);truth=torch.rand(10,6,7,3,dtype=torch.float64)
        roi=torch.zeros(6,7,dtype=torch.bool);roi[1:5,2:6]=True
        weights=((truth[1:]-truth[:1]).abs().mean(-1)/.1).clamp(0,1)
        total=0.
        for h in range(1,10):
            loss=w.frame_loss(prediction[h-1],fixed0,truth[h],truth[0],roi,weights[h-1],h)
            total+=float(loss.detach());loss.backward()
        error=(replica-truth[1:]).abs().mean(-1)
        delta=((replica-fixed0)-(truth[1:]-truth[0])).abs().mean(-1)
        direct=error.mean((-1,-2)).sum()/10+error[:,roi].mean(-1).sum()/10
        direct+=((error*weights).sum((-1,-2))/weights.sum((-1,-2)).clamp_min(1e-6)).mean()
        direct+=delta.mean()+delta[:,roi].mean()
        direct.backward()
        self.assertAlmostEqual(total,float(direct.detach()),places=12)
        torch.testing.assert_close(prediction.grad,replica.grad,rtol=1e-12,atol=1e-12)
        self.assertIsNone(fixed0.grad)

    def test_zero_motion_weight_does_not_make_nan(self):
        image=torch.rand(3,4,3,requires_grad=True);roi=torch.ones(3,4,dtype=torch.bool)
        loss=w.frame_loss(image,image.detach(),torch.zeros_like(image),torch.zeros_like(image),roi,torch.zeros(3,4),1)
        loss.backward();self.assertTrue(bool(torch.isfinite(image.grad).all()))

    def test_independent_disk_packet_preserves_all_attributes_and_sh(self):
        packet=self.packet();original={k:v.clone() for k,v in packet.items() if torch.is_tensor(v)}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packet.pt';w.save_tensor(path,packet)
            del packet
            reloaded=torch.load(path,map_location='cpu',weights_only=True);w.validate_packet(reloaded)
            for k,v in original.items():self.assertTrue(torch.equal(v,reloaded[k]))
            with self.assertRaises(FileExistsError):w.save_tensor(path,reloaded)
        self.assertEqual(reloaded['frame'],44)
        self.assertNotIn('displacement',reloaded)

    def test_packet_rejects_missing_sh_nonfinite_and_wrong_mapping(self):
        for mutate in (lambda p:p.update(sh=p['sh'][:-1]),lambda p:p['gs'].__setitem__((0,0),float('nan')),
                       lambda p:p.update(frame=45),lambda p:p.update(resolution_hw=[126,224]),
                       lambda p:p['gs'].__setitem__((slice(None),slice(9,13)),0)):
            packet=self.packet();mutate(packet)
            with self.assertRaises(RuntimeError):w.validate_packet(packet)

    def test_replay_uses_both_saved_cameras_and_sh(self):
        packet=self.packet();calls=[]
        def renderer(gs,camera,intrinsics,near,far,sh):
            calls.append((camera.clone(),sh.clone()))
            image=torch.ones(252,448,3)*(gs[:,:3].sum()+sh.sum()+camera[0,3])
            return image,torch.ones(252,448,1)
        actual=w.render_pair(packet['gs'],packet['sh'],packet,renderer)
        self.assertEqual(len(calls),2);self.assertTrue(torch.equal(calls[0][1],packet['sh']))
        self.assertNotEqual(float(actual[0,0,0,0]),float(actual[1,0,0,0]))
        w.compare_render(actual,actual.clone(),dict(max_abs=1e-6,mae=1e-7))
        corrupted=actual.clone();corrupted[1,4,5,2]+=1e-3
        with self.assertRaises(RuntimeError):w.compare_render(corrupted,actual,dict(max_abs=1e-6,mae=1e-7))

    def test_h0_outputs_do_not_reference_oracle_variable(self):
        baseline=[(self.make_gs(),torch.zeros(7,4,3)) for _ in range(10)]
        variables=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(7,3)) for _ in range(9)])
        output=w.outputs_for({},dict(baseline=baseline),'xyz_only',variables)
        self.assertIs(output(0)[0],baseline[0][0]);self.assertFalse(output(0)[0].requires_grad)
        output(5)[0][:,:3].sum().backward()
        self.assertIsNone(variables[0].grad);self.assertIsNotNone(variables[4].grad)

    def test_parity_needs_all_ten_frames(self):
        rows=[dict(mae=.1,psnr=20.) for _ in range(10)]
        w.parity(rows,rows,dict(mae=1e-4,psnr=.02))
        with self.assertRaises(RuntimeError):w.parity(rows[:-1],rows,dict(mae=1e-4,psnr=.02))
        bad=copy.deepcopy(rows);bad[9]['mae']+=.001
        with self.assertRaises(RuntimeError):w.parity(bad,rows,dict(mae=1e-4,psnr=.02))

    def test_scope_protocol_rejects_smaller_or_reinterpreted_probe(self):
        p=json.loads((Path(__file__).parent/'protocol.json').read_text())
        w.protocol_contract(p)
        for section,key,value in [('scope','source_frames',10),('scope','frames_per_window',5),
            ('optimization','steps',50),('optimization','xyz_lr',.01),('scope','h0_frozen',False)]:
            changed=copy.deepcopy(p);changed[section][key]=value
            with self.assertRaises(RuntimeError):w.protocol_contract(changed)

    def test_output_paths_cannot_escape_and_writes_never_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):w.owned_path(root,'../old_run/FAILED.json')
            path=w.owned_path(root,'new.json');w.write_new(path,dict(value=1))
            with self.assertRaises(FileExistsError):w.write_new(path,dict(value=2))
            self.assertEqual(w.read(path),dict(value=1))

    def test_frozen_parameters_and_buffers_are_not_accidentally_trainable(self):
        module=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)).eval().requires_grad_(False)
        w.assert_frozen(module)
        module[0].weight.grad=torch.zeros_like(module[0].weight)
        with self.assertRaises(RuntimeError):w.assert_frozen(module)


if __name__=='__main__':unittest.main()
