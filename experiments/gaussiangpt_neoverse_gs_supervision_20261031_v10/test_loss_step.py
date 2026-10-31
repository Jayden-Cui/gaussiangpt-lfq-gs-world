"""CPU witnesses for the changed loss and complete staged image derivatives."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch
import loss_step

PATH=Path(__file__).resolve().parent.parent/'gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py'
SPEC=importlib.util.spec_from_file_location('_v10_test_bound_temporal',PATH)
TEMPORAL=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=TEMPORAL;SPEC.loader.exec_module(TEMPORAL)


class LossTests(unittest.TestCase):
    def test_zero_weights_exact_original_dispatch(self):
        temporal=Mock();temporal.step_native.return_value={'loss':3}
        result=loss_step.step_native('m','b','o','d',[0,1,2,3],'cfg',temporal,[0,0])
        temporal.step_native.assert_called_once_with('m','b','o','d',[0,1,2,3],'cfg',delta_weight=1.)
        self.assertEqual(result['loss'],3)

    def test_mask_uses_only_valid_nearby_pixels(self):
        image=torch.ones(2,3,4,3,requires_grad=True);target=torch.zeros_like(image)
        mask=torch.zeros(3,4,dtype=torch.bool);mask[1,2]=True
        loss,_=loss_step.teacher_loss(image,target,[0,.25],mask)
        loss.backward();self.assertAlmostEqual(float(loss.detach()),.025)
        self.assertEqual(float(image.grad[0].abs().sum()),0)
        self.assertEqual(float(image.grad[1][~mask].abs().sum()),0)
        self.assertAlmostEqual(float(image.grad[1][mask].sum()),.025)

    def test_invalid_pixels_do_not_change_value(self):
        image=torch.zeros(2,3,4,3);target=torch.zeros_like(image);mask=torch.zeros(3,4,dtype=torch.bool);mask[0,0]=True
        image[1][~mask]=100.;image[0]=500.
        loss,_=loss_step.teacher_loss(image,target,[0,.25],mask)
        self.assertEqual(float(loss),0)

    def test_empty_or_malformed_mask_rejected(self):
        image=torch.ones(2,3,4,3);target=torch.zeros_like(image)
        for mask in (torch.zeros(3,4,dtype=torch.bool),torch.ones(3,4),torch.ones(4,3,dtype=torch.bool)):
            with self.assertRaises(ValueError):loss_step.teacher_loss(image,target,[0,.25],mask)

    def test_teacher_target_cannot_receive_gradients(self):
        image=torch.ones(2,3,4,3);target=torch.zeros_like(image,requires_grad=True)
        with self.assertRaises(ValueError):loss_step.teacher_loss(image,target,[0,.25],torch.ones(3,4,dtype=torch.bool))

    def test_main_teacher_and_reduced_window_forbidden(self):
        image=torch.ones(2,3,4,3);target=torch.zeros_like(image);mask=torch.ones(3,4,dtype=torch.bool)
        with self.assertRaises(ValueError):loss_step.teacher_loss(image,target,[.5,.25],mask)
        with self.assertRaises(ValueError):loss_step.teacher_loss(image,target,[0,.25],mask,length=9)

    def test_bad_weights_rejected(self):
        for weights in ([0],[-1,0],[0,float('inf')],[0,float('nan')]):
            with self.assertRaises(ValueError):loss_step.checked_weights(weights)

    def _grads(self,staged):
        torch.manual_seed(901)
        dtype=torch.float64;B,T=2,10
        upstream=torch.randn(3,dtype=dtype,requires_grad=True)
        native=torch.randn(2,2,3,3,3,dtype=dtype,requires_grad=True)
        feature_seed=torch.randn(B,T,3,dtype=dtype)
        features=[[upstream*feature_seed[b,h] for h in range(T)] for b in range(B)]
        targets=torch.rand(B,T,2,2,3,3,dtype=dtype)
        mask=torch.tensor([[True,False,True],[False,True,True]])
        def render(b,h,f):return (native*f).sum(-1).sigmoid()
        def base(b,h,image):
            original=(image[0]-targets[b,h,0]).square().mean()/T
            teacher,_=loss_step.teacher_loss(image,targets[b,h],[0,.25],mask)
            return original+teacher
        def delta(b,h,image,initial):
            return ((image[0]-initial[0])-(targets[b,h,0]-targets[b,0,0])).square().mean()
        auxiliary=upstream.square().mean()*.01
        if staged:
            result=TEMPORAL.staged_temporal_backward(features,render,base,delta,auxiliary=auxiliary)
            self.assertEqual(len(result.initial_rgb_gradients),B)
            self.assertTrue(all(float(g[1].norm())>0 for g in result.initial_rgb_gradients))
        else:
            loss=auxiliary
            for b in range(B):
                images=[render(b,h,features[b][h]) for h in range(T)]
                loss=loss+sum(base(b,h,images[h])/B for h in range(T))
                loss=loss+sum(delta(b,h,images[h],images[0])/(B*(T-1)) for h in range(1,T))
            loss.backward()
        return upstream.grad,native.grad

    def test_dual_view_staged_gradients_match_full_graph(self):
        expected=self._grads(False);actual=self._grads(True)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main(verbosity=2)
