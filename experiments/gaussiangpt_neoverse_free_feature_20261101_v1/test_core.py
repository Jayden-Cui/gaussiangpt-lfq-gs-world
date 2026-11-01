import unittest
import numpy as np
import torch
from core import future_features, object_proxy, coordinate_overlap


class DiagnosticContracts(unittest.TestCase):
    def test_equal_start_gradient_but_different_reachable_ranges(self):
        initial = torch.randn(3, 4)
        std = torch.tensor([2., 0., 1., 3.])
        for mode in ('bounded', 'unbounded'):
            r = torch.zeros_like(initial, requires_grad=True)
            f = future_features(initial, r, std, mode)
            self.assertTrue(torch.equal(f, initial))
            f.sum().backward()
            expected = torch.ones_like(initial)
            if mode == 'bounded': expected[:, 1] = 0
            torch.testing.assert_close(r.grad, expected)
        r = torch.ones_like(initial) * 100
        bounded = future_features(initial, r, std, 'bounded') - initial
        self.assertTrue(bool((bounded.abs() <= 3*std + 1e-5).all()))
        self.assertTrue(bool(((future_features(initial,r,std,'unbounded')-initial) > 3*std).all()))

    def test_color_proxy_detects_translation_and_rotation_and_empty_mask(self):
        target = np.zeros((50,50,3),dtype=np.uint8)
        image = target.copy()
        target[5:35,10:15] = [230,200,0]
        image[30:35,15:45] = [230,200,0]
        spec = dict(hue_uint8=[28,52],saturation_min=90,value_min=35,roi_xyxy=[0,0,50,50])
        exact = object_proxy(target,target,spec)
        self.assertEqual(exact['iou'],1.)
        self.assertEqual(exact['centroid_error_px'],0.)
        changed = object_proxy(image,target,spec)
        self.assertGreater(changed['centroid_error_px'],10)
        self.assertAlmostEqual(changed['axis_error_degrees'],90.)
        empty = object_proxy(np.zeros_like(target),target,spec)
        self.assertFalse(empty['prediction']['valid'])
        self.assertIsNone(empty['axis_error_degrees'])

    def test_coordinate_overlap_is_order_invariant(self):
        a = torch.tensor([[0,0,0],[1,2,3]])
        b = torch.tensor([[1,2,3],[4,5,6]])
        self.assertEqual(coordinate_overlap(a,b),coordinate_overlap(a.flip(0),b.flip(0)))
        self.assertEqual(coordinate_overlap(a,b)['target_only_nodes'],1)


if __name__ == '__main__': unittest.main()
