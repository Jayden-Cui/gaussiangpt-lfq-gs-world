import unittest
import torch
from teacher_assets import valid_mask

class TeacherMaskTests(unittest.TestCase):
    def test_holes_borders_and_detachment(self):
        alpha=torch.ones(9,9,requires_grad=True)
        with torch.no_grad(): alpha[4,4]=.89
        mask=valid_mask(alpha)
        expected=torch.ones(9,9,dtype=torch.bool)
        expected[[0,-1],:]=False; expected[:,[0,-1]]=False; expected[3:6,3:6]=False
        self.assertTrue(torch.equal(mask,expected)); self.assertFalse(mask.requires_grad)
    def test_alpha_threshold_is_inclusive(self):
        self.assertEqual(int(valid_mask(torch.full((5,5),.90)).sum()),9)
    def test_nan_rejected(self):
        with self.assertRaises(RuntimeError): valid_mask(torch.full((5,5),float('nan')))

if __name__=='__main__': unittest.main()
