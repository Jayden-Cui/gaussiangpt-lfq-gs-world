"""Protect action identity at the continuous/LFQ training boundary."""
import unittest
import torch
from paired_step import validate_encoded


class SimpleQuantizer:
    def indices_to_codes(self, ids, project_out=False):
        return torch.ones((*ids.shape,16))


class PairedBoundaryTest(unittest.TestCase):
    def test_continuous_actual_code_does_not_require_id(self):
        value = dict(action_mode='continuous16',action_codes=torch.randn(4,9,16),
                     z_quantized=torch.randn(4,9,64),indices=None,aux_loss=torch.tensor(.1))
        validate_encoded(value,'continuous16',4,SimpleQuantizer())
        value['indices'] = torch.zeros(4,9,dtype=torch.long)
        with self.assertRaises(ValueError): validate_encoded(value,'continuous16',4,SimpleQuantizer())

    def test_lfq_id_sign_must_agree(self):
        value = dict(action_mode='lfq16',action_codes=torch.ones(4,9,16),
                     z_quantized=torch.randn(4,9,64),indices=torch.zeros(4,9,dtype=torch.long),aux_loss=torch.tensor(.1))
        validate_encoded(value,'lfq16',4,SimpleQuantizer())
        value['action_codes'][0,0,0] = -1
        with self.assertRaises(ValueError): validate_encoded(value,'lfq16',4,SimpleQuantizer())


if __name__ == '__main__': unittest.main()
