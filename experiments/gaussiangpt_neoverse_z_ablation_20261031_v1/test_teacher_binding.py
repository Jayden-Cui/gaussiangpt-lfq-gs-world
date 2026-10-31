"""Synthetic raw-record identity checks; no renderer or training."""
import copy
import unittest

import torch
from codec_binding import CLIP
from teacher_binding import validate_raw


class TeacherBindingTest(unittest.TestCase):
    def setUp(self):
        self.row = dict(pts=1.0, normalization={'scale': .5})
        self.raw = dict(clip=CLIP, frame=30, normalization={'scale': .5}, n_input=112896,
                        metrics={'pts_seconds': 1.0}, near=.005, far=5e9,
                        cameras=torch.zeros(2,4,4), intrinsics=torch.ones(4),
                        targets=torch.zeros(2,252,448,3), real=torch.zeros(252,448,3),
                        counts=torch.tensor([112895,1]))

    def test_valid_raw_and_mismatches(self):
        self.assertIs(validate_raw(self.raw, 30, self.row), self.raw)
        for key, value in [('frame',31), ('n_input',112895), ('normalization',{'scale':1.0})]:
            altered = copy.deepcopy(self.raw)
            altered[key] = value
            with self.assertRaises(ValueError): validate_raw(altered, 30, self.row)
        altered = copy.deepcopy(self.raw)
        altered['metrics']['pts_seconds'] = 1.1
        with self.assertRaises(ValueError): validate_raw(altered, 30, self.row)
        altered = copy.deepcopy(self.raw)
        altered['cameras'] = altered['cameras'][:1]
        with self.assertRaises(ValueError): validate_raw(altered, 30, self.row)


if __name__ == '__main__': unittest.main()
