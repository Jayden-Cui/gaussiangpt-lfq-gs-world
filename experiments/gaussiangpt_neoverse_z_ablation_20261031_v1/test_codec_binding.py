"""Test identity, metadata promotion, and failure cases without GPU or source edits."""
import copy
import unittest

import torch
from codec_binding import CHECKPOINT_SHA, ENCODER_SHA, DECODER_SHA, CLIP, validate_state_view


class BindingViewTest(unittest.TestCase):
    def setUp(self):
        meta = dict(clip=CLIP, frame=44, arm='baseline', checkpoint_sha256=CHECKPOINT_SHA,
                    encoder_sha256=ENCODER_SHA, decoder_sha256=DECODER_SHA, source_cache_sha256='c'*64,
                    normalization={'scale': .5}, pts=1.5)
        self.state = dict(schema='multisource_continuous_encoder_state_v1', coordinates=torch.tensor([[0,0,0],[1,0,0]],dtype=torch.int32),
                          features=torch.ones(2,64), stride=4, grid=.008, metadata=meta)
        self.row = dict(**meta, state_sha256='a'*64, shape=[2,64], replay_passed=True,
                        cold_manager_replay_verified=True)

    def test_preserves_values_and_promotes_real_metadata(self):
        view = validate_state_view(self.state, self.row)
        self.assertEqual((view['frame_index'], view['clip'], view['pts']), (44, CLIP, 1.5))
        self.assertIs(view['coordinates'], self.state['coordinates'])
        self.assertIs(view['features'], self.state['features'])
        self.assertNotIn('pts', self.state)

    def test_rejects_wrong_identity_and_duplicate_coordinates(self):
        bad = copy.deepcopy(self.state)
        bad['metadata']['encoder_sha256'] = '0'*64
        with self.assertRaises(ValueError): validate_state_view(bad, self.row)
        bad = copy.deepcopy(self.state)
        bad['coordinates'][1] = bad['coordinates'][0]
        with self.assertRaises(ValueError): validate_state_view(bad, self.row)
        bad = copy.deepcopy(self.row)
        bad['pts'] += .1
        with self.assertRaises(ValueError): validate_state_view(self.state, bad)
        bad = copy.deepcopy(self.row)
        bad['replay_passed'] = False
        with self.assertRaises(ValueError): validate_state_view(self.state, bad)


if __name__ == '__main__': unittest.main()
