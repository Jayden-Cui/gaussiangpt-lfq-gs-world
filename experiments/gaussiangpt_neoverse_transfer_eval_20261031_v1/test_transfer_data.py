"""CPU-only transfer metric, isolation and metadata rejection checks."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch

import transfer_data as data

torch.set_num_threads(1)


def normalization():
    return dict(center=[0., 0., 0.], scale=.5, source_frame=0, fixed_for_all_frames=True,
                uses_future_frames=False, changes_codec_output_std=False)


def fixture_data():
    value = data.TransferData.__new__(data.TransferData)
    value.torch = torch
    value.records = [{'real': torch.zeros(data.HEIGHT, data.WIDTH, 3)},
                     {'real': torch.full((data.HEIGHT, data.WIDTH, 3), .2)}]
    value.roi = torch.zeros((data.HEIGHT, data.WIDTH), dtype=torch.bool)
    x0, y0, x1, y1 = data.ROI_XYXY
    value.roi[y0:y1, x0:x1] = True
    return value


def frame_fixture():
    cache_hash, native_hash = 'a' * 64, 'b' * 64
    row = dict(latent_positions=1, input_voxels=1, cache_sha256=cache_hash,
               native_gs_sha256=native_hash, codec_real={'mae': 0., 'psnr': 120.})
    meta = dict(clip=data.CLIP, frame=0, pts=0., normalization=normalization(),
        source_cache_sha256=cache_hash, native_gs_sha256=native_hash,
        codec_checkpoint_sha256=data.CODEC_CHECKPOINT, input_binding_sha256=data.INPUT_BINDING)
    state = dict(schema='gsworld_continuous_state_v1', feature_dtype='float32',
        coordinates=torch.tensor([[1, 2, 3]], dtype=torch.int32), features=torch.ones(1, 64),
        stride=4, grid=.008, clip=data.CLIP, frame=0, pts=0., normalization=normalization(),
        source_cache_sha256=cache_hash, metadata=meta, baseline_real=row['codec_real'],
        reconstruction=torch.zeros(2, data.HEIGHT, data.WIDTH, 3))
    raw = dict(clip=data.CLIP, frame=0, pts=0., normalization=normalization(), n_input=112896,
        counts=torch.tensor([112896]), coordinates=torch.zeros(1, 3, dtype=torch.int32),
        real=torch.zeros(data.HEIGHT, data.WIDTH, 3), targets=torch.zeros(2, data.HEIGHT, data.WIDTH, 3),
        cameras=torch.eye(4).repeat(2, 1, 1), intrinsics=torch.ones(4), near=.01, far=1e10)
    value = fixture_data()
    value.context = dict(quality_rows=[row], pts=[0.], normalization=normalization())
    return value, state, raw


class TransferContracts(unittest.TestCase):
    def test_metric_math_and_static_weight_null(self):
        value = fixture_data()
        image = torch.zeros_like(value.records[0]['real'])
        result = value.metrics(image, 0, 1)
        self.assertEqual(set(result), set(data.METRICS))
        self.assertAlmostEqual(result['mae'], .2, places=6)
        self.assertAlmostEqual(result['workspace_roi_mae'], .2, places=6)
        self.assertAlmostEqual(result['motion_mae'], .2, places=6)
        self.assertAlmostEqual(result['psnr'], -10 * math.log10(.04), places=5)
        self.assertIsNone(value.metrics(image, 0, 0)['motion_mae'])
        value.records[1]['real'].zero_()
        self.assertIsNone(value.metrics(image + .2, 0, 1)['motion_mae'])
        with self.assertRaises(ValueError):
            value.metrics(image.half(), 0, 1)

    def test_delta_uses_own_h0(self):
        value = fixture_data()
        zero = value.records[0]['real']
        shifted_h0 = zero + .3
        shifted_h1 = value.records[1]['real'] + .3
        result = value.delta_metrics(shifted_h1, shifted_h0, 0, 1)
        self.assertLess(result['mae'], 1e-7)
        self.assertLess(result['motion_mae'], 1e-7)
        wrong_h0 = value.delta_metrics(shifted_h1, zero, 0, 1)
        self.assertAlmostEqual(wrong_h0['mae'], .3, places=6)
        static = value.delta_metrics(zero, zero, 0, 1)
        self.assertAlmostEqual(static['mae'], .2, places=6)

    def test_model_allow_list_preserves_geometry_and_excludes_targets(self):
        _, state, _ = frame_fixture()
        state.update(real=torch.zeros(1), targets=torch.ones(1), image='never',
                     auxiliary={'future_pixels': torch.ones(3)})
        model = data.model_state(state, 0, 'cpu')
        self.assertEqual(set(model), {'coordinates', 'features', 'pts', 'frame_index', 'stride', 'grid'})
        self.assertIs(model['coordinates'], state['coordinates'])
        self.assertIs(model['features'], state['features'])
        self.assertEqual(model['pts'], state['pts'])

    def test_frame_metadata_and_full_point_rejections(self):
        value, state, raw = frame_fixture()
        value._validate_frame(state, raw, 0)
        for name, bad in [('source_cache_sha256', 'c' * 64), ('codec_checkpoint_sha256', 'c' * 64),
                          ('input_binding_sha256', 'c' * 64), ('frame', 1)]:
            changed = dict(state, metadata=dict(state['metadata'], **{name: bad}))
            with self.assertRaises(ValueError):
                value._validate_frame(changed, raw, 0)
        with self.assertRaises(ValueError):
            value._validate_frame(state, dict(raw, counts=torch.tensor([1])), 0)
        with self.assertRaises(ValueError):
            value._validate_frame(dict(state, features=state['features'].half()), raw, 0)

    def test_no_future_normalization_or_frame_resampling(self):
        data.validate_normalization(normalization())
        for key, value in [('uses_future_frames', True), ('source_frame', 4), ('scale', .3),
                           ('center', [1, 0, 0]), ('changes_codec_output_std', True)]:
            with self.assertRaises(ValueError):
                data.validate_normalization(dict(normalization(), **{key: value}))
        pts = [frame / data.FPS for frame in range(177)]
        mapping = dict(clip=data.CLIP, source_sha256=data.SOURCE_VIDEO, source_frame_indices=list(range(177)),
            model_timestamps=list(range(177)), adjacent_pairs=[[t, t + 1] for t in range(176)], source_pts_seconds=pts)
        data.validate_mapping(mapping, pts)
        bad = copy.deepcopy(mapping)
        bad['source_frame_indices'][7] = 6
        with self.assertRaises(ValueError):
            data.validate_mapping(bad, pts)
        with self.assertRaises(ValueError):
            data.validate_mapping(mapping, pts[:-1])

    def test_missing_terminal_and_changed_input_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / data.PREPARATION_NAME
            root.mkdir()
            with self.assertRaises(FileNotFoundError):
                data.load_prepared(root)
            path = root / 'sample.bin'
            path.write_bytes(b'123')
            context = dict(root=root, hashes={str(path): data.sha(path)}, file_sizes={str(path): 3})
            data.verify_unchanged(context)
            path.write_bytes(b'124')
            with self.assertRaises(ValueError):
                data.verify_unchanged(context)


if __name__ == '__main__':
    unittest.main(verbosity=2)
