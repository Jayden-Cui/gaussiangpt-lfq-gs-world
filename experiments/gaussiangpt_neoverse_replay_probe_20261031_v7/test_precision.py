"""Real torch CPU regression tests; never import MinkowskiEngine or invoke CUDA."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

import precision


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = nn.Module()
        self.ae.encoder = nn.Sequential(nn.Linear(3, 64), nn.BatchNorm1d(64))
        self.ae.decoder = nn.Sequential(nn.Linear(64, 8), nn.BatchNorm1d(8))
        self.ae.feature_heads = nn.ModuleDict({'rgb': nn.Linear(8, 3)})
        self.ae.gaussian_multiplier = nn.Linear(8, 16)
        self.set_encoder = nn.Linear(5, 64)
        self.register_buffer('global_counter', torch.tensor(7, dtype=torch.int64))
        self.ae.encoder.register_buffer('observed_count', torch.tensor([3.], dtype=torch.float32))
        self.ae.decoder.register_buffer('decoder_count', torch.tensor(11, dtype=torch.int64))
        self.ae.encoder.register_buffer('nonpersistent_stat', torch.tensor([-.25]), persistent=False)
        self.ae.decoder.register_buffer('signed_zeros', torch.tensor([0., -0.], dtype=torch.float32))
        self.eval()


def tensor_signature(tensor):
    t = tensor.detach().contiguous()
    return str(t.dtype), tuple(t.shape), t.reshape(-1).view(torch.uint8).numpy().tobytes()


def all_state(model):
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


class FakeSparseTensor:
    # This intentionally accepts no coordinates argument: revoxelization is not
    # part of a precision cast and must not silently enter the constructor call.
    def __init__(self, *, features, coordinate_map_key, coordinate_manager):
        self.F = features
        self.coordinate_map_key = coordinate_map_key
        self.coordinate_manager = coordinate_manager


def sparse_fixture():
    return SimpleNamespace(F=torch.arange(192, dtype=torch.float32).reshape(3, 64) / 64,
                           coordinate_map_key=object(), coordinate_manager=object())


class DecoderPrecisionTests(unittest.TestCase):
    def test_precision_module_import_does_not_load_torch_or_me(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys, precision; assert 'torch' not in sys.modules; assert 'MinkowskiEngine' not in sys.modules"],
            cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_double_castback_is_exact_and_all_other_parameters_buffers_are_unchanged(self):
        model = TinyModel()
        original = {name: value.detach().clone() for name, value in all_state(model).items()}
        unselected_objects = {name: value for name, value in all_state(model).items() if not precision.selected(name)}
        adapter = precision.DecoderPrecision(model, 'float64')
        self.assertEqual(adapter.decoder_precision, 'float64')
        for name, value in all_state(model).items():
            if precision.selected(name):
                self.assertEqual(value.dtype, torch.float64 if original[name].is_floating_point() else original[name].dtype)
                self.assertEqual(tensor_signature(value.to(original[name].dtype)), tensor_signature(original[name]))
            else:
                self.assertIs(value, unselected_objects[name])
                self.assertEqual(tensor_signature(value), tensor_signature(original[name]))
        self.assertTrue(adapter.report['castback_exact'])
        self.assertTrue(adapter.report['encoder_exact'])
        self.assertTrue(adapter.report['nonselected_state_exact'])
        self.assertIn('ae.encoder.nonpersistent_stat', adapter.original_state)
        self.assertIn('torch.float64', adapter.report['mode_dtype_counts'])
        self.assertIn('torch.float32', adapter.report['mode_dtype_counts'])
        self.assertIn('torch.int64', adapter.report['mode_dtype_counts'])
        self.assertNotEqual(adapter.report['original_state_digest'], adapter.report['mode_state_digest'])
        self.assertTrue(adapter.assert_unchanged()['unchanged'])

    def test_float32_baseline_preserves_every_tensor_and_digest(self):
        model = TinyModel()
        original = {name: (value, tensor_signature(value)) for name, value in all_state(model).items()}
        adapter = precision.DecoderPrecision(model, 'float32')
        for name, value in all_state(model).items():
            self.assertIs(value, original[name][0])
            self.assertEqual(tensor_signature(value), original[name][1])
        self.assertEqual(adapter.report['original_state_digest'], adapter.report['mode_state_digest'])
        self.assertTrue(adapter.assert_unchanged()['unchanged'])

    def test_original_non_float32_floating_state_is_rejected_before_conversion(self):
        model = TinyModel()
        model.ae.encoder.observed_count = model.ae.encoder.observed_count.double()
        decoder_before = tensor_signature(model.ae.decoder[0].weight)
        with self.assertRaisesRegex(RuntimeError, 'Original floating model state must be float32'):
            precision.DecoderPrecision(model, 'float64')
        self.assertEqual(tensor_signature(model.ae.decoder[0].weight), decoder_before)

    def test_registered_state_changes_are_detected_in_each_mode(self):
        for mode in ('float32', 'float64'):
            for name in ('ae.decoder.0.weight', 'ae.encoder.1.running_mean', 'global_counter', 'ae.encoder.nonpersistent_stat'):
                with self.subTest(mode=mode, name=name):
                    model = TinyModel()
                    adapter = precision.DecoderPrecision(model, mode)
                    with torch.no_grad():
                        all_state(model)[name].add_(1)
                    with self.assertRaisesRegex(RuntimeError, 'Mode model state changed'):
                        adapter.assert_unchanged()

    def test_prepare_sparse_double_keeps_coordinates_manager_and_original_features(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float64')
        original = sparse_fixture()
        original_f = original.F
        signature = tensor_signature(original.F)
        prepared = adapter.prepare_sparse(original, SimpleNamespace(SparseTensor=FakeSparseTensor))
        self.assertIs(prepared.coordinate_map_key, original.coordinate_map_key)
        self.assertIs(prepared.coordinate_manager, original.coordinate_manager)
        self.assertIs(original.F, original_f)
        self.assertEqual(tensor_signature(original.F), signature)
        self.assertEqual(prepared.F.dtype, torch.float64)
        self.assertTrue(torch.equal(prepared.F, original.F.double()))
        self.assertNotEqual(prepared.F.data_ptr(), original.F.data_ptr())

    def test_prepare_sparse_float32_returns_original_without_constructor(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float32')
        original = sparse_fixture()

        def forbidden_constructor(**kwargs):
            raise AssertionError('Float32 baseline must not reconstruct sparse coordinates')

        self.assertIs(adapter.prepare_sparse(original, SimpleNamespace(SparseTensor=forbidden_constructor)), original)

    def test_prepare_sparse_rejects_changed_manager_or_map_key_identity(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float64')
        for changed in ('coordinate_map_key', 'coordinate_manager'):
            with self.subTest(changed=changed):
                def changed_constructor(**kwargs):
                    kwargs[changed] = object()
                    return FakeSparseTensor(**kwargs)
                with self.assertRaisesRegex(RuntimeError, 'coordinate key or manager identity'):
                    adapter.prepare_sparse(sparse_fixture(), SimpleNamespace(SparseTensor=changed_constructor))

    def test_prepare_sparse_rejects_wrong_dtype_width_and_nonfinite_values(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float64')
        for value in (torch.zeros(3, 64, dtype=torch.float64), torch.zeros(3, 63),
                      torch.full((3, 64), float('nan'))):
            with self.subTest(dtype=value.dtype, shape=tuple(value.shape)):
                original = sparse_fixture()
                original.F = value
                with self.assertRaises(RuntimeError):
                    adapter.prepare_sparse(original, SimpleNamespace(SparseTensor=FakeSparseTensor))

    def test_renderer_casts_float_tensors_and_preserves_integer_objects_and_input(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float64')
        points = {'coords': torch.tensor([[.125, .25, -.5]], dtype=torch.float64),
                  'sh': torch.ones(1, 9, dtype=torch.float32),
                  'batch': torch.zeros(1, dtype=torch.int32), 'counter': torch.tensor(3, dtype=torch.int64)}
        originals = {key: (value, tensor_signature(value)) for key, value in points.items()}
        rendered = adapter.to_render_points(points)
        self.assertIsNot(rendered, points)
        self.assertEqual(list(rendered), list(points))
        self.assertEqual(rendered['coords'].dtype, torch.float32)
        self.assertEqual(rendered['sh'].dtype, torch.float32)
        self.assertIs(rendered['batch'], points['batch'])
        self.assertIs(rendered['counter'], points['counter'])
        for key, value in points.items():
            self.assertIs(value, originals[key][0])
            self.assertEqual(tensor_signature(value), originals[key][1])

    def test_renderer_rejects_nonfinite_and_float32_overflow(self):
        adapter = precision.DecoderPrecision(TinyModel(), 'float64')
        for value in (float('nan'), float('inf'), 1e100):
            with self.subTest(value=value):
                points = {'coords': torch.tensor([[value, 0., 0.]], dtype=torch.float64)}
                original = tensor_signature(points['coords'])
                with self.assertRaises(RuntimeError):
                    adapter.to_render_points(points)
                self.assertEqual(tensor_signature(points['coords']), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
