"""CPU numerical/contract tests only; no torch, CUDA or ME imports."""
import copy
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import probe


def state():
    return {'coordinates': np.array([[4, 1, 2], [-2, 3, 4], [0, 0, 0]], dtype=np.int32),
            'features': np.arange(192, dtype=np.float32).reshape(3, 64), 'stride': 4, 'grid': .008}


def packet():
    keys = SimpleNamespace(**{k: k.lower() for k in
        ('COORDS', 'COORD_OFFSET', 'SCALES', 'QUATS', 'OPACITIES', 'SH0', 'SH', 'BATCH')})
    widths = {'coords': 3, 'coord_offset': 3, 'scales': 3, 'quats': 4,
              'opacities': 1, 'sh0': 3, 'sh': 9, 'embeddings': 64}
    result = {k: np.arange(3 * n, dtype=np.float32).reshape(3, n) for k, n in widths.items()}
    result['batch'] = np.zeros(3, dtype=np.int32)
    return result, keys


class BoundKeyTypes(str, Enum):
    """Values/types verified in vendor conf/dataclasses.py:28-39."""
    COORDS = 'coords'
    COORD_OFFSET = 'coord_offset'
    SCALES = 'scales'
    QUATS = 'quats'
    OPACITIES = 'opacities'
    SH0 = 'sh0'
    SH = 'sh'
    BATCH = 'batch'
    EMBEDDINGS = 'embeddings'


class ComparisonTests(unittest.TestCase):
    def test_bound_mixed_enum_and_string_keys_are_equivalent(self):
        raw, _ = packet()
        mixed = {(BoundKeyTypes(k) if k in ('coords', 'batch', 'embeddings') else k): v
                 for k, v in raw.items()}
        self.assertNotEqual(str(BoundKeyTypes.SH), 'sh')
        arrays, mapping = probe.gs_arrays(mixed, BoundKeyTypes)
        self.assertEqual(mapping['SH'], 'sh')
        self.assertEqual(mapping['BATCH'], 'batch')
        self.assertEqual(set(arrays), set(raw))
        self.assertTrue(all(np.array_equal(arrays[k], raw[k]) for k in raw))
        all_enums = {BoundKeyTypes(k): v for k, v in raw.items()}
        enum_arrays, enum_mapping = probe.gs_arrays(all_enums, BoundKeyTypes)
        comparison = probe.compare_gs(arrays, enum_arrays, mapping)
        self.assertEqual(mapping, enum_mapping)
        self.assertTrue(all(v['exact_equal'] for v in comparison['rowwise'].values()))

    def test_mixed_key_fix_keeps_batch_required(self):
        raw, _ = packet()
        del raw['batch']
        with self.assertRaisesRegex(RuntimeError, 'missing required attributes'):
            probe.gs_arrays(raw, BoundKeyTypes)

    def test_schema_metadata_records_types_without_tensor_values(self):
        raw, _ = packet()
        mixed = {(BoundKeyTypes(k) if k in ('coords', 'batch') else k): v for k, v in raw.items()}
        schema = probe.gs_schema(mixed)
        by_name = {r['normalized_key']: r for r in schema['attributes']}
        self.assertIn('BoundKeyTypes', by_name['coords']['key_type'])
        self.assertEqual(by_name['sh']['key_type'], 'builtins.str')
        self.assertEqual(by_name['sh']['shape'], [3, 9])
        self.assertEqual(by_name['sh']['dtype'], 'float32')
        self.assertNotIn('values', by_name['sh'])
        json.dumps(schema, allow_nan=False)

    def test_import_does_not_load_gpu_stack(self):
        command = [sys.executable, '-B', '-c',
                   "import sys, probe; assert not any(k in sys.modules for k in ['torch','MinkowskiEngine','worker','pool_index'])"]
        result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_state_permutation_preserves_canonical_pairing(self):
        a = state()
        b = {**a, 'coordinates': a['coordinates'][[2, 0, 1]], 'features': a['features'][[2, 0, 1]]}
        comparison = probe.compare_states(a, b)
        self.assertFalse(comparison['coordinates']['exact_equal'])
        self.assertFalse(comparison['features']['exact_equal'])
        self.assertTrue(comparison['canonical_coordinates']['exact_equal'])
        self.assertTrue(comparison['canonical_features']['exact_equal'])
        self.assertEqual(comparison['reference']['duplicate_coordinate_rows'], 0)

    def test_wrong_coordinate_feature_correspondence_is_detected(self):
        a = state()
        b = {**a, 'coordinates': a['coordinates'][[2, 0, 1]]}
        comparison = probe.compare_states(a, b)
        self.assertTrue(comparison['canonical_coordinates']['exact_equal'])
        self.assertFalse(comparison['canonical_features']['exact_equal'])

    def test_duplicate_coordinate_tie_is_order_invariant(self):
        a = state()
        a['coordinates'][1] = a['coordinates'][0]
        b = {**a, 'coordinates': a['coordinates'][[1, 2, 0]], 'features': a['features'][[1, 2, 0]]}
        comparison = probe.compare_states(a, b)
        self.assertTrue(comparison['canonical_features']['exact_equal'])
        self.assertEqual(comparison['reference']['duplicate_coordinate_rows'], 1)

    def test_state_contract_rejects_wrong_width_and_dtype(self):
        a = state()
        with self.assertRaises(RuntimeError):
            probe.state_summary(a['coordinates'], a['features'][:, :32])
        with self.assertRaises(RuntimeError):
            probe.state_summary(a['coordinates'].astype(np.float32), a['features'])

    def test_shape_mismatch_is_reported_without_broadcasting(self):
        d = probe.difference(np.zeros((3, 2)), np.zeros((2, 2)))
        self.assertFalse(d['same_shape'])
        self.assertFalse(d['exact_equal'])
        self.assertNotIn('mean_abs', d)

    def test_difference_records_known_perturbation(self):
        a = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        b = a.copy()
        b[1, 0] += .25
        d = probe.difference(a, b)
        self.assertEqual(d['changed_elements'], 1)
        self.assertEqual(d['max_abs'], .25)
        self.assertEqual(d['mean_abs'], .0625)
        self.assertEqual(d['rmse'], .125)

    def test_full_gs_dictionary_handles_batch_and_extra_embeddings(self):
        raw, keys = packet()
        a, mapping = probe.gs_arrays(raw, keys)
        b = {k: v[[2, 0, 1]] for k, v in a.items()}
        d = probe.compare_gs(a, b, mapping)
        self.assertTrue(d['count_equal'])
        self.assertTrue(d['canonical_anchor_exact_equal'])
        self.assertTrue(d['canonical_attribute_correspondence_valid'])
        self.assertTrue(all(v['exact_equal'] for v in d['canonical'].values()))
        self.assertFalse(d['rowwise']['coords']['exact_equal'])
        self.assertIn('embeddings', d['canonical'])
        b['sh'][0, 0] += .5
        self.assertFalse(probe.compare_gs(a, b, mapping)['canonical']['sh']['exact_equal'])

    def test_gs_coordinate_drift_does_not_claim_attribute_alignment(self):
        raw, keys = packet()
        a, mapping = probe.gs_arrays(raw, keys)
        b = {k: v.copy() for k, v in a.items()}
        b['coords'][1, 0] += .001
        d = probe.compare_gs(a, b, mapping)
        self.assertFalse(d['canonical_anchor_exact_equal'])
        self.assertFalse(d['canonical_attribute_correspondence_valid'])

    def test_gs_count_mismatch_is_reported(self):
        raw, keys = packet()
        a, mapping = probe.gs_arrays(raw, keys)
        d = probe.compare_gs(a, {k: v[:2] for k, v in a.items()}, mapping)
        self.assertFalse(d['count_equal'])
        self.assertFalse(d['canonical']['coords']['same_shape'])

    def test_threshold_failure_is_a_record_not_an_exception(self):
        a = np.zeros((252, 448, 3), dtype=np.float32)
        result = probe.render_metrics(a, a + 2e-5)
        self.assertFalse(result['passed'])
        self.assertGreater(result['psnr'], 60)
        self.assertGreater(result['mae'], 1e-5)
        self.assertEqual(probe.render_metrics(a, a)['psnr'], 120.)

    def test_nonfinite_is_json_serializable_and_fails_threshold(self):
        a = np.zeros((2, 2, 3), dtype=np.float32)
        a[0, 0, 0] = np.nan
        result = probe.render_metrics(np.zeros_like(a), a)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['mae'])
        json.dumps(result, allow_nan=False)
        json.dumps(probe.difference(np.zeros_like(a), a), allow_nan=False)

    def test_digest_includes_dtype_shape_and_exact_bytes(self):
        a = np.zeros((2, 3), dtype=np.float32)
        self.assertNotEqual(probe.array_digest(a), probe.array_digest(a.reshape(3, 2)))
        self.assertNotEqual(probe.array_digest(a), probe.array_digest(a.astype(np.float64)))
        b = a.copy()
        b[0, 0] = -0.
        self.assertNotEqual(probe.array_digest(a), probe.array_digest(b))

    def test_manifest_detects_tampering_and_unbound_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'protocol.json').write_text('{}', encoding='utf-8')
            (root / 'probe.py').write_text('# test fixture', encoding='utf-8')
            manifest = {name: probe.sha(root / name) for name in ('probe.py', 'protocol.json')}
            (root / 'CODE_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')
            probe.verify_manifest(root)
            (root / 'extra.py').write_text('', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                probe.verify_manifest(root)
            (root / 'extra.py').unlink()
            (root / 'protocol.json').write_text('{"drift": true}', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                probe.verify_manifest(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
