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
    def test_historical_state_failures_do_not_mask_fresh_same_state_passes(self):
        rows = [{'lane': lane, 'replay_thresholds_passed': lane != 'historical_disk_rebuild'}
                for lane in probe.LANES for _ in range(3)]
        summary = probe.replay_group_summary(rows)
        self.assertEqual(len(rows), 15)
        self.assertFalse(summary['all_lanes']['passed'])
        self.assertEqual(summary['all_lanes']['passed'], all(r['replay_thresholds_passed'] for r in rows))
        self.assertTrue(summary['fresh_same_state']['passed'])
        self.assertFalse(summary['historical_state_comparison']['passed'])
        passing_mode = probe.replay_group_summary(
            [{'lane': r['lane'], 'replay_thresholds_passed': True} for r in rows])
        aggregate = probe.aggregate_replay_summaries([passing_mode, summary])
        self.assertEqual({key: value['passed'] for key, value in aggregate.items()},
                         {'all_lanes': False, 'fresh_same_state': True, 'historical_state_comparison': False})
        self.assertEqual(aggregate['fresh_same_state']['input_identity'], summary['fresh_same_state']['input_identity'])
        self.assertEqual(aggregate['historical_state_comparison']['input_identity'],
                         summary['historical_state_comparison']['input_identity'])

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


class PruningTests(unittest.TestCase):
    """CPU checks for raw classifier decisions and exact coordinate joins only."""

    @staticmethod
    def coordinates(count):
        c = np.zeros((count, 4), dtype=np.int32)
        c[:, 1] = np.arange(count, dtype=np.int32)
        return c

    def test_strict_boundary_and_negative_threshold(self):
        c = self.coordinates(4)
        logits = np.array([-.75, -.5, -.25, 0.], dtype=np.float32)
        result = probe.pruning_summary(c, logits, -.5)
        self.assertEqual(result['comparison'], 'raw_logit > threshold')
        self.assertEqual(result['candidates'], 4)
        self.assertEqual(result['kept'], 2)
        self.assertEqual(result['pruned'], 2)
        self.assertEqual(result['min_abs_margin'], 0.)
        closest = result['closest_to_threshold'][0]
        self.assertEqual(closest['coordinate'], [0, 1, 0, 0])
        self.assertEqual(closest['logit'], -.5)
        self.assertEqual(closest['signed_margin'], 0.)
        self.assertFalse(closest['kept'])

    def test_threshold_uses_classifier_dtype_at_float32_boundary(self):
        logits = np.array([[.1]], dtype=np.float32)
        # float32(.1) is above Python's float64 .1; the decoder compares in
        # classifier dtype, so equality at that rounded threshold must be pruned.
        self.assertGreater(float(logits[0, 0]), .1)
        result = probe.pruning_summary(self.coordinates(1), logits, .1)
        self.assertEqual(result['effective_threshold'], float(logits[0, 0]))
        self.assertEqual(result['kept'], 0)
        self.assertEqual(result['min_abs_margin'], 0.)

    def test_nearest_threshold_points_have_known_order_and_ten_limit(self):
        c = self.coordinates(12)
        logits = np.arange(-6, 6, dtype=np.float32).reshape(-1, 1)
        result = probe.pruning_summary(c, logits, .25)
        closest = result['closest_to_threshold']
        self.assertEqual(len(closest), 10)
        self.assertEqual([r['coordinate'][1] for r in closest], [6, 7, 5, 8, 4, 9, 3, 10, 2, 11])
        self.assertEqual(result['min_abs_margin'], .25)
        self.assertEqual([r['signed_margin'] for r in closest[:3]], [-.25, .75, -1.25])
        self.assertEqual(result['kept'] + result['pruned'], result['candidates'])
        json.dumps(result, allow_nan=False)

    def test_permutation_joins_by_batch_and_coordinates(self):
        c = np.array([[0, 2, -1, 4], [1, 2, -1, 4], [0, -2, 3, 0]], dtype=np.int32)
        f = np.array([[-.5], [.5], [.125]], dtype=np.float32)
        permutation = [2, 0, 1]
        reference = {'coordinates': c, 'logits': f, 'threshold': 0.}
        candidate = {'coordinates': c[permutation], 'logits': f[permutation], 'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['coordinate_alignment'], 'exact_batch_and_integer_coordinates')
        self.assertEqual(result['common_coordinates'], 3)
        self.assertEqual(result['reference_only_coordinates'], 0)
        self.assertEqual(result['candidate_only_coordinates'], 0)
        self.assertEqual(result['crossed_threshold_count'], 0)
        self.assertTrue(result['common_logits_diff']['exact_equal'])

    def test_true_threshold_crossings_are_at_the_correct_coordinates(self):
        c = np.array([[0, 1, 2, 3], [0, -1, 0, 4], [1, 1, 2, 3]], dtype=np.int32)
        reference = {'coordinates': c, 'logits': np.array([[.25], [-.125], [0.]], dtype=np.float32),
                     'threshold': 0.}
        candidate = {'coordinates': c[[2, 0, 1]],
                     'logits': np.array([[.125], [0.], [-.125]], dtype=np.float32), 'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['crossed_threshold_count'], 2)
        by_coordinate = {tuple(row['coordinate']): row for row in result['crossed_threshold']}
        self.assertEqual(set(by_coordinate), {(0, 1, 2, 3), (1, 1, 2, 3)})
        became_pruned = by_coordinate[(0, 1, 2, 3)]
        self.assertTrue(became_pruned['reference_kept'])
        self.assertFalse(became_pruned['candidate_kept'])
        self.assertEqual(became_pruned['reference_logit'], .25)
        self.assertEqual(became_pruned['candidate_logit'], 0.)
        became_kept = by_coordinate[(1, 1, 2, 3)]
        self.assertFalse(became_kept['reference_kept'])
        self.assertTrue(became_kept['candidate_kept'])
        self.assertEqual(result['common_logits_diff']['changed_elements'], 2)
        json.dumps(result, allow_nan=False)

    def test_missing_coordinate_is_not_counted_as_threshold_crossing(self):
        c = self.coordinates(3)
        reference = {'coordinates': c[:2], 'logits': np.array([[.125], [.25]], dtype=np.float32),
                     'threshold': 0.}
        candidate = {'coordinates': c[1:], 'logits': np.array([[.5], [-.5]], dtype=np.float32),
                     'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['common_coordinates'], 1)
        self.assertEqual(result['reference_only_coordinates'], 1)
        self.assertEqual(result['candidate_only_coordinates'], 1)
        self.assertEqual(result['reference_only_kept'], 1)
        self.assertEqual(result['candidate_only_kept'], 0)
        self.assertEqual(result['crossed_threshold_count'], 0)
        self.assertEqual(result['crossed_threshold'], [])
        self.assertEqual(result['common_logits_diff']['changed_elements'], 1)
        self.assertEqual(result['common_logits_diff']['max_abs'], .25)

    def test_duplicate_summary_preserves_every_row_and_reports_multiplicity(self):
        c = np.array([[0, 2, 0, 0]] * 3 + [[1, 2, 0, 0]] * 2, dtype=np.int32)
        logits = np.array([[-.25], [.5], [1.], [-.125], [0.]], dtype=np.float32)
        result = probe.pruning_summary(c, logits, 0.)
        self.assertEqual(result['candidates'], 5)
        self.assertEqual(result['kept'], 2)
        self.assertEqual(result['pruned'], 3)
        self.assertEqual(result['coordinate_multiplicity'], {
            'rows': 5, 'unique_coordinates': 2, 'duplicate_rows': 3,
            'duplicate_groups': 2, 'max_multiplicity': 3,
            'duplicate_coordinates': [{'coordinate': [0, 2, 0, 0], 'multiplicity': 3},
                                      {'coordinate': [1, 2, 0, 0], 'multiplicity': 2}]})
        closest = result['closest_to_threshold']
        self.assertEqual({r['row_index'] for r in closest}, set(range(5)))
        self.assertEqual(sorted(r['logit'] for r in closest), [-.25, -.125, 0., .5, 1.])
        json.dumps(result, allow_nan=False)

    def test_duplicate_permutation_preserves_logit_multiset_without_pairing(self):
        c = np.zeros((3, 4), dtype=np.int32)
        f = np.array([[-.25], [.125], [.125]], dtype=np.float32)
        reference = {'coordinates': c, 'logits': f, 'threshold': 0.}
        candidate = {'coordinates': c[[2, 0, 1]], 'logits': f[[2, 0, 1]], 'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['common_coordinates'], 1)
        self.assertEqual(result['common_unique_coordinates'], 0)
        self.assertEqual(result['ambiguous_common_coordinates'], 1)
        self.assertEqual(result['crossed_threshold_count'], 0)
        self.assertEqual(result['crossed_threshold_scope'], 'coordinates_unique_on_both_sides_only')
        self.assertEqual(result['common_logits_diff']['reference_shape'], [0])
        group = result['ambiguous_groups'][0]
        self.assertEqual(group['coordinate'], [0, 0, 0, 0])
        self.assertEqual((group['reference_count'], group['candidate_count']), (3, 3))
        self.assertEqual((group['reference_kept'], group['candidate_kept'], group['kept_count_delta']), (2, 2, 0))
        self.assertTrue(group['logit_multiset_equal'])

    def test_duplicate_count_changes_and_missing_rows_do_not_invent_flips(self):
        a, b, c = self.coordinates(3)
        reference = {'coordinates': np.array([a, a, b, b]),
                     'logits': np.array([[.5], [-.25], [.25], [-.5]], dtype=np.float32), 'threshold': 0.}
        candidate = {'coordinates': np.array([a, a, a, c, c, c]),
                     'logits': np.array([[.5], [-.25], [.25], [-1.], [0.], [1.]], dtype=np.float32), 'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['reference_only_coordinates'], 1)
        self.assertEqual(result['candidate_only_coordinates'], 1)
        self.assertEqual(result['reference_only_rows'], 2)
        self.assertEqual(result['candidate_only_rows'], 3)
        self.assertEqual(result['reference_only_kept'], 1)
        self.assertEqual(result['candidate_only_kept'], 1)
        self.assertEqual(result['crossed_threshold_count'], 0)
        self.assertEqual(result['crossed_threshold'], [])
        group = result['ambiguous_groups'][0]
        self.assertEqual((group['reference_count'], group['candidate_count']), (2, 3))
        self.assertEqual((group['reference_kept'], group['candidate_kept'], group['kept_count_delta']), (1, 2, 1))
        self.assertFalse(group['logit_multiset_equal'])

    def test_mixed_duplicate_and_unique_coordinates_still_detect_unique_flip(self):
        a, b, c = self.coordinates(3)
        reference = {'coordinates': np.array([a, a, b, c]),
                     'logits': np.array([[-.25], [.25], [-.125], [.5]], dtype=np.float32), 'threshold': 0.}
        candidate = {'coordinates': np.array([c, a, b, a]),
                     'logits': np.array([[.5], [.125], [.125], [.25]], dtype=np.float32), 'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        self.assertEqual(result['common_coordinates'], 3)
        self.assertEqual(result['common_unique_coordinates'], 2)
        self.assertEqual(result['ambiguous_common_coordinates'], 1)
        self.assertEqual(result['common_logits_diff']['reference_shape'], [2])
        self.assertEqual(result['common_logits_diff']['changed_elements'], 1)
        self.assertEqual(result['crossed_threshold_count'], 1)
        self.assertEqual(result['crossed_threshold'][0]['coordinate'], b.tolist())
        self.assertFalse(result['crossed_threshold'][0]['reference_kept'])
        self.assertTrue(result['crossed_threshold'][0]['candidate_kept'])
        group = result['ambiguous_groups'][0]
        self.assertEqual(group['coordinate'], a.tolist())
        self.assertEqual((group['reference_kept'], group['candidate_kept'], group['kept_count_delta']), (1, 2, 1))
        self.assertFalse(group['logit_multiset_equal'])

    def test_duplicate_logits_change_kept_counts_without_individual_correspondence(self):
        c = np.zeros((3, 4), dtype=np.int32)
        reference = {'coordinates': c, 'logits': np.array([[-.5], [0.], [.5]], dtype=np.float32),
                     'threshold': 0.}
        candidate = {'coordinates': c, 'logits': np.array([[.125], [.25], [.5]], dtype=np.float32),
                     'threshold': 0.}
        result = probe.compare_pruning(reference, candidate)
        group = result['ambiguous_groups'][0]
        self.assertEqual((group['reference_count'], group['candidate_count']), (3, 3))
        self.assertEqual((group['reference_kept'], group['candidate_kept'], group['kept_count_delta']), (1, 3, 2))
        self.assertFalse(group['logit_multiset_equal'])
        self.assertEqual(result['crossed_threshold_count'], 0)
        self.assertEqual(result['crossed_threshold'], [])
        json.dumps(result, allow_nan=False)

    def test_signed_zero_multiset_is_permutation_invariant_and_byte_exact(self):
        c = np.zeros((3, 4), dtype=np.int32)
        f = np.array([[0.], [-0.], [.25]], dtype=np.float32)
        reference = {'coordinates': c, 'logits': f, 'threshold': 0.}
        permutation = {'coordinates': c, 'logits': f[[1, 2, 0]], 'threshold': 0.}
        result = probe.compare_pruning(reference, permutation)
        self.assertTrue(result['ambiguous_groups'][0]['logit_multiset_equal'])
        changed = {'coordinates': c, 'logits': np.array([[0.], [0.], [.25]], dtype=np.float32),
                   'threshold': 0.}
        result = probe.compare_pruning(reference, changed)
        self.assertFalse(result['ambiguous_groups'][0]['logit_multiset_equal'])
        self.assertEqual(result['ambiguous_groups'][0]['kept_count_delta'], 0)
        self.assertEqual(result['crossed_threshold_count'], 0)

    def test_capture_snapshot_owns_independent_storage(self):
        class FakeTensor:
            def __init__(self, values, requires_grad=True):
                self.values = np.asarray(values)
                self.requires_grad = requires_grad

            def detach(self):
                return FakeTensor(self.values, requires_grad=False)

            def clone(self):
                return FakeTensor(self.values.copy(), requires_grad=self.requires_grad)

        source = FakeTensor(np.array([[.25], [-.5]], dtype=np.float32))
        snapshot = probe.capture_snapshot_tensor(source)
        source.values[0, 0] = 99.
        self.assertEqual(snapshot.values[0, 0], .25)
        snapshot.values[1, 0] = 88.
        self.assertEqual(source.values[1, 0], -.5)
        self.assertFalse(snapshot.requires_grad)
        self.assertTrue(source.requires_grad)

    def test_nonfinite_pruning_inputs_are_rejected(self):
        c = self.coordinates(2)
        f = np.array([[.125], [.25]], dtype=np.float32)
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(kind='logit', invalid=invalid):
                invalid_logits = f.copy()
                invalid_logits[0, 0] = invalid
                with self.assertRaises(RuntimeError):
                    probe.pruning_summary(c, invalid_logits, 0.)
                with self.assertRaises(RuntimeError):
                    probe.compare_pruning({'coordinates': c, 'logits': f, 'threshold': 0.},
                                          {'coordinates': c, 'logits': invalid_logits, 'threshold': 0.})
            with self.subTest(kind='threshold', invalid=invalid):
                with self.assertRaises(RuntimeError):
                    probe.pruning_summary(c, f, invalid)
            with self.subTest(kind='coordinate', invalid=invalid):
                invalid_coordinates = c.astype(np.float64)
                invalid_coordinates[0, 1] = invalid
                with self.assertRaises(RuntimeError):
                    probe.pruning_summary(invalid_coordinates, f, 0.)


class FakeOperationMode(Enum):
    SEPARATE_COORDINATE_MANAGER = 0
    SHARE_COORDINATE_MANAGER = 1


class FakeME:
    """Process-local state double; does not import torch or MinkowskiEngine."""
    SparseTensorOperationMode = FakeOperationMode

    def __init__(self, manager=None, flag=False, mode=FakeOperationMode.SEPARATE_COORDINATE_MANAGER):
        self.manager, self.flag, self.mode = manager, flag, mode
        self.writes = []

    def sparse_tensor_operation_mode(self):
        return self.mode

    def set_sparse_tensor_operation_mode(self, mode):
        self.writes.append(('set_mode', mode))
        self.mode = mode

    def global_coordinate_manager(self):
        return self.manager

    def set_global_coordinate_manager(self, manager):
        self.writes.append(('set_manager', manager))
        self.manager = manager

    def clear_global_coordinate_manager(self):
        self.writes.append(('clear_manager', self.manager))
        self.manager = None

    def is_deterministic(self):
        return self.flag

    def set_deterministic(self, flag):
        self.writes.append(('set_flag', flag))
        self.flag = bool(flag)


class CaseManagerScopeTests(unittest.TestCase):
    def assert_restored(self, scope, engine, original_manager, original_mode, original_flag):
        self.assertIs(engine.global_coordinate_manager(), original_manager)
        self.assertEqual(engine.sparse_tensor_operation_mode(), original_mode)
        self.assertEqual(engine.is_deterministic(), original_flag)
        self.assertTrue(scope.record['restored'])
        self.assertTrue(scope.record['manager_identity_restored'])
        self.assertTrue(scope.record['mode_restored'])
        self.assertTrue(scope.record['flag_restored'])

    def test_nonempty_original_manager_is_restored_by_identity(self):
        original = {'maps': ['original-map']}
        engine = FakeME(original, flag=False)
        original_mode = engine.mode
        case_manager = {'maps': ['case-map']}
        cleanup_observed = []
        scope = probe.CaseManagerScope(engine, enabled=True,
                                       cleanup=lambda: cleanup_observed.append((engine.manager, engine.mode, engine.flag)))
        with scope as entered:
            self.assertIs(entered, scope)
            self.assertIsNone(engine.manager)
            self.assertEqual(engine.mode, FakeOperationMode.SHARE_COORDINATE_MANAGER)
            self.assertTrue(engine.flag)
            self.assertEqual(scope.record['before']['global_manager_python_id'], id(original))
            self.assertFalse(scope.record['before']['global_manager_is_none'])
            self.assertTrue(scope.record['after_enter']['global_manager_is_none'])
            engine.set_global_coordinate_manager(case_manager)
        self.assert_restored(scope, engine, original, original_mode, False)
        self.assertEqual(len(cleanup_observed), 1)
        self.assertIs(cleanup_observed[0][0], case_manager)
        self.assertEqual(cleanup_observed[0][1], FakeOperationMode.SHARE_COORDINATE_MANAGER)
        self.assertTrue(cleanup_observed[0][2])
        self.assertEqual(scope.record['before_exit']['global_manager_python_id'], id(case_manager))
        self.assertEqual(scope.record['after_exit']['global_manager_python_id'], id(original))
        self.assertEqual(original, {'maps': ['original-map']})
        self.assertEqual(case_manager, {'maps': ['case-map']})
        self.assertTrue(scope.record['body_completed'])
        json.dumps(scope.record, allow_nan=False)

    def test_none_original_manager_and_true_original_flag_are_restored(self):
        engine = FakeME(None, flag=True)
        original_mode = engine.mode
        scope = probe.CaseManagerScope(engine, enabled=True)
        with scope:
            engine.set_global_coordinate_manager(object())
        self.assert_restored(scope, engine, None, original_mode, True)
        self.assertTrue(scope.record['before']['global_manager_is_none'])
        self.assertIsNone(scope.record['after_exit']['global_manager_python_id'])
        self.assertTrue(scope.record['after_exit']['deterministic'])

    def test_body_exception_restores_state_and_propagates_original_exception(self):
        original = object()
        engine = FakeME(original, flag=False)
        original_mode = engine.mode
        error = LookupError('diagnostic body failed')
        cleanup_called = []
        scope = probe.CaseManagerScope(engine, enabled=True, cleanup=lambda: cleanup_called.append(True))
        with self.assertRaises(LookupError) as caught:
            with scope:
                engine.set_global_coordinate_manager(object())
                raise error
        self.assertIs(caught.exception, error)
        self.assert_restored(scope, engine, original, original_mode, False)
        self.assertEqual(cleanup_called, [True])
        self.assertFalse(scope.record['body_completed'])
        self.assertEqual(scope.record['body_error'], repr(error))

    def test_cleanup_exception_still_restores_all_state(self):
        original = object()
        engine = FakeME(original, flag=False)
        original_mode = engine.mode

        def failing_cleanup():
            self.assertEqual(engine.mode, FakeOperationMode.SHARE_COORDINATE_MANAGER)
            self.assertTrue(engine.flag)
            raise ArithmeticError('cleanup failed')

        scope = probe.CaseManagerScope(engine, enabled=True, cleanup=failing_cleanup)
        with self.assertRaisesRegex(RuntimeError, 'cleanup/restoration failed'):
            with scope:
                engine.set_global_coordinate_manager(object())
        self.assert_restored(scope, engine, original, original_mode, False)
        self.assertEqual(scope.record['cleanup_errors'][0]['operation'], 'cleanup')
        self.assertIn('cleanup failed', scope.record['cleanup_errors'][0]['error'])

    def test_false_control_does_not_write_state_or_call_cleanup(self):
        original = object()
        engine = FakeME(original, flag=False)
        original_mode = engine.mode
        cleanup_called = []
        scope = probe.CaseManagerScope(engine, enabled=False, cleanup=lambda: cleanup_called.append(True))
        with scope:
            self.assertIs(engine.manager, original)
            self.assertEqual(engine.mode, original_mode)
            self.assertFalse(engine.flag)
        self.assertEqual(engine.writes, [])
        self.assertEqual(cleanup_called, [])
        self.assert_restored(scope, engine, original, original_mode, False)
        self.assertEqual(scope.record['before'], scope.record['after_enter'])
        self.assertEqual(scope.record['before'], scope.record['after_exit'])

    def test_consecutive_cases_start_isolated_without_destroying_old_references(self):
        original = {'maps': ['prior']}
        engine = FakeME(original, flag=True)
        original_mode = engine.mode
        case_managers = []
        for index in range(2):
            scope = probe.CaseManagerScope(engine, enabled=True)
            with scope:
                self.assertIsNone(engine.manager)
                case_manager = {'maps': [index]}
                case_managers.append(case_manager)
                engine.set_global_coordinate_manager(case_manager)
            self.assert_restored(scope, engine, original, original_mode, True)
        self.assertIsNot(case_managers[0], case_managers[1])
        self.assertEqual(case_managers, [{'maps': [0]}, {'maps': [1]}])
        self.assertEqual(original, {'maps': ['prior']})

    def test_failed_manager_restoration_still_attempts_mode_and_flag_restore(self):
        original = object()

        class FailingRestoreME(FakeME):
            def set_global_coordinate_manager(self, manager):
                if manager is original:
                    self.writes.append(('failed_restore_manager', manager))
                    raise RuntimeError('restore manager failed')
                super().set_global_coordinate_manager(manager)

        engine = FailingRestoreME(original, flag=False)
        original_mode = engine.mode
        scope = probe.CaseManagerScope(engine, enabled=True)
        with self.assertRaisesRegex(RuntimeError, 'cleanup/restoration failed'):
            with scope:
                engine.set_global_coordinate_manager(object())
        self.assertFalse(scope.record['restored'])
        self.assertFalse(scope.record['manager_identity_restored'])
        self.assertTrue(scope.record['mode_restored'])
        self.assertTrue(scope.record['flag_restored'])
        self.assertEqual(engine.mode, original_mode)
        self.assertFalse(engine.flag)
        self.assertEqual(scope.record['cleanup_errors'][0]['operation'], 'restore_original_manager')


if __name__ == '__main__':
    unittest.main(verbosity=2)
