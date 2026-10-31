"""CPU integrity regressions; synthetic tensor fixtures are never real evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import admission as a
from test_prepare_clip import completed_fixture, ITEM, CLIP, POOL, save


class FileAndMappingTests(unittest.TestCase):
    def test_hash_checks_actual_file_and_detects_mid_admission_change(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'data'; p.write_bytes(b'original')
            audit = a.Audit(); record = audit.add(p)
            with self.assertRaisesRegex(RuntimeError, 'hash changed'):
                audit.add(p, {'sha256': '0' * 64})
            p.write_bytes(b'changed!')
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                audit.recheck()
            self.assertEqual(record['bytes'], 8)

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'Missing input'):
                a.Audit().add(Path(temp) / 'missing')

    def test_source_indices_pts_and_source_identity_are_independent_checks(self):
        mapping = dict(source_frame_indices=list(range(177)), model_timestamps=list(range(177)),
                       source_sha256=ITEM['video_sha256'], source_pts_seconds=[i / 29.97 for i in range(177)])
        a.validate_mapping(mapping, ITEM)
        for key, value in [('source_frame_indices', list(range(0, 177, 3))),
                           ('model_timestamps', list(range(176, -1, -1))), ('source_sha256', '0' * 64),
                           ('source_pts_seconds', [i / 5 for i in range(177)])]:
            changed = dict(mapping, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                a.validate_mapping(changed, ITEM)


class TensorContractTests(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.multiple(a, POINTS=8, HEIGHT=2, WIDTH=3)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.native = {key: np.ones(shape, np.float32) for key, shape in
                       {'means': (8, 3), 'scales': (8, 3), 'rotations': (8, 4),
                        'harmonics': (8, 1, 3), 'opacities': (8,)}.items()}
        self.native['metadata_json'] = np.array(json.dumps(dict(timestamp=0, forward_vel=None, backward_vel=None)))
        self.normal = dict(center=[0., 0., 0.], scale=.5, source_frame=0, fixed_for_all_frames=True)
        self.rgb = np.full((2, 3, 3), 127, np.uint8); self.camera = np.eye(4, dtype=np.float32)
        self.intr = np.eye(3, dtype=np.float32)
        self.payload = {key: torch.ones(shape) for key, shape in
                        {'representatives': (2, 14), 'set_statistics': (2, 29), 'targets': (2, 2, 3, 3)}.items()}
        cams = torch.stack([torch.eye(4), torch.eye(4)]); cams[1, 0, 3] = .02
        self.payload.update(clip=ITEM['clip_id'], frame=0, n_input=8, normalization=self.normal,
                            metrics=dict(frame=0, source_count=8, pts_seconds=0),
                            coordinates=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
                            source_indices=torch.tensor([0, 4]), counts=torch.tensor([4, 4]),
                            real=torch.from_numpy(self.rgb).float() / 255, cameras=cams,
                            intrinsics=torch.tensor([1/3, 1/2, 0, 0]), near=.005, far=5e9)

    def audit_cache(self, payload=None):
        a.validate_cache(self.payload if payload is None else payload, ITEM, 0, self.normal,
                         self.rgb, self.camera, self.intr)

    def test_native_arrays_count_finiteness_and_frame_checked(self):
        a.validate_native(self.native, 0)
        for key, value in [('means', np.ones((7, 3), np.float32)),
                           ('scales', np.zeros((8, 3), np.float32)),
                           ('harmonics', np.full((8, 1, 3), np.nan)),
                           ('metadata_json', np.array(json.dumps(dict(timestamp=1))))]:
            changed = dict(self.native, **{key: value})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                a.validate_native(changed, 0)

    def test_complete_cache_shape_and_identity(self):
        self.audit_cache()
        for key, value in [('n_input', 7), ('clip', 'wrong'), ('counts', torch.tensor([3, 4])),
                           ('source_indices', torch.tensor([0, 0])), ('set_statistics', torch.ones((2, 28))),
                           ('real', torch.zeros((2, 3, 3))), ('normalization', dict(self.normal, scale=1.))]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.audit_cache(dict(self.payload, **{key: value}))


class FreshReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.directory = self.root / 'clips' / CLIP
        self.directory.mkdir(parents=True)
        self.item, binding_hash = completed_fixture(self.directory)
        save(self.root / 'CODE_MANIFEST.json', {}); save(self.root / 'INPUT_POOL.json', POOL)
        save(self.directory / 'HELPER_SUMMARY.json', a.read(self.directory / 'SUMMARY.json'))
        actual = a.prep.verify_outputs(self.directory, self.item, binding_hash)
        self.receipt = dict(schema='gs_fulltrain_clip_preparation_receipt_v1', status='complete', clip=CLIP,
                            source_group=self.item['source_group'], original_role=self.item['original_role'],
                            current_role='TRAIN', data_origin='new_full177_export', training_performed=False,
                            goal_complete=False, helper_sha256=a.prep.HELPER_SHA256,
                            code_manifest_sha256=a.bindings.sha256(self.root / 'CODE_MANIFEST.json'),
                            input_pool_sha256=a.bindings.sha256(self.root / 'INPUT_POOL.json'),
                            input_binding_sha256=binding_hash,
                            summary_sha256=a.bindings.sha256(self.directory / 'SUMMARY.json'),
                            helper_summary_sha256=a.bindings.sha256(self.directory / 'HELPER_SUMMARY.json'), **actual)
        self.write_receipt()

    def write_receipt(self):
        save(self.directory / 'PREPARATION_RECEIPT.json', self.receipt)

    def audit(self):
        # Static binding validation is covered by test_bindings; this fixture uses
        # actual tiny files to exercise the full downstream hash/coverage validator.
        with mock.patch.object(a.bindings, 'validate_binding'):
            return a.admit_fresh(self.root, self.item, a.Audit())

    def test_complete_preparation_receipt_is_independently_reverified(self):
        result = self.audit()
        self.assertEqual(result['actual_frames'], 177)
        self.assertFalse(result['state_encodings_reused_for_new_codec'])

    def test_tampered_data_rejected_despite_complete_receipt(self):
        (self.directory / 'cache' / CLIP / '044.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'hash/size changed'):
            self.audit()

    def test_missing_frame_rejected_despite_complete_receipt(self):
        (self.directory / 'states' / CLIP / '176.pt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'states coverage'):
            self.audit()

    def test_wrong_role_or_invented_coverage_not_admitted(self):
        self.receipt['original_role'] = 'TEST'; self.write_receipt()
        with self.assertRaisesRegex(RuntimeError, 'original_role'):
            self.audit()
        self.receipt['original_role'] = self.item['original_role']
        self.receipt['actual_frames'] = 59; self.write_receipt()
        with self.assertRaisesRegex(RuntimeError, 'actual_frames'):
            self.audit()

    def test_failed_marker_dominates_successful_receipt(self):
        save(self.directory / 'ADAPTER_FAILED.json', {'status': 'failed'})
        with self.assertRaisesRegex(RuntimeError, 'Failed preparation'):
            self.audit()


class ReuseReceiptTests(unittest.TestCase):
    def test_duplicate_reuse_clips_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); save(root/'CODE_MANIFEST.json', {}); save(root/'INPUT_POOL.json', POOL)
            receipt = dict(a.base_receipt(root), schema='gs_fulltrain_reuse_admission_v1',
                           clips=[{'clip_id': 'buildingblock_tower_10_176'}] * 2)
            with self.assertRaisesRegex(RuntimeError, 'coverage/duplicates'):
                a.validate_reuse_receipt(receipt, root, POOL)

    def test_reference_only_record_cannot_count_as_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); save(root/'CODE_MANIFEST.json', {}); save(root/'INPUT_POOL.json', POOL)
            receipt = dict(a.base_receipt(root), schema='gs_fulltrain_reuse_reference_v1', clips=[])
            with self.assertRaisesRegex(RuntimeError, 'reuse schema'):
                a.validate_reuse_receipt(receipt, root, POOL)


if __name__ == '__main__':
    unittest.main()
