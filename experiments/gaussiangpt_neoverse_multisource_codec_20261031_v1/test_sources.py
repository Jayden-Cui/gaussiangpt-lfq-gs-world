"""CPU contract tests only: no CUDA, teacher export, or vendor model tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sources as s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal():
    return dict(center=[0.0, 0.0, 0.0], scale=0.5, source_frame=0,
                fixed_for_all_frames=True, calibration_only_alpha_threshold=0.01)


def index(clip):
    times = [t / 29.97 for t in range(s.FRAMES)]
    mapping = dict(clip=clip, source_frame_indices=list(range(s.FRAMES)),
                   model_timestamps=list(range(s.FRAMES)), source_pts_seconds=times,
                   source_sha256=s.VIDEO_SHA[clip])
    rows = [dict(clip=clip, frame=t, pts_seconds=times[t], source_count=s.N,
                 cache_sha256='1' * 64, cache_bytes=123, native_gs_sha256='2' * 64)
            for t in range(s.FRAMES)]
    return rows, mapping


class ShapeOnly:
    """Small shape/point-count fixture, not a tensor or fake model execution."""
    def __init__(self, shape, total=None, minimum=None):
        self.shape, self.total, self.minimum = shape, total, minimum

    def __len__(self):
        return self.shape[0]

    def sum(self):
        return self.total

    def min(self):
        return self.minimum


def frame_fixture(clip=s.TRAIN_CLIPS[0]):
    rows, mapping = index(clip)
    count, frame = 2, 0
    source = SimpleNamespace(clip=clip, role=s.ROLES[clip], normalization=normal(),
                             frames=[dict(pts_seconds=0.0, native_gs_sha256='2' * 64)])
    shapes = {'representatives': (count, 14), 'coordinates': (count, 3), 'set_statistics': (count, 29),
              'counts': (count,), 'source_indices': (count,), 'cameras': (2, 4, 4), 'intrinsics': (4,),
              'targets': (2, s.H, s.W, 3), 'real': (s.H, s.W, 3)}
    payload = {key: ShapeOnly(shape) for key, shape in shapes.items()}
    payload.update(clip=clip, frame=frame, normalization=normal(), n_input=s.N,
                   metrics=rows[0], near=0.005, far=5e9)
    payload['counts'] = ShapeOnly((count,), s.N, s.N // count)
    return payload, source


class SamplingTests(unittest.TestCase):
    def test_balanced_full_first_epoch_and_eval_exclusion(self):
        plan = s.balanced_plan(20261031)
        self.assertEqual(len(plan), 300)
        seen = {sample for step in plan[:177] for sample in step}
        self.assertEqual(seen, {(clip, t) for clip in s.TRAIN_CLIPS for t in range(177)})
        for pair in plan:
            self.assertEqual(tuple(clip for clip, _ in pair), s.TRAIN_CLIPS)
            self.assertNotIn(s.EVAL_CLIP, [clip for clip, _ in pair])
        self.assertEqual(plan, s.balanced_plan(20261031))
        self.assertNotEqual(plan, s.balanced_plan(20261032))

    def test_incomplete_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError, '177'):
            s.balanced_plan(1, steps=176)

    def test_protocol_role_width_and_reduction_rejected(self):
        cfg = s.read(Path(__file__).parent / 'protocol.json')
        s.validate_protocol(cfg)
        cases = [('data', 'train_clips', [*s.TRAIN_CLIPS, s.EVAL_CLIP]),
                 ('data', 'frames_per_clip', 9), ('data', 'teacher_points_per_frame', 10000),
                 ('data', 'width', 224), ('model', 'latent_width', 32),
                 ('model', 'state_quantization', True), ('dependency', 'require_done_complete', False)]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                altered = deepcopy(cfg)
                altered[section][key] = value
                with self.assertRaisesRegex(ValueError, 'Fixed protocol changed'):
                    s.validate_protocol(altered)


class FullFrameTests(unittest.TestCase):
    def test_source_identity_and_all_points_required(self):
        payload, source = frame_fixture()
        s.validate_frame_metadata(payload, source, 0)
        for key, value in [('n_input', s.N - 1), ('clip', s.EVAL_CLIP), ('frame', 1),
                           ('counts', ShapeOnly((2,), s.N - 1, 1)),
                           ('real', ShapeOnly((126, 224, 3))), ('features', ShapeOnly((2, 64))),
                           ('pts', 0.1), ('normalization', dict(normal(), scale=0.25))]:
            with self.subTest(key=key):
                altered = dict(payload)
                altered[key] = value
                with self.assertRaises(ValueError):
                    s.validate_frame_metadata(altered, source, 0)

    def test_eval_frame_remains_eval_only(self):
        payload, source = frame_fixture(s.EVAL_CLIP)
        s.validate_frame_metadata(payload, source, 0)
        source.role = 'TRAIN'
        with self.assertRaisesRegex(ValueError, 'role'):
            s.validate_frame_metadata(payload, source, 0)

    def test_index_no_frame_skip_or_clip_mix(self):
        rows, mapping = index(s.TRAIN_CLIPS[1])
        s.validate_index(s.TRAIN_CLIPS[1], rows, mapping, normal())
        cases = [('rows', rows[:-1]), ('rows', [rows[1], *rows[1:]]),
                 ('mapping', dict(mapping, source_sha256=s.VIDEO_SHA[s.EVAL_CLIP])),
                 ('mapping', dict(mapping, source_frame_indices=list(range(0, 354, 2)))),
                 ('mapping', dict(mapping, source_pts_seconds=[t / 15 for t in range(177)]))]
        for kind, value in cases:
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    s.validate_index(s.TRAIN_CLIPS[1], value if kind == 'rows' else rows,
                                     value if kind == 'mapping' else mapping, normal())

    def test_actual_pinned_old_and_transfer_indexes(self):
        """Use mirrored metadata only; do not claim the remote caches are present."""
        experiments = Path(__file__).resolve().parent.parent
        for clip, root, key in [(s.TRAIN_CLIPS[0], experiments / s.CODEC_ROOT.parent.name / 'state', 'rows'),
                                 (s.EVAL_CLIP, experiments / s.TOWER14_ROOT.name, 'frames')]:
            index_path = root / ('prepare' if key == 'rows' else 'quality') / f'{clip}.json'
            value = s.read(index_path)
            s.validate_index(clip, value[key], s.read(root / 'mappings' / f'{clip}.json'),
                             s.read(root / 'normalization.json'))
            expected = s.STATIC_METADATA.get(s.CODEC_ROOT / 'prepare' / f'{clip}.json', s.TOWER14_QUALITY_SHA)
            self.assertEqual(digest(index_path), expected)
        for path, expected in s.STATIC_METADATA.items():
            relative = path.relative_to(s.BASE)
            mirror = experiments / relative
            if mirror.is_file():
                self.assertEqual(digest(mirror), expected)


class DynamicPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write(self.root / 'CODE_MANIFEST.json', {'prepare.py': '7' * 64})
        write(self.root / 'input_binding.json', {'schema': 'fixture'})
        manifest_sha, binding_sha = digest(self.root / 'CODE_MANIFEST.json'), digest(self.root / 'input_binding.json')
        self.patches = patch.multiple(s, TOWER10_MANIFEST_SHA=manifest_sha, TOWER10_BINDING_SHA=binding_sha)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.summary = dict(status='complete', preparation_contract_passed=True, frames=177,
            adjacent_pairs=176, clip=s.TRAIN_CLIPS[1], source_points_per_frame=s.N,
            all_input_points_aggregated=True, codec_frozen=True, training_performed=False,
            input_binding_sha256=binding_sha, state_quantization='Identity; continuous float32 Nx64',
            source_role='TRAIN', source_group='buildingblock_tower_10', official_independent_test_split=False,
            codec_state_sha256_before_and_after='3' * 64, normalization=normal())
        write(self.root / 'SUMMARY.json', self.summary)
        self.done = dict(status='complete', preparation_contract_passed=True, training_performed=False,
            source_role='TRAIN', admission={'code_manifest_sha256': manifest_sha, 'input_binding_sha256': binding_sha},
            summary_sha256=digest(self.root / 'SUMMARY.json'), codec_state_sha256_before_and_after='3' * 64,
            outcomes=[dict(name=name, exit_code=0, timed_out=False)
                      for name in ('teacher_export', 'frozen_codec_preparation')])
        write(self.root / 'queue/DONE.json', self.done)

    def test_complete_records_are_frozen(self):
        records = {}
        self.assertEqual(s.verify_tower10(self.root, records), self.summary)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(row['bytes'] > 0 for row in records.values()))

    def test_missing_done_rejected_even_with_summary(self):
        (self.root / 'queue/DONE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'no DONE'):
            s.verify_tower10(self.root, {})

    def test_empty_terminal_rejected(self):
        (self.root / 'queue/DONE.json').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'Empty'):
            s.verify_tower10(self.root, {})

    def test_failure_or_incomplete_terminal_rejected(self):
        for path in ('FAILED.json', 'queue/FAILED.json'):
            with self.subTest(path=path):
                write(self.root / path, {'status': 'failed'})
                with self.assertRaisesRegex(ValueError, 'failure receipt'):
                    s.verify_tower10(self.root, {})
                (self.root / path).unlink()
        write(self.root / 'queue/DONE.json', dict(self.done, status='waiting'))
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            s.verify_tower10(self.root, {})

    def test_exact_manifest_and_summary_sha_required(self):
        original = (self.root / 'CODE_MANIFEST.json').read_bytes()
        (self.root / 'CODE_MANIFEST.json').write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            s.verify_tower10(self.root, {})
        (self.root / 'CODE_MANIFEST.json').write_bytes(original)
        write(self.root / 'SUMMARY.json', dict(self.summary, frames=176))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            s.verify_tower10(self.root, {})

    def test_other_preparation_and_failed_children_rejected(self):
        changed = deepcopy(self.done)
        changed['admission']['code_manifest_sha256'] = '0' * 64
        write(self.root / 'queue/DONE.json', changed)
        with self.assertRaisesRegex(ValueError, 'another preparation'):
            s.verify_tower10(self.root, {})
        changed = deepcopy(self.done)
        changed['outcomes'][1]['timed_out'] = True
        write(self.root / 'queue/DONE.json', changed)
        with self.assertRaisesRegex(ValueError, 'both complete'):
            s.verify_tower10(self.root, {})

    def test_post_phase_hash_detects_same_length_mutation(self):
        path = self.root / 'actual_cache.pt'
        path.write_bytes(b'original')
        records = {path: s.fingerprint(path)}
        binding = {'files': list(records.values())}
        write(self.root / 'DATA_BINDING.json', binding)
        bundle = SimpleNamespace(root=self.root, records=records, binding=binding,
                                 binding_sha256=digest(self.root / 'DATA_BINDING.json'))
        with patch.object(s, 'TOWER10_ROOT', self.root):
            self.assertEqual(s.verify_sources(bundle)['status'], 'passed')
            path.write_bytes(b'mutated!')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                s.verify_sources(bundle)


if __name__ == '__main__':
    unittest.main()
