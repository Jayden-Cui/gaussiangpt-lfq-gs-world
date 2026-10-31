"""Complete synthetic on-disk receipts; CPU only, no production data access.

Only the pinned original checkpoint digest is substituted with a fixture digest.
Every input/trace/packet/screenshot hash is calculated and checked for real.
The dummy checkpoint/packet contents are intentionally never deserialized.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import compare
import sources


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return file_record(path)


def png():
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 448, 252, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * (448 * 3)) * 252)) + chunk(b'IEND', b''))


def make_fixture(root):
    """Build three full evaluations and two traces independently of validator logic."""
    original = put(root / 'original' / 'codec_final.pt', b'fixture-original-codec')
    source_digest = original['sha256']
    cfg = read(Path(__file__).with_name('protocol.json'))
    cfg['source']['checkpoint_sha256'] = source_digest
    write(root / 'protocol.json', cfg)
    for name in ('worker.py', 'sources.py', 'compare.py'):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
    write(root / 'CODE_MANIFEST.json', {name: sha(root / name) for name in ('protocol.json', 'worker.py', 'sources.py', 'compare.py')})
    bound = [original] + [file_record(root / name) for name in
                          ('protocol.json', 'worker.py', 'sources.py', 'compare.py', 'CODE_MANIFEST.json')]
    cache_hash = {}
    for clip in sources.ALL_CLIPS:
        for frame in range(177):
            record = put(root / 'original' / 'cache' / clip / f'{frame:03d}.pt', f'{clip}/{frame}'.encode())
            bound.append(record)
            cache_hash[(clip, frame)] = record['sha256']
    write(root / 'DATA_BINDING.json', dict(schema='gs_multisource_codec_data_binding_v1', frames_per_clip=177,
        source_points_per_frame=112896, height=252, width=448, train_clips=list(sources.TRAIN_CLIPS),
        eval_only_clips=[sources.EVAL_CLIP], historical_roles=sources.HISTORICAL_ROLES,
        official_independent_holdout=False, historical_latents_reused=False,
        initial_codec_sha256=source_digest, state_quantization='Identity; continuous float32 Nx64', files=bound))
    plan = sources.balanced_plan(20261031, 300)
    source = dict(binding_sha256=sha(root / 'DATA_BINDING.json'), initial_checkpoint_sha256=source_digest,
        protocol_sha256=sha(root / 'protocol.json'), worker_sha256=sha(root / 'worker.py'),
        sources_sha256=sha(root / 'sources.py'),
        plan_sha256=hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest(),
        inherited_codec_steps=3000, additional_codec_updates=300, seed=20261031, batch_frames=2,
        forcing_updates=240, unforced_updates=60, goal_complete=False)
    preflight = dict(status='passed', source=source, arms={arm: dict(
        initial_model_sha256='a' * 64, initial_encoder_sha256='b' * 64,
        final_encoder_sha256=('b' if arm == 'decoder_only' else 'c') * 64,
        passed=True, discarded_updates=2, production_updates=0,
        frozen_encoder_parameters_and_buffers_unchanged=arm == 'decoder_only') for arm in compare.ARMS},
        initial_arms_identical=True, goal_complete=False)
    write(root / 'PREFLIGHT.json', preflight)
    dones = {}
    for arm in compare.ARMS:
        folder = root / 'arms' / arm
        folder.mkdir(parents=True)
        write(folder / 'SOURCE.json', dict(source, arm=arm, preflight_sha256=sha(root / 'PREFLIGHT.json')))
        trace, seen = [], set()
        for step, pair in enumerate(plan, 1):
            seen.update(pair)
            samples = [dict(clip=clip, frame=frame, loss=3.1, terms={name: 1. for name in compare.LOSS},
                            latent_count=20, output_count=200) for clip, frame in pair]
            groups = {'ae.decoder': dict(parameters=2, with_grad=2, nonzero_grad=2)}
            if arm == 'full_codec':
                groups['ae.encoder'] = dict(parameters=1, with_grad=1, nonzero_grad=1)
            trace.append(dict(step=step, arm=arm, seen_frames=len(seen), force_target=step <= 240,
                              frames=samples, loss=3.1, terms={name: 1. for name in compare.LOSS},
                              gradient_norm=1., gradient_groups=groups))
        write_rows(folder / 'trace.jsonl', trace)
        ckpt = put(folder / 'step_0300.pt', (arm + 'step300').encode())
        partial = put(folder / 'step_0150.pt', (arm + 'step150').encode())
        done = dict(status='complete', arm=arm, source=source, steps=300, inherited_codec_steps=3000,
            additional_codec_updates=300, frames_seen=354, first_epoch_coverage_at_step=177,
            no_target_forcing_steps=60, initial_model_sha256='a' * 64, checkpoint_sha256=ckpt['sha256'],
            initial_encoder_sha256='b' * 64, final_encoder_sha256=('b' if arm == 'decoder_only' else 'c') * 64,
            initial_encoder_parameters_sha256='d' * 64,
            final_encoder_parameters_sha256=('d' if arm == 'decoder_only' else 'e') * 64,
            fresh_optimizer_verified=True, trace_sha256=sha(folder / 'trace.jsonl'),
            checkpoint_0150_sha256=partial['sha256'], frozen_encoder_verified=arm == 'decoder_only', goal_complete=False)
        write(folder / 'TRAIN_DONE.json', done)
        dones[arm] = done
    for label, error in (('baseline', .06), ('decoder_only', .05), ('full_codec', .04)):
        folder = root / 'evaluations' / label
        folder.mkdir(parents=True)
        write(folder / 'SOURCE.json', dict(source, arm=label))
        checkpoint = source_digest if label == 'baseline' else dones[label]['checkpoint_sha256']
        encoder = ('c' if label == 'full_codec' else 'b') * 64
        step = 0 if label == 'baseline' else 300
        metrics = {
            'codec_real': dict(mae=error, psnr=18., workspace_roi_mae=error + .01, workspace_roi_psnr=17.),
            'teacher_real': dict(mae=.02, psnr=30., workspace_roi_mae=.03, workspace_roi_psnr=29.),
            'codec_teacher': dict(mae=error, psnr=20., workspace_roi_mae=error + .01, workspace_roi_psnr=19.),
            'pseudo_view_teacher': dict(mae=error + .01, psnr=18., workspace_roi_mae=error + .02, workspace_roi_psnr=17.)}
        replay = {view: dict(mae=0., psnr=120., workspace_roi_mae=0., workspace_roi_psnr=120.)
                  for view in ('real_view', 'nearby_view')}
        targets, packets, pictures = [], [], []
        for clip in sources.ALL_CLIPS:
            for frame in range(177):
                common = dict(clip=clip, frame=frame, split='TRAIN' if clip in sources.TRAIN_CLIPS else 'EVAL',
                              pts=frame / 29.97, checkpoint_sha256=checkpoint, encoder_sha256=encoder,
                              additional_codec_updates=step)
                targets.append(dict(common, **metrics, replay=replay, latent_positions=20, latent_channels=64,
                                    input_gaussians=112896, input_voxels=200, decoded_gaussians=200))
                relative = f'states/{clip}/{frame:03d}.pt'
                record = put(folder / relative, f'{label}/{clip}/{frame}'.encode())
                packets.append(dict(common, arm=label, normalization={'scale': 2.},
                    source_cache_sha256=cache_hash[(clip, frame)], binding_sha256=source['binding_sha256'],
                    inherited_codec_steps=3000, decoder_target_key=None, fresh_encode=True, frame_count=177,
                    decoded_gaussians=200, state_path=relative, state_sha256=record['sha256'],
                    state_bytes=record['bytes'], shape=[20, 64], saved_packet_replay=replay))
                if frame in (0, 44, 88, 132, 176):
                    for kind in ('real', 'teacher', 'codec', 'nearby', 'nearby_teacher'):
                        relative = f'screenshots/{clip}/{frame:03d}_{kind}.png'
                        record = put(folder / relative, png())
                        pictures.append(dict(clip=clip, frame=frame, kind=kind, path=relative,
                                             bytes=record['bytes'], sha256=record['sha256']))
        write_rows(folder / 'targets.jsonl', targets)
        write_rows(folder / 'state_manifest.jsonl', packets)
        write_rows(folder / 'screenshot_manifest.jsonl', pictures)
        write(folder / 'EVALUATION.json', dict(status='evaluated', arm=label, frames=531, frames_per_clip=177,
            additional_codec_updates=step, inherited_codec_steps=3000, checkpoint_sha256=checkpoint,
            encoder_sha256=encoder, source=source, summary_by_clip={clip: metrics for clip in sources.ALL_CLIPS},
            summary_all=metrics, roi_xyxy=[67, 38, 380, 227], every_packet_saved_replay_passed=True,
            all_frames_freshly_encoded=True, decoder_eval=True, decoder_target_key=None,
            targets_sha256=sha(folder / 'targets.jsonl'), state_manifest_sha256=sha(folder / 'state_manifest.jsonl'),
            screenshot_manifest_sha256=sha(folder / 'screenshot_manifest.jsonl'), goal_complete=False))
    return source_digest


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='multisource-compare-test-')
        cls.root = Path(cls.temporary.name)
        cls.original = make_fixture(cls.root)
        cls.patches = [patch.object(compare, 'INITIAL_SHA256', cls.original),
                       patch.object(sources, 'CODEC_SHA', cls.original)]
        for change in cls.patches:
            change.start()

    @classmethod
    def tearDownClass(cls):
        for change in reversed(cls.patches):
            change.stop()
        cls.temporary.cleanup()

    def setUp(self):
        self.restore = {}

    def tearDown(self):
        for path, data in self.restore.items():
            path.write_bytes(data)

    def mutate(self, relative, operation):
        path = self.root / relative
        self.restore.setdefault(path, path.read_bytes())
        document = read_rows(path) if path.suffix == '.jsonl' else read(path)
        operation(document)
        (write_rows if path.suffix == '.jsonl' else write)(path, document)

    def refresh_receipt(self, directory, stem, receipt='EVALUATION.json'):
        self.mutate(f'{directory}/{receipt}', lambda doc: doc.update({f'{stem}_sha256': sha(self.root / directory / f'{stem}.jsonl')}))

    def rejected(self, expected):
        result = compare.build_summary(self.root)
        self.assertFalse(result['comparison_eligibility']['eligible'])
        self.assertEqual({}, result['paired_comparisons'])
        self.assertIn(expected, '\n'.join(result['comparison_eligibility']['reasons']))
        self.assertFalse(result['goal_complete'])

    def test_complete_artifacts_bad_science_still_engineering_complete(self):
        result = compare.build_summary(self.root)
        self.assertEqual(result['comparison_eligibility']['reasons'], [])
        self.assertEqual('complete', result['engineering_status'])
        self.assertFalse(result['goal_complete'])
        self.assertIsNone(result['production_codec_selected'])
        self.assertEqual(15, len(result['visual_acceptance']['cases']))
        self.assertFalse(result['absolute_quality_reference']['results']['full_codec'][sources.EVAL_CLIP]['reference_passed'])
        ci = result['paired_comparisons']['full_codec_vs_decoder_only'][sources.TRAIN_CLIPS[0]]['codec_real']['mae']
        self.assertAlmostEqual(.01, ci['mean_improvement'])
        self.assertAlmostEqual(.01, ci['ci95_low'])
        self.assertAlmostEqual(.01, ci['ci95_high'])
        self.assertEqual(177, ci['physical_frames'])

    def test_role_leakage_rejected_even_after_trace_hash_refreshed(self):
        self.mutate('arms/full_codec/trace.jsonl', lambda doc: doc[0]['frames'][0].update(clip=sources.EVAL_CLIP))
        self.refresh_receipt('arms/full_codec', 'trace', 'TRAIN_DONE.json')
        self.rejected('EVAL leakage')

    def test_missing_frame_rejected_even_after_metric_hash_refreshed(self):
        self.mutate('evaluations/baseline/targets.jsonl', lambda doc: doc.pop())
        self.refresh_receipt('evaluations/baseline', 'targets')
        self.rejected('incomplete 531-frame')

    def test_short_training_and_short_trace_both_rejected(self):
        self.mutate('arms/full_codec/TRAIN_DONE.json', lambda doc: doc.update(steps=299))
        self.rejected('TRAIN_DONE.steps')
        self.mutate('arms/full_codec/TRAIN_DONE.json', lambda doc: doc.update(steps=300))
        self.mutate('arms/full_codec/trace.jsonl', lambda doc: doc.pop())
        self.refresh_receipt('arms/full_codec', 'trace', 'TRAIN_DONE.json')
        self.rejected('exactly 300')

    def test_encoder_buffer_drift_rejected(self):
        self.mutate('arms/decoder_only/TRAIN_DONE.json', lambda doc: doc.update(final_encoder_sha256='9' * 64))
        self.rejected('parameters or buffers changed')

    def test_full_encoder_unchanged_rejected(self):
        self.mutate('arms/full_codec/TRAIN_DONE.json', lambda doc: doc.update(final_encoder_parameters_sha256='d' * 64))
        self.rejected('encoder parameters did not update')

    def test_packet_tamper_rejected(self):
        path = self.root / 'evaluations/full_codec/states' / sources.EVAL_CLIP / '000.pt'
        self.restore[path] = path.read_bytes()
        path.write_bytes(b'tampered')
        self.rejected('SHA256 mismatch')

    def test_checkpoint_tamper_rejected(self):
        path = self.root / 'arms/full_codec/step_0300.pt'
        self.restore[path] = path.read_bytes()
        path.write_bytes(b'tampered')
        self.rejected('SHA256 mismatch')

    def test_replay_failure_rejected_after_receipt_hash_refreshed(self):
        self.mutate('evaluations/full_codec/targets.jsonl', lambda doc: doc[0]['replay']['nearby_view'].update(psnr=59.))
        self.refresh_receipt('evaluations/full_codec', 'targets')
        self.rejected('replay failed')

    def test_wrong_target_role_rejected(self):
        self.mutate('evaluations/full_codec/targets.jsonl', lambda doc: doc[-1].update(split='TRAIN'))
        self.refresh_receipt('evaluations/full_codec', 'targets')
        self.rejected('wrong role')

    def test_missing_metric_rejected(self):
        self.mutate('evaluations/full_codec/targets.jsonl', lambda doc: doc[-1]['codec_real'].pop('mae'))
        self.refresh_receipt('evaluations/full_codec', 'targets')
        self.rejected('missing/nonfinite')

    def test_missing_evidence_cli_writes_refusal_and_returns_two(self):
        with tempfile.TemporaryDirectory(prefix='multisource-compare-empty-') as directory:
            code = compare.main(['--root', directory])
            self.assertEqual(2, code)
            result = read(Path(directory) / 'SUMMARY.json')
            self.assertEqual('inconclusive', result['scientific_status'])
            self.assertFalse(result['goal_complete'])
            self.assertTrue((Path(directory) / 'SUMMARY.md').is_file())

    def test_import_does_not_load_gpu_runtime(self):
        command = 'import sys, compare; assert "torch" not in sys.modules; assert "MinkowskiEngine" not in sys.modules'
        result = subprocess.run([sys.executable, '-c', command], cwd=Path(__file__).parent,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_missing_hash_cannot_disable_verification(self):
        with self.assertRaisesRegex(ValueError, 'missing/invalid SHA256'):
            compare.check_hash(self.root / 'original/codec_final.pt', None)


if __name__ == '__main__':
    unittest.main()
