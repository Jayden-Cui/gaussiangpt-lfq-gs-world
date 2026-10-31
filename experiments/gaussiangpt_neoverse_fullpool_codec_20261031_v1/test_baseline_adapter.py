"""Synthetic CPU-only contract tests; no SSH, production data, codec, or GPU."""
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import baseline_adapter as a


def values(error=.1):
    return dict(mae=error, psnr=20., workspace_roi_mae=error, workspace_roi_psnr=20.)


def all_rows():
    return [dict(clip=clip, frame=frame, split='EVAL-only' if clip==a.EVAL_CLIP else 'TRAIN',
                 **{c: values(.2 if clip==a.EVAL_CLIP else .1) for c in a.CATEGORIES})
            for clip in [*a.pool.EXPECTED, a.EVAL_CLIP] for frame in range(177)]


def state(prepared=True):
    normal = dict(center=[0.,0.,0.], scale=.25, fixed_for_all_frames=True)
    metadata = dict(clip='buildingblock_tower_7_0', frame=1, pts=1/29.97,
                    normalization=normal, source_cache_sha256='1'*64)
    if prepared:
        metadata['codec_checkpoint_sha256'] = a.INITIAL_SHA
    else:
        metadata.update(checkpoint_sha256=a.INITIAL_SHA, arm='baseline', additional_codec_updates=0,
                        inherited_codec_steps=3000, decoder_target_key=None, binding_sha256=a.BINDING_SHA,
                        encoder_sha256=a.ENCODER_SHA)
    result = dict(schema=a.PREPARED_SCHEMA if prepared else 'multisource_continuous_encoder_state_v1',
                  metadata=metadata, coordinates=torch.tensor([[0,0,0],[1,0,0]],dtype=torch.int32),
                  features=torch.zeros(2,64), stride=4, grid=.008, feature_dtype='float32')
    if prepared:
        result.update(clip=metadata['clip'], frame=1, pts=1/29.97, normalization=normal, source_cache_sha256='1'*64)
    return result, normal


def old_fixture():
    receipt = dict(status='evaluated', arm='baseline', frames=531, checkpoint_sha256=a.INITIAL_SHA,
        encoder_sha256=a.ENCODER_SHA, additional_codec_updates=0, decoder_eval=True,
        decoder_target_key=None, every_packet_saved_replay_passed=True, roi_xyxy=list(a.ROI),
        targets_sha256=a.OLD_TARGETS_SHA, state_manifest_sha256=a.OLD_MANIFEST_SHA,
        source=dict(binding_sha256=a.BINDING_SHA))
    replay = {view:dict(mae=1e-7,psnr=100.,workspace_roi_mae=1e-7,workspace_roi_psnr=100.)
              for view in ('real_view','nearby_view')}
    rows=[]; manifests=[]
    for clip in [*a.HISTORICAL, 'buildingblock_tower_10_176']:
        for frame in range(177):
            shared=dict(clip=clip,frame=frame,checkpoint_sha256=a.INITIAL_SHA,
                        encoder_sha256=a.ENCODER_SHA,additional_codec_updates=0)
            rows.append(dict(**shared, **{c:values() for c in a.CATEGORIES},replay=deepcopy(replay)))
            manifests.append(dict(**shared,arm='baseline',decoder_target_key=None,saved_packet_replay=deepcopy(replay)))
    return receipt, rows, manifests


class BaselineAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    def test_metrics_match_exact_original_worker_on_cpu(self):
        path=Path(__file__).resolve().parent.parent/a.pool.LEGACY_RUN/'worker.py'
        spec=importlib.util.spec_from_file_location('_baseline_test_legacy_worker',path)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        generator=torch.Generator().manual_seed(47)
        image=torch.rand((252,448,3),generator=generator);target=torch.rand((252,448,3),generator=generator)
        self.assertEqual(a.image_metrics(image,target),legacy.image_metrics(image,target))

    def test_roi_formula_and_identical_images_psnr_floor(self):
        image=torch.zeros(252,448,3);target=image.clone();image[38:227,67:380]=.5
        measured=a.image_metrics(image,target)
        self.assertAlmostEqual(measured['workspace_roi_mae'],.5)
        self.assertAlmostEqual(measured['workspace_roi_psnr'],-10*math.log10(.25),places=5)
        self.assertAlmostEqual(measured['mae'],.5*189*313/(252*448),places=6)
        identical=a.image_metrics(target,target)
        self.assertEqual(identical['psnr'],120.)
        self.assertEqual(identical['mae'],0.)

    def test_saved_reconstruction_must_be_float32_finite_full_resolution(self):
        image=torch.zeros(252,448,3)
        for bad in [image.double(),image[:251],image.clone().fill_(float('nan'))]:
            with self.subTest(shape=bad.shape,dtype=bad.dtype), self.assertRaises(ValueError):
                a.image_metrics(bad,image)

    def test_four_categories_teacher_clamp_and_no_quantization(self):
        images=torch.full((2,252,448,3),.1234567)
        cache=dict(real=torch.zeros(252,448,3),targets=torch.full((2,252,448,3),2.))
        result=a.compute_metrics(images,cache)
        self.assertEqual(set(result),set(a.CATEGORIES))
        self.assertAlmostEqual(result['codec_real']['mae'],.1234567,places=6)
        self.assertEqual(result['teacher_real']['mae'],1.)
        with self.assertRaises(ValueError): a.compute_metrics(images+1,cache)

    def test_tolerance_is_explicit_not_bitwise_and_drift_rejected(self):
        actual={c:values() for c in a.CATEGORIES}
        recorded={c:{m:actual[c][m] for m in a.TOLERANCE} for c in a.CATEGORIES}
        recorded['codec_real']['mae']+=1e-7
        differences=a.compare_recorded(actual,recorded)
        self.assertGreater(differences['codec_real.mae'],0)
        recorded['codec_real']['mae']+=1e-3
        with self.assertRaises(ValueError):a.compare_recorded(actual,recorded)

    def test_nonfinite_or_extra_historical_metric_fails(self):
        actual={c:values() for c in a.CATEGORIES}
        recorded={c:{m:actual[c][m] for m in a.TOLERANCE} for c in a.CATEGORIES}
        recorded['codec_teacher']['mae']=float('nan')
        with self.assertRaises(ValueError):a.compare_recorded(actual,recorded)

    def test_actual_bytes_are_hashed_before_tensor_load_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.pt';torch.save(dict(feature=torch.tensor([.1234567])),path)
            record=dict(bytes=path.stat().st_size,sha256=a.sha(path));audit=a.pool.Audit()
            loaded=a.load_tensor(path,record,audit)
            self.assertEqual(loaded['feature'].device.type,'cpu');audit.finish()
            data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with patch.object(torch,'load',side_effect=AssertionError('must not deserialize bad bytes')):
                with self.assertRaises(ValueError):a.load_tensor(path,record,audit)

    def test_wrong_byte_count_and_previous_audit_binding_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x';path.write_bytes(b'abc');audit=a.pool.Audit()
            digest=hashlib.sha256(b'abc').hexdigest()
            with self.assertRaises(ValueError):a.bound_bytes(path,digest,4,audit)
            a.bound_bytes(path,digest,3,audit);path.write_bytes(b'abd')
            with self.assertRaises(ValueError):a.bound_bytes(path,a.sha(path),3,audit)

    def test_prepared_state_accepts_legal_quarter_scale(self):
        packet,normal=state()
        a.validate_state(packet,'buildingblock_tower_7_0',1,normal,'1'*64,prepared=True)

    def test_state_rejects_wrong_weight_normalization_cache_and_time(self):
        packet,normal=state()
        for key,value in [('codec_checkpoint_sha256','2'*64),('normalization',dict(scale=.5)),
                          ('source_cache_sha256','2'*64),('pts',5.)]:
            with self.subTest(key=key):
                bad=deepcopy(packet);bad['metadata'][key]=value
                with self.assertRaises(ValueError):
                    a.validate_state(bad,'buildingblock_tower_7_0',1,normal,'1'*64,prepared=True)

    def test_state_rejects_wrong_width_grid_and_duplicate_coords(self):
        packet,normal=state()
        for key,value in [('features',torch.zeros(2,512)),('grid',.004),('coordinates',torch.zeros(2,3,dtype=torch.int32))]:
            with self.subTest(key=key):
                bad=deepcopy(packet);bad[key]=value
                with self.assertRaises(ValueError):a.validate_state(bad,packet['clip'],1,normal,'1'*64,prepared=True)

    def test_historical_state_requires_target_free_original_checkpoint(self):
        packet,normal=state(False)
        a.validate_state(packet,'buildingblock_tower_7_0',1,normal,'1'*64,prepared=False)
        for key,value in [('decoder_target_key','forced'),('additional_codec_updates',300),('binding_sha256','2'*64)]:
            bad=deepcopy(packet);bad['metadata'][key]=value
            with self.assertRaises(ValueError):a.validate_state(bad,'buildingblock_tower_7_0',1,normal,'1'*64,prepared=False)

    def test_full_9027_coverage_and_eval_split_aggregation(self):
        result=a.summaries(all_rows())
        self.assertEqual(len(result['summary_by_clip']),51)
        self.assertAlmostEqual(result['summary_train_frames']['codec_real']['mae'],.1)
        self.assertAlmostEqual(result['summary_exposed_eval']['codec_real']['mae'],.2)

    def test_missing_duplicate_foreign_frame_and_eval_leakage_fail(self):
        good=all_rows()
        bad_cases=[good[:-1],good[:-1]+[good[0]]]
        for field,value in [('clip','foreign'),('frame',True),('split','TRAIN')]:
            bad=list(good);bad[-1]=dict(bad[-1],**{field:value});bad_cases.append(bad)
        for bad in bad_cases:
            with self.assertRaises(ValueError):a.check_coverage(bad)

    def test_old_baseline_exact_metrics_and_two_view_replay_contract(self):
        receipt,rows,manifests=old_fixture()
        tables=a.historical_rows(receipt,rows,manifests)
        self.assertEqual(len(tables[0]),531)
        self.assertEqual(tables[0][rows[0]['clip'],0]['codec_real'],rows[0]['codec_real'])

    def test_old_baseline_changed_weight_counts_forcing_or_replay_fails(self):
        receipt,rows,manifests=old_fixture()
        for field,value in [('frames',354),('checkpoint_sha256','2'*64),('decoder_target_key','forced')]:
            bad=dict(receipt,**{field:value})
            with self.assertRaises(ValueError):a.historical_rows(bad,rows,manifests)
        bad=deepcopy(manifests);bad[0]['saved_packet_replay']['nearby_view']['mae']=.1
        with self.assertRaises(ValueError):a.historical_rows(receipt,rows,bad)

    def test_budget_fails_if_cuda_was_initialized(self):
        budget=a.Budget(1)
        with patch.object(torch.cuda,'is_initialized',return_value=True):
            with self.assertRaises(ValueError):budget.check()
        with self.assertRaises(ValueError):a.Budget(0)

    def test_read_jsonl_detects_duplicate_json_keys_and_wrong_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl';path.write_bytes(b'{"frame":1,"frame":2}\n')
            with self.assertRaises(ValueError):a.read_lines(path,a.sha(path),a.pool.Audit())
            with self.assertRaises(ValueError):a.read_lines(path,'0'*64,a.pool.Audit())


if __name__=='__main__':
    unittest.main()
