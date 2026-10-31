"""Bounded synthetic CPU evidence tests; no GPU, SSH, tensors or full artifact tree."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
import compare as c

def metric_row(value):
    return {g:dict(mae=value,workspace_roi_mae=value,psnr=20+value,workspace_roi_psnr=19+value) for g in c.GROUPS}

class CompareTests(unittest.TestCase):
    def test_exact9027_coverage_rejects_missing_duplicate_foreign_and_boolean_frames(self):
        rows=[dict(clip=clip,frame=frame) for clip in c.CLIPS for frame in range(177)]
        self.assertEqual(len(c.index_frames(rows)),9027)
        for invalid in (rows[:-1],rows+[deepcopy(rows[0])],rows+[dict(clip='foreign',frame=0)]):
            with self.assertRaises(ValueError):c.index_frames(invalid)
        invalid=deepcopy(rows);invalid[1]['frame']=True
        with self.assertRaises(ValueError):c.index_frames(invalid)

    def test_hash_and_size_tampering_detected_with_small_files(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'packet.pt';p.write_bytes(b'synthetic bytes, not a Torch packet')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            c.check_hash(p,digest,p.stat().st_size)
            with self.assertRaises(ValueError):c.check_hash(p,'f'*64,p.stat().st_size)
            with self.assertRaises(ValueError):c.check_hash(p,digest,p.stat().st_size+1)
            p.write_bytes(b'different same-length value')
            with self.assertRaises(ValueError):c.check_hash(p,digest)

    def test_summary_must_match_physical_frame_metrics_not_reordered_labels(self):
        rows=[metric_row(.1),metric_row(.3)]
        means=c.mean_metrics(rows);c.compare_means(means,deepcopy(means),'valid')
        wrong=deepcopy(means);wrong['codec_real']['mae']=.3
        with self.assertRaises(ValueError):c.compare_means(means,wrong,'wrong clip summary')
        wrong=deepcopy(means);wrong['teacher_real']['psnr']=float('nan')
        with self.assertRaises(ValueError):c.compare_means(means,wrong,'nonfinite')

    def test19_source_groups_equal_weight_differs_from50_clip_mean_and_excludes14(self):
        first=next(iter(c.pool.EXPECTED.values()))
        by_clip={clip:metric_row(.9 if group==first else .1) for clip,group in c.pool.EXPECTED.items()}
        by_clip[c.EVAL_CLIP]=metric_row(.8)
        a=c.aggregates(by_clip)
        self.assertEqual(len(a['by_train_source_group']),19)
        self.assertEqual(sum(a['group_clip_counts'].values()),50)
        group_mean=(.9+18*.1)/19
        clip_mean=sum(v['codec_real']['mae'] for k,v in by_clip.items() if k!=c.EVAL_CLIP)/50
        self.assertAlmostEqual(a['train_source_group_equal']['codec_real']['mae'],group_mean)
        self.assertAlmostEqual(a['train_clip_equal']['codec_real']['mae'],clip_mean)
        self.assertNotAlmostEqual(group_mean,clip_mean)
        self.assertEqual(a['exposed_eval_only']['codec_real']['mae'],.8)

    def test_teacher_alignment_allows_declared_numeric_noise_but_rejects_different_targets(self):
        a,b=metric_row(.2),metric_row(.2)
        b['teacher_real']['mae']+=1e-6;b['teacher_real']['psnr']+=.001
        c.teacher_alignment(a,b)
        b['teacher_real']['mae']+=.001
        with self.assertRaises(ValueError):c.teacher_alignment(a,b)

    def test_replay_requires_both_views_and_both_thresholds(self):
        scores=dict(mae=0.,psnr=120.,workspace_roi_mae=0.,workspace_roi_psnr=120.)
        replay=dict(real_view=deepcopy(scores),nearby_view=deepcopy(scores));c.check_replay(replay)
        for bad in ({'real_view':scores},dict(real_view=scores,nearby_view={**scores,'psnr':59.}),
                    dict(real_view={**scores,'mae':2e-5},nearby_view=scores)):
            with self.assertRaises(ValueError):c.check_replay(bad)

    def test_original_pinned_bootstrap_covers177_and_keeps_direction(self):
        bootstrap=c.bootstrap_function(Path(__file__).resolve().parent)
        result=bootstrap({f:.125 for f in range(177)})
        self.assertEqual(result['mean_improvement'],.125)
        self.assertEqual(result['ci95_low'],.125);self.assertEqual(result['ci95_high'],.125)
        self.assertTrue(result['positive_interval'])
        self.assertEqual((result['block_length'],result['replicates'],result['seed']),(16,2000,20261031))
        with self.assertRaises(ValueError):bootstrap({f:.125 for f in range(176)})

    def test_metadata_field_types_do_not_accept_true_as_one_or_swap_checkpoint(self):
        with self.assertRaises(ValueError):c.fields({'frames':True},{'frames':1},'count')
        with self.assertRaises(ValueError):c.fields({'checkpoint':'a'*64},{'checkpoint':'b'*64},'checkpoint')

    def test_fixed_visual_cardinality_is1275_and_no_target_tensor_imports(self):
        expected={(clip,frame,kind) for clip in c.CLIPS for frame in c.SNAPSHOTS for kind in c.PICTURES}
        self.assertEqual(len(expected),1275)
        self.assertNotIn('torch',c.__dict__)
        self.assertNotIn('worker',c.__dict__)

if __name__=='__main__':unittest.main()
