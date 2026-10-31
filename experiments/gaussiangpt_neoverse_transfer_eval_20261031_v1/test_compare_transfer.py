"""New comparison boundaries only; never executes the old bootstrap or GPU."""
import copy
import hashlib
import json
import unittest

import compare_transfer as c


def values(value=0., motion=0.):
    return dict(mae=value, psnr=20., workspace_roi_mae=value, workspace_roi_psnr=20., motion_mae=motion)


def rows(missing_frame=None):
    result=[]
    for s,h in sorted(c.KEYS):
        motion=None if s+h==missing_frame else 1.
        row=dict(start=s,end=s+h,horizon=h,clip=c.CLIP,
                 motion_ids=[s]*9,blinded_motion_ids=[s]*9)
        row.update({control:values(1.,motion) for control in c.CONTROLS})
        row['delta']={control:dict(mae=1.,workspace_roi_mae=1.,motion_mae=motion) for control in c.CONTROLS}
        result.append(row)
    return result


class ComparisonTests(unittest.TestCase):
    def test_physical_frame_weighting_and_null_counts(self):
        data=[dict(start=0,end=1,horizon=1,correct=values(0.,None)),
              dict(start=0,end=2,horizon=2,correct=values(1.,2.)),
              dict(start=1,end=2,horizon=1,correct=values(1.,None))]
        reduced=c.reduce_rows(data,'correct')
        self.assertEqual(reduced['frame_uniform']['mae'],.5)
        self.assertAlmostEqual(reduced['exposure_mean']['mae'],2/3)
        self.assertEqual(reduced['frame_uniform']['motion_mae'],2.)
        self.assertEqual(reduced['valid_physical_frames']['motion_mae'],1)
        self.assertEqual(reduced['valid_exposures']['motion_mae'],1)

    def test_missing_motion_frame_has_no_bootstrap(self):
        data=rows(missing_frame=73)
        def forbidden(_):raise AssertionError('Missing frame must not enter old bootstrap')
        result=c.paired(data,data,'shuffle84',forbidden,metrics=('motion_mae',))['motion_mae']
        self.assertFalse(result['available'])
        self.assertFalse(result['positive_interval'])
        self.assertIsNone(result['ci95_low'])
        self.assertEqual(result['physical_frames'],175)
        self.assertEqual(result['missing_physical_frames'],[73])

    def test_complete_series_dispatches_only_176_frame_means(self):
        left=rows();right=copy.deepcopy(left)
        for row in right:row['correct']['mae']+=row['end']/176
        calls=[]
        def witness(frame_means):
            calls.append(frame_means)
            return dict(mean_improvement=sum(frame_means.values())/176,ci95_low=.1,ci95_high=1.,
                        positive_interval=True,physical_frames=176)
        result=c.paired(left,right,'correct',witness,metrics=('mae',))['mae']
        self.assertEqual(len(calls),1)
        self.assertEqual(sorted(calls[0]),list(range(1,177)))
        self.assertAlmostEqual(calls[0][73],73/176)
        self.assertTrue(result['available'])
        self.assertNotIn('TRAIN',result['interval'])

    def test_row_masks_ids_and_complete_coverage(self):
        data=rows();c.validate_rows(data)
        bad=copy.deepcopy(data);bad[0]['shuffle84']['motion_mae']=None
        with self.assertRaises(ValueError):c.validate_rows(bad)
        bad=copy.deepcopy(data);bad[0]['motion_ids'][0]=65536
        with self.assertRaises(ValueError):c.validate_rows(bad)
        with self.assertRaises(ValueError):c.validate_rows(data[:-1])

    def test_same_step_source_and_frozen_state_rejections(self):
        digest='a'*64
        selection=dict(selected_step=300,checkpoints={'z_rgb':dict(path='/fixture/z_rgb/step_0300.pt',sha256=digest)})
        inputs={'/fixture/summary.json':'b'*64}
        context=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        expected=dict(code_manifest_sha256='c'*64,protocol_sha256='d'*64)
        evaluation=dict(schema='gs_tower14_transfer_evaluation_v1',status='complete',arm='z_rgb',
            selected_step=300,checkpoint_path=selection['checkpoints']['z_rgb']['path'],checkpoint_sha256=digest,
            windows=168,future_exposures=1512,unique_future_frames=176,initial_frames=168,optimizer_updates=0,
            all_parameters_frozen=True,goal_complete=False,scientific_transfer_passed=None,
            target_input_sha256=inputs,target_context_sha256=context,clip=c.CLIP,resolution_hw=[252,448],
            workspace_roi_xyxy=[67,38,380,227],all_initial_latent_points_retained=True,
            native_internal_hard_pruning_retained=True,gpu_peak_gib=3.,code_mode='posterior',**expected)
        for family in ('model_state','codec_state'):
            evaluation[family+'_sha256_before']=evaluation[family+'_sha256_after']=digest
        for key in ('source','before','after'):evaluation['output_std_sha256_'+key]=digest
        self.assertEqual(c.evaluation_contract(evaluation,'z_rgb',selection,expected),context)
        for key,value in (('selected_step',200),('protocol_sha256','e'*64),
                          ('output_std_sha256_after','f'*64),('optimizer_updates',1),('target_context_sha256','0'*64)):
            with self.assertRaises(ValueError):
                c.evaluation_contract(dict(evaluation,**{key:value}),'z_rgb',selection,expected)
        with self.assertRaises(ValueError):c.evaluation_contract(evaluation,'z_rgb',selection,expected,'0'*64)


if __name__=='__main__':unittest.main(verbosity=2)
