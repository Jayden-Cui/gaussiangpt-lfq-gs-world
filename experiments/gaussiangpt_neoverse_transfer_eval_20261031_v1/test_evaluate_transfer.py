"""Pure CPU checks of target coverage, missing motion values and overlap reduction."""
import copy
import unittest

import evaluate_transfer as evaluation


def rows(initial=False):
    result = []
    for start in range(168):
        for h in ([0] if initial else range(1, 10)):
            motion = None if initial else .25
            row = dict(clip=evaluation.CLIP, start=start, end=start+h, horizon=h,
                       motion_ids=[17]*9, blinded_motion_ids=[18]*9)
            row.update({control: dict(mae=.1, psnr=20., workspace_roi_mae=.2,
                                      workspace_roi_psnr=15., motion_mae=motion)
                        for control in evaluation.CONTROLS})
            row['delta'] = {control: dict(mae=.11, workspace_roi_mae=.21, motion_mae=motion)
                            for control in evaluation.CONTROLS}
            result.append(row)
    return result


class TransferEvaluation(unittest.TestCase):
    def test_complete_physical_coverage_and_bundles(self):
        complete = rows()
        evaluation.validate_rows(complete)
        evaluation.validate_rows(rows(True), initial=True)
        for bad in (complete[:-1], complete[:-1]+[complete[0]]):
            with self.assertRaises(ValueError):
                evaluation.validate_rows(bad)
        invalid = copy.deepcopy(complete)
        invalid[1]['motion_ids'][0] = 19
        with self.assertRaises(ValueError):
            evaluation.validate_rows(invalid)
        invalid = copy.deepcopy(complete)
        invalid[0]['blinded_motion_ids'][0] = 65536
        with self.assertRaises(ValueError):
            evaluation.validate_rows(invalid)

    def test_motion_null_mask_is_target_dependent(self):
        complete = rows()
        for control in evaluation.CONTROLS:
            complete[0][control]['motion_mae'] = None
            complete[0]['delta'][control]['motion_mae'] = None
        evaluation.validate_rows(complete)
        complete[0]['delta']['blinded']['motion_mae'] = 0.
        with self.assertRaises(ValueError):
            evaluation.validate_rows(complete)
        initial = rows(True)
        initial[0]['correct']['motion_mae'] = 0.
        with self.assertRaises(ValueError):
            evaluation.validate_rows(initial, initial=True)

    def test_physical_frame_means_and_nulls(self):
        # Frame1 contributes once, frame2 has two overlapping predictions.
        observations = [dict(end=1, correct=dict(mae=0., psnr=20., workspace_roi_mae=0.,
            workspace_roi_psnr=20., motion_mae=None)),
            dict(end=2, correct=dict(mae=2., psnr=20., workspace_roi_mae=2.,
            workspace_roi_psnr=20., motion_mae=3.)),
            dict(end=2, correct=dict(mae=4., psnr=20., workspace_roi_mae=4.,
            workspace_roi_psnr=20., motion_mae=5.))]
        result = evaluation.aggregate(observations, 'correct')
        self.assertEqual(result['frame_uniform']['mae'], 1.5)
        self.assertEqual(result['frame_uniform']['motion_mae'], 4.)
        self.assertEqual(result['valid_physical_frames']['motion_mae'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
