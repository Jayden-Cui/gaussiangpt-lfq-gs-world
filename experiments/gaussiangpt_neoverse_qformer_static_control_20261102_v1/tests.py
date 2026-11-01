"""Check the same-camera control metrics on analytically known images."""
import copy
import torch
import static_probe as s


def raises(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid control was accepted')


still=torch.zeros((2,2,3));moving=still.clone();moving[0,0]=.5
correct=s.weighted_controls(moving,still,moving,still)
assert correct['motion_code_to_motion_reference']==0
assert correct['static_code_to_static_reference']==0
assert correct['static_code_to_motion_reference']==.5
wrong=s.weighted_controls(moving,moving,moving,still)
assert wrong['static_code_to_static_reference']==.5
assert wrong['static_code_to_motion_reference']==0
raises(lambda:s.weighted_controls(still,still,still,still))
bad=still.clone();bad[0,0,0]=float('nan')
raises(lambda:s.weighted_controls(bad,still,moving,still))
rows=[{'start':start,'horizon':h,'metrics':copy.deepcopy(correct)} for start in (35,123) for h in range(1,10)]
out=s.aggregate(rows)
assert out['overall']['static_prediction_closer_to_static_than_motion'] is True
assert out['h9']['static_code_to_static_reference']==0
raises(lambda:s.aggregate(rows[:-1]+[rows[0]]))
print('PASS: same-camera motion weighting, ideal/wrong controls, finite and 18-frame coverage')
