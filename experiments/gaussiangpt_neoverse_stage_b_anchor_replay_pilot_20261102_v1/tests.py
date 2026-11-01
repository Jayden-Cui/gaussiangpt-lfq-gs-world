"""Real invariants for the matched pilot's bounded data exposure."""
from collections import Counter
import copy
import sampling
import pilot
import torch


def raises(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Invalid pilot schedule was accepted')


items=sampling.make_schedule(20261103,20261104)
witness=sampling.audit(items)
assert witness['main_pairs']==756 and witness['updates']==189
assert items==sampling.make_schedule(20261103,20261104)
assert items!=sampling.make_schedule(20261103,20261105)
assert Counter(v['horizon'] for x in items for v in x['batch'])=={h:84 for h in range(1,10)}
bad=copy.deepcopy(items);bad[-1]['batch'][-1]=bad[0]['batch'][0]
raises(lambda:sampling.audit(bad))
bad=copy.deepcopy(items);bad[0]['anchor']['start']=0
raises(lambda:sampling.audit(bad))
bad=copy.deepcopy(items);bad[0]['batch']=bad[0]['batch'][:3]
raises(lambda:sampling.audit(bad))

# Adam's CPU step counters can alias a supplied state dict; every trial arm
# must receive its own deep copy so the second arm starts from the same state.
a=torch.nn.Parameter(torch.tensor([1.0]));opt=torch.optim.Adam([a],lr=.01)
(a.square().sum()).backward();opt.step()
checkpoint=copy.deepcopy(opt.state_dict());saved=copy.deepcopy(checkpoint)
b=torch.nn.Parameter(torch.tensor([1.0]));other=torch.optim.Adam([b],lr=.01)
other.load_state_dict(copy.deepcopy(checkpoint))
assert pilot.equal_tree(other.state_dict(),checkpoint)
(b.square().sum()).backward();other.step()
assert pilot.equal_tree(checkpoint,saved)
assert not pilot.equal_tree(other.state_dict(),saved)
print('PASS: balanced matched schedules, corrupted-sample rejection, isolated Adam state')
