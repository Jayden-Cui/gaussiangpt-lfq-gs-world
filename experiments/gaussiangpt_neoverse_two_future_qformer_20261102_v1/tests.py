"""Check balanced same-initial-state training and native packet coverage."""
from collections import Counter
import copy
import sys
from q_utils import ROOT, expected_packet_names, read, schedule, validate_protocol

p=read(ROOT/'protocol.json')
rows=schedule(p)
assert len(rows)==1194
assert sum(x['phase']=='distill' for x in rows)==600
assert sum(x['phase']=='joint' for x in rows)==594
assert Counter(x['horizon'] for x in rows if x['phase']=='joint')=={h:66 for h in range(1,10)}
expected={(s,c) for s in (35,123) for c in ('motion','static')}
assert all(len(r['examples'])==4 and set(r['examples'])==expected for r in rows)
assert len(expected_packet_names(p))==36
for change in ({'quantized':True},{'joint_updates':595},{'conditions':['motion']},
               {'native_replay_sha256':'truncated'}):
    bad=copy.deepcopy(p);bad.update(change)
    try:
        validate_protocol(bad)
        raise AssertionError('Invalid paired-Q protocol accepted')
    except ValueError:
        pass
import worker
assert 'common' not in sys.modules, 'Paired-Q helper shadows native renderer common module'
print('PASS: 600 distill + 594 joint updates; every update covers four conditions and each horizon 66 times')
print('PASS: 36 final packets, invalid protocol rejection and native renderer module isolation')
