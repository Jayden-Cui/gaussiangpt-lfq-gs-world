"""Check replay packet coverage and frozen protocol."""
import copy
from common import ROOT, expected_names, read, validate_protocol

p = read(ROOT/'protocol.json')
validate_protocol(p)
names = expected_names(p)
assert len(names) == 36
assert all(f's{s:03d}_h{h:02d}_{branch}' in names
           for s in (35,123) for h in range(1,10) for branch in ('motion','static'))
for changes in ({'branches':['motion']},{'packets':18},{'views':36},{'quantized':True},{'horizons':[1,9]}):
    candidate = copy.deepcopy(p)
    candidate.update(changes)
    try:
        validate_protocol(candidate)
        raise AssertionError('Invalid replay protocol accepted')
    except ValueError:
        pass
print('PASS: 36 named motion/static GS packets, 72 views and invalid protocol rejection')
