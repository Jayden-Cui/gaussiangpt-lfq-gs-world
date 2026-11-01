"""Validate fixed-gauge code metric behavior on analytic sequences."""
import torch
from replay_cpu import code_stats

zeros=torch.zeros((9,16));ones=torch.ones((9,16))
same=code_stats(zeros,zeros,torch.tensor(1.))
assert same['mse']==0 and same['rms']==0 and same['normalized_mse']==0
known=code_stats(ones,zeros,torch.tensor(4.))
assert known['mse']==1 and known['rms']==1 and known['normalized_mse']==.25
for q,t,s in ((zeros[:8],zeros,1.),(zeros,zeros,0.)):
    try:
        code_stats(q,t,s)
        raise AssertionError('Invalid code stats input accepted')
    except ValueError:
        pass
print('PASS: code-shape, finite-value and fixed-teacher-normalization checks')
