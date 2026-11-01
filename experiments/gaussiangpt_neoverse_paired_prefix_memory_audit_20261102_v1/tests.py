"""Sparse alignment must be invariant to row order and preserve repeated keys."""
import numpy as np
from audit import summarize_world

coords=np.array([[0,0,0],[1,2,3],[4,5,6]],dtype=np.int32)
features=np.stack([np.full(64,k,dtype=np.float32) for k in (1,2,3)])
order=np.array([2,0,1])
same=summarize_world(coords,features,coords[order],features[order],unique_per_input=True)
assert same['shared_world_cells']==3 and same['world_cell_jaccard']==1
assert same['shared_cell_mean_feature_difference_rms']==0
other=coords.copy();other[0]=[9,9,9]
diff=summarize_world(coords,features,other,features,unique_per_input=True)
assert diff['first_only_world_cells']==1 and diff['second_only_world_cells']==1 and diff['shared_world_cells']==2
dup=np.concatenate((coords,coords[:1]));dup_features=np.concatenate((features,features[:1]))
permuted=np.array([3,2,0,1])
memory=summarize_world(dup,dup_features,dup[permuted],dup_features[permuted],unique_per_input=False)
assert memory['world_cell_token_distribution_tv']==0 and memory['shared_cell_mean_feature_difference_rms']==0
try:
    summarize_world(dup,dup_features,coords,features,unique_per_input=True)
    raise AssertionError('Duplicate state coordinate accepted')
except ValueError:
    pass
bad=features.copy();bad[0,0]=np.nan
try:
    summarize_world(coords,bad,coords,features,unique_per_input=True)
    raise AssertionError('Nonfinite input accepted')
except ValueError:
    pass
print('PASS: world-coordinate alignment is order invariant, duplicate-token aware and finite')
