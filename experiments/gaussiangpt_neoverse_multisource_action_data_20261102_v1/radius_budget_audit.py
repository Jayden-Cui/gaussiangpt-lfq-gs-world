"""Count exact initial-only candidates for radius 6 on the second TRAIN video."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import maximum_filter
import torch

from source_store import SourceStore,TRAIN,sha
from coverage_audit import candidate_count

ROOT=Path(__file__).resolve().parent


def exact_candidate_count(coords,radius):
    points=coords.astype(np.int64)
    low=points.min(axis=0)-radius;high=points.max(axis=0)+radius
    shape=tuple((high-low+1).tolist())
    if np.prod(shape)>10_000_000:raise RuntimeError('Candidate-count grid too large; no truncation')
    grid=np.zeros(shape,dtype=np.uint8);indices=points-low
    grid[indices[:,0],indices[:,1],indices[:,2]]=1
    return int(maximum_filter(grid,size=2*radius+1,mode='constant',cval=0).sum())


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    audit_path=ROOT/'COVERAGE_AUDIT.json';audit=json.loads(audit_path.read_text())
    if (audit['status']!='complete' or audit['code_sha256']!=sha(ROOT/'coverage_audit.py')
            or audit['loader_manifest_sha256']!=sha(ROOT/'CODE.sha256')):
        raise RuntimeError('Prior radius-4 data audit changed')
    clip=TRAIN[1];prior=audit['sources'][clip]
    if prior['split']!='TRAIN' or prior['max_chebyshev_distance']!=6.:
        raise RuntimeError('Expected TRAIN-only radius evidence differs')
    probe=np.array([[0,0,0],[4,2,1],[4,2,2]],dtype=np.int32)
    for radius in (4,6):
        if exact_candidate_count(probe,radius)!=candidate_count(probe,radius):
            raise RuntimeError('Morphological and explicit candidate counts differ')
    store=SourceStore();counts={}
    for start in range(168):
        coords=store.load_frame(clip,start,purpose='train')['state']['coordinates'].numpy()
        counts[str(start)]={str(radius):exact_candidate_count(coords,radius) for radius in (4,6)}
        if start in (0,35,123,167):
            for radius in (4,6):
                if counts[str(start)][str(radius)]!=candidate_count(coords,radius):
                    raise RuntimeError('Full-size explicit candidate count differs')
    result={}
    for radius in (4,6):
        values=np.array([counts[str(s)][str(radius)] for s in range(168)],dtype=np.int64)
        result[str(radius)]=dict(min=int(values.min()),max=int(values.max()),mean=float(values.mean()),
            median=float(np.median(values)),p95=float(np.quantile(values,.95)),
            windows_above_25000=int((values>25000).sum()),
            starts_above_25000=[s for s,v in enumerate(values) if v>25000])
    output=dict(status='complete',clip=clip,split='TRAIN',initial_states=168,
      counts=counts,by_radius=result,radius6_covers_all_prior_train_targets=prior['max_chebyshev_distance']<=6,
      radius_chosen_from_train_data_only=True,evaluation_source_used=False,
      radius4_audit_sha256=sha(audit_path),code_sha256=sha(ROOT/'radius_budget_audit.py'),
      elapsed_seconds=time.monotonic()-began,
      limitation='Candidate count is not a GPU memory measurement; changing radius or max_candidates requires a new protocol and real GPU preflight.')
    with (ROOT/'RADIUS_BUDGET_AUDIT.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: exact radius-4/radius-6 candidate counts for all 168 second-source TRAIN starts')


if __name__=='__main__':main()
