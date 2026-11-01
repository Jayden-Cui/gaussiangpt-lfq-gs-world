"""Offline cross-source candidate-radius audit; no target geometry enters a model."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
import torch

from source_store import SourceStore,TRAIN,EVAL,sha

ROOT=Path(__file__).resolve().parent
ORIGINAL=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json')
PROTOCOL_SHA='4600a2a7c4f32c4ad4803dd709089583270649034f5e443e836eab3dea8658fd'


def candidate_count(coordinates,radius):
    offsets=np.stack(np.meshgrid(*([np.arange(-radius,radius+1)]*3),indexing='ij'),-1).reshape(-1,3)
    return len(np.unique((coordinates.astype(np.int64)[:,None]+offsets[None]).reshape(-1,3),axis=0))


def verify_distance_equivalence():
    initial=np.array([[0,0,0],[5,0,0]],dtype=np.int32)
    targets=np.array([[4,4,4],[10,0,0],[-4,-4,-4],[0,5,0],[8,4,3]],dtype=np.int32)
    offsets=np.stack(np.meshgrid(*([np.arange(-4,5)]*3),indexing='ij'),-1).reshape(-1,3)
    explicit={tuple(x) for x in (initial[:,None]+offsets[None]).reshape(-1,3)}
    from_tree=cKDTree(initial).query(targets,k=1,p=np.inf)[0]<=4
    direct=np.array([tuple(x) in explicit for x in targets])
    if not np.array_equal(from_tree,direct):raise RuntimeError('Candidate-distance equivalence failed')


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    witness=json.loads((ROOT/'CPU_TESTS.json').read_text())
    if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256'):
        raise RuntimeError('Source loader witness changed')
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Source code changed')
    if sha(ORIGINAL)!=PROTOCOL_SHA:raise RuntimeError('Original candidate protocol changed')
    protocol=json.loads(ORIGINAL.read_text());radius=protocol['candidate_radius'];budget=protocol['max_candidates']
    if radius!=4 or budget!=25000:raise RuntimeError('Unexpected initial-only candidate policy')
    verify_distance_equivalence();store=SourceStore();results={}
    for clip in (*TRAIN,*EVAL):
        purpose='train' if clip in TRAIN else 'eval'
        states=[store.load_frame(clip,frame,purpose=purpose)['state']['coordinates'].numpy() for frame in range(177)]
        total=missing=0;incomplete=[];by_h={str(h):dict(target_nodes=0,outside_nodes=0) for h in range(1,10)}
        max_distance=0.
        for start in range(168):
            tree=cKDTree(states[start]);bad=[]
            for h in range(1,10):
                distances=tree.query(states[start+h],k=1,p=np.inf,workers=1)[0]
                if not np.isfinite(distances).all():raise RuntimeError('Invalid coordinate distances')
                count=int((distances>radius).sum());size=len(distances)
                total+=size;missing+=count;max_distance=max(max_distance,float(distances.max()))
                by_h[str(h)]['target_nodes']+=size;by_h[str(h)]['outside_nodes']+=count
                if count:bad.append(dict(horizon=h,outside_nodes=count,target_nodes=size))
            if bad:incomplete.append(dict(start=start,horizons=bad))
        candidate_samples={str(start):candidate_count(states[start],radius) for start in (0,35,123,167)}
        for v in by_h.values():v['outside_fraction']=v['outside_nodes']/v['target_nodes']
        results[clip]=dict(split='TRAIN' if clip in TRAIN else 'EVAL',frames=177,windows=168,
          window_horizon_pairs=1512,target_nodes=total,outside_nodes=missing,
          outside_fraction=missing/total,incomplete_windows=len(incomplete),max_chebyshev_distance=max_distance,
          sampled_candidate_counts=candidate_samples,sampled_counts_above_budget=[int(k) for k,v in candidate_samples.items() if v>budget],
          by_horizon=by_h,incomplete_window_details=incomplete)
        store._read_frame.cache_clear()
    output=dict(status='complete',radius=radius,max_candidates=budget,
      training_performed=False,model_or_decoder_executed=False,ground_truth_used_only_for_offline_coverage=True,
      candidate_set_uses_initial_coordinates_only=True,exact_chebyshev_equivalence_verified=True,
      code_sha256=sha(ROOT/'coverage_audit.py'),loader_manifest_sha256=sha(ROOT/'CODE.sha256'),
      protocol_sha256=PROTOCOL_SHA,sources=results,elapsed_seconds=time.monotonic()-began,
      limitation='Candidate counts are sampled at four starts per video, not exhaustively bounded. EVAL video was historically exposed and is not pristine.')
    with (ROOT/'COVERAGE_AUDIT.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: exact radius-4 coverage audit on three source videos; no model training')


if __name__=='__main__':main()
