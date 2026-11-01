"""Check a TRAIN-chosen radius against every source's initial-only candidates."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from source_store import SourceStore,TRAIN,EVAL,sha
from coverage_audit import candidate_count
from radius_budget_audit import exact_candidate_count

ROOT=Path(__file__).resolve().parent


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    prior=ROOT/'RADIUS_BUDGET_AUDIT.json';evidence=json.loads(prior.read_text())
    if (evidence['status']!='complete' or evidence['code_sha256']!=sha(ROOT/'radius_budget_audit.py')
            or not evidence['radius_chosen_from_train_data_only'] or evidence['evaluation_source_used']
            or not evidence['radius6_covers_all_prior_train_targets']):
        raise RuntimeError('TRAIN-only radius evidence differs')
    coverage_path=ROOT/'COVERAGE_AUDIT.json';coverage=json.loads(coverage_path.read_text())
    if coverage['status']!='complete' or coverage['code_sha256']!=sha(ROOT/'coverage_audit.py'):
        raise RuntimeError('Prior full coverage audit changed')
    store=SourceStore();results={}
    for clip in (*TRAIN,*EVAL):
        purpose='train' if clip in TRAIN else 'eval';counts=[]
        for start in range(168):
            coords=store.load_frame(clip,start,purpose=purpose)['state']['coordinates'].numpy()
            count=exact_candidate_count(coords,6);counts.append(count)
            if start==35 and count!=candidate_count(coords,6):
                raise RuntimeError('Exact candidate-count cross-check failed')
        values=np.asarray(counts)
        results[clip]=dict(split='TRAIN' if clip in TRAIN else 'EVAL',initial_states=168,
            min=int(values.min()),max=int(values.max()),mean=float(values.mean()),p95=float(np.quantile(values,.95)),
            windows_above_25000=int((values>25000).sum()),starts_above_25000=np.flatnonzero(values>25000).tolist(),
            all_prior_targets_within_radius6=coverage['sources'][clip]['max_chebyshev_distance']<=6,
            per_start_counts=counts)
        store._read_frame.cache_clear()
    output=dict(status='complete',radius=6,max_candidates=25000,sources=results,
      radius_selected_using_train_only=True,eval_initials_used_for_feasibility_check_only=True,
      prior_train_audit_sha256=sha(prior),prior_coverage_sha256=sha(coverage_path),
      code_sha256=sha(ROOT/'global_radius6_audit.py'),elapsed_seconds=time.monotonic()-began,
      limitation='Counts do not measure GPU memory; no training policy has been changed. Evaluation source was historically exposed.')
    with (ROOT/'GLOBAL_RADIUS6_AUDIT.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: exact radius-6 candidate counts for all 504 initial states across three videos')


if __name__=='__main__':main()
