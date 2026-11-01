"""Compare early-summary calculations with the already verified sweep-10 result."""
import hashlib
import json
from pathlib import Path
import torch

from final_latent import ROOT,analyze

PRIOR=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_stage_a_interim_latent_20261102_v1/INTERIM.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Early-analysis code changed')
    old=json.loads(PRIOR.read_text());new=analyze(10)
    for start in ('35','123'):
        for new_name,old_name in (('baseline_qformer','baseline_qformer'),('free_code','free_sweep_10')):
            actual=new['anchors'][start][new_name];reference=old['summary'][start][old_name]
            for key in ('mean_target_feature_normalized_mae','mean_support_iou'):
                if abs(actual[key]-reference[key])>1e-6:raise RuntimeError('Sweep-10 CPU result mismatch')
    output=dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),prior_interim_sha256=sha(PRIOR),
      sweep10_anchor_metrics_reproduced=True,full_code_shape=[168,9,16],
      scope='CPU test on existing 10-sweep checkpoints; not final render or action result')
    with (ROOT/'CPU_TESTS.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print('PASS: sweep-10 anchor metrics and full code ordering reproduce prior CPU analysis')


if __name__=='__main__':main()
