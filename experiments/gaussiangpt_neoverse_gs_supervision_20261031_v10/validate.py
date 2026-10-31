"""Fresh CPU document witness; no server access, environment rebuild or GPU dispatch."""
import argparse
import ast
from datetime import datetime,timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from contract import verify,require,write_new,sha

def main(root):
    root=Path(root).resolve();require(not (root/'CPU_VALIDATION.json').exists(),'Preserve previous witness')
    _,proof=verify(root,cpu_witness=False);begun=time.monotonic()
    for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for rel,digest in {
        'gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py':'6a1de737b3e87d88c6678352236258c031f9e8d66f17d79a04e2b944560562ef',
        'gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py':'14fa085be07f8ab3f6ef79ec671362f062d6b2726f3968b02f8c4d95ff91c1b8',
        'gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py':'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2',
    }.items():require(sha(root.parent/rel)==digest,'Reused helper changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-B','-m','unittest','test_loss_step','test_teacher_assets','-v']
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=240)
    count=re.findall(r'Ran (\d+) tests? in ',result.stdout+result.stderr)
    passed=result.returncode==0 and count and int(count[-1])==11
    _,after=verify(root,cpu_witness=False);require(after==proof,'Frozen source changed during witness')
    report=dict(status='passed' if passed else 'failed',**proof,gpu_executed=False,
        actual_tests_executed=int(count[-1]) if count else 0,command=command,exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,elapsed_seconds=time.monotonic()-begun,
        finished_at=datetime.now(timezone.utc).isoformat(),goal_complete=False)
    write_new(root/'CPU_VALIDATION.json',report)
    print(report['status'],report['actual_tests_executed'],'CPU tests',flush=True)
    return 0 if passed else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().root))

