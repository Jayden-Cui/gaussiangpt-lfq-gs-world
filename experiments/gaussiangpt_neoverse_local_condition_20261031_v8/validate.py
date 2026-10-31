"""Fresh CPU document invocation for the local-context experiment."""
import argparse
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import pipeline

def main(root):
    root=root.resolve();begun=time.monotonic();out=root/'CPU_VALIDATION.json'
    if out.exists():raise FileExistsError('CPU validation was already attempted')
    cfg,binding=pipeline.verify(root,cpu=False,gpu=False)
    for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    checks=[]
    for script in ('test_local_skip.py','test_worker.py','test_compare.py','test_pipeline.py'):
        if not (root/script).is_file():raise FileNotFoundError(script)
        command=[sys.executable,'-B',str(root/script)]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        checks.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:break
    pipeline.verify(root,binding,cpu=False,gpu=False)
    passed=len(checks)==4 and all(x['exit_code']==0 for x in checks)
    with out.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='gs_local_condition_cpu_witness_v1',status='passed' if passed else 'failed',**binding,checks=checks,checked_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begun,gpu_executed=False,gpu_budget_verified=False,training_state_read=False,goal_complete=False),f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(status='passed' if passed else 'failed',**binding,gpu_executed=False)))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
