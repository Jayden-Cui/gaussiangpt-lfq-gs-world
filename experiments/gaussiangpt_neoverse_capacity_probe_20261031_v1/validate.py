"""One local CPU document witness; never initializes the GS/GPU environment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import pipeline

def main(root):
    root=root.resolve();begun=time.monotonic()
    cfg,binding=pipeline.verify(root,witness=False)
    output=root/'CPU_VALIDATION.json'
    if output.exists(): raise FileExistsError(output)
    checks=[]
    for script,extra in [('test_probe.py',[]),('pipeline.py',['--self-test'])]:
        command=[sys.executable,'-B',str(root/script)]+extra
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        checks.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode: break
    passed=len(checks)==2 and all(x['exit_code']==0 for x in checks)
    pipeline.verify(root,binding,witness=False)
    report=dict(schema='gs_capacity_probe_cpu_witness_v1',status='passed' if passed else 'failed',**binding,checks=checks,checked_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begun,gpu_executed=False,training_state_read=False,gpu_budget_verified=False,goal_complete=False)
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:report[k] for k in ['status','code_manifest_sha256','protocol_sha256','gpu_executed']}))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
