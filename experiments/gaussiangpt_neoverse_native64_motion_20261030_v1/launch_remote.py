"""Launch exactly once after witness; bounded startup acknowledgement only."""
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_native64_motion_20261030_v1')
SCREEN='gg_native64_lfq_20261030_v1'
sys.path.insert(0,str(ROOT/'code'))
from contracts import verify_sources
verify_sources(ROOT)
assert json.loads((ROOT/'witness.json').read_text())['status']=='passed'
assert not (ROOT/'STATUS.json').exists() and not (ROOT/'SUBMITTED.json').exists()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
gpus={int(x[0]):{'used_mib':int(x[1]),'total_mib':int(x[2])} for x in [r.split(',') for r in rows.strip().splitlines()]}
assert (gpus[0]['total_mib']-gpus[0]['used_mib'])/1024 >= 55, gpus
with (ROOT/'SUBMITTED.json').open('x') as f:
    json.dump({'screen':SCREEN,'submitted_unix':time.time(),'gpu_snapshot':gpus},f,indent=2)
command='exec timeout --signal=TERM --kill-after=30s 28800 /data/CJH/envs/gaussiangpt/bin/python -u -B '+str(ROOT/'code/pipeline.py')+' --root '+str(ROOT)+' > '+str(ROOT/'pipeline.log')+' 2>&1'
subprocess.run(['screen','-dmS',SCREEN,'bash','-lc',command],check=True)
started=time.monotonic()
while time.monotonic()-started<55:
    if (ROOT/'FAILED.json').exists() or (ROOT/'lfq64_k4/train_progress.json').exists():break
    time.sleep(1)
result={'screen':SCREEN,'root':str(ROOT),'startup_wait_seconds':time.monotonic()-started,'gpu_snapshot':gpus}
for name in ['STATUS.json','launch_receipt.json','lfq64_k4/train_progress.json','FAILED.json']:
    p=ROOT/name
    if p.exists():result[name]=json.loads(p.read_text())
print(json.dumps(result,indent=2))
assert 'launch_receipt.json' in result and result['STATUS.json']['status']=='running',result
assert 'lfq64_k4/train_progress.json' in result and 'FAILED.json' not in result,result
