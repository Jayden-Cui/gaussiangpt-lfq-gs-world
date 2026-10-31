"""Exactly one submission and bounded startup check; never poll training."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_vw1_h9_lfq_condentropy_20261030_v4')
SCREEN='gg_vw1_h9_condentropy_20261030_v4'
sys.path.insert(0,str(ROOT/'code'))
from contracts import verify_sources
verify_sources(ROOT)
assert json.loads((ROOT/'witness.json').read_text())['status']=='passed'
assert not (ROOT/'SUBMITTED.json').exists() and not (ROOT/'STATUS.json').exists()
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
gpus={int(x[0]):dict(used_mib=int(x[1]),total_mib=int(x[2])) for x in [r.split(',') for r in rows.strip().splitlines()]}
assert (gpus[0]['total_mib']-gpus[0]['used_mib'])/1024>=55,gpus
with (ROOT/'SUBMITTED.json').open('x') as f:json.dump(dict(screen=SCREEN,submitted_unix=time.time(),gpu_snapshot=gpus),f,indent=2)
command='exec timeout --signal=TERM --kill-after=30s 28800 /data/CJH/envs/gaussiangpt/bin/python -u -B '+str(ROOT/'code/pipeline.py')+' --root '+str(ROOT)+' > '+str(ROOT/'pipeline.log')+' 2>&1'
subprocess.run(['screen','-dmS',SCREEN,'bash','-lc',command],check=True)
started=time.monotonic()
while time.monotonic()-started<55:
    if (ROOT/'FAILED.json').exists() or (ROOT/'vw1_h9_lfq/train_progress.json').exists():break
    time.sleep(1)
result=dict(screen=SCREEN,root=str(ROOT),startup_wait_seconds=time.monotonic()-started,gpu_snapshot=gpus)
for name in ['STATUS.json','launch_receipt.json','vw1_h9_lfq/train_progress.json','FAILED.json']:
    path=ROOT/name
    if path.exists():result[name]=json.loads(path.read_text())
receipt=result.get('launch_receipt.json',{})
if 'pid' in receipt:
    try:__import__('os').kill(receipt['pid'],0);result['pipeline_pid_live']=True
    except ProcessLookupError:result['pipeline_pid_live']=False
print(json.dumps(result,indent=2))
assert result.get('pipeline_pid_live') and result['STATUS.json']['status']=='running' and 'FAILED.json' not in result
assert 'vw1_h9_lfq/train_progress.json' in result
