"""Run on134 once after witness passes; bounded startup acknowledgment, no monitoring."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_state_motion_20261030_v1')
SCREEN='gg_neoverse_adj_cont_20261030_v1'
assert not (ROOT/'STATUS.json').exists(),'Already submitted; do not relaunch'
assert json.loads((ROOT/'witness.json').read_text())['status']=='passed'
command='exec timeout --signal=TERM --kill-after=30s 64800 /data/CJH/envs/gaussiangpt/bin/python -u -B '+str(ROOT/'code/pipeline.py')+' --root '+str(ROOT)+' > '+str(ROOT/'pipeline.log')+' 2>&1'
subprocess.run(['screen','-dmS',SCREEN,'bash','-lc',command],check=True)
started=time.monotonic()
while time.monotonic()-started<55:
    if (ROOT/'FAILED.json').exists() or (ROOT/'state/train_progress.json').exists():break
    time.sleep(1)
result={'screen':SCREEN,'root':str(ROOT),'startup_wait_seconds':time.monotonic()-started}
for name in ['STATUS.json','launch_receipt.json','state/train_progress.json','FAILED.json']:
    p=ROOT/name
    if p.exists():result[name]=json.loads(p.read_text())
print(json.dumps(result,indent=2))
if (ROOT/'FAILED.json').exists():sys.exit(1)
assert 'launch_receipt.json' in result and result['STATUS.json']['status']=='running',result
