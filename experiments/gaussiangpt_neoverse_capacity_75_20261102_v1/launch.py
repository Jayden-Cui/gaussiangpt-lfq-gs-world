from runtime_env import configure,environment,PYTHON
configure()
import os,subprocess,select
from pathlib import Path
from bindings import ROOT,read,write,sha,require
from worker import verify

p=verify();require(ROOT.is_relative_to(Path('/data/CJH/1')),'Deployment path outside authorized root')
require(not (ROOT/'run').exists() and not (ROOT/'LAUNCHED.json').exists(),'Refusing duplicate launch')
require(read(ROOT/'ENV_AGENT_WITNESS.json')['status']=='passed' and read(ROOT/'CPU_TESTS.json')['status']=='passed','Missing validation')
for arm in p['arms']:
    pre=read(ROOT/'preflight'/arm/'PREFLIGHT.json');require(pre['status']=='passed' and pre['source']['code_sha256']==sha(ROOT/'CODE.sha256'),'Preflight stale')
r,w=os.pipe()
with (ROOT/'coordinator.log').open('x') as log:
    process=subprocess.Popen([PYTHON,'-B',str(ROOT/'coordinator.py'),'--ready-fd',str(w)],env=environment(),cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(w,))
os.close(w)
write(ROOT/'LAUNCHED.json',dict(pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
    code_sha256=sha(ROOT/'CODE.sha256'),protocol_sha256=sha(ROOT/'protocol.json'),result_root=str(ROOT/'run')))
ready=bool(select.select([r],[],[],60)[0]);message=os.read(r,32).decode() if ready else 'observer timeout, do not relaunch'
os.close(r);print(dict(pid=process.pid,message=message,exit_code=process.poll(),result_root=str(ROOT/'run')),flush=True)
