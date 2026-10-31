"""One bounded queue: two isolated GPU arms, then paired comparison."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from codec_binding import require,sha
from formal_common import verify_sources,write_new


def main():
    root=Path(__file__).resolve().parent;p,source=verify_sources(root)
    require(shutil.disk_usage(root).free>=p['minimum_disk_free_gib']*2**30,'Insufficient disk space')
    smoke=json.loads((root/'SMOKE_DONE.json').read_text())
    require(smoke['status']=='passed' and smoke['source']==source and smoke['formal_updates']==0,
            'Exact bound integration smoke is required')
    for mode,entry in smoke['arms'].items():
        require(sha(root/'integration_smoke'/mode/'EVALUATION.json')==entry['evaluation_sha256']
                and sha(root/'integration_smoke'/mode/'NATIVE_REPLAY.json')==entry['native_replay_sha256'],
                'Integration evidence changed')
    out=root/'queue';out.mkdir(exist_ok=False)
    (root/'logs').mkdir(exist_ok=False)
    start=time.monotonic();children=[];handles=[]
    try:
        for mode,gpu in [('continuous16','0'),('lfq16','1')]:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
            handle=(root/'logs'/f'{mode}.log').open('x',encoding='utf8',buffering=1)
            child=subprocess.Popen([sys.executable,'-B','run_arm.py',mode],cwd=root,env=env,
                                   stdout=handle,stderr=subprocess.STDOUT)
            children.append((mode,gpu,child));handles.append(handle)
        write_new(out/'STARTED.json',dict(status='running',source=source,pid=os.getpid(),
                  children=[dict(mode=m,gpu=g,pid=c.pid) for m,g,c in children],smoke_sha256=sha(root/'SMOKE_DONE.json')))
        statuses={mode:child.wait() for mode,_,child in children}
        for handle in handles: handle.close()
        require(all(code==0 for code in statuses.values()),'At least one isolated arm failed: '+str(statuses))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        subprocess.run([sys.executable,'-B','compare.py'],cwd=root,env=env,check=True,timeout=1800)
        write_new(out/'DONE.json',dict(status='complete',source=source,exit_codes=statuses,
                  summary_sha256=sha(root/'SUMMARY.json'),elapsed_seconds=time.monotonic()-start,
                  visual_review_pending=True,goal_complete=False))
    except BaseException as error:
        for handle in handles:
            if not handle.closed: handle.close()
        write_new(out/'FAILED.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
                  source=source,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':main()
