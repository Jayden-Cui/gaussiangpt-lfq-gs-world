"""Fresh-document native preflight wrapper; four discarded B4 updates, no training run."""
import argparse
import os
from pathlib import Path
import signal
import sys
import traceback
import pipeline

def main(root):
    root=root.resolve();output=root/'GPU_WITNESS.json'
    if output.exists() or (root/'preflight_process.log').exists() or (root/'PREFLIGHT.json').exists():
        raise FileExistsError('Preflight already attempted; do not retry in place')
    result={}
    def startup_stop(signum,frame):raise TimeoutError('GPU witness admission exceeded670 seconds')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,startup_stop)
    signal.alarm(670)
    try:
        cfg,binding=pipeline.verify(root,gpu=False)
        helper=pipeline.load('_local_condition_witness_groups',root/pipeline.HELPER)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,helper.terminate_queue)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        command=[sys.executable,'-B',str(root/'worker.py'),'--root',str(root),'--mode','preflight']
        result=helper.run_child(command,env,root/'preflight_process.log',650)
        if result['exit_code']!=0 or result['timed_out']:raise RuntimeError('Native preflight failed')
        pipeline.verify(root,binding,gpu=False);proof=pipeline.read(root/'PREFLIGHT.json')
        if proof.get('status')!='passed' or any(proof.get(k)!=binding[k] for k in ('code_manifest_sha256','protocol_sha256')):
            raise RuntimeError('Missing exact-source native preflight result')
        pipeline.write(output,dict(status='passed',**binding,process=result,preflight_sha256=pipeline.sha(root/'PREFLIGHT.json'),gpu_executed=True,formal_training_started=False,goal_complete=False))
        print('Native discarded B4 preflight passed; formal training not started')
    except BaseException as e:
        pipeline.write(output,dict(status='failed',process=result,error=repr(e),traceback=traceback.format_exc(),formal_training_started=False,goal_complete=False));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    main(p.parse_args().root)
