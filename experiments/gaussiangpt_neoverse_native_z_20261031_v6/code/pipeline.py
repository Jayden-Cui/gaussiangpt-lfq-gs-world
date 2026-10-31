"""One submission, two GPU workers, bounded arms and terminal summary."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def run_worker(root,gpu,arms,cfg):
    outcomes=[]
    for arm in arms:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
        path=root/(arm+'_process.log')
        with path.open('x') as log:
            command=[sys.executable,'-B',str(root/'code/native_experiment.py'),'--root',str(root),'--mode','train','--arm',arm]
            started=time.monotonic()
            try:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=cfg['resources']['per_arm_process_timeout_seconds'])
                outcome=dict(arm=arm,gpu=gpu,exit_code=result.returncode,elapsed_seconds=time.monotonic()-started)
            except subprocess.TimeoutExpired:
                outcome=dict(arm=arm,gpu=gpu,exit_code=None,error='process_timeout',elapsed_seconds=time.monotonic()-started)
        outcomes.append(outcome);write(root/(arm+'_PROCESS.json'),outcome)
        if outcome.get('exit_code')!=0:
            # Do not launch another arm on a worker whose prerequisite failed.
            break
    return outcomes

def main(root):
    cfg=json.loads((root/'protocol.json').read_text())
    if json.loads((root/'PREFLIGHT.json').read_text())['status']!='passed':raise RuntimeError('Preflight required')
    write(root/'PIPELINE_STATUS.json',dict(status='running',pid=os.getpid(),started_unix=time.time(),
          workers=cfg['resources']['gpu_workers']))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_worker,root,int(gpu),arms,cfg) for gpu,arms in cfg['resources']['gpu_workers'].items()]
        outcomes=[item for f in futures for item in f.result()]
    command=[sys.executable,'-B',str(root/'code/compare_arms.py'),'--root',str(root)]
    result=subprocess.run(command,timeout=180,capture_output=True,text=True)
    (root/'comparison.log').write_text(result.stdout+result.stderr)
    write(root/'PIPELINE_STATUS.json',dict(status='complete' if result.returncode==0 and len(outcomes)==3 and all(x.get('exit_code')==0 for x in outcomes) else 'failed',
          outcomes=outcomes,comparison_exit_code=result.returncode,finished_unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    try:main(a.root)
    except BaseException as e:
        write(a.root/'PIPELINE_STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise
