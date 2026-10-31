"""Bounded independent validation invocation, never starts scientific arms."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main(root):
    begun=time.monotonic(); phases=[]
    jobs=[('anchored_cpu',[sys.executable,'-B','-m','unittest','-v','test_anchored_ldm'],180,True),
          ('native_b4',[sys.executable,'-B',str(root/'code/native_experiment.py'),'--root',str(root),'--mode','preflight'],900,False)]
    for name,command,timeout,cpu in jobs:
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2' if cpu else '6',
            MKL_NUM_THREADS='2' if cpu else '6',OPENBLAS_NUM_THREADS='2' if cpu else '6')
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        with (root/(name+'.log')).open('x') as log:
            try:
                result=subprocess.run(command,cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                code=result.returncode;error=None
            except subprocess.TimeoutExpired:
                code=None;error='timeout'
        phases.append(dict(phase=name,exit_code=code,error=error))
        (root/'VALIDATION.json').write_text(json.dumps(dict(status='running' if code==0 else 'failed',phases=phases,
            elapsed_seconds=time.monotonic()-begun),indent=2))
        if code!=0:raise SystemExit(1)
    (root/'VALIDATION.json').write_text(json.dumps(dict(status='passed',phases=phases,elapsed_seconds=time.monotonic()-begun),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
