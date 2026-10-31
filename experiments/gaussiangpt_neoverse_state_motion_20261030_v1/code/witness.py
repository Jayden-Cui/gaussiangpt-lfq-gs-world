"""Documented entry: validate sources, all-state preparation, actual CUDA witness."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from binding import read,write,verify
def main(root):
    start=time.monotonic();verify(root)
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert min(int(x) for x in usage.split())>=55*1024,usage
    state=root/'state';env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    for script,limit in [('prepare_neoverse.py',1800),('witness_state.py',900)]:
        subprocess.run([sys.executable,'-B',str(state/'code'/script),'--root',str(state)],env=env,check=True,timeout=limit)
    write(root/'witness.json',{'status':'passed','seconds':time.monotonic()-start,'state':read(state/'preflight.json'),
        'prepared':read(state/'prepare_done.json'),'environment':'Reused unchanged; real NeoVerse loss backward verified',
        'motion':'Deferred actual trained-codec witness inside bounded pipeline before motion optimization'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
