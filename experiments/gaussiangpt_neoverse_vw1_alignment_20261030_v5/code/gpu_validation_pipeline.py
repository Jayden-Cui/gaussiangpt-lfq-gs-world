"""Run the two fixed GPU validation phases once. No optimizer or retry."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def write(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def run(root):
    manifest=json.loads((root/'GPU_CODE_MANIFEST.json').read_text())
    for relative,digest in manifest.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:raise ValueError('GPU validation code changed: '+relative)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();phases=[]
    python='/data/CJH/envs/gaussiangpt/bin/python'
    phases_to_run=[('cpu_changed_components',180,[python,'-B','-m','unittest','-v',
        'test_aligned_ldm','test_online_state','test_multiscale64','test_soft_occupancy','test_reconstruction_objective'],True),
        ('gpu_objective_witness',300,[python,'-B',str(root/'code/gpu_objective_witness.py'),'--root',str(root)],False),
        ('gpu_render_witness',900,[python,'-B',str(root/'code/gpu_render_witness.py'),'--root',str(root)],False)]
    for name,limit,command,cpu_only in phases_to_run:
        log=root/(name+'.log')
        write(root/'GPU_VALIDATION_STATUS.json',dict(status='running',phase=name,phases=phases))
        with log.open('x') as out:
            try:
                phase_environment=dict(environment)
                if cpu_only:phase_environment['CUDA_VISIBLE_DEVICES']=''
                result=subprocess.run(command,cwd=root/'code',
                    env=phase_environment,stdout=out,stderr=subprocess.STDOUT,timeout=limit,check=False)
                code=result.returncode;error=None
            except subprocess.TimeoutExpired:code=124;error='phase timeout'
        phases.append(dict(phase=name,exit_code=code,error=error,log=str(log)))
        if code:
            write(root/'GPU_VALIDATION_STATUS.json',dict(status='failed',phases=phases,elapsed_seconds=time.monotonic()-start))
            raise SystemExit(code)
    write(root/'GPU_VALIDATION_STATUS.json',dict(status='passed',phases=phases,elapsed_seconds=time.monotonic()-start,
        training_started=False,full_goal_complete=False))
    print('GPU_VALIDATION_PASSED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

