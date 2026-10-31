"""CPU-only validation for the conditional common-checkpoint evaluation."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
from queue_equal_steps import sha,verify


def main(root):
    started=time.monotonic();verify(root)
    hashes={name:sha(root/name) for name in ('CODE_MANIFEST.json','POSTFIT_MANIFEST.json','EQUAL_STEP_MANIFEST.json')}
    stages=[]
    for path in (root/'equal_step_code').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=path.name)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for script in ('queue_equal_steps.py','evaluate_checkpoint.py','compare_selected.py'):
        result=subprocess.run([sys.executable,'-B',str(root/'equal_step_code'/script),'--self-test'],
                              env=env,capture_output=True,text=True,timeout=30)
        stages.append(dict(name=script,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError('CPU self-test failed: '+script+' '+result.stderr)
    verify(root)
    if any(sha(root/name)!=digest for name,digest in hashes.items()):raise RuntimeError('Sources changed during validation')
    result=dict(status='passed',stages=stages,elapsed_seconds=time.monotonic()-started,
                equal_step_manifest_sha256=hashes['EQUAL_STEP_MANIFEST.json'],
                main_manifest_sha256=hashes['CODE_MANIFEST.json'],postfit_manifest_sha256=hashes['POSTFIT_MANIFEST.json'],
                gpu_executed=False,training_queries=0,
                scope='Common-step selection, strict source/coverage contracts, physical-frame aggregation and unchanged comparison-rule adapter; no actual checkpoint/GPU outcome claim')
    with (root/'EQUAL_STEP_VALIDATION.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);a=p.parse_args();main(a.root.resolve())
