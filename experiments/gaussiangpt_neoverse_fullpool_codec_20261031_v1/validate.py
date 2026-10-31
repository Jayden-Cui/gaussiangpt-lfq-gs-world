"""One fresh CPU document witness for this exact frozen source set."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(root):
    root=Path(root).resolve(strict=True)
    out=root/'CPU_VALIDATION.json'
    if out.exists():raise RuntimeError('Existing CPU witness; no overwrite or automatic retry')
    manifest=json.loads((root/'CODE_MANIFEST.json').read_text())
    if not manifest or not {p.name for p in root.glob('*.py')} <= set(manifest):
        raise RuntimeError('Incomplete frozen source set')
    for name,pinned in manifest.items():
        if Path(name).name != name or digest(root/name)!=pinned:raise RuntimeError('Frozen source changed: '+name)
    cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(root),'-p','test_*.py','-v']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
    started=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(cmd,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
    print(result.stdout,flush=True)
    match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',result.stdout)
    count=int(match.group(1)) if match else 0
    skips=[line for line in result.stdout.splitlines() if ' ... skipped ' in line]
    passed=result.returncode==0 and count>0 and not skips
    proof=dict(status='passed' if passed else 'failed',gpu_executed=False,environment_ready=False,
        code_manifest_sha256=digest(root/'CODE_MANIFEST.json'),protocol_sha256=digest(root/'protocol.json'),
        started_at_utc=started,actual_command=cmd,exit_code=result.returncode,actual_tests_executed=count,
        skips=skips,output=result.stdout,scope='Synthetic CPU source/data/gradient/queue tests; real full-pool GPU preflight still required')
    with out.open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
