"""CPU documentation witness; validates frozen code without exporting or training."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def main(root):
    root = Path(root).resolve(strict=True)
    output = root / 'CPU_VALIDATION.json'
    if output.exists():
        raise FileExistsError('Preserve existing CPU witness; no silent rerun')
    sys.path.insert(0, str(root))
    from pipeline import verify
    start = time.monotonic()
    _, proof = verify(root, require_cpu=False)
    for path in sorted(root.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2')
    checks = []
    count = 0
    suites = ['test_bindings', 'test_prepare_clip', 'test_admission', 'test_pipeline']
    for suite in suites:
        argv = [sys.executable, '-B', '-m', 'unittest', suite, '-v']
        try:
            p = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=240)
            matches = re.findall(r'Ran (\d+) tests? in ', p.stdout + p.stderr)
            actual = int(matches[-1]) if matches else 0
            count += actual
            checks.append(dict(suite=suite, command=argv, exit_code=p.returncode,
                               actual_tests=actual, stdout=p.stdout, stderr=p.stderr))
            if p.returncode or actual == 0:
                break
        except subprocess.TimeoutExpired as e:
            checks.append(dict(suite=suite, command=argv, exit_code=None,
                               actual_tests=0, error=type(e).__name__))
            break
    verify(root, proof, require_cpu=False)
    passed = len(checks) == len(suites) and all(c['exit_code']==0 and c['actual_tests']>0 for c in checks)
    record = dict(schema='gs_fulltrain_pool_cpu_document_witness_v1',
                  status='passed' if passed else 'failed', **proof,
                  gpu_executed=False, training_performed=False, goal_complete=False,
                  actual_tests_executed=count, elapsed_seconds=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat(), checks=checks)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k!='checks'}), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().root))
