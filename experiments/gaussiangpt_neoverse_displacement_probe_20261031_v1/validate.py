"""Run the frozen displacement probe's CPU document checks exactly once."""
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
    result_path = root / 'CPU_VALIDATION.json'
    if result_path.exists():
        raise FileExistsError('Preserve previous witness; no silent rerun')
    sys.path.insert(0, str(root))
    from worker import verify
    start = time.monotonic()
    _, binding = verify(root, cpu_witness=False)
    for path in sorted(root.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2')
    checks = []
    count = 0
    for name in ['test_worker', 'test_pipeline']:
        command = [sys.executable, '-B', '-m', 'unittest', name, '-v']
        try:
            p = subprocess.run(command, cwd=root, env=env, capture_output=True,
                               text=True, encoding='utf-8', errors='replace', timeout=240)
            matches = re.findall(r'Ran (\d+) tests? in ', p.stdout + p.stderr)
            actual = int(matches[-1]) if matches else 0
            count += actual
            checks.append(dict(name=name, command=command, exit_code=p.returncode,
                               actual_tests=actual, stdout=p.stdout, stderr=p.stderr))
            if p.returncode != 0 or actual == 0:
                break
        except subprocess.TimeoutExpired as e:
            checks.append(dict(name=name, command=command, exit_code=None,
                               actual_tests=0, error=type(e).__name__))
            break
    _, after = verify(root, cpu_witness=False)
    if after != binding:
        raise RuntimeError('Frozen sources changed during CPU witness')
    passed = len(checks) == 2 and all(x['exit_code']==0 and x['actual_tests']>0 for x in checks)
    record = dict(schema='gs_displacement_probe_cpu_document_witness_v1',
                  status='passed' if passed else 'failed', **binding,
                  actual_tests_executed=count, gpu_executed=False,
                  model_updates=0, oracle_updates=0, goal_complete=False,
                  elapsed_seconds=time.monotonic()-start,
                  finished_at=datetime.now(timezone.utc).isoformat(), checks=checks)
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in record.items() if k!='checks'}), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
