"""One local CPU document witness for the exact frozen spatial-action run."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def main(root):
    root = Path(root).resolve()
    output = root / 'CPU_VALIDATION.json'
    if output.exists():
        raise FileExistsError('Preserve existing CPU witness; do not overwrite or silently rerun')
    sys.path.insert(0, str(root))
    from pipeline import verify
    started = time.monotonic()
    _, binding = verify(root, cpu=False)
    checks = []
    for path in sorted(root.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    for name in ('test_spatial_action', 'test_worker', 'test_compare', 'test_pipeline'):
        command = [sys.executable, '-B', '-m', 'unittest', name, '-v']
        done = subprocess.run(command, cwd=root, env=env, capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=180)
        checks.append(dict(name=name, command=command, exit_code=done.returncode,
                           stdout=done.stdout, stderr=done.stderr))
        if done.returncode:
            break
    verify(root, binding, cpu=False)
    passed = len(checks) == 4 and all(item['exit_code'] == 0 for item in checks)
    record = dict(schema='gs_spatial_action_cpu_document_witness_v1',
                  status='passed' if passed else 'failed', **binding,
                  gpu_executed=False, training_performed=False, goal_complete=False,
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, checks=checks)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in record.items() if key != 'checks'}))
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().root))
