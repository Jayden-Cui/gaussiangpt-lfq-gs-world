"""Independent CPU document witness; native GS execution remains unproven."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import pipeline


def main(root):
    root = root.resolve()
    output = root / 'CPU_VALIDATION.json'
    if output.exists():
        raise FileExistsError('CPU witness already attempted; preserve the existing result')
    started = time.monotonic()
    _, binding = pipeline.verify(root, cpu=False)
    for path in root.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    checks = []
    for name in ('test_sources.py', 'test_worker.py', 'test_compare.py', 'test_pipeline.py'):
        command = [sys.executable, '-B', str(root / name)]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=300 if name == 'test_compare.py' else 180)
        checks.append(dict(command=command, exit_code=result.returncode,
                           stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            break
    pipeline.verify(root, binding, cpu=False)
    passed = len(checks) == 4 and all(row['exit_code'] == 0 for row in checks)
    proof = dict(schema='gs_multisource_codec_cpu_document_witness_v1',
                 status='passed' if passed else 'failed', **binding, checks=checks,
                 gpu_executed=False, training_performed=False, goal_complete=False,
                 checked_at=datetime.now(timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic() - started)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], **binding, gpu_executed=False)))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
