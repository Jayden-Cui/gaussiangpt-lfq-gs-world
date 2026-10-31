"""Single CPU document witness, without reading live or future experiment artifacts."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def main(root):
    if (root / 'CPU_VALIDATION.json').exists() or (root / 'CPU_VALIDATION_FAILED.json').exists():
        raise FileExistsError('Preserve the original document witness; no automatic retry')
    begun = time.monotonic()
    spec = importlib.util.spec_from_file_location('_transfer_admission', root / 'pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    _, binding = pipeline.verify(root, witness=False)
    scripts = ['test_transfer_data.py', 'test_evaluate_transfer.py', 'test_compare_transfer.py']
    commands = [[sys.executable, '-B', str(root / name)] for name in scripts]
    commands.append([sys.executable, '-B', str(root / 'pipeline.py'), '--self-test'])
    outcomes = []
    for name in ['pipeline.py', 'validate.py', 'transfer_data.py', 'evaluate_transfer.py', 'compare_transfer.py'] + scripts:
        ast.parse((root / name).read_text(encoding='utf-8'), filename=name)
    for command in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=60)
        outcome = dict(command=command, exit_code=result.returncode, stdout=result.stdout[-6000:], stderr=result.stderr[-6000:])
        outcomes.append(outcome)
        if result.returncode:
            raise RuntimeError('CPU test failed: ' + json.dumps(outcome, ensure_ascii=False))
    pipeline.verify(root, expected=binding, witness=False)
    result = dict(schema='gs_transfer_eval_cpu_witness_v1', status='passed', gpu_executed=False,
                  training_state_read=False, gpu_budget_verified=False, optimizer_updates=0,
                  checked_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-begun,
                  checks=outcomes, goal_complete=False, **binding)
    with (root / 'CPU_VALIDATION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status='passed', checks=len(outcomes), gpu_executed=False, **binding)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    try:
        main(root)
    except BaseException as error:
        if not (root / 'CPU_VALIDATION.json').exists() and not (root / 'CPU_VALIDATION_FAILED.json').exists():
            with (root / 'CPU_VALIDATION_FAILED.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(status='failed', error=repr(error), gpu_executed=False, goal_complete=False), stream, indent=2)
        raise
