"""CPU-only executable witness for the independent v7 postfit admission.

Reads immutable source files, never live training state, checkpoints, or CUDA.
This does not assert that a real postfit GPU run will succeed within its budget.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
MAIN_MANIFEST = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
TESTS = ('queue_postfit.py', 'render_six_panel.py', 'nearby_view_probe.py')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, postfit_hash=None):
    if sha(root / 'CODE_MANIFEST.json') != MAIN_MANIFEST:
        raise RuntimeError('Wrong immutable main manifest')
    current = sha(root / 'POSTFIT_MANIFEST.json')
    if postfit_hash is not None and current != postfit_hash:
        raise RuntimeError('Postfit manifest changed during validation')
    counts = {}
    for name in ('CODE_MANIFEST.json', 'POSTFIT_MANIFEST.json'):
        manifest = json.loads((root / name).read_text(encoding='utf-8'))
        if not isinstance(manifest, dict) or not manifest:
            raise RuntimeError('Empty or invalid source manifest')
        for relative, expected in manifest.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.parent) or sha(path) != expected:
                raise RuntimeError('Immutable source mismatch: ' + relative)
        counts[name] = len(manifest)
    return current, counts


def main(root):
    destination = root / 'POSTFIT_VALIDATION.json'
    if destination.exists() or (root / 'POSTFIT_VALIDATION_FAILED.json').exists():
        raise RuntimeError('Existing validation receipt must be preserved; no automatic retry')
    begun = time.monotonic()
    manifest, counts = verify(root)
    spec = importlib.util.spec_from_file_location('_postfit_cpu_admission', root / 'postfit_code/queue_postfit.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.verify(root, require_validation=False)
    checks = []
    for name in TESTS:
        path = root / 'postfit_code' / name
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        command = [sys.executable, '-B', str(path), '--self-test']
        result = subprocess.run(command, capture_output=True, text=True, timeout=45,
                                encoding='utf-8', errors='replace', cwd=str(root))
        check = dict(script=name, ast_passed=True, exit_code=result.returncode,
                     stdout=result.stdout[-5000:], stderr=result.stderr[-5000:])
        checks.append(check)
        if result.returncode != 0:
            raise RuntimeError('CPU self-test failed: ' + json.dumps(check, ensure_ascii=False))
    verify(root, manifest)
    receipt = dict(schema='gs_v7_postfit_cpu_validation_v1', status='passed',
                   checked_at=datetime.now(timezone.utc).isoformat(),
                   main_manifest_sha256=MAIN_MANIFEST, postfit_manifest_sha256=manifest,
                   gpu_executed=False, cpu_only=True, optimizer_updates=0,
                   training_state_read=False, gpu_runtime_budget_verified=False,
                   source_counts=counts, checks=checks, elapsed_seconds=time.monotonic()-begun,
                   goal_complete=False)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='passed', checks=len(checks), gpu_executed=False,
                         postfit_manifest_sha256=manifest, receipt=str(destination))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    try:
        main(root)
    except BaseException as error:
        failed = root / 'POSTFIT_VALIDATION_FAILED.json'
        if not failed.exists() and not (root / 'POSTFIT_VALIDATION.json').exists():
            with failed.open('x', encoding='utf-8') as stream:
                json.dump(dict(status='failed', error=repr(error), gpu_executed=False,
                               goal_complete=False), stream, ensure_ascii=False, indent=2)
        raise
