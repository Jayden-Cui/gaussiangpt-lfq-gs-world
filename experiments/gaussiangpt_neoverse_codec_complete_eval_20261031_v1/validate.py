"""One fresh CPU-only validation of the frozen complete evaluation source."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve(strict=True)
    if root != Path(__file__).resolve().parent:
        raise RuntimeError('Validation must target its own frozen directory')
    manifest_path = root / 'CODE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not {p.name for p in root.glob('*.py')} <= set(manifest):
        raise RuntimeError('Unbound executable')
    for name, digest in manifest.items():
        if Path(name).name != name or sha(root / name) != digest:
            raise RuntimeError('Source mismatch: ' + name)
    out = root / 'CPU_VALIDATION.json'
    log = root / 'cpu_validation.log'
    if out.exists() or log.exists():
        raise RuntimeError('Validation already attempted; preserve existing evidence')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    sys.path.insert(0, str(root))
    stream = io.StringIO()
    started = time.monotonic()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        suite = unittest.defaultTestLoader.discover(str(root), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log.write_text(stream.getvalue(), encoding='utf-8')
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    receipt = dict(status='passed' if passed else 'failed', gpu_executed=False,
        code_manifest_sha256=sha(manifest_path), protocol_sha256=sha(root/'protocol.json'),
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), elapsed_seconds=time.monotonic()-started,
        python=sys.version, platform=platform.platform(), log_sha256=sha(log))
    out.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(stream.getvalue(), end='')
    print(json.dumps(receipt))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
