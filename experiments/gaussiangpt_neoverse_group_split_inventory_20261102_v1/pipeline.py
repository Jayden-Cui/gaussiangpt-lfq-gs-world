"""One bounded CPU run: inventory first, then known-group boundary checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from inventory import ROOT, GUARD, sha


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    phase = 'validating_sources'
    began = time.monotonic()
    try:
        witness, config = load(ROOT / 'CPU_TESTS.json'), load(ROOT / 'protocol.json')
        if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
            raise RuntimeError('Inventory CPU witness differs')
        for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            if Path(name).name != name or sha(ROOT / name) != digest:
                raise RuntimeError('Inventory source changed: ' + name)
        guard = load(GUARD / 'REAL_CHECK.json')
        if (guard.get('status') != 'passed' or guard.get('code_sha256') != sha(GUARD / 'CODE.sha256')
                or config['identity_guard_real_check_sha256'] != sha(GUARD / 'REAL_CHECK.json')):
            raise RuntimeError('Real source-identity guard preflight differs')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        result = ROOT / 'result'
        phase = 'inventory'
        subprocess.run([sys.executable, '-B', str(ROOT / 'inventory.py'), '--output', str(result)],
                       cwd=ROOT, env=env, check=True, timeout=config['max_seconds'] + 30)
        inventory = load(result / 'INVENTORY.json')
        if inventory.get('status') != 'complete' or (inventory.get('clips'), inventory.get('source_groups')) != (60, 22):
            raise RuntimeError('Unexpected inventory output')
        phase = 'boundary_comparison'
        subprocess.run([sys.executable, '-B', str(ROOT / 'boundary.py'), '--inventory', str(result / 'INVENTORY.json'),
                        '--expected-inventory-sha256', sha(result / 'INVENTORY.json'),
                        '--output', str(result / 'BOUNDARY.json')], cwd=ROOT, env=env, check=True,
                       timeout=config['boundary_max_seconds'] + 30)
        boundary = load(result / 'BOUNDARY.json')
        if (boundary.get('status') != 'complete' or len(boundary.get('rows', [])) != 3
                or boundary.get('inventory_sha256') != sha(result / 'INVENTORY.json')):
            raise RuntimeError('Unexpected boundary output')
        write(ROOT / 'DONE.json', dict(status='complete', inventory_sha256=sha(result / 'INVENTORY.json'),
            boundary_sha256=sha(result / 'BOUNDARY.json'), code_sha256=sha(ROOT / 'CODE.sha256'),
            gpu_used=False, training_performed=False, clean_holdout_established=False,
            elapsed_seconds=time.monotonic() - began))
        print('PASS: inventory and known boundary comparisons completed; no training launched')
    except BaseException as error:
        write(ROOT / 'FAILED.json', dict(status='failed', phase=phase, error=repr(error),
            traceback=traceback.format_exc(), no_automatic_retry=True, gpu_used=False, training_performed=False))
        raise


if __name__ == '__main__':
    main()
