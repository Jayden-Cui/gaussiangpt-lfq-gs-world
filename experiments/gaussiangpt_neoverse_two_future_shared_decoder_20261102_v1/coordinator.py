"""Launch one bounded paired-diagnostic worker without signalling other jobs."""
import os
from pathlib import Path
import subprocess
import sys
import traceback

from worker import ROOT, read, sha, verify_self, write


def main():
    phase = 'source_check'
    try:
        verify_self()
        protocol = read(ROOT/'protocol.json')
        write(ROOT/'LAUNCHED.json', {'pid': os.getpid(), 'code_sha256': sha(ROOT/'CODE.sha256'),
              'protocol_sha256': sha(ROOT/'protocol.json'), 'main_training_signalled': False})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(protocol['cuda_visible_device']), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        phase = 'paired_diagnostic'
        with (ROOT/'WORKER.log').open('x', encoding='utf8') as log:
            run = subprocess.run([sys.executable, '-B', str(ROOT/'worker.py')], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=protocol['maximum_gpu_seconds']+60)
        summary = ROOT/'result/SUMMARY.json'
        if run.returncode or read(summary).get('status') != 'complete':
            raise RuntimeError('Paired worker did not complete')
        write(ROOT/'DONE.json', {'status': 'complete', 'summary_sha256': sha(summary),
              'main_training_signalled': False, 'visual_review': 'pending'})
    except BaseException as error:
        write(ROOT/'COORDINATOR_FAILED.json', {'status': 'failed', 'phase': phase, 'error': repr(error),
            'traceback': traceback.format_exc(), 'main_training_signalled': False, 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    main()
