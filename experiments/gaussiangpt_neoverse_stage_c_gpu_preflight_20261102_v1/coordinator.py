"""Run the three short GPU preflights sequentially on the spare device."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import pilot

ROOT = Path(__file__).resolve().parent
B_RUN = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_b_20261102_v1')


def execute(launch):
    for variant in ('C0', 'C1', 'C2'):
        launch(variant)


def main():
    phase = 'initializing'
    try:
        witness = pilot.read(ROOT / 'CPU_TESTS.json')
        if witness.get('status') != 'passed':
            raise RuntimeError('Pilot CPU tests missing')
        pilot.verify_manifest(ROOT, witness['code_sha256'])
        training = pilot.read(B_RUN / 'training/STATUS.json')
        if (training.get('status') != 'running' or training.get('completed_updates', 999999) >= 900):
            raise RuntimeError('Not enough separation from B checkpoint previews; do not start pilot')
        pilot.write(ROOT / 'STARTED.json', {'pid': os.getpid(), 'gpu': 1,
            'b_completed_updates_at_start': training['completed_updates'],
            'maximum_seconds_per_variant': 300, 'b_training_modified': False,
            'code_sha256': pilot.sha(ROOT / 'CODE.sha256')})
        phase = 'preflight'
        pilot.write(ROOT / 'STATUS.json', {'status': 'running', 'phase': phase})
        def launch(variant):
            pilot.write(ROOT / 'STATUS.json', {'status': 'running', 'phase': phase, 'variant': variant}, replace=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
            with (ROOT / (variant + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-B', str(ROOT / 'pilot.py'), '--variant', variant],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise RuntimeError(variant + ' exceeded its fixed GPU preflight deadline')
            if code != 0 or pilot.read(ROOT / variant / 'GPU_PREFLIGHT.json').get('status') != 'passed':
                raise RuntimeError(variant + ' GPU preflight failed')
        execute(launch)
        results = {v: pilot.read(ROOT / v / 'GPU_PREFLIGHT.json') for v in ('C0', 'C1', 'C2')}
        pilot.write(ROOT / 'SUMMARY.json', {'status': 'complete', 'variants': results,
                    'formal_stage_c_training': 'not started', 'validated_teacher': 'not selected',
                    'b_training_modified': False})
        pilot.write(ROOT / 'STATUS.json', {'status': 'complete', 'phase': 'engineering_preflights_done'}, replace=True)
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True,
                   'b_training_modified': False}
        pilot.write(ROOT / 'FAILED.json', failure)
        pilot.write(ROOT / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
