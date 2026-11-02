"""Generate axis diagnostics after all stage-B evaluations and replays finish."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
RUN = EXP / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
AXIS = EXP / 'gaussiangpt_neoverse_visible_axis_reference_20261102_v1'
PREVIOUS = EXP / 'gaussiangpt_neoverse_eight_window_object_audit_20261102_v1/watcher.py'
AXIS_MANIFEST_SHA = 'f2b5ef152256a879d09eec9e7a1077942bda8081558a0fa4fcc246f68d08538c'
WAIT_SECONDS = 86400
SWEEPS = (5, 10, 20)


def previous_watcher():
    spec = importlib.util.spec_from_file_location('_visible_axis_previous_watcher', PREVIOUS)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_sources():
    previous = previous_watcher()
    old, _ = previous.validate_sources()
    for folder in (ROOT, AXIS):
        witness = old.read(folder / 'CPU_TESTS.json')
        manifest = folder / 'CODE.sha256'
        if witness.get('status') != 'passed' or witness.get('code_sha256') != old.sha(manifest):
            raise RuntimeError('CPU source witness missing or changed: ' + str(folder))
        for line in manifest.read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            if Path(name).name != name or old.sha(folder / name) != digest:
                raise RuntimeError('Bound source changed: ' + name)
    if old.sha(AXIS / 'CODE.sha256') != AXIS_MANIFEST_SHA:
        raise RuntimeError('Visible axis reference version changed')
    reference = old.read(AXIS / 'source_axis_reference.json')
    if reference.get('baseline_object_audit_sha256') != old.sha(previous.BASELINE):
        raise RuntimeError('Axis reference and eight-window baseline differ')
    return previous, old, reference


def validate_report(report, evaluation, evaluation_sha, reference, reference_sha, sweep):
    if (report.get('status') != 'complete' or report.get('schema') != 'visible_source_axis_diagnostic_v1'
            or report.get('evaluation_sha256') != evaluation_sha
            or report.get('evaluation_source') != evaluation['source']
            or evaluation['source'].get('sweep') != sweep
            or evaluation['source'].get('quantized') is not False
            or report.get('reference_sha256') != reference_sha
            or report.get('manual_motion_review_required') is not True
            or report.get('no_acceptance_decision') is not True):
        raise RuntimeError('Axis report provenance or review scope differs')
    target = {(r['start'], r['horizon'], r['object']): r for r in reference['rows']}
    rows = report.get('rows', [])
    observed = {(r['start'], r['horizon'], r['object']) for r in rows}
    if len(rows) != 18 or len(target) != 18 or observed != set(target):
        raise RuntimeError('Expected the 18 distinct source-axis frames')
    for row in rows:
        key = row['start'], row['horizon'], row['object']
        if row.get('source_target_rgb_sha256') != target[key]['source_target_rgb_sha256']:
            raise RuntimeError('Axis report source pixels differ')
        arms = row['axis_color_coverage_by_radius_px']
        if set(arms) != {'source', 'stage_b_free_motion', 'actual_state_codec'}:
            raise RuntimeError('Axis report arms differ')
        for result in arms.values():
            if set(result) != {'2', '4', '8'}:
                raise RuntimeError('Axis radius sensitivity is incomplete')
            values = [result[str(radius)] for radius in (2, 4, 8)]
            if (any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                    or values != sorted(values)):
                raise RuntimeError('Invalid axis coverage')


def run(ready_fd):
    previous, old, reference = verify_sources()
    descriptor = old.open_watch(RUN)
    phase = 'waiting_for_complete_B'
    try:
        old.write(ROOT / 'WATCH_STARTED.json', dict(pid=os.getpid(), method='inotify close-write/move',
            run_root=str(RUN), max_wait_seconds=WAIT_SECONDS, gpu_used=False, training_signalled=False))
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='running', phase=phase,
            gpu_used=False, training_signalled=False))
        if ready_fd is not None:
            os.write(ready_fd, b'ready\n')
            os.close(ready_fd)
        summary = previous.await_summary(old, descriptor, WAIT_SECONDS)
        phase = 'validating_full_B_evaluations_and_replays'
        previous, old, reference = verify_sources()
        evaluations = old.validate_summary(summary, RUN)
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='running', phase='running_axis_diagnostics',
            gpu_used=False, training_signalled=False), replace=True)
        phase = 'running_axis_diagnostics'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        reports = []
        for sweep in SWEEPS:
            evaluation = evaluations[sweep]
            evaluation_sha = old.sha(evaluation)
            output = ROOT / ('result_b_sweep_%02d' % sweep)
            command = [sys.executable, '-B', str(AXIS / 'axis_audit.py'),
                '--evaluation-root', str(evaluation.parent), '--expected-evaluation-sha256', evaluation_sha,
                '--output', str(output)]
            with (ROOT / ('axis_sweep_%02d.log' % sweep)).open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError('Axis diagnostic failed for sweep ' + str(sweep))
            path = output / 'AXIS_DIAGNOSTIC.json'
            report = old.read(path)
            validate_report(report, old.read(evaluation), evaluation_sha, reference,
                            old.sha(AXIS / 'source_axis_reference.json'), sweep)
            reports.append(dict(sweep=sweep, path=str(path), sha256=old.sha(path),
                                summaries=report['summaries']))
        old.write(ROOT / 'WATCH_RESULT.json', dict(status='complete', B_summary_sha256=old.sha(RUN / 'SUMMARY.json'),
            reports=reports, gpu_used=False, training_signalled=False, manual_dynamic_review_required=True,
            action_codec_complete=False, scope='Two approximate source-image axes only; not physical ground truth'))
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='complete', phase='awaiting_visual_review',
            gpu_used=False, training_signalled=False), replace=True)
    except BaseException as error:
        failure = dict(status='failed', phase=phase, error=repr(error), traceback=traceback.format_exc(),
                       gpu_used=False, training_signalled=False, no_automatic_retry=True)
        old.write(ROOT / 'WATCH_FAILED.json', failure)
        old.write(ROOT / 'WATCH_STATUS.json', failure, replace=True)
        raise
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--ready-fd', type=int)
    args = parser.parse_args()
    if args.preflight:
        _, old, _ = verify_sources()
        import numpy, scipy, PIL, torch  # noqa: F401; CPU-only imports.
        descriptor = old.open_watch(RUN)
        os.close(descriptor)
        print('PASS: bound sources, CPU dependencies and event registration')
    else:
        run(args.ready_fd)
