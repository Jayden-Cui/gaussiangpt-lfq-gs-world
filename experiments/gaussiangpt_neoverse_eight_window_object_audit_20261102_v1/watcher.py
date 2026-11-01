"""Event-driven CPU-only eight-window reports after stage B fully finishes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
RUN = EXP / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PREVIOUS_WATCH = EXP / 'gaussiangpt_neoverse_motion_groups_watch_20261102_v1/watcher.py'
SENSITIVITY = EXP / 'gaussiangpt_neoverse_object_proxy_sensitivity_20261102_v1'
BASELINE = ROOT / 'result_baseline/OBJECT_AUDIT.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def old_watcher():
    spec = importlib.util.spec_from_file_location('_bound_existing_b_watcher', PREVIOUS_WATCH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_sources():
    old = old_watcher()
    old.verify_sources()
    for folder in (ROOT, SENSITIVITY):
        witness = load(folder / 'CPU_TESTS.json')
        key = 'code_sha256' if folder == ROOT else 'code_manifest_sha256'
        if witness.get('status') != 'passed' or witness.get(key) != old.sha(folder / 'CODE.sha256'):
            raise RuntimeError('CPU source witness missing or changed: ' + str(folder))
        for line in (folder / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            if Path(name).name != name or old.sha(folder / name) != digest:
                raise RuntimeError('Bound source changed: ' + name)
    protocol = load(ROOT / 'protocol.json')
    if (protocol.get('schema') != 'stage_b_eight_window_cpu_watch_v1'
            or protocol.get('run_root') != str(RUN) or protocol.get('sweeps') != [5, 10, 20]
            or protocol.get('max_wait_seconds') != 86400 or protocol.get('per_sweep_timeout_seconds') != 300
            or protocol.get('gpu_used') is not False or protocol.get('training_signalled') is not False
            or protocol.get('baseline_audit_sha256') != old.sha(BASELINE)):
        raise RuntimeError('Fixed watcher protocol or reference audit changed')
    baseline = load(BASELINE)
    witness = load(ROOT / 'CPU_TESTS.json')
    if (baseline.get('status') != 'complete' or baseline.get('audited_object_rows') != 144
            or baseline.get('reference_targets_checked') is not False
            or baseline.get('analysis_code_sha256') != old.sha(ROOT / 'audit.py')
            or baseline.get('core_sha256') != old.sha(EXP / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/core.py')
            or baseline.get('config_sha256') != old.sha(EXP / 'gaussiangpt_neoverse_qformer_full177_20261101_v1/protocol.json')
            or baseline.get('sensitivity_sha256') != old.sha(SENSITIVITY / 'sensitivity.py')
            or witness.get('baseline_audit_sha256') != old.sha(BASELINE)):
        raise RuntimeError('Baseline report is not bound to the current audit sources')
    return old, protocol


def await_summary(old, fd, timeout):
    poll = select.poll()
    poll.register(fd, select.POLLIN | select.POLLERR | select.POLLHUP)
    deadline = time.monotonic() + timeout
    if (RUN / 'FAILED.json').exists():
        raise RuntimeError('Stage B reported failure')
    if (RUN / 'SUMMARY.json').is_file():
        try:
            return load(RUN / 'SUMMARY.json')
        except json.JSONDecodeError:
            pass  # A close-write event will follow the in-progress write.
    while True:
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise TimeoutError('Stage B completion was not observed within the fixed budget')
        events = poll.poll(remaining)
        if not events:
            raise TimeoutError('Stage B completion was not observed within the fixed budget')
        for _, flags in events:
            if flags & (select.POLLERR | select.POLLHUP):
                raise RuntimeError('Stage B event watcher failed')
            try:
                names = old.event_names(os.read(fd, 65536))
            except BlockingIOError:
                continue
            if 'FAILED.json' in names:
                raise RuntimeError('Stage B reported failure')
            if 'SUMMARY.json' in names:
                return load(RUN / 'SUMMARY.json')


def run(ready_fd):
    old, protocol = validate_sources()
    fd = old.open_watch(RUN)  # Register before checking whether the summary already exists.
    phase = 'waiting_for_complete_B'
    try:
        old.write(ROOT / 'WATCH_STARTED.json', dict(pid=os.getpid(), method='inotify close-write/move',
            run_root=str(RUN), max_wait_seconds=protocol['max_wait_seconds'], gpu_used=False,
            training_signalled=False))
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='running', phase=phase,
            gpu_used=False, training_signalled=False))
        if ready_fd is not None:
            os.write(ready_fd, b'ready\n')
            os.close(ready_fd)
        summary = await_summary(old, fd, protocol['max_wait_seconds'])
        phase = 'validating_completed_B'
        evaluations = old.validate_summary(summary, RUN)
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='running', phase='auditing_snapshots',
            gpu_used=False, training_signalled=False), replace=True)
        phase = 'auditing_snapshots'
        reports = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        for sweep in protocol['sweeps']:
            evaluation = evaluations[sweep]
            output = ROOT / f'result_b_sweep_{sweep:02d}'
            command = [sys.executable, '-B', str(ROOT / 'audit.py'), '--evaluation-root', str(evaluation.parent),
                '--expected-evaluation-sha256', old.sha(evaluation), '--reference-audit', str(BASELINE),
                '--expected-reference-audit-sha256', protocol['baseline_audit_sha256'], '--output', str(output)]
            with (ROOT / f'audit_sweep_{sweep:02d}.log').open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=protocol['per_sweep_timeout_seconds'])
            if result.returncode:
                raise RuntimeError('Eight-window audit failed for sweep ' + str(sweep))
            report_path = output / 'OBJECT_AUDIT.json'
            report = load(report_path)
            if (report.get('status') != 'complete' or report.get('audited_object_rows') != 144
                    or report.get('reference_targets_checked') is not True
                    or report.get('reference_audit_sha256') != protocol['baseline_audit_sha256']
                    or report.get('evaluation_source', {}).get('sweep') != sweep
                    or report.get('evaluation_sha256') != old.sha(evaluation)):
                raise RuntimeError('Eight-window report provenance differs for sweep ' + str(sweep))
            selected = {}
            for start, name in [(35, 'yellow'), (112, 'green'), (123, 'green')]:
                value = report['summaries'][str(start)][name]['correct']
                selected[f's{start:03d}_{name}'] = dict(
                    target_mask_detected_frames=value['target_mask_detected_frames'],
                    prediction_mask_detected_frames=value['prediction_mask_detected_frames'])
            reports.append(dict(sweep=sweep, path=str(report_path), sha256=old.sha(report_path), selected=selected))
        old.write(ROOT / 'WATCH_RESULT.json', dict(status='complete', B_summary_sha256=old.sha(RUN / 'SUMMARY.json'),
            baseline_audit_sha256=protocol['baseline_audit_sha256'], reports=reports, gpu_used=False,
            training_signalled=False, manual_dynamic_review_required=True, action_codec_complete=False,
            scope='Eight fixed windows only; color proxies are not physical object-tracking labels'))
        old.write(ROOT / 'WATCH_STATUS.json', dict(status='complete', phase='awaiting_manual_visual_review',
            gpu_used=False, training_signalled=False), replace=True)
    except BaseException as error:
        failure = dict(status='failed', phase=phase, error=repr(error), traceback=traceback.format_exc(),
                       gpu_used=False, training_signalled=False, no_automatic_retry=True)
        old.write(ROOT / 'WATCH_FAILED.json', failure)
        old.write(ROOT / 'WATCH_STATUS.json', failure, replace=True)
        raise
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--ready-fd', type=int)
    args = parser.parse_args()
    if args.preflight:
        old, _ = validate_sources()
        import numpy, scipy, PIL, torch  # noqa: F401; CPU-only dependency preflight.
        descriptor = old.open_watch(RUN)
        os.close(descriptor)
        print('PASS: bound sources, CPU dependencies and Linux event registration')
    else:
        run(args.ready_fd)
