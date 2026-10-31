"""One bounded postfit submission; the program waits, never the agent.

Wait for the exact admitted main pipeline and its native workers to leave.
Then run each completed posterior arm's future-information diagnostic and
visualization once. Never train, resume, mutate main artifacts, or retry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
STOP_REQUESTED = threading.Event()
ACTIVE_CHILDREN = {}


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def terminate_queue(signum, frame):
    STOP_REQUESTED.set()
    # These groups were created by this queue, never main or shared workers.
    children = list(ACTIVE_CHILDREN.values())
    for process in children:
        signal_group(process.pid, signal.SIGTERM)
    for process in children:
        signal_group(process.pid, signal.SIGKILL)
    raise SystemExit(128 + signum)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify(root):
    for name in ('CODE_MANIFEST.json', 'POSTFIT_MANIFEST.json'):
        for relative, expected in read(root / name).items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError(f'Immutable manifest mismatch: {name}: {relative}')
    cfg = read(root / 'analysis_code/POSTFIT_PROTOCOL.json')
    if sha(root / 'CODE_MANIFEST.json') != cfg['main_manifest_sha256']:
        raise ValueError('Wrong main experiment')
    return cfg


def process_snapshot(pid):
    """Only retain identity for the explicitly named process."""
    directory = Path('/proc') / str(pid)
    try:
        argv = directory.joinpath('cmdline').read_bytes().split(b'\0')
        argv = [item.decode(errors='replace') for item in argv if item]
        stat = directory.joinpath('stat').read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    fields = stat[stat.rfind(')') + 2:].split()
    if not argv or fields[0] == 'Z':
        return None
    return dict(pid=pid, argv=argv, start_ticks=fields[19])


def is_main_process(snapshot, root):
    return bool(snapshot and str(root / 'code/pipeline.py') in snapshot['argv']
                and str(root) in snapshot['argv'])


def matching_native_workers(root):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (str(root / 'code/native_experiment.py').encode() in argv
                and str(root).encode() in argv):
            matches.append(int(entry.name))
    return sorted(matches)


def wait_for_main(root, output, cfg):
    pid = cfg['parent_pipeline_pid']
    admitted = process_snapshot(pid)
    if admitted is not None and not is_main_process(admitted, root):
        raise RuntimeError('Admitted pipeline PID belongs to a different command')
    write(output / 'WAIT_IDENTITY.json', dict(admitted_process=admitted, checked_at=timestamp(),
          expected_pid=pid, expected_script=str(root / 'code/pipeline.py')))
    deadline = time.monotonic() + cfg['wait_timeout_seconds']
    last_key = None
    while True:
        current = process_snapshot(pid)
        same = (is_main_process(current, root) and
                (admitted is None or current['start_ticks'] == admitted['start_ticks']))
        workers = matching_native_workers(root)
        state_path = root / 'PIPELINE_STATUS.json'
        state = read(state_path) if state_path.is_file() else {}
        terminal = state.get('status') in ('complete', 'failed')
        if not same and not workers:
            if not terminal:
                raise RuntimeError('Main processes ended without a terminal pipeline record; no postfit work started')
            write(output / 'MAIN_TERMINAL.json', dict(status=state, observed_at=timestamp(),
                  pipeline_status_sha256=sha(state_path), main_process_exited=True, native_workers=[]))
            return state
        key = (same, tuple(workers), state.get('status'))
        if key != last_key:
            write(output / 'STATUS.json', dict(status='waiting_for_main', pipeline_alive=same,
                  native_workers=workers, checked_at=timestamp(), pid=os.getpid()))
            last_key = key
        if time.monotonic() >= deadline:
            raise TimeoutError('Bounded main-pipeline wait expired; no process restarted or stopped')
        time.sleep(min(cfg['wait_interval_seconds'], max(0.01, deadline - time.monotonic())))


def run_child(command, env, log_path, timeout):
    begun = time.monotonic()
    if STOP_REQUESTED.is_set():
        raise RuntimeError('Queue stopping; no new diagnostic may start')
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        ACTIVE_CHILDREN[process.pid] = process
        # Close the race if termination occurred while Popen was starting.
        if STOP_REQUESTED.is_set():
            signal_group(process.pid, signal.SIGKILL)
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Only the new diagnostic's own process group, including ffmpeg.
            signal_group(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_group(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
            # A group leader may exit before a child that ignores SIGTERM.
            signal_group(process.pid, signal.SIGKILL)
        finally:
            # On abnormal interruption never leave this queue's child group.
            if process.poll() is None:
                signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            ACTIVE_CHILDREN.pop(process.pid, None)
    return dict(command=command, pid=process.pid, exit_code=returncode, timed_out=timed_out,
                elapsed_seconds=time.monotonic()-begun)


def run_arm(root, output, arm, gpu, cfg):
    from future_information_probe import load_completed_evidence
    try:
        verify(root)
        evidence = load_completed_evidence(root, arm)
    except Exception as error:
        result = dict(arm=arm, gpu=gpu, status='skipped_incomplete_arm', reason=repr(error), goal_complete=False)
        write(output / (arm + '_OUTCOME.json'), result)
        return result
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
    result = dict(arm=arm, gpu=gpu, status='running', steps=evidence['train']['steps'],
                  checkpoint_sha256=evidence['final_hash'], jobs=[], goal_complete=False)
    jobs = [
        ('future_information', 'future_information_probe.py', cfg['future_probe_seconds'], cfg['future_process_seconds']),
        ('render_comparison', 'render_comparison.py', cfg['render_seconds'], cfg['render_process_seconds']),
    ]
    for name, script, budget, timeout in jobs:
        if STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping before next diagnostic')
        verify(root)
        destination = root / 'diagnostics' / name / arm
        command = [sys.executable, '-B', str(root / 'analysis_code' / script), '--root', str(root),
                   '--arm', arm, '--output', str(destination), '--max-seconds', str(budget)]
        outcome = run_child(command, env, output / f'{arm}_{name}.log', timeout)
        outcome.update(name=name, output=str(destination))
        result['jobs'].append(outcome)
        write(output / (arm + '_OUTCOME.json'), result)
        if outcome['exit_code'] != 0 or outcome['timed_out']:
            result['status'] = 'failed'
            write(output / (arm + '_OUTCOME.json'), result)
            return result
    result['status'] = 'complete'
    write(output / (arm + '_OUTCOME.json'), result)
    return result


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        cfg = verify(root)
        signal.signal(signal.SIGTERM, terminate_queue)
        signal.signal(signal.SIGINT, terminate_queue)
        signal.signal(signal.SIGALRM, terminate_queue)
        signal.alarm(cfg['queue_outer_timeout_seconds'])
        validation = read(root / 'POSTFIT_VALIDATION.json')
        if (validation.get('status') != 'passed' or validation.get('gpu_executed') is not False
                or validation.get('postfit_manifest_sha256') != sha(root / 'POSTFIT_MANIFEST.json')
                or validation.get('main_manifest_sha256') != sha(root / 'CODE_MANIFEST.json')):
            raise RuntimeError('Exact postfit code has no successful CPU document witness')
        write(output / 'QUEUE_RECEIPT.json', dict(status='submitted', pid=os.getpid(), started_at=timestamp(),
              main_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
              postfit_manifest_sha256=sha(root / 'POSTFIT_MANIFEST.json'),
              protocol=cfg, goal_complete=False))
        wait_for_main(root, output, cfg)
        write(output / 'STATUS.json', dict(status='postfit_running', pid=os.getpid(), started_at=timestamp()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_arm, root, output, arm, gpu, cfg)
                       for arm, gpu in cfg['arms_and_gpus'].items()]
            outcomes = [future.result() for future in futures]
        verify(root)
        status = 'complete' if all(item['status']=='complete' for item in outcomes) else 'completed_with_failures_or_skips'
        write(output / 'SUMMARY.json', dict(status=status, outcomes=outcomes, finished_at=timestamp(),
              goal_complete=False, interpretation='Postfit diagnostics do not replace primary controls or establish action semantics alone'))
        write(output / 'STATUS.json', dict(status=status, finished_at=timestamp(), goal_complete=False))
    except BaseException as error:
        write(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
              finished_at=timestamp(), goal_complete=False))
        write(output / 'STATUS.json', dict(status='failed', error=repr(error), goal_complete=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    allowed = root / 'diagnostics'
    if not root.is_dir() or not output.is_relative_to(allowed) or output == allowed:
        parser.error('Output must be a new child directory of ROOT/diagnostics')
    if output.exists():
        parser.error('Queue output already exists; refusing duplicate submission or resume')
    main(root, output)
