"""Bounded independent evaluation queue. The external trainer is never owned."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import traceback
import worker as w
import inference_profile as inference

HELPER = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
QUEUE = dict(training_pid=479060, training_start_ticks=40929788, training_wait_seconds=8500,
             evaluation_process_seconds=8500, compare_seconds=1800, total_seconds=20000,
             minimum_gpu_free_gib=16, minimum_disk_free_gib=100,
             owned_helper_path=HELPER, owned_helper_sha256=HELPER_SHA)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_queue(p):
    q = p['queue']
    for key, expected in QUEUE.items():
        w.require(type(q.get(key)) is type(expected) and q[key] == expected, 'Queue protocol drift: ' + key)
    return q


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_identity(snapshot, cfg, root):
    """None means exited; a reused PID or changed command is a hard failure."""
    if snapshot is None:
        return False
    w.require(snapshot.get('pid') == cfg['training_pid']
              and str(snapshot.get('start_ticks')) == str(cfg['training_start_ticks']), 'External training PID identity differs')
    argv = snapshot.get('argv', [])
    w.require(str(root / 'worker.py') in argv and '--root' in argv
              and argv[argv.index('--root') + 1:argv.index('--root') + 2] == [str(root)],
              'External training command differs')
    return True


def completion_probe(root):
    """Read only. No checkpoint loading, process adoption, or signals."""
    folder = root / 'arms' / 'full_codec'
    w.require(not any(p.exists() for p in (root / 'FAILED.json', folder / 'FAILURE.json',
              folder / 'FAILED_DO_NOT_EVALUATE.pt')), 'External training failed; no evaluation eligibility')
    paths = (root / 'DONE.json', folder / 'TRAIN_DONE.json')
    if not all(p.is_file() for p in paths):
        return None
    done, train_done, source = w.full_receipts(root)
    # The source must match the pinned old preflight even before a GPU is allocated.
    old = w.read(Path(w.OLD_ROOT) / 'PREFLIGHT.json')['source']
    w.require(old.get('code_manifest_sha256') == w.OLD_MANIFEST, 'Old training source differs')
    digest = w.validate_full_done(done, train_done, source, old, has_failure=False)
    w.require(w.sha(folder / 'step_4426.pt') == digest, 'Completed full-codec checkpoint differs')
    return dict(status='eligible', done_sha256=w.sha(paths[0]), train_done_sha256=w.sha(paths[1]),
                checkpoint_sha256=digest, training_source=source, old_source=old)


def wait_for_training(root, cfg, outer_deadline, snapshot, stop, *, probe=completion_probe,
                      clock=time.monotonic, sleep=time.sleep):
    deadline = min(clock() + cfg['training_wait_seconds'], outer_deadline)
    while True:
        w.require(not stop.is_set(), 'Queue stopped while waiting for external trainer')
        w.require(clock() < deadline, 'Bounded external training wait expired')
        alive = check_identity(snapshot(cfg['training_pid']), cfg, root)
        complete = probe(root)
        if not alive:
            w.require(complete is not None, 'External trainer exited without successful DONE and TRAIN_DONE')
            return dict(**complete, external_training_pid=cfg['training_pid'],
                        external_training_start_ticks=cfg['training_start_ticks'],
                        external_process_exited=True, unrelated_processes_signalled=False)
        sleep(min(5., max(0., deadline - clock())))


def check_resources(text, disk_free_bytes, gpu, cfg):
    readings = [tuple(int(v.strip()) for v in line.split(',')) for line in text.strip().splitlines()]
    w.require(len(readings) == 1 and len(readings[0]) == 2 and readings[0][0] == gpu,
              'One assigned GPU memory reading required')
    w.require(readings[0][1] >= cfg['minimum_gpu_free_gib'] * 1024, 'Insufficient actual free GPU memory')
    w.require(disk_free_bytes >= cfg['minimum_disk_free_gib'] * 2**30, 'Insufficient free disk space')
    return dict(gpu=gpu, gpu_free_mib=readings[0][1], disk_free_bytes=disk_free_bytes, measured_at=now())


def quality_receipt(root, arm):
    folder = root / 'evaluations' / arm
    w.require(not (folder / 'FAILURE.json').exists(), 'Evaluation has a failure receipt')
    receipt = w.read(folder / 'EVALUATION.json')
    w.require(receipt.get('status') == 'evaluated' and receipt.get('arm') == arm
              and receipt.get('quality_evaluation_complete') is True and receipt.get('frames') == 9027
              and receipt.get('codec_accepted') is False, 'Complete quality receipt required')
    return dict(evaluation_sha256=w.sha(folder / 'EVALUATION.json'),
                replay_passed_all=receipt['replay_passed_all'], replay_failed_frames=receipt['replay_failed_frames'])


def final_status(outcomes, comparison_complete=False):
    successful = {r['arm'] for r in outcomes if r.get('status') == 'complete'}
    if successful == set(w.ARMS) and len(outcomes) == 3 and comparison_complete:
        return 'complete'
    return 'partial' if successful else 'failed'


def main(root):
    begun = time.monotonic()
    root = Path(root).resolve()
    w.require(str(root) == '/data/CJH/AGS-V6Direct/outputs/' + w.RUN
              and root == Path(__file__).resolve().parent, 'Dedicated remote evaluation root required')
    p, source = w.verify_root(root)
    # No worker process is created until the exact CPU-only diagnostic gate passes.
    _, diagnostic_admission = inference.load_diagnostic(p['inference'])
    cfg = validate_queue(p)
    queue = root / 'queue'
    queue.mkdir(exist_ok=False)
    w.require(not (root / 'evaluations').exists() and not (root / 'SUMMARY.json').exists(), 'No retry or existing evaluations')
    w.require(w.sha(HELPER) == HELPER_SHA, 'Owned-child helper changed')
    spec = importlib.util.spec_from_file_location('_complete_eval_owned_children', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    w.require(not helper.ACTIVE_CHILDREN, 'Owned-child registry must start empty')
    w.verify_manifest(Path(w.OLD_ROOT), w.OLD_MANIFEST)
    runtime = w.read(Path(w.OLD_ROOT) / 'protocol.json')['queue']['runtime']
    deadline = begun + cfg['total_seconds']
    records, outcomes, lock = [], [], threading.Lock()
    prior = {sig: signal.signal(sig, helper.terminate_queue) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    w.require(time.monotonic() < deadline, 'Queue initialization exhausted total budget')
    signal.alarm(max(1, int(deadline - time.monotonic())))

    def stop_owned():
        try:
            helper.terminate_queue(signal.SIGTERM, None)
        except SystemExit:
            pass

    def verify():
        w.require(not helper.STOP_REQUESTED.is_set(), 'Queue stopped')
        w.require(w.verify_root(root)[1] == source, 'Evaluation source changed')

    def run(name, gpu=None, arm=None):
        verify()
        argv = [runtime, '-B', '-u', str(root / ('compare.py' if arm is None else 'worker.py')), '--root', str(root)]
        if arm is not None:
            argv += ['--arm', arm]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        limit = cfg['compare_seconds'] if arm is None else cfg['evaluation_process_seconds']
        seconds = min(limit, deadline - time.monotonic())
        w.require(seconds > 0, 'Total queue budget expired')
        result = helper.run_child(argv, env, queue / (name + '.log'), seconds)
        record = dict(name=name, gpu=gpu, process_budget_seconds=seconds, **result)
        with lock:
            records.append(record)
        write_new(queue / (name + '_PROCESS.json'), record)
        w.require(result['exit_code'] == 0 and result['timed_out'] is False, name + ' failed; no retry')
        verify()

    def evaluate(arm, gpu, wait=False):
        result = dict(arm=arm, gpu=gpu, status='failed', quality_evaluation_complete=False, codec_accepted=False)
        try:
            if wait:
                evidence = wait_for_training(Path(w.FULL_ROOT), cfg, deadline, helper.process_snapshot, helper.STOP_REQUESTED)
                write_new(queue / 'TRAINING_ELIGIBILITY.json', evidence)
            verify()
            query = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-gpu=index,memory.free',
                                             '--format=csv,noheader,nounits'], text=True, timeout=15)
            resource = check_resources(query, shutil.disk_usage(root).free, gpu, cfg)
            write_new(queue / (arm + '_RESOURCE_ADMISSION.json'), resource)
            run('evaluate_' + arm, gpu, arm)
            result.update(status='complete', quality_evaluation_complete=True, **quality_receipt(root, arm))
        except Exception as error:
            # No terminate_queue here: an independent failure must not kill another arm.
            result.update(error=repr(error), traceback=traceback.format_exc())
        with lock:
            outcomes.append(result)
        write_new(queue / (arm + '_OUTCOME.json'), result)
        return result

    def gpu_zero():
        evaluate('baseline', 0)
        evaluate('decoder_only', 0)

    executor = None
    try:
        write_new(queue / 'RECEIPT.json', dict(status='running', started_at=now(), source=source,
                  process=helper.process_snapshot(os.getpid()), queue_protocol=cfg, training_updates=0,
                  diagnostic_admission=diagnostic_admission,
                  external_training_is_owned=False, owned_helper_sha256=HELPER_SHA,
                  gpu_assignment={'baseline': 0, 'decoder_only': 0, 'full_codec': 1}))
        executor = ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(gpu_zero), executor.submit(evaluate, 'full_codec', 1, True)]
        for future in futures:
            future.result()
        compare_complete, compare_error = False, None
        if len(outcomes) == 3 and all(r['status'] == 'complete' for r in outcomes):
            try:
                run('compare')
                summary = w.read(root / 'SUMMARY.json')
                w.require(summary.get('status') == 'complete' and summary.get('codec_accepted') is False,
                          'Comparison did not produce complete audit')
                compare_complete = True
            except Exception as error:
                compare_error = repr(error)
        status = final_status(outcomes, compare_complete)
        report = dict(status=status, finished_at=now(), source=source, outcomes=outcomes, processes=records,
                      comparison_complete=compare_complete, comparison_error=compare_error,
                      quality_evaluation_complete=all(r['status'] == 'complete' for r in outcomes) and len(outcomes) == 3,
                      codec_accepted=False, codec_acceptance_status='pending_separate_review',
                      automatic_retry=False, unrelated_processes_signalled=False, training_updates=0,
                      action_Z_trained=False, automatic_LDM_codec_replacement=False, goal_complete=False)
        if compare_complete:
            report['summary_sha256'] = w.sha(root / 'SUMMARY.json')
        write_new(queue / ('DONE.json' if status == 'complete' else 'PARTIAL.json' if status == 'partial' else 'FAILED.json'), report)
        w.write(queue / 'STATUS.json', report)
        return 0 if status == 'complete' else 1
    except BaseException as error:
        stop_owned()
        report = dict(status='failed', error=repr(error), at=now(), source=source, outcomes=outcomes,
                      processes=records, quality_evaluation_complete=False, codec_accepted=False,
                      automatic_retry=False, unrelated_processes_signalled=False, goal_complete=False)
        w.write(queue / 'FAILED.json', report)
        w.write(queue / 'STATUS.json', report)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        signal.alarm(0)
        for sig, handler in prior.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().root))
