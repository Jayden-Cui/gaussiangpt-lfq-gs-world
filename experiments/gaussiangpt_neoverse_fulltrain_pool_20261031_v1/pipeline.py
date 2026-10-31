"""One bounded, non-resumable two-GPU export of the complete authorized pool.

This prepares data only. Process groups belong to this queue, never its resource
dependency. Source snapshots, source roles and every original frame are retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
import bindings
import prepare_clip

RUN = bindings.RUN
BASE = bindings.BASE
ROOT = BASE + '/' + RUN
RUNTIME = '/data/CJH/envs/gaussiangpt/bin/python'
HELPER_PATH = BASE + '/gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
QUEUE = dict(wait_seconds=7200, wait_interval_seconds=60, reuse_seconds=600,
             binding_seconds=300, teacher_seconds=1200, prepare_seconds=1230,
             export_seconds=9000, final_seconds=1800, outer_seconds=18600)
RESOURCES = dict(cpu_threads=6, minimum_free_gib=55, initial_disk_free_gib=500,
                 per_clip_disk_free_gib=64)
DEPENDENCY = dict(root=BASE + '/gaussiangpt_neoverse_spatial_action_20261031_v9',
                  pid=441516, start_ticks='39022821',
                  manifest_sha256='b40b3353bdcc322c0f40497b1eb17ebeec293e07261a0668d23c04ed31fbdc40',
                  summary_sha256='961a28478dc10e6a651d60ab29ea15fed89f0ee1ee9a7b56ef32580a8ffe09f7',
                  done_sha256='20b8c61e73b3d04b0e46460d7f5808bdf95a80d5182d577cad5613f91b1153c9')
REQUIRED = {'pipeline.py', 'test_pipeline.py', 'admission.py', 'test_admission.py',
            'validate.py', 'protocol.json', 'EXECUTION.md'}
OUTCOME_LOCK = threading.Lock()
read = bindings.read_json
sha = bindings.sha256
require = bindings.require


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def status(output, stage, **fields):
    path = output / 'STATUS.json'
    temporary = output / 'STATUS.tmp'
    temporary.write_text(json.dumps(dict(status=stage, pid=os.getpid(),
        updated_at=timestamp(), goal_complete=False, **fields), indent=2), encoding='utf-8')
    temporary.replace(path)


def protocol_contract(cfg):
    require(cfg.get('schema') == 'gs_fulltrain_pool_v1', 'Wrong pool protocol')
    require(cfg.get('queue') == QUEUE and cfg.get('resources') == RESOURCES,
            'Fixed queue budget or resources changed')
    require(all(cfg.get(k) is False for k in ('automatic_retry', 'training_executed', 'goal_complete')),
            'Export must not retry, train, or complete the research goal')
    require(cfg.get('process_helper') == {'path': HELPER_PATH, 'sha256': HELPER_SHA},
            'Immutable process helper changed')
    dependency = cfg.get('dependency', {})
    require(all(dependency.get(k) == v for k, v in DEPENDENCY.items()), 'Exact resource dependency changed')
    require(bindings._sha(dependency.get('done_sha256')), 'Missing pinned dependency DONE digest')


def helper_file(root):
    # The same pinned sibling is available in local CPU witnesses and on 134.
    return Path(root).parent / 'gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'


def verify(root, expected=None, require_cpu=True):
    manifest = bindings.verify_manifest(root)
    require(REQUIRED <= set(manifest), 'Manifest omits queue, admission or document sources')
    cfg = read(root / 'protocol.json')
    protocol_contract(cfg)
    require(sha(helper_file(root)) == HELPER_SHA, 'Immutable process helper bytes changed')
    proof = dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
                 protocol_sha256=sha(root / 'protocol.json'), input_pool_sha256=sha(root / 'INPUT_POOL.json'),
                 process_helper_sha256=HELPER_SHA)
    if require_cpu:
        validation = read(root / 'CPU_VALIDATION.json')
        require(validation.get('status') == 'passed' and validation.get('gpu_executed') is False
                and type(validation.get('actual_tests_executed')) is int
                and validation['actual_tests_executed'] > 0
                and all(validation.get(k) == proof[k] for k in ('code_manifest_sha256', 'protocol_sha256')),
                'No passed CPU document witness for exact sources and protocol')
        proof['cpu_validation_sha256'] = sha(root / 'CPU_VALIDATION.json')
    require(expected is None or proof == expected, 'Frozen queue inputs changed')
    return cfg, proof


def load_helper(root):
    path = helper_file(root)
    source = path.read_bytes()
    import hashlib
    require(hashlib.sha256(source).hexdigest() == HELPER_SHA, 'Process helper changed before import')
    spec = importlib.util.spec_from_file_location('_fullpool_owned_groups', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def command_parts(snapshot):
    if not isinstance(snapshot, dict):
        return None
    argv = snapshot.get('argv')
    if (not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv)
            or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', PurePosixPath(argv[0]).name)):
        return None
    i = 1
    while i < len(argv) and argv[i] in ('-B', '-u'):
        i += 1
    return (argv[i], argv[i + 1:]) if i < len(argv) and not argv[i].startswith('-') else None


def classify_parent(snapshot, dependency):
    if snapshot is None:
        return 'exited'
    require(snapshot.get('pid') == dependency['pid']
            and str(snapshot.get('start_ticks')) == dependency['start_ticks']
            and command_parts(snapshot) == (dependency['root'] + '/pipeline.py', ['--root', dependency['root']]),
            'Dependency PID/start-ticks/command mismatch; shared processes are never signalled')
    return 'alive'


def dependency_command(snapshot, dependency):
    parts = command_parts(snapshot)
    if parts is None:
        return False
    script, argv = parts
    root = dependency['root']
    return (script in {root + '/' + name for name in ('pipeline.py', 'worker.py', 'compare.py')}
            and argv.count('--root') == 1 and argv.index('--root') + 1 < len(argv)
            and argv[argv.index('--root') + 1] == root)


def stable_snapshot(helper, pid):
    first, second = helper.process_snapshot(pid), helper.process_snapshot(pid)
    if second is None:
        try:
            text = (Path('/proc') / str(pid) / 'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        require(text[text.rfind(')') + 2:].split()[0] == 'Z',
                'Live dependency identity unavailable; exit not inferred')
        return None
    require(first == second, 'Dependency identity raced during observation')
    return second


def observe(helper, dependency):
    require(sha(Path(dependency['root']) / 'CODE_MANIFEST.json') == dependency['manifest_sha256'],
            'Dependency manifest changed')
    parent_state = classify_parent(stable_snapshot(helper, dependency['pid']), dependency)
    workers = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        snapshot = helper.process_snapshot(int(entry.name))
        if not dependency_command(snapshot, dependency):
            continue
        current = stable_snapshot(helper, int(entry.name))
        if current is None:
            continue
        require(current == snapshot, 'Dependency worker identity raced')
        if command_parts(current)[0].endswith('/pipeline.py'):
            parent_state = classify_parent(current, dependency)
        else:
            workers.append(current)
    return dict(parent_state=parent_state, workers=workers,
                all_exited=parent_state == 'exited' and not workers)


def read_terminal(dependency):
    root = Path(dependency['root'])
    require(not (root / 'queue/FAILED.json').exists(), 'Expected successful v9 dependency, found FAILED')
    paths = {'done_sha256': root / 'queue/DONE.json', 'summary_sha256': root / 'SUMMARY.json',
             'manifest_sha256': root / 'CODE_MANIFEST.json'}
    require(all(sha(path) == dependency[key] for key, path in paths.items()), 'Dependency terminal bytes changed')
    done = read(paths['done_sha256'])
    require(done.get('status') == 'complete'
            and done.get('admission', {}).get('code_manifest_sha256') == dependency['manifest_sha256']
            and done.get('summary_sha256') == dependency['summary_sha256'], 'Dependency terminal identity mismatch')
    return dict(done_sha256=dependency['done_sha256'], summary_sha256=dependency['summary_sha256'],
                resource_dependency_only=True, goal_complete=False)


def wait_for_parent(output, cfg, helper):
    deadline = time.monotonic() + cfg['queue']['wait_seconds']
    while True:
        require(not helper.STOP_REQUESTED.is_set(), 'Queue stopped before resource admission')
        observation = observe(helper, cfg['dependency'])
        status(output, 'waiting_dependency', **observation)
        if observation['all_exited']:
            terminal = read_terminal(cfg['dependency'])
            write_new(output / 'DEPENDENCY_TERMINAL.json', dict(observation=observation, **terminal))
            return terminal
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Bounded resource wait expired; no shared process stopped or restarted')
        helper.STOP_REQUESTED.wait(min(cfg['queue']['wait_interval_seconds'], remaining))


def assignments(pool):
    bindings.validate_pool(pool)
    clips = [row for row in pool['clips'] if row['requires_new_full177_export']]
    clips.sort(key=lambda row: (int(row['source_group'].rsplit('_', 1)[1]), row['index_window_offset']))
    lanes = {gpu: [row['clip_id'] for row in clips[gpu::2]] for gpu in (0, 1)}
    require(all(len(values) == 24 for values in lanes.values())
            and len(set(lanes[0] + lanes[1])) == 48, 'Expected 48 unique fresh clips, 24 per GPU')
    return lanes


def disk_admission(root, minimum_gib):
    free = shutil.disk_usage(root).free
    require(free >= minimum_gib * 2**30, f'Disk below fixed {minimum_gib} GiB reserve; no retry or reduction')
    return dict(free_bytes=free, minimum_free_gib=minimum_gib, checked_at=timestamp())


def gpu_admission(output):
    result = subprocess.run(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=15, check=True)
    parsed = {}
    for line in result.stdout.strip().splitlines():
        match = re.fullmatch(r'\s*([01])\s*,\s*(\d+)\s*', line)
        require(match is not None and int(match[1]) not in parsed, 'Unexpected GPU admission response')
        parsed[int(match[1])] = int(match[2])
    require(set(parsed) == {0, 1}, 'Both assigned GPUs must be present')
    write_new(output / 'GPU_ADMISSION.json', dict(free_mib=parsed, required_mib=55 * 1024, queries=1))
    require(all(free >= 55 * 1024 for free in parsed.values()), 'GPU free memory below 55 GiB; no retry')


def stop_owned(helper):
    try:
        helper.terminate_queue(signal.SIGTERM, None)
    except SystemExit:
        pass


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def run_job(name, command, gpu, timeout, output, helper, outcomes, deadline=None):
    row = dict(name=name, command=command, gpu=gpu, maximum_process_seconds=timeout)
    try:
        require(not helper.STOP_REQUESTED.is_set(), 'Queue stopping; no new child')
        remaining = timeout if deadline is None else min(timeout, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Fixed export-stage deadline reached')
        row['actual_timeout_seconds'] = remaining
        row.update(helper.run_child(command, environment(gpu), output / (name + '.log'), remaining))
        require(row.get('exit_code') == 0 and row.get('timed_out') is False,
                name + ' failed or timed out; no retry')
        row['status'] = 'complete'
        return row
    except BaseException as error:
        row.update(status='failed', error=repr(error))
        stop_owned(helper)
        raise
    finally:
        with OUTCOME_LOCK:
            outcomes.append(row)
            write_new(output / (name + '_PROCESS.json'), row)


def receipt_contract(root, name, expected_count, proof):
    value = read(root / name)
    require(value.get('status') == 'complete' and value.get('goal_complete') is False
            and value.get('training_executed') is False, 'Admission status or scope mismatch')
    require(all(value.get(key) == proof[key] for key in ('code_manifest_sha256', 'input_pool_sha256')),
            'Admission bound to different inputs')
    require(isinstance(value.get('clips'), list) and len(value['clips']) == expected_count,
            'Incomplete admission membership')
    identities = [row.get('clip_id') for row in value['clips'] if isinstance(row, dict)]
    expected = bindings.REUSE if expected_count == 2 else set(bindings.EXPECTED_CLIPS)
    require(len(identities) == expected_count and len(set(identities)) == expected_count
            and set(identities) == expected, 'Duplicate or incorrect admission membership')
    if name == 'POOL_COMPLETE.json':
        require(value.get('actual_clip_internal_frames_total') == 8850
                and value.get('all_clips_complete') is True
                and 'cross_clip_physical_frame_unique_count' in value
                and value['cross_clip_physical_frame_unique_count'] is None,
                'Final coverage or physical-frame semantics changed')
    return sha(root / name)


def run_lane(root, output, gpu, clips, helper, outcomes, deadline, before_stage):
    try:
        for clip in clips:
            before_stage()
            require(not helper.STOP_REQUESTED.is_set(), 'Queue stopped before next clip')
            directory = bindings.clip_root(root, clip)
            require(not directory.exists(), 'Existing clip attempt; no resume or overwrite')
            disk = disk_admission(root, RESOURCES['per_clip_disk_free_gib'])
            write_new(output / (clip + '_DISK.json'), disk)
            run_job(clip + '_binding', [RUNTIME, '-B', str(root / 'bindings.py'), '--root', str(root), '--clip', clip],
                    None, QUEUE['binding_seconds'], output, helper, outcomes, deadline)
            before_stage()
            run_job(clip + '_teacher', prepare_clip.teacher_command(root, clip), gpu, QUEUE['teacher_seconds'],
                    output, helper, outcomes, deadline)
            before_stage()
            run_job(clip + '_prepare', [RUNTIME, '-B', '-u', str(root / 'prepare_clip.py'), '--root', str(root),
                                      '--clip', clip], gpu, QUEUE['prepare_seconds'], output, helper, outcomes, deadline)
            receipt = read(directory / 'PREPARATION_RECEIPT.json')
            require(receipt.get('status') == 'complete' and receipt.get('clip') == clip
                    and receipt.get('training_performed') is False and receipt.get('goal_complete') is False,
                    'Completed child omitted preparation receipt or changed scope')
            write_new(output / (clip + '_COMPLETE.json'), dict(clip=clip, gpu=gpu, status='complete',
                      preparation_receipt_sha256=sha(directory / 'PREPARATION_RECEIPT.json'), goal_complete=False))
        before_stage()
    except BaseException:
        stop_owned(helper)
        raise


def execute_exports(root, output, lanes, helper, outcomes, before_stage):
    deadline = time.monotonic() + QUEUE['export_seconds']
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_lane, root, output, gpu, clips, helper, outcomes, deadline, before_stage)
                   for gpu, clips in lanes.items()]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop_owned(helper)
            raise


def main(root):
    root = Path(root).resolve(strict=True)
    require(root.as_posix() == ROOT and Path(__file__).resolve().parent == root,
            'Only the dedicated fulltrain pool queue may execute')
    require(not (root / 'queue').is_symlink() and not (root / 'clips').exists()
            and all(not (root / name).exists() for name in ('REUSE_ADMISSION.json', 'POOL_COMPLETE.json')),
            'Existing export attempt; no retry or resume')
    output = root / 'queue'
    output.mkdir(exist_ok=False)
    helper, outcomes, proof, terminal = None, [], {}, None
    def startup_stop(signum, frame):
        raise TimeoutError('Queue interrupted during bounded startup')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, startup_stop)
        signal.alarm(QUEUE['outer_seconds'])
        cfg, proof = verify(root)
        helper = load_helper(root)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        pool = bindings.load_pool(root)
        lanes = assignments(pool)
        write_new(output / 'RECEIPT.json', dict(status='submitted', pid=os.getpid(), started_at=timestamp(),
                  admission=proof, assignments=lanes, dependency=cfg['dependency'], automatic_retry=False,
                  training_executed=False, goal_complete=False, only_owned_process_groups_may_be_signalled=True))
        terminal = wait_for_parent(output, cfg, helper)
        def before_stage():
            verify(root, proof)
            require(not helper.STOP_REQUESTED.is_set(), 'Queue stopped')
            require(observe(helper, cfg['dependency'])['all_exited']
                    and read_terminal(cfg['dependency']) == terminal, 'Dependency resumed or changed')
        before_stage()
        write_new(output / 'DISK_ADMISSION.json', disk_admission(root, RESOURCES['initial_disk_free_gib']))
        status(output, 'reuse_admission')
        run_job('reuse_admission', [RUNTIME, '-B', str(root / 'admission.py'), '--root', str(root), '--mode', 'reuse'],
                None, QUEUE['reuse_seconds'], output, helper, outcomes)
        reuse_sha = receipt_contract(root, 'REUSE_ADMISSION.json', 2, proof)
        before_stage()
        gpu_admission(output)
        status(output, 'export', fresh_clips=48, reuse_clips=2)
        execute_exports(root, output, lanes, helper, outcomes, before_stage)
        before_stage()
        require(sha(root / 'REUSE_ADMISSION.json') == reuse_sha, 'Reuse admission changed during export')
        status(output, 'final_admission')
        run_job('final_admission', [RUNTIME, '-B', str(root / 'admission.py'), '--root', str(root), '--mode', 'final'],
                None, QUEUE['final_seconds'], output, helper, outcomes)
        complete_sha = receipt_contract(root, 'POOL_COMPLETE.json', 50, proof)
        before_stage()
        require(len(outcomes) == 146 and all(row['status'] == 'complete' for row in outcomes),
                'Incomplete child execution ledger')
        status(output, 'complete')
        write_new(output / 'DONE.json', dict(status='complete', finished_at=timestamp(), admission=proof,
                  reuse_admission_sha256=reuse_sha, pool_complete_sha256=complete_sha, outcomes=outcomes,
                  dependency_terminal=terminal, clips=50, fresh_exports=48, reused_exports=2,
                  clip_internal_frames=8850, cross_clip_physical_frame_unique_count=None,
                  automatic_retry=False, training_executed=False, goal_complete=False))
    except BaseException as error:
        if helper is not None:
            stop_owned(helper)
        write_new(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                  admission=proof, outcomes=outcomes, dependency_terminal=terminal, finished_at=timestamp(),
                  automatic_retry=False, old_processes_signalled=False, training_executed=False, goal_complete=False))
        status(output, 'failed', error=repr(error))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
