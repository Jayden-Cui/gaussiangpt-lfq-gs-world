"""Bounded, read-only-source tower14 teacher export and frozen-codec preparation queue."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = PurePosixPath('/data/CJH/AGS-V6Direct/outputs')
RUN = 'gaussiangpt_neoverse_tower14_transfer_20261031_v1'
V7 = BASE / 'gaussiangpt_neoverse_temporal_z_20261031_v7'
HELPER = BASE / 'gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
MAIN_PID, MAIN_TICKS = 429150, '37740478'
PREVIEW = BASE / 'neoverse_gs_preview_20261028_v1'
RUNTIME = PREVIEW / 'runtime/bin/python'
REPO = PREVIEW / 'NeoVerse_source'
CHECKPOINT = PREVIEW / 'weights_cache/models/Yuppie1204--NeoVerse/snapshots/master/reconstructor.ckpt'
RENDERER = BASE / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py'
CODEC = BASE / 'gaussiangpt_neoverse_state_motion_20261030_v1/state'
GS_PYTHON = PurePosixPath('/data/CJH/envs/gaussiangpt/bin/python')
CLIP = 'buildingblock_tower_14_0'
VIDEO = BASE / 'buildingblock_vw2_h4_20261011_v1/datasets/Video-CraftBench/Paper_and_Block_clips' / (CLIP + '.mp4')
VIDEO_SHA = '456d178c177d6790196f2e27f3bf5e1f5eebbb9fdaecc716794be90fbbab57f8'
CHECKPOINT_SHA = 'fd87ca703cf026c19e345e08f68292ae5a43e34d77ff20c7b49b78907e10f5c9'
REQUIRED_LOCAL = {'queue_prepare.py', 'PREPARATION_PROTOCOL.json', 'EXECUTION.md', 'test_queue_prepare.py',
                  'prepare_frozen_codec.py', 'README_preparation.md', 'input_binding.json'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return result


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def local_manifest(root):
    """CPU witness checks only local relative manifest entries, never binding paths."""
    manifest = read(root / 'CODE_MANIFEST.json')
    if not REQUIRED_LOCAL.issubset(manifest):
        raise ValueError('Manifest omits required code, documents, protocol or binding')
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError('Local immutable source mismatch: ' + relative)
    return dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
                input_binding_sha256=sha(root / 'input_binding.json'))


def protocol_contract(cfg):
    expected = dict(schema='gs_tower14_preparation_queue_protocol_v1', wait_timeout_seconds=14000,
        wait_interval_seconds=60, queue_outer_timeout_seconds=16600, renderer_process_seconds=1200,
        preparation_internal_seconds=1200, preparation_process_seconds=1230, gpu=0,
        minimum_free_gib=60, main_pid=MAIN_PID, main_start_ticks=MAIN_TICKS,
        parent_script=str(V7 / 'postfit_code/queue_postfit.py'), parent_root=str(V7),
        process_helper_sha256=HELPER_SHA, automatic_retry=False, training_performed=False,
        scientific_transfer_passed=None, job_order=['teacher_export', 'frozen_codec_preparation'])
    if cfg != expected:
        raise ValueError('Fixed preparation queue protocol changed')


def verify_sources(root, expected=None):
    admission = local_manifest(root)
    cfg = read(root / 'PREPARATION_PROTOCOL.json')
    protocol_contract(cfg)
    witness = read(root / 'CPU_VALIDATION.json')
    if (witness.get('status') != 'passed' or witness.get('gpu_executed') is not False
            or witness.get('goal_complete') is not False
            or any(witness.get(k) != v for k, v in admission.items())):
        raise ValueError('Exact code/binding has no successful fresh CPU witness')
    admission['cpu_validation_sha256'] = sha(root / 'CPU_VALIDATION.json')
    binding = read(root / 'input_binding.json')
    if binding.get('schema') != 'gs_frozen_codec_transfer_binding_v1' or not binding.get('files'):
        raise ValueError('Invalid immutable input binding')
    files = {}
    for row in binding['files']:
        path = Path(row['path'])
        if (not path.is_absolute() or str(path) in files or path.stat().st_size != row['bytes']
                or sha(path) != row['sha256']):
            raise ValueError('Bound input changed: ' + str(path))
        files[str(path)] = row['sha256']
    teacher = binding.get('expected_teacher', {})
    fixed = dict(clip=CLIP, frames=177, height=252, width=448, frame_stride=1, use_motion=False,
                 source_video_path=str(VIDEO), source_video_sha256=VIDEO_SHA,
                 renderer_path=str(RENDERER), checkpoint_sha256=CHECKPOINT_SHA)
    if any(teacher.get(k) != v for k, v in fixed.items()):
        raise ValueError('Fixed full source video/teacher contract changed')
    required = {VIDEO: VIDEO_SHA, CHECKPOINT: CHECKPOINT_SHA, RENDERER: teacher.get('renderer_sha256')}
    for name in ('codec_final.pt', 'selection.json', 'protocol.json', 'code/common.py', 'code/codec.py'):
        required[CODEC / name] = files.get(str(CODEC / name))
    if any(digest is None or files.get(str(path)) != digest for path, digest in required.items()):
        raise ValueError('Required command input is not fully bound')
    if sha(HELPER) != HELPER_SHA:
        raise ValueError('Verified process-group helper changed')
    if expected is not None and admission != expected:
        raise ValueError('Admission manifest, input binding or CPU witness changed')
    return cfg, admission


def command_parts(snapshot):
    if not isinstance(snapshot, dict):
        return None
    argv = snapshot.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        return None
    if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?', PurePosixPath(argv[0]).name):
        return None
    index = 1
    while index < len(argv) and argv[index] in ('-B', '-u'):
        index += 1
    if index == len(argv) or argv[index].startswith('-'):
        return None
    return argv[index], argv[index + 1:]


def parent_command(snapshot, script):
    return command_parts(snapshot) == (str(script), ['--root', str(V7)])


def classify_parent(snapshot, pid, ticks, script):
    if snapshot is None:
        return 'exited'
    if snapshot.get('pid') != pid or snapshot.get('stable', True) is not True:
        return 'identity_mismatch'
    if str(snapshot.get('start_ticks')) != str(ticks):
        return 'replacement' if parent_command(snapshot, script) else 'pid_reused_unrelated'
    return 'alive' if parent_command(snapshot, script) else 'identity_mismatch'


def worker_match(snapshot):
    parsed = command_parts(snapshot)
    if parsed is None:
        return False
    script, arguments = parsed
    scripts = {str(V7 / p) for p in ('runner.py', 'compare.py',
              'postfit_code/render_six_panel.py', 'postfit_code/nearby_view_probe.py')}
    if script not in scripts or arguments.count('--root') != 1:
        return False
    index = arguments.index('--root')
    return index + 1 < len(arguments) and arguments[index + 1] == str(V7)


def relevant_process(snapshot):
    """Only already matched same-root work merits a strict second identity read."""
    return worker_match(snapshot) or any(parent_command(snapshot, V7 / script)
        for script in ('pipeline.py', 'postfit_code/queue_postfit.py'))


def terminal_decision(main_done, main_failed, postfit_done, postfit_failed):
    if main_failed is not None or not isinstance(main_done, dict) or main_done.get('status') != 'complete':
        raise ValueError('Main queue must have DONE complete and no FAILED receipt')
    if (postfit_done is None) == (postfit_failed is None):
        raise ValueError('Require exactly one postfit DONE or FAILED terminal receipt')
    kind, receipt = ('DONE', postfit_done) if postfit_done is not None else ('FAILED', postfit_failed)
    if not isinstance(receipt, dict) or receipt.get('status') != ('complete' if kind == 'DONE' else 'failed'):
        raise ValueError('Invalid postfit terminal status')
    return dict(main_status='complete', postfit_terminal=kind, postfit_failure_does_not_block_preparation=True)


def stable_snapshot(helper, pid):
    """Reuse pinned reader, reject unreadable live identities and raced PID reuse."""
    first = helper.process_snapshot(pid)
    second = helper.process_snapshot(pid)
    if second is None:
        try:
            text = (Path('/proc') / str(pid) / 'stat').read_text()
            state = text[text.rfind(')') + 2:].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return None
        if state == 'Z':
            return None
        raise RuntimeError('Live process identity is unavailable; not assuming exit')
    if first is None or first != second:
        raise RuntimeError('Process identity changed during observation')
    return dict(second, stable=True)


def observe(helper, parent_pid, parent_ticks):
    parents = ((MAIN_PID, MAIN_TICKS, V7 / 'pipeline.py'),
               (parent_pid, parent_ticks, V7 / 'postfit_code/queue_postfit.py'))
    states, workers = {}, []
    for pid, ticks, script in parents:
        state = classify_parent(stable_snapshot(helper, pid), pid, ticks, script)
        if state in ('identity_mismatch', 'replacement'):
            raise RuntimeError('Parent identity changed; no process will be signalled: ' + str(pid))
        states[str(pid)] = state
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        candidate = helper.process_snapshot(int(directory.name))
        if not relevant_process(candidate):
            continue  # Unrelated/kernel/vanished processes never receive strict rereads.
        snapshot = stable_snapshot(helper, int(directory.name))
        if snapshot is None:
            continue
        if any(snapshot[key] != candidate[key] for key in ('pid', 'start_ticks', 'argv')):
            raise RuntimeError('Matched same-root worker/parent identity changed during observation')
        for pid, ticks, script in parents:
            if parent_command(snapshot, script) and (snapshot['pid'] != pid or str(snapshot['start_ticks']) != str(ticks)):
                raise RuntimeError('A replacement same-root parent exists; refusing admission')
        if worker_match(snapshot):
            workers.append(snapshot)
    return dict(parents=states, workers=workers, all_exited=all(x != 'alive' for x in states.values()) and not workers)


def read_terminal():
    paths = {key: Path(V7 / relative) for key, relative in dict(main_done='queue/DONE.json',
        main_failed='queue/FAILED.json', postfit_done='queue_postfit/DONE.json', postfit_failed='queue_postfit/FAILED.json').items()}
    values = {key: read(path) if path.is_file() else None for key, path in paths.items()}
    return dict(decision=terminal_decision(**values), hashes={str(path): sha(path) for path in paths.values() if path.is_file()})


def wait_for_parents(output, cfg, helper, parent_pid, parent_ticks):
    begun = time.monotonic()
    while True:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping')
        observation = observe(helper, parent_pid, parent_ticks)
        write(output / 'STATUS.json', dict(status='waiting', elapsed_seconds=time.monotonic() - begun, **observation))
        if observation['all_exited']:
            terminal = read_terminal()  # Missing receipts fail immediately once all related work exited.
            write(output / 'PARENTS_TERMINAL.json', dict(observation=observation, **terminal))
            return terminal
        remaining = cfg['wait_timeout_seconds'] - (time.monotonic() - begun)
        if remaining <= 0:
            raise TimeoutError('14000-second parent wait expired; no old/shared process stopped')
        helper.STOP_REQUESTED.wait(min(cfg['wait_interval_seconds'], remaining))


def unchanged_parents(helper, parent_pid, parent_ticks, terminal):
    if not observe(helper, parent_pid, parent_ticks)['all_exited'] or read_terminal() != terminal:
        raise RuntimeError('Parent work resumed or terminal receipts changed')


def commands(root):
    teacher = root / 'teacher/native448'
    return (
        ('teacher_export', [str(RUNTIME), '-B', '-u', str(RENDERER), '--repo', str(REPO), '--checkpoint', str(CHECKPOINT),
            '--input', str(VIDEO), '--output', str(teacher), '--num-frames', '177', '--start-frame', '0',
            '--frame-stride', '1', '--fps', '29.97', '--height', '252', '--width', '448', '--save-gs', '177',
            '--novel-view-shift', '0.03', '--device', 'cuda:0', '--dtype', 'bfloat16']),
        ('frozen_codec_preparation', [str(GS_PYTHON), '-B', '-u', str(root / 'prepare_frozen_codec.py'),
            '--root', str(root), '--teacher', str(teacher), '--codec-root', str(CODEC), '--clip', CLIP, '--max-seconds', '1200']))


def prepared_contract(summary, binding_hash):
    fixed = dict(status='complete', preparation_contract_passed=True, frames=177, clip=CLIP, codec_frozen=True,
                 training_performed=False, scientific_transfer_passed=None, input_binding_sha256=binding_hash)
    if (any(k not in summary or summary[k] != v for k, v in fixed.items())
            or not re.fullmatch('[0-9a-f]{64}', summary.get('codec_state_sha256_before_and_after', ''))):
        raise ValueError('Frozen preparation did not provide the actual complete/no-training/model-unchanged contract')


def main(root, parent_pid, parent_ticks):
    root = root.resolve()
    if root != BASE / RUN or parent_pid <= 1 or parent_pid == MAIN_PID or not re.fullmatch('[0-9]+', parent_ticks):
        raise ValueError('Require dedicated remote root and explicit postfit parent PID/start ticks')
    output = root / 'queue'
    output.mkdir(exist_ok=False)
    helper, outcomes = None, []

    def startup_deadline(signum, frame):
        raise TimeoutError('Preparation queue exceeded its 16600-second outer budget')

    signal.signal(signal.SIGALRM, startup_deadline)
    signal.alarm(16600)  # Includes initial full source hashing; never reset after admission.
    try:
        cfg, admission = verify_sources(root)
        spec = importlib.util.spec_from_file_location('_tower14_owned_groups', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        write(output / 'RECEIPT.json', dict(status='waiting', pid=os.getpid(), admission=admission,
            parent_pid=parent_pid, parent_start_ticks=parent_ticks, parent_script=str(V7 / 'postfit_code/queue_postfit.py'),
            parent_root=str(V7), only_own_new_session_groups_may_be_signalled=True, scientific_transfer_passed=None))
        terminal = wait_for_parents(output, cfg, helper, parent_pid, parent_ticks)
        verify_sources(root, admission)
        unchanged_parents(helper, parent_pid, parent_ticks, terminal)
        gpu = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15, check=True)
        fields = gpu.stdout.strip().splitlines()
        if len(fields) != 1 or not re.fullmatch(r'\s*\d+\s*', fields[0]):
            raise ValueError('Unexpected one-shot GPU0 free-memory response')
        free_mib = int(fields[0])
        write(output / 'GPU_ADMISSION.json', dict(gpu=0, free_mib=free_mib, required_mib=60 * 1024, queries=1))
        if free_mib < 60 * 1024:
            raise RuntimeError('GPU0 has less than 60 GiB free; fail without waiting or retry')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6',
                   MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for name, command in commands(root):
            verify_sources(root, admission)
            unchanged_parents(helper, parent_pid, parent_ticks, terminal)
            if helper.STOP_REQUESTED.is_set():
                raise RuntimeError('Queue stopping before child launch')
            if name == 'teacher_export' and (root / 'teacher').exists():
                raise ValueError('Teacher output exists; no overwrite or retry')
            timeout = cfg['renderer_process_seconds'] if name == 'teacher_export' else cfg['preparation_process_seconds']
            outcome = helper.run_child(command, env, output / (name + '.log'), timeout)
            outcomes.append(dict(name=name, **outcome))
            write(output / (name + '_PROCESS.json'), outcomes[-1])
            if outcome['exit_code'] != 0 or outcome['timed_out']:
                raise RuntimeError(name + ' failed or exceeded its fixed time budget; no retry')
        verify_sources(root, admission)
        unchanged_parents(helper, parent_pid, parent_ticks, terminal)
        if (root / 'FAILED.json').exists():
            raise ValueError('Preparation wrote a failure receipt')
        summary = read(root / 'SUMMARY.json')
        prepared_contract(summary, admission['input_binding_sha256'])
        write(output / 'DONE.json', dict(status='complete', preparation_contract_passed=True,
            scientific_transfer_passed=None, training_performed=False, goal_complete=False,
            summary_sha256=sha(root / 'SUMMARY.json'), codec_state_sha256_before_and_after=summary['codec_state_sha256_before_and_after'],
            admission=admission, terminal=terminal, outcomes=outcomes))
        write(output / 'STATUS.json', dict(status='prepared', scientific_transfer_passed=None, training_performed=False))
    except BaseException as error:
        signal.alarm(0)
        write(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
            outcomes=outcomes, automatic_retry=False, old_processes_signalled=False, scientific_transfer_passed=None))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--parent-start-ticks', required=True)
    args = parser.parse_args()
    main(args.root, args.parent_pid, args.parent_start_ticks)
