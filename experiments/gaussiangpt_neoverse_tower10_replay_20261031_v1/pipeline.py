"""One bounded TRAIN-source preparation queue; never train, resume or retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
RUN = 'gaussiangpt_neoverse_tower10_replay_20261031_v1'
CLIP = 'buildingblock_tower_10_176'
PARENT = dict(root=str(BASE / 'gaussiangpt_neoverse_local_condition_20261031_v8'),
              pid=436766, start_ticks='38503415',
              manifest_sha256='169764ae89e7bbc589b27b749beb2de7e1105fd9036d122d20eaeae2be8ad7cd')
HELPER = '../gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
PREPARE_HELPER = '../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py'
PREPARE_HELPER_SHA = '3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85'
PREVIEW = BASE / 'neoverse_gs_preview_20261028_v1'
RUNTIME = PREVIEW / 'runtime/bin/python'
REPO = PREVIEW / 'NeoVerse_source'
CHECKPOINT = PREVIEW / 'weights_cache/models/Yuppie1204--NeoVerse/snapshots/master/reconstructor.ckpt'
CHECKPOINT_SHA = 'fd87ca703cf026c19e345e08f68292ae5a43e34d77ff20c7b49b78907e10f5c9'
RENDERER = BASE / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py'
RENDERER_SHA = '7f35ca4563c31db1f70117f4eb9575dfea5f5f175e180242bf1a441884080cd9'
CODEC = BASE / 'gaussiangpt_neoverse_state_motion_20261030_v1/state'
GS_PYTHON = PurePosixPath('/data/CJH/envs/gaussiangpt/bin/python')
VIDEO = BASE / 'buildingblock_vw2_h4_20261011_v1/datasets/Video-CraftBench/Paper_and_Block_clips' / (CLIP + '.mp4')
VIDEO_SHA = 'a8d76842e1d7de31cb8bedfe437264a28b3203c9f3bfbaebf2a6f0864a051dab'
REQUIRED = {'pipeline.py', 'prepare.py', 'test_pipeline.py', 'test_prepare.py', 'validate.py',
            'protocol.json', 'input_binding.json', 'EXECUTION.md', HELPER, PREPARE_HELPER}


def read(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def protocol_contract(cfg):
    fixed = dict(schema='gs_tower10_full_preparation_v1', run=RUN, clip=CLIP,
        source_group='buildingblock_tower_10', source_role='TRAIN', official_independent_test_split=False,
        frames=177, height=252, width=448, stride=1, fps=29.97, source_video_sha256=VIDEO_SHA,
        prepare_helper_relative=PREPARE_HELPER, prepare_helper_sha256=PREPARE_HELPER_SHA,
        codec_root=str(CODEC), parent=PARENT, wait_seconds=10800, wait_interval_seconds=60,
        teacher_process_seconds=1200, preparation_seconds=1200, preparation_process_seconds=1230,
        outer_seconds=13400, gpu=0, minimum_free_gib=60, cpu_threads=6, automatic_retry=False,
        training_performed=False, scientific_transfer_passed=None, goal_complete=False)
    if any(key not in cfg or type(cfg[key]) is not type(value) or cfg[key] != value
           for key, value in fixed.items()):
        raise ValueError('Fixed preparation source, parent identity or resource limits changed')


def binding_contract(binding, cfg):
    """Check path/hash consistency without opening any remote source on CPU."""
    if binding.get('schema') != 'gs_frozen_codec_transfer_binding_v1' or not binding.get('files'):
        raise ValueError('Invalid immutable input binding')
    rows = {}
    for row in binding['files']:
        path = row['path']
        if (not isinstance(path, str) or not PurePosixPath(path).is_absolute() or path in rows
                or type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Invalid, duplicate or unbound input: ' + str(path))
        rows[path] = row
    teacher = binding.get('expected_teacher', {})
    fixed = dict(clip=CLIP, frames=177, height=252, width=448, frame_stride=1, use_motion=False,
                 source_video_path=str(VIDEO), source_video_sha256=cfg['source_video_sha256'],
                 renderer_path=str(RENDERER), renderer_sha256=RENDERER_SHA, checkpoint_sha256=CHECKPOINT_SHA)
    if any(key not in teacher or type(teacher[key]) is not type(value) or teacher[key] != value
           for key, value in fixed.items()):
        raise ValueError('Full177 teacher export parameters do not match bound inputs')
    required = {str(VIDEO): VIDEO_SHA, str(RENDERER): RENDERER_SHA, str(CHECKPOINT): CHECKPOINT_SHA}
    for name in ('codec_final.pt', 'selection.json', 'protocol.json', 'code/common.py', 'code/codec.py'):
        path = str(CODEC / name)
        required[path] = rows.get(path, {}).get('sha256')
    if any(digest is None or rows.get(path, {}).get('sha256') != digest for path, digest in required.items()):
        raise ValueError('Command source/checkpoint/codec input is not fully bound')
    provenance = binding.get('verification', {})
    if (provenance.get('source_role') != 'TRAIN' or provenance.get('source_group') != 'buildingblock_tower_10'
            or provenance.get('official_independent_test_split') is not False):
        raise ValueError('Source identity must remain original TRAIN group10')
    return rows


def verify(root, expected=None, cpu=True):
    """Local manifest/protocol witness only; cpu=False is safe on Windows."""
    root = Path(root).resolve()
    manifest = read(root / 'CODE_MANIFEST.json')
    required = REQUIRED | {p.relative_to(root).as_posix() for p in root.glob('*.py')}
    required |= {p.relative_to(root).as_posix() for p in root.glob('*.md')}
    if not required.issubset(manifest):
        raise ValueError('Manifest omits executed sources, protocol, binding, tests or documents')
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if (Path(relative).is_absolute() or PurePosixPath(relative).is_absolute()
                or (relative not in (HELPER, PREPARE_HELPER) and not path.is_relative_to(root))
                or not path.is_relative_to(root.parent) or sha(path) != digest):
            raise ValueError('Immutable source mismatch: ' + relative)
    if manifest[HELPER] != HELPER_SHA or manifest[PREPARE_HELPER] != PREPARE_HELPER_SHA:
        raise ValueError('Pinned immutable process/preparation helper changed')
    cfg = read(root / 'protocol.json')
    protocol_contract(cfg)
    binding_contract(read(root / 'input_binding.json'), cfg)
    admission = dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
                     protocol_sha256=sha(root / 'protocol.json'), input_binding_sha256=sha(root / 'input_binding.json'))
    if cpu:
        proof = read(root / 'CPU_VALIDATION.json')
        if (not proof.get('schema') or proof.get('status') != 'passed' or proof.get('gpu_executed') is not False
                or any(proof.get(key) != admission[key] for key in ('code_manifest_sha256', 'protocol_sha256'))):
            raise ValueError('Exact code/protocol has no passed CPU validation receipt')
        admission['cpu_validation_sha256'] = sha(root / 'CPU_VALIDATION.json')
    if expected is not None and expected != admission:
        raise ValueError('Immutable admission changed')
    return cfg, admission


def verify_sources(root, cfg):
    rows = binding_contract(read(root / 'input_binding.json'), cfg)
    for row in rows.values():
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Bound source changed: ' + str(path))


def command_parts(snapshot):
    if not isinstance(snapshot, dict):
        return None
    argv = snapshot.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) for value in argv):
        return None
    if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?', PurePosixPath(argv[0]).name):
        return None
    index = 1
    while index < len(argv) and argv[index] in ('-B', '-u'):
        index += 1
    if index == len(argv) or argv[index].startswith('-'):
        return None
    return argv[index], argv[index + 1:]


def parent_command(snapshot, parent):
    return command_parts(snapshot) == (str(PurePosixPath(parent['root']) / 'pipeline.py'), ['--root', parent['root']])


def classify_parent(snapshot, parent):
    if snapshot is None:
        return 'exited'
    if (snapshot.get('pid') != parent['pid'] or snapshot.get('stable', True) is not True
            or str(snapshot.get('start_ticks')) != parent['start_ticks'] or not parent_command(snapshot, parent)):
        raise RuntimeError('Parent PID/start ticks/command identity mismatch; no process signalled')
    return 'alive'


def worker_match(snapshot, parent):
    parsed = command_parts(snapshot)
    if parsed is None:
        return False
    script, arguments = parsed
    if script not in {str(PurePosixPath(parent['root']) / name) for name in ('worker.py', 'compare.py')}:
        return False
    if arguments.count('--root') != 1:
        return False
    index = arguments.index('--root')
    return index + 1 < len(arguments) and arguments[index + 1] == parent['root']


def stable_snapshot(helper, pid):
    first, second = helper.process_snapshot(pid), helper.process_snapshot(pid)
    if second is None:
        try:
            stat = (Path('/proc') / str(pid) / 'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        if stat[stat.rfind(')') + 2:].split()[0] == 'Z':
            return None
        raise RuntimeError('Live process identity unavailable; not assuming exit')
    if first is None or first != second:
        raise RuntimeError('Process identity changed during observation')
    return dict(second, stable=True)


def observe(helper, parent):
    if sha(Path(parent['root']) / 'CODE_MANIFEST.json') != parent['manifest_sha256']:
        raise RuntimeError('Admitted parent manifest changed')
    state = classify_parent(stable_snapshot(helper, parent['pid']), parent)
    workers = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        candidate = helper.process_snapshot(int(directory.name))
        if not (worker_match(candidate, parent) or parent_command(candidate, parent)):
            continue
        current = stable_snapshot(helper, int(directory.name))
        if current is None:
            continue
        if any(candidate[key] != current[key] for key in ('pid', 'start_ticks', 'argv')):
            raise RuntimeError('Same-root process identity raced during observation')
        if parent_command(current, parent):
            classify_parent(current, parent)  # A new same-root parent is forbidden, even with a different PID.
            state = 'alive'
        if worker_match(current, parent):
            workers.append(current)
    return dict(parent_state=state, workers=workers, all_exited=state == 'exited' and not workers)


def terminal_decision(done, failed):
    if (done is None) == (failed is None):
        raise ValueError('Require exactly one parent queue/DONE.json or FAILED.json; no missing-receipt inference')
    kind, receipt = ('DONE', done) if done is not None else ('FAILED', failed)
    if not isinstance(receipt, dict) or receipt.get('status') != ('complete' if kind == 'DONE' else 'failed'):
        raise ValueError('Invalid parent terminal receipt')
    return dict(parent_terminal=kind, parent_failure_does_not_block_preparation=True)


def read_terminal(parent):
    directory = Path(parent['root']) / 'queue'
    paths = [directory / name for name in ('DONE.json', 'FAILED.json')]
    values = [read(path) if path.is_file() else None for path in paths]
    return dict(decision=terminal_decision(*values), hashes={str(path): sha(path) for path in paths if path.is_file()})


def wait_for_parent(output, cfg, helper):
    begun = time.monotonic()
    while True:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping')
        observation = observe(helper, cfg['parent'])
        write(output / 'STATUS.json', dict(status='waiting', checked_at=timestamp(),
              elapsed_seconds=time.monotonic() - begun, goal_complete=False, **observation))
        if observation['all_exited']:
            terminal = read_terminal(cfg['parent'])
            write(output / 'PARENT_TERMINAL.json', dict(observation=observation, **terminal))
            return terminal
        remaining = cfg['wait_seconds'] - (time.monotonic() - begun)
        if remaining <= 0:
            raise TimeoutError('10800-second parent wait expired; no old/shared process stopped')
        helper.STOP_REQUESTED.wait(min(cfg['wait_interval_seconds'], remaining))


def unchanged_parent(helper, cfg, terminal):
    if not observe(helper, cfg['parent'])['all_exited'] or read_terminal(cfg['parent']) != terminal:
        raise RuntimeError('Parent work resumed or terminal receipt changed')


def commands(root, binding):
    # source_video_path comes from the hash-verified binding, never a sparse existing GS export.
    teacher = binding['expected_teacher']
    destination = root / 'teacher' / CLIP
    return (
        ('teacher_export', [str(RUNTIME), '-B', '-u', teacher['renderer_path'], '--repo', str(REPO),
            '--checkpoint', str(CHECKPOINT), '--input', teacher['source_video_path'], '--output', str(destination),
            '--num-frames', '177', '--start-frame', '0', '--frame-stride', '1', '--fps', '29.97',
            '--height', '252', '--width', '448', '--save-gs', '177', '--novel-view-shift', '0.03',
            '--device', 'cuda:0', '--dtype', 'bfloat16']),
        ('frozen_codec_preparation', [str(GS_PYTHON), '-B', '-u', str(root / 'prepare.py'), '--root', str(root)]))


def stop_owned_children(helper):
    try:
        helper.terminate_queue(signal.SIGTERM, None)
    except SystemExit:
        pass


def execute_jobs(root, output, cfg, binding, helper, outcomes, before_launch):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6',
               MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for name, command in commands(root, binding):
        before_launch()
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping before child launch')
        if name == 'teacher_export' and (root / 'teacher').exists():
            raise FileExistsError('Teacher output already exists; no overwrite or retry')
        write(output / 'STATUS.json', dict(status=name, pid=os.getpid(), training_performed=False, goal_complete=False))
        timeout = cfg['teacher_process_seconds'] if name == 'teacher_export' else cfg['preparation_process_seconds']
        result = helper.run_child(command, env, output / (name + '.log'), timeout)
        outcomes.append(dict(name=name, **result))
        write(output / (name + '_PROCESS.json'), outcomes[-1])
        if result['exit_code'] != 0 or result['timed_out']:
            stop_owned_children(helper)
            raise RuntimeError(name + ' failed or timed out; no automatic retry')


def prepared_contract(summary, binding_hash):
    fixed = dict(status='complete', preparation_contract_passed=True, frames=177, adjacent_pairs=176,
        clip=CLIP, source_points_per_frame=112896, all_input_points_aggregated=True, codec_frozen=True,
        training_performed=False, scientific_transfer_passed=None, goal_complete=False,
        input_binding_sha256=binding_hash, source_role='TRAIN', source_group='buildingblock_tower_10',
        official_independent_test_split=False)
    if (any(key not in summary or type(summary[key]) is not type(value) or summary[key] != value
            for key, value in fixed.items())
            or not re.fullmatch('[0-9a-f]{64}', summary.get('codec_state_sha256_before_and_after', ''))):
        raise ValueError('No complete177 TRAIN/frozen-model/no-training preparation contract')


def gpu_admission(output):
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15, check=True)
    fields = result.stdout.strip().splitlines()
    if len(fields) != 1 or not re.fullmatch(r'\s*\d+\s*', fields[0]):
        raise ValueError('Unexpected one-shot GPU0 free-memory response')
    free_mib = int(fields[0])
    write(output / 'GPU_ADMISSION.json', dict(gpu=0, free_mib=free_mib, required_mib=60 * 1024, queries=1))
    if free_mib < 60 * 1024:
        raise RuntimeError('GPU0 has less than60 GiB free; no waiting or retry')


def main(root):
    root = root.resolve()
    if str(root) != str(BASE / RUN):
        raise ValueError('Require dedicated remote preparation root')
    output = root / 'queue'
    output.mkdir(exist_ok=False)
    outcomes, helper = [], None

    def startup_stop(signum, frame):
        raise TimeoutError('Preparation queue stopped during admission or exceeded13400 seconds')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, startup_stop)
        signal.alarm(13400)  # Includes admission; never reset after parent wait.
        cfg, admission = verify(root)
        spec = importlib.util.spec_from_file_location('_tower10_owned_groups', root / HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        write(output / 'RECEIPT.json', dict(status='waiting', pid=os.getpid(), started_at=timestamp(),
            admission=admission, parent=cfg['parent'], only_own_new_session_groups_may_be_signalled=True,
            automatic_retry=False, training_performed=False, goal_complete=False))
        terminal = wait_for_parent(output, cfg, helper)
        verify(root, admission)
        verify_sources(root, cfg)  # All bound hashes checked before any GPU child/import.
        unchanged_parent(helper, cfg, terminal)
        gpu_admission(output)

        def before_launch():
            verify(root, admission)
            unchanged_parent(helper, cfg, terminal)

        execute_jobs(root, output, cfg, read(root / 'input_binding.json'), helper, outcomes, before_launch)
        before_launch()
        verify_sources(root, cfg)
        if (root / 'FAILED.json').exists():
            raise ValueError('Preparation wrote a failure receipt')
        summary = read(root / 'SUMMARY.json')
        prepared_contract(summary, admission['input_binding_sha256'])
        write(output / 'DONE.json', dict(status='complete', preparation_contract_passed=True,
            training_performed=False, scientific_transfer_passed=None, goal_complete=False, source_role='TRAIN',
            summary_sha256=sha(root / 'SUMMARY.json'), finished_at=timestamp(), admission=admission,
            codec_state_sha256_before_and_after=summary['codec_state_sha256_before_and_after'], terminal=terminal, outcomes=outcomes))
        write(output / 'STATUS.json', dict(status='prepared', training_performed=False,
            scientific_transfer_passed=None, goal_complete=False))
    except BaseException as error:
        if helper is not None:
            stop_owned_children(helper)
        write(output / 'STATUS.json', dict(status='failed', error=repr(error), goal_complete=False))
        write(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
            finished_at=timestamp(), outcomes=outcomes, automatic_retry=False, old_processes_signalled=False,
            training_performed=False, scientific_transfer_passed=None, goal_complete=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
