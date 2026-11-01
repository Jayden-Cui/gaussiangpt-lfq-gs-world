"""Event-driven CPU-only grouping after B's full evaluations and native replay."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
RUN = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
GROUPS = OUTPUTS/'gaussiangpt_neoverse_motion_groups_20261102_v1'
EXPECTED_GROUPS_SHA = 'b283b40fbda8a11095a769b82692da7c8ce5aaa6733fcb90a0b35895d8a9db57'
MASK = 0x00000008 | 0x00000080  # close-write or atomic move into directory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value, replace=False):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(content, encoding='utf8')
        tmp.replace(path)
    else:
        with path.open('x', encoding='utf8') as f:
            f.write(content)


def event_names(data):
    offset = 0
    result = []
    while offset < len(data):
        if offset+16 > len(data):
            raise ValueError('Truncated inotify event')
        _, _, _, size = struct.unpack_from('iIII', data, offset)
        offset += 16
        if offset+size > len(data):
            raise ValueError('Truncated inotify name')
        name = data[offset:offset+size].split(b'\0', 1)[0]
        offset += size
        if name:
            result.append(os.fsdecode(name))
    return result


def open_watch(path):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        n = ctypes.get_errno()
        raise OSError(n, os.strerror(n))
    if libc.inotify_add_watch(fd, os.fsencode(path), MASK) < 0:
        n = ctypes.get_errno()
        os.close(fd)
        raise OSError(n, os.strerror(n))
    return fd


def validate_summary(summary, run):
    run = Path(run)
    if (summary.get('status') != 'complete' or summary.get('stage') != 'B'
            or summary.get('action_codec_complete') is not False
            or summary.get('manual_dynamic_review_required') is not True):
        raise ValueError('B summary is incomplete or misrepresents review')
    rows = summary['results']
    if len(rows) != 3 or {r['sweep'] for r in rows} != {5,10,20}:
        raise ValueError('Expected exactly three B checkpoints')
    if sha(run/'training/TRAIN_DONE.json') != summary['training_sha256']:
        raise ValueError('B training receipt changed')
    if sha(run/'REVIEW_ASSETS/CONTACTS.json') != summary['visual_review_assets_sha256']:
        raise ValueError('B visual review assets changed')
    paths = {}
    for row in rows:
        sweep = row['sweep']
        folder = run/'training'/f'sweep_{sweep:02d}'/'free_z'/'evaluation'
        expected = folder/'EVALUATION.json'
        if Path(row['evaluation_path']).resolve() != expected.resolve() or sha(expected) != row['evaluation_sha256']:
            raise ValueError('B evaluation path or hash changed')
        evaluation = read(expected)
        replay = read(folder/'REPLAY.json')
        if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
                or evaluation.get('source', {}).get('sweep') != sweep
                or row['replay']['sha256'] != sha(folder/'REPLAY.json')
                or replay.get('status') != 'passed' or replay.get('packets') != 104
                or replay.get('views') != 208 or replay.get('encoder_or_codec_loaded') is not False
                or replay.get('packets_sha256') != evaluation.get('packets_sha256')
                or sha(folder/'PACKETS.json') != evaluation.get('packets_sha256')):
            raise ValueError('B full evaluation or independent GS replay did not pass')
        paths[sweep] = expected
    return paths


def verify_sources():
    for root in (ROOT, GROUPS):
        witness = read(root/'CPU_TESTS.json')
        if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(root/'CODE.sha256'):
            raise ValueError('Missing source CPU witness')
        for line in (root/'CODE.sha256').read_text(encoding='utf8').splitlines():
            digest, name = line.split('  ',1)
            if Path(name).name != name or sha(root/name) != digest:
                raise ValueError('Source changed: '+name)
    if sha(GROUPS/'MOTION_GROUPS.json') != EXPECTED_GROUPS_SHA:
        raise ValueError('Frozen grouping changed')


def main():
    fd = None
    phase = 'initializing'
    try:
        verify_sources()
        fd = open_watch(RUN)
        poll = select.poll()
        poll.register(fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        write(ROOT/'WATCH_STARTED.json', {'pid': os.getpid(), 'method': 'inotify close-write/move',
            'run': str(RUN), 'max_wait_seconds': 86400, 'gpu_used': False, 'training_signalled': False})
        write(ROOT/'WATCH_STATUS.json', {'status': 'running', 'phase': 'waiting_for_complete_B',
            'gpu_used': False, 'training_signalled': False})
        phase = 'waiting_for_complete_B'
        deadline = time.monotonic()+86400
        # Register before checking existing files, so no completion event is lost.
        if (RUN/'FAILED.json').is_file():
            raise RuntimeError('B coordinator reported failure; no grouped report produced')
        summary_ready = False
        if (RUN/'SUMMARY.json').is_file():
            try:
                read(RUN/'SUMMARY.json')
                summary_ready = True
            except json.JSONDecodeError:
                # The summary was being written when monitoring began; wait for close-write.
                pass
        while not summary_ready:
            remaining = int((deadline-time.monotonic())*1000)
            if remaining <= 0:
                raise TimeoutError('B completion event exceeded 24 hours')
            events = poll.poll(remaining)
            if not events:
                raise TimeoutError('B completion event exceeded 24 hours')
            for _, flags in events:
                if flags & (select.POLLERR | select.POLLHUP):
                    raise RuntimeError('B event watcher failed')
                try:
                    names = event_names(os.read(fd, 65536))
                    if 'FAILED.json' in names:
                        raise RuntimeError('B coordinator reported failure; no grouped report produced')
                    if 'SUMMARY.json' in names:
                        summary_ready = True
                except BlockingIOError:
                    pass
        phase = 'validating_B_and_replay'
        summary = read(RUN/'SUMMARY.json')
        evaluations = validate_summary(summary, RUN)
        write(ROOT/'WATCH_STATUS.json', {'status': 'running', 'phase': 'grouping_completed_evaluations',
            'gpu_used': False, 'training_signalled': False}, replace=True)
        phase = 'grouping_completed_evaluations'
        sys.path.insert(0, str(GROUPS))
        import motion_groups
        rows = []
        for sweep in (5,10,20):
            output = ROOT/f'STAGE_B_SWEEP{sweep:02d}.json'
            motion_groups.evaluate(GROUPS/'MOTION_GROUPS.json', evaluations[sweep], output)
            rows.append({'sweep': sweep, 'path': str(output), 'sha256': sha(output)})
        write(ROOT/'WATCH_RESULT.json', {'status': 'complete', 'B_summary_sha256': sha(RUN/'SUMMARY.json'),
            'groups_sha256': EXPECTED_GROUPS_SHA, 'grouped_reports': rows, 'gpu_used': False,
            'training_signalled': False, 'manual_dynamic_review_required': True,
            'action_codec_complete': False})
        write(ROOT/'WATCH_STATUS.json', {'status': 'complete', 'phase': 'awaiting_dynamic_review',
            'gpu_used': False, 'training_signalled': False}, replace=True)
    except BaseException as error:
        result = {'status': 'failed', 'phase': phase, 'error': repr(error),
                  'traceback': traceback.format_exc(), 'gpu_used': False,
                  'training_signalled': False, 'no_automatic_retry': True}
        write(ROOT/'WATCH_FAILED.json', result)
        write(ROOT/'WATCH_STATUS.json', result, replace=True)
        raise
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
