"""Wait for B's checkpoint-file and trainer-exit events, then preview 5 and 10."""
import ctypes
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import traceback

from process_handle import pidfd_open

ROOT = Path(__file__).resolve().parent
RUN = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_b_20261102_v1')
TRAINING = RUN / 'training'
MASK = 0x00000008 | 0x00000080  # IN_CLOSE_WRITE | IN_MOVED_TO


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def open_watch(path):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    if libc.inotify_add_watch(fd, os.fsencode(path), MASK) < 0:
        number = ctypes.get_errno()
        os.close(fd)
        raise OSError(number, os.strerror(number))
    return fd


def event_names(data):
    names = []
    offset = 0
    while offset < len(data):
        if offset + 16 > len(data):
            raise RuntimeError('Truncated inotify event')
        _, _, _, size = struct.unpack_from('iIII', data, offset)
        offset += 16
        if offset + size > len(data):
            raise RuntimeError('Truncated inotify filename')
        name = data[offset:offset + size].split(b'\0', 1)[0]
        if name:
            names.append(os.fsdecode(name))
        offset += size
    return names


def main():
    from preview import sha, verify_sources
    phase = 'initializing'
    watch_fd = process_fd = None
    try:
        verify_sources()
        test = read(ROOT / 'WATCH_TEST.json')
        if (test.get('status') != 'passed' or test.get('code_sha256') != sha(ROOT / 'CODE.sha256')
                or read(ROOT / 'warm_start/PREVIEW.json').get('status') != 'complete'):
            raise RuntimeError('Watcher test or real warm-start GPU preview is incomplete')
        process = read(RUN / 'train_PROCESS.json')
        pid = process['pid']
        proc = Path('/proc') / str(pid)
        if not proc.exists():
            raise RuntimeError('Trainer already exited before watcher start')
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'b_worker_v3.py' not in cmd or str(TRAINING) not in cmd:
            raise RuntimeError('Stage-B trainer PID identity differs')
        watch_fd = open_watch(TRAINING)
        process_fd = pidfd_open(pid)
        poll = select.poll()
        poll.register(watch_fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        poll.register(process_fd, select.POLLIN | select.POLLERR | select.POLLHUP)
        pending = [5, 10]
        write(ROOT / 'WATCH_STARTED.json', {'pid': os.getpid(), 'training_pid': pid,
              'method': 'inotify checkpoint-close plus pidfd exit events',
              'gpu': 1, 'training_signalled': False, 'code_sha256': sha(ROOT / 'CODE.sha256')})
        phase = 'waiting_for_checkpoint_events'
        write(ROOT / 'WATCH_STATUS.json', {'status': 'running', 'phase': phase, 'pending': pending,
              'training_signalled': False})
        deadline = time.monotonic() + 21600
        trainer_exited = False
        while pending:
            ready = [s for s in pending if (TRAINING / f'sweep_{s:02d}.json').is_file()]
            if ready:
                sweep = ready[0]
                phase = 'previewing_checkpoint'
                write(ROOT / 'WATCH_STATUS.json', {'status': 'running', 'phase': phase, 'sweep': sweep,
                      'pending': pending, 'training_signalled': False}, replace=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
                with (ROOT / f'preview_{sweep:02d}.log').open('x', encoding='utf-8') as log:
                    child = subprocess.Popen([sys.executable, '-B', str(ROOT / 'preview.py'), '--label', f'{sweep:02d}'],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        exit_code = child.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise RuntimeError('Preview exceeded its fixed deadline')
                if exit_code or read(ROOT / f'sweep_{sweep:02d}/PREVIEW.json').get('status') != 'complete':
                    raise RuntimeError('Preview failed for sweep ' + str(sweep))
                pending.remove(sweep)
                phase = 'waiting_for_checkpoint_events'
                write(ROOT / 'WATCH_STATUS.json', {'status': 'running', 'phase': phase, 'pending': pending,
                      'training_signalled': False}, replace=True)
                continue
            if trainer_exited:
                raise RuntimeError('Trainer exited before all preview checkpoints were available')
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise RuntimeError('Checkpoint event wait exceeded fixed deadline')
            events = poll.poll(remaining)
            if not events:
                raise RuntimeError('Checkpoint event wait exceeded fixed deadline')
            for fd, _ in events:
                if fd == process_fd:
                    trainer_exited = True
                elif fd == watch_fd:
                    try:
                        event_names(os.read(watch_fd, 65536))
                    except BlockingIOError:
                        pass
        write(ROOT / 'WATCH_STATUS.json', {'status': 'complete', 'phase': 'previews_ready',
              'training_signalled': False, 'visual_dynamic_review': 'pending'}, replace=True)
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'training_signalled': False, 'no_automatic_retry': True}
        write(ROOT / 'WATCH_FAILED.json', failure)
        write(ROOT / 'WATCH_STATUS.json', failure, replace=True)
        raise
    finally:
        for fd in (watch_fd, process_fd):
            if fd is not None:
                os.close(fd)


if __name__ == '__main__':
    main()
