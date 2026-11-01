"""Wait for the existing B10 preview to exit, then launch the diagnostic once."""
import os
from pathlib import Path
import select
import subprocess
import sys
import traceback

from probe import ROOT, PREVIEW, read, verify_self, write


def is_preview_watcher(pid, command):
    args = [os.fsdecode(x) for x in command.split(b'\0') if x]
    candidates = [Path(x) for x in args if Path(x).name == 'watcher.py']
    if len(candidates) != 1:
        return False
    path = candidates[0]
    if not path.is_absolute():
        try:
            path = (Path('/proc') / str(pid) / 'cwd').resolve(strict=True) / path
        except FileNotFoundError:
            return False
    return path.resolve() == (PREVIEW / 'watcher.py').resolve()


def main():
    phase = 'verifying_self'
    fd = None
    try:
        sha = verify_self()
        protocol = read(ROOT / 'protocol.json')
        phase = 'binding_preview_watcher'
        receipt = PREVIEW / 'WATCH_STARTED.json'
        if sha(receipt) != protocol['preview_watcher_started_sha256']:
            raise RuntimeError('Preview watcher receipt changed')
        pid = read(receipt)['pid']
        command_path = Path('/proc') / str(pid) / 'cmdline'
        try:
            command = command_path.read_bytes()
        except FileNotFoundError:
            command = b''
        if command:
            if not is_preview_watcher(pid, command):
                raise RuntimeError('Preview watcher PID was reused')
            sys.path.insert(0, str(PREVIEW))
            from process_handle import pidfd_open
            try:
                fd = pidfd_open(pid)
            except ProcessLookupError:
                fd = None
            if fd is not None:
                try:
                    after = command_path.read_bytes()
                except FileNotFoundError:
                    after = b''
                if after and not is_preview_watcher(pid, after):
                    raise RuntimeError('Preview watcher identity changed while opening pidfd')
                if not after:
                    check = select.poll()
                    check.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
                    if not check.poll(0):
                        raise RuntimeError('Preview watcher identity could not be verified')
        write(ROOT / 'QUEUE_STARTED.json', {'status': 'waiting' if fd is not None else 'preview_already_exited',
              'pid': os.getpid(), 'preview_watcher_pid': pid, 'method': 'pidfd process-exit event',
              'preview_signalled': False, 'main_training_signalled': False})
        if fd is not None:
            phase = 'waiting_for_preview'
            poll = select.poll()
            poll.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
            if not poll.poll(protocol['maximum_wait_seconds'] * 1000):
                raise TimeoutError('Preview did not exit in bounded wait')
        if read(PREVIEW / 'WATCH_STATUS.json').get('status') != 'complete':
            raise RuntimeError('Preview did not complete; do not start on GPU 1')
        write(ROOT / 'QUEUE_RELEASED.json', {'status': 'released', 'preview_watcher_pid': pid,
              'preview_signalled': False, 'main_training_signalled': False})
        phase = 'running_static_code_probe'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(protocol['cuda_visible_device']), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        with (ROOT / 'PROBE.log').open('x', encoding='utf8') as log:
            run = subprocess.run([sys.executable, '-B', str(ROOT / 'probe.py')], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=protocol['maximum_gpu_seconds'])
        output = ROOT / 'result/SUMMARY.json'
        if run.returncode or read(output).get('status') != 'complete':
            raise RuntimeError('Static code probe did not complete')
        write(ROOT / 'QUEUE_DONE.json', {'status': 'complete', 'result_sha256': sha(output),
              'preview_signalled': False, 'main_training_signalled': False, 'visual_review': 'pending'})
    except BaseException as error:
        write(ROOT / 'QUEUE_FAILED.json', {'status': 'failed', 'phase': phase, 'error': repr(error),
              'traceback': traceback.format_exc(), 'preview_signalled': False,
              'main_training_signalled': False, 'no_automatic_retry': True})
        raise
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
