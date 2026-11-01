"""Wait on the paired coordinator's process-exit event, then create assets once."""
import ctypes
import os
from pathlib import Path
import platform
import select
import traceback

from visual import ROOT, create, read, sha, verify_self, write


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Linux x86_64 pidfd is required')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(fd)


def is_source_coordinator(pid, command, source):
    args = [os.fsdecode(x) for x in command.split(b'\0') if x]
    candidates = [Path(x) for x in args if Path(x).name == 'coordinator.py']
    if len(candidates) != 1:
        return False
    path = candidates[0]
    if not path.is_absolute():
        try:
            path = (Path('/proc')/str(pid)/'cwd').resolve(strict=True)/path
        except FileNotFoundError:
            return False
    return path.resolve() == (source/'coordinator.py').resolve()


def main():
    fd = None
    phase = 'source_check'
    try:
        verify_self()
        protocol = read(ROOT/'protocol.json')
        source = Path(protocol['source_directory'])
        launch = source/'LAUNCHED.json'
        if sha(launch) != protocol['launch_sha256']:
            raise ValueError('Paired coordinator launch receipt changed')
        pid = read(launch)['pid']
        command_path = Path('/proc')/str(pid)/'cmdline'
        try:
            command = command_path.read_bytes()
        except FileNotFoundError:
            command = b''
        if command:
            if not is_source_coordinator(pid, command, source):
                raise RuntimeError('Paired coordinator PID was reused')
            try:
                fd = pidfd_open(pid)
            except ProcessLookupError:
                fd = None
            if fd is not None:
                try:
                    after = command_path.read_bytes()
                except FileNotFoundError:
                    after = b''
                if after and not is_source_coordinator(pid, after, source):
                    raise RuntimeError('Paired coordinator identity changed while acquiring pidfd')
                if not after:
                    check = select.poll()
                    check.register(fd, select.POLLIN | select.POLLERR | select.POLLHUP)
                    if not check.poll(0):
                        raise RuntimeError('Unable to verify coordinator identity')
        write(ROOT/'WAIT_STARTED.json', {'pid': os.getpid(), 'source_pid': pid,
            'status': 'waiting' if fd is not None else 'source_already_exited',
            'method': 'pidfd process-exit event', 'source_signalled': False, 'gpu_used': False})
        if fd is not None:
            phase = 'waiting_for_source_exit'
            poll = select.poll()
            poll.register(fd, select.POLLIN | select.POLLERR | select.POLLHUP)
            if not poll.poll(protocol['maximum_wait_seconds']*1000):
                raise TimeoutError('Paired experiment did not exit before deadline')
        phase = 'validating_and_creating_visuals'
        manifest = create(protocol)
        write(ROOT/'DONE.json', {'status': 'complete', 'manifest_sha256': sha(ROOT/'result/MANIFEST.json'),
            'frames': len(manifest['frames']), 'source_signalled': False, 'gpu_used': False,
            'visual_review': 'pending'})
    except BaseException as error:
        write(ROOT/'FAILED.json', {'status': 'failed', 'phase': phase, 'error': repr(error),
            'traceback': traceback.format_exc(), 'source_signalled': False, 'gpu_used': False,
            'no_automatic_retry': True})
        raise
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
