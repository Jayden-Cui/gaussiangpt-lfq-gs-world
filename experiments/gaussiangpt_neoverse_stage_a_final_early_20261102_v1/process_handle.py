"""Bind Linux process identities even when Python was built without pidfd_open."""
import ctypes
import os
import platform
import signal


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):
        return os.pidfd_open(pid)
    # This experiment server is Linux x86_64. Do not guess syscall numbers on
    # a different platform or fall back to kill(pid), which risks PID reuse.
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('pidfd_open unavailable on this platform')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if result<0:
        number=ctypes.get_errno()
        raise OSError(number,os.strerror(number))
    return int(result)


def send_signal(fd,sig):
    if not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('pidfd_send_signal unavailable; refusing unsafe PID-only signalling')
    signal.pidfd_send_signal(fd,sig)
