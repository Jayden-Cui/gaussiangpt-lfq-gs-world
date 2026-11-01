"""Open a Linux pidfd without sending signals or trusting a reused PID."""
import ctypes
import os
import platform


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('pidfd_open unavailable on this platform')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)
