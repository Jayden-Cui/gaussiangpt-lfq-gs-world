"""Use a pidfd exit event to wait for the existing GPU-1 pilot, then run once."""
import ctypes
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import traceback

from static_probe import ROOT,OUTPUTS,read,sha,verify_self,write


PILOT=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_replay_pilot_20261102_v1'


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Linux x86_64 pidfd support required')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if result<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return int(result)


def main():
    verify_self();protocol=read(ROOT/'protocol.json');fd=None;phase='checking_pilot'
    try:
        started=PILOT/'result/STARTED.json'
        if sha(started)!=protocol['queued_pilot_started_sha256']:
            raise ValueError('Pilot launch receipt changed')
        pid=protocol['queued_after_pilot_pid']
        if read(started)['pid']!=pid:raise ValueError('Pilot PID differs from launch receipt')
        cmdline=Path('/proc')/str(pid)/'cmdline'
        try:command=cmdline.read_bytes()
        except FileNotFoundError:command=b''
        if command:
            expected=b'gaussiangpt_neoverse_stage_b_anchor_replay_pilot_20261102_v1/pilot.py'
            if expected not in command:raise RuntimeError('Pilot PID was reused')
            try:fd=pidfd_open(pid)
            except ProcessLookupError:fd=None
            if fd is not None:
                try:after=cmdline.read_bytes()
                except FileNotFoundError:after=b''
                if after and expected not in after:raise RuntimeError('Pilot PID changed while acquiring pidfd')
                if not after:
                    finished=select.poll();finished.register(fd,select.POLLIN|select.POLLHUP|select.POLLERR)
                    if not finished.poll(0):raise RuntimeError('Pilot identity cannot be verified')
        write(ROOT/'QUEUE_STARTED.json',{'status':'waiting' if fd is not None else 'pilot_already_exited',
              'pid':os.getpid(),'pilot_pid':pid,'method':'pidfd process-exit event',
              'pilot_signalled':False,'main_training_signalled':False})
        if fd is not None:
            phase='waiting_for_pilot_exit'
            poll=select.poll();poll.register(fd,select.POLLIN|select.POLLERR|select.POLLHUP)
            if not poll.poll(protocol['maximum_wait_seconds']*1000):
                raise TimeoutError('Pilot process did not exit within the bounded wait')
        phase='running_same_camera_probe'
        write(ROOT/'QUEUE_RELEASED.json',{'status':'released','pilot_pid':pid,'pilot_signalled':False,
              'main_training_signalled':False})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(protocol['cuda_visible_device']),OMP_NUM_THREADS='4',
                 MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        with (ROOT/'PROBE.log').open('x',encoding='utf8') as log:
            run=subprocess.run([sys.executable,'-B',str(ROOT/'static_probe.py')],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,timeout=protocol['maximum_gpu_seconds'])
        if run.returncode!=0 or read(ROOT/'result/STATIC_CONTROL.json').get('status')!='complete':
            raise RuntimeError('Static-control probe did not complete')
        write(ROOT/'QUEUE_DONE.json',{'status':'complete','result_sha256':sha(ROOT/'result/STATIC_CONTROL.json'),
              'pilot_signalled':False,'main_training_signalled':False,'visual_review':'pending'})
    except BaseException as error:
        write(ROOT/'QUEUE_FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'pilot_signalled':False,'main_training_signalled':False,
              'no_automatic_retry':True})
        raise
    finally:
        if fd is not None:os.close(fd)


if __name__=='__main__':main()
