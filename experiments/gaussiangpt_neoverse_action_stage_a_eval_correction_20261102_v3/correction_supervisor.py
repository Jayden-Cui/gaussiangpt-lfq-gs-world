"""Wait for the existing coordinator to exit, then correct its saved controls."""
import os
from pathlib import Path
import select
import time
import traceback

import correction
from process_handle import pidfd_open


def main():
    began=time.monotonic();phase='initializing';fd=None
    try:
        c,training,baseline,old,packets,replay=correction.setup()
        witness=correction.read(correction.ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_sha256']!=correction.sha(correction.ROOT/'CODE.sha256'):
            raise RuntimeError('Correction CPU witness changed')
        integration=correction.read(correction.ROOT/'INTEGRATION_TEST.json')
        if integration['status']!='passed' or integration['code_sha256']!=correction.sha(correction.ROOT/'CODE.sha256'):
            raise RuntimeError('Correction integration witness changed')
        main_test=correction.read(correction.ROOT/'MAIN_TEST.json')
        if main_test['status']!='passed' or main_test['code_sha256']!=correction.sha(correction.ROOT/'CODE.sha256'):
            raise RuntimeError('Correction multi-checkpoint summary test changed')
        pid=correction.read(training/'STARTED.json')['pid']
        proc=Path('/proc')/str(pid)
        if proc.exists():
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'driver.py coordinate' not in cmd:raise RuntimeError('Coordinator PID was reused or changed')
            fd=pidfd_open(pid)
        else:
            state=correction.read(training/'STATUS.json')
            if state['status']!='complete':raise RuntimeError('Original coordinator exited without success')
        correction.write(correction.ROOT/'STARTED.json',dict(pid=os.getpid(),original_coordinator_pid=pid,
             method='read-only pidfd exit event; no signal or process mutation',
             code_sha256=correction.sha(correction.ROOT/'CODE.sha256')))
        phase='waiting_for_original_coordinator'
        correction.write(correction.ROOT/'STATUS.json',dict(status='running',phase=phase,
             original_experiment_signalled=False))
        if fd is not None:
            poll=select.poll();poll.register(fd,select.POLLIN|select.POLLERR|select.POLLHUP)
            if not poll.poll(c['wait_max_seconds']*1000):raise RuntimeError('Original coordinator wait deadline')
        phase='correcting_completed_results'
        correction.write(correction.ROOT/'STATUS.json',dict(status='running',phase=phase,
             original_experiment_signalled=False),replace=True)
        correction.main()
        correction.write(correction.ROOT/'STATUS.json',dict(status='complete',phase='corrected',
             action_codec_complete=False,manual_visual_review_required=True,
             original_experiment_signalled=False,elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),
                     original_experiment_signalled=False,no_automatic_retry=True)
        correction.write(correction.ROOT/'FAILED.json',failure)
        correction.write(correction.ROOT/'STATUS.json',failure,replace=True)
        raise
    finally:
        if fd is not None:os.close(fd)


if __name__=='__main__':main()
