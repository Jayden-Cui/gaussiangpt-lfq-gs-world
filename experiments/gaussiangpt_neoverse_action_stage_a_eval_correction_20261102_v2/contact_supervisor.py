"""Wait for corrected A results, then prepare review images without deciding."""
import json
import os
from pathlib import Path
import select
import traceback

from process_handle import pidfd_open
from review_assets import create,sha

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def main():
    fd=None;phase='initializing'
    try:
        test=read(ROOT/'REVIEW_ASSETS_TEST.json')
        if (test['status']!='passed' or test['script_sha256']!=sha(ROOT/'review_assets.py')
                or test['test_sha256']!=sha(ROOT/'review_assets_tests.py')):
            raise RuntimeError('Visual contact-sheet witness changed')
        pid=read(ROOT/'STARTED.json')['pid'];proc=Path('/proc')/str(pid)
        if proc.exists():
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if 'correction_supervisor.py' not in cmd:raise RuntimeError('Correction PID changed')
            fd=pidfd_open(pid)
        else:
            if read(ROOT/'STATUS.json')['status']!='complete':raise RuntimeError('Correction did not complete')
        write(ROOT/'CONTACT_SUPERVISOR_STARTED.json',dict(pid=os.getpid(),correction_pid=pid,
            method='pidfd exit event; no signal or training mutation'))
        phase='waiting_for_corrected_results'
        write(ROOT/'CONTACT_SUPERVISOR_STATUS.json',dict(status='running',phase=phase))
        if fd is not None:
            poll=select.poll();poll.register(fd,select.POLLIN|select.POLLERR|select.POLLHUP)
            if not poll.poll(40000*1000):raise RuntimeError('Correction wait deadline')
        if read(ROOT/'STATUS.json')['status']!='complete':raise RuntimeError('Corrected evaluation failed')
        phase='preparing_review_images'
        write(ROOT/'CONTACT_SUPERVISOR_STATUS.json',dict(status='running',phase=phase),replace=True)
        result=create(ROOT/'sweep_20',ROOT/'REVIEW_ASSETS')
        write(ROOT/'CONTACT_SUPERVISOR_STATUS.json',dict(status='complete',phase='review_images_ready',
            visual_verdict=result['visual_verdict'],contacts_sha256=sha(ROOT/'REVIEW_ASSETS/CONTACTS.json')),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(ROOT/'CONTACT_SUPERVISOR_FAILED.json',failure)
        write(ROOT/'CONTACT_SUPERVISOR_STATUS.json',failure,replace=True)
        raise
    finally:
        if fd is not None:os.close(fd)


if __name__=='__main__':main()
