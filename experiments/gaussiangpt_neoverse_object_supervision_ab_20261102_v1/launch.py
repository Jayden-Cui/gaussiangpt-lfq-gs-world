import os
import select
import subprocess
import sys
from trial_utils import ROOT,verify,require,write,sha


def main():
    verify(need_masks=True,need_tests=True)
    require(not (ROOT/'run').exists() and not (ROOT/'LAUNCHED.json').exists(),'Run already launched; no retry')
    read_fd,write_fd=os.pipe()
    with (ROOT/'coordinator.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-B',str(ROOT/'coordinator.py'),'--ready-fd',str(write_fd)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                             PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(write_fd,))
    os.close(write_fd)
    write(ROOT/'LAUNCHED.json',dict(pid=process.pid,code_sha256=sha(ROOT/'CODE.sha256'),protocol_sha256=sha(ROOT/'protocol.json'),
        process_start_ticks=open(f'/proc/{process.pid}/stat').read().split()[21],result_root=str(ROOT/'run')))
    ready=bool(select.select([read_fd],[],[],60)[0]);message=os.read(read_fd,128) if ready else b'';os.close(read_fd)
    print(dict(pid=process.pid,ready=message==b'ready\n',exit_code=process.poll(),result_root=str(ROOT/'run')))
    # A handshake timeout is only an observation failure; never relaunch automatically.


if __name__=='__main__':main()
