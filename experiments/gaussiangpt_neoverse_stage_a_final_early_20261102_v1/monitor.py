"""Wait for existing free-code workers, then run the final CPU-only summary."""
import json
import os
from pathlib import Path
import select
import time
import traceback
import torch

from final_latent import ROOT,STAGE_A,analyze
from process_handle import pidfd_open


def sha(path):
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def completed(shard):
    path=STAGE_A/f'shard_{shard}/DONE.json'
    if not path.is_file():return False
    value=read(path)
    return (value['status']=='complete' and value['updates']==3780 and value['sweeps']==20
            and value['decoder_and_codec_frozen'])


def main():
    phase='initializing';fds=[];began=time.monotonic()
    try:
        witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256'):
            raise RuntimeError('Early CPU analysis witness changed')
        for line in (ROOT/'CODE.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            if Path(name).name!=name or sha(ROOT/name)!=digest:raise RuntimeError('Early source changed: '+name)
        poller=select.poll();waiting={};pids={}
        for shard in (0,1):
            pid=read(STAGE_A/f'shard_{shard}_PROCESS.json')['pid'];pids[str(shard)]=pid
            proc=Path('/proc')/str(pid)
            if not proc.exists():
                if not completed(shard):raise RuntimeError('Shard exited without complete result')
                continue
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if not cmd and completed(shard):continue
            if str(STAGE_A/'optimize.py') not in cmd or f'--shard {shard}' not in cmd:
                raise RuntimeError('Shard process identity changed')
            try:fd=pidfd_open(pid)
            except ProcessLookupError:
                if completed(shard):continue
                raise
            fds.append(fd);waiting[fd]=shard;poller.register(fd,select.POLLIN|select.POLLERR|select.POLLHUP)
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),shard_pids=pids,method='pidfd exit events; no process signals'))
        phase='waiting_for_training_completion'
        write(ROOT/'STATUS.json',dict(status='running',phase=phase,processes_signalled=False))
        deadline=time.monotonic()+21600
        while waiting:
            timeout=max(0,int((deadline-time.monotonic())*1000))
            if not timeout:raise RuntimeError('Training exit wait exceeded deadline')
            events=poller.poll(timeout)
            if not events:raise RuntimeError('Training exit wait exceeded deadline')
            for fd,event in events:
                if fd not in waiting:continue
                poller.unregister(fd);waiting.pop(fd)
        if not all(completed(shard) for shard in (0,1)):
            raise RuntimeError('At least one training shard failed or did not complete')
        phase='cpu_final_latent_analysis'
        write(ROOT/'STATUS.json',dict(status='running',phase=phase,processes_signalled=False),replace=True)
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        result=analyze(20);result['analysis_code_sha256']=sha(ROOT/'CODE.sha256')
        write(ROOT/'FINAL_LATENT.json',result)
        write(ROOT/'STATUS.json',dict(status='complete',phase='final_latent_ready',processes_signalled=False,
              visual_verdict='not evaluated',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(ROOT/'FAILED.json',failure);write(ROOT/'STATUS.json',failure,replace=True)
        raise
    finally:
        for fd in fds:os.close(fd)


if __name__=='__main__':main()
