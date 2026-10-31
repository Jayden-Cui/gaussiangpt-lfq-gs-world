"""One bounded full-pool codec queue; only this queue's child groups are owned."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import traceback
import worker

ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1'
HELPER = 'gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'

def now():
    return datetime.now(timezone.utc).isoformat()

def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)

def check_resources(text, disk_free_bytes, p):
    free = {int(a):int(b) for a,b in (line.split(',') for line in text.strip().splitlines())}
    worker.require(set(free)=={0,1},'Expected the two assigned GPUs')
    worker.require(min(free.values()) >= p['resources']['minimum_free_gib']*1024,
                   'Insufficient actual free GPU memory; no other task is stopped')
    worker.require(disk_free_bytes >= p['resources']['minimum_disk_free_gib']*2**30,
                   'Insufficient disk space; do not reduce coverage or delete old data')
    return dict(free_mib=free,disk_free_bytes=disk_free_bytes,at=now())

def successful(outcomes, expected):
    return (len(outcomes)==len(expected) and {r['name'] for r in outcomes}==set(expected)
            and all(r['exit_code']==0 and not r['timed_out'] for r in outcomes))

def main(root):
    root=Path(root).resolve();worker.require(str(root)==ROOT,'Dedicated remote root required')
    p,proof=worker.verify_root(root)
    queue=root/'queue';queue.mkdir(exist_ok=False)
    helper_path=root.parent/HELPER
    worker.require(worker.sha(helper_path)==HELPER_SHA,'Owned-process implementation changed')
    spec=importlib.util.spec_from_file_location('_fullpool_owned_children',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    records=[];lock=threading.Lock();deadline=time.monotonic()+p['queue']['outer_seconds']
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,helper.terminate_queue)
    signal.alarm(p['queue']['outer_seconds'])

    def run(mode,gpu=None,arm=None):
        worker.require(not helper.STOP_REQUESTED.is_set(),'Queue stopped')
        script = 'baseline_adapter.py' if mode=='baseline' else 'compare.py' if mode=='compare' else 'worker.py'
        argv=[p['queue']['runtime'],'-B','-u',str(root/script),'--root',str(root)]
        if mode in ('preflight','train','evaluate'):argv+=['--mode',mode]
        if arm is not None:argv+=['--arm',arm]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        name=mode+('' if arm is None else '_'+arm)
        seconds=min(p['queue'][mode+'_seconds'],deadline-time.monotonic())
        worker.require(seconds>0,'Queue budget expired')
        result=helper.run_child(argv,env,queue/(name+'.log'),seconds)
        record=dict(name=name,gpu=gpu,**result)
        with lock:
            records.append(record);write_new(queue/(name+'_PROCESS.json'),record)
        worker.require(result['exit_code']==0 and result['timed_out'] is False,name+' failed; no retry')
        _,current=worker.verify_root(root)
        worker.require(current==proof,'Frozen run changed')

    def stop_owned():
        try:helper.terminate_queue(signal.SIGTERM,None)
        except SystemExit:pass

    def parallel(jobs):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(job) for job in jobs]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                stop_owned();raise

    def chain(gpu,arm):
        run('train',gpu,arm);run('evaluate',gpu,arm)

    try:
        query=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'],text=True,timeout=15)
        resources=check_resources(query,shutil.disk_usage(root).free,p)
        write_new(queue/'RESOURCE_ADMISSION.json',resources)
        stat=Path('/proc',str(os.getpid()),'stat').read_text()
        ticks=stat[stat.rfind(')')+2:].split()[19]
        write_new(queue/'RECEIPT.json',dict(status='running',pid=os.getpid(),start_ticks=ticks,
                  started_at=now(),source=proof,process_helper_sha256=HELPER_SHA,
                  assignments=p['resources']['gpu_assignment'],goal_complete=False))
        parallel([lambda:run('preflight',0),lambda:run('baseline')])
        pre=worker.read(root/'PREFLIGHT.json');base=worker.read(root/'evaluations/baseline/EVALUATION.json')
        worker.require(pre['status']=='passed' and base['status']=='evaluated' and base['frames']==9027,
                       'GPU preflight and complete CPU baseline are both required')
        write_new(queue/'READY_FOR_TRAINING.json',dict(status='accepted',at=now(),source=proof,
                  preflight_sha256=worker.sha(root/'PREFLIGHT.json'),
                  baseline_sha256=worker.sha(root/'evaluations/baseline/EVALUATION.json')))
        print('FULLPOOL_CODEC_READY_FOR_TRAINING',flush=True)
        parallel([lambda:chain(0,'decoder_only'),lambda:chain(1,'full_codec')])
        run('compare')
        expected=['preflight','baseline','train_decoder_only','evaluate_decoder_only',
                  'train_full_codec','evaluate_full_codec','compare']
        worker.require(successful(records,expected),'Incomplete child ledger')
        write_new(queue/'DONE.json',dict(status='complete',finished_at=now(),source=proof,outcomes=records,
                  summary_sha256=worker.sha(root/'SUMMARY.json'),formal_updates=8852,
                  exposures_per_arm=17700,training_clips=50,source_groups=19,goal_complete=False,
                  action_Z_trained=False,automatic_LDM_codec_replacement=False,
                  visual_acceptance='pending_fixed_frame_review'))
    except BaseException as error:
        stop_owned()
        write_new(queue/'FAILED.json',dict(status='failed',at=now(),source=proof,outcomes=records,
                  error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
                  unrelated_processes_signalled=False,goal_complete=False))
        raise
    finally:
        signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
