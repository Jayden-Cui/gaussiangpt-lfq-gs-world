"""One bounded queue: assets, full GPU preflight, visual gate, four matched arms."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
from contract import read,sha,require,verify,verify_sources,write_new

ROOT='/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_gs_supervision_20261031_v10'
PYTHON='/data/CJH/envs/gaussiangpt/bin/python'
HELPER='gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA='ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
LIMITS={'teacher_assets':930,'preflight':930,'train':2430,'evaluate':2430,'summary':300,'visual_gate':1800,'outer':14400}

def now():return datetime.now(timezone.utc).isoformat()

def main(root):
    root=Path(root).resolve();require(str(root)==ROOT,'Use dedicated remote run only')
    p,proof=verify(root);verify_sources(p)
    require(not (root/'queue').exists(),'Preserve original queue; no automatic retry')
    queue=root/'queue';queue.mkdir();records=[];lock=threading.Lock();deadline=time.monotonic()+LIMITS['outer']
    helper_path=root.parent/HELPER;require(sha(helper_path)==HELPER_SHA,'Owned-process helper changed')
    spec=importlib.util.spec_from_file_location('_v10_owned_children',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,helper.terminate_queue)
    signal.alarm(LIMITS['outer'])
    def run(mode,gpu,arm=None):
        require(not helper.STOP_REQUESTED.is_set(),'Queue already stopped')
        script='teacher_assets.py' if mode=='teacher_assets' else ('summary.py' if mode=='summary' else 'worker.py')
        argv=[PYTHON,'-B','-u',str(root/script),'--root',str(root)]
        if mode not in ('teacher_assets','summary'):argv+=['--mode',mode]
        if arm is not None:argv+=['--arm',arm]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        name=mode+('' if arm is None else '_'+arm)
        result=helper.run_child(argv,env,queue/(name+'.log'),min(LIMITS[mode],deadline-time.monotonic()))
        record=dict(name=name,gpu=gpu,**result)
        with lock:records.append(record);write_new(queue/(name+'_PROCESS.json'),record)
        require(result['exit_code']==0 and not result['timed_out'],name+' failed; no retry')
        _,after=verify(root);require(after==proof,'Sources changed during queue')
    def stop_owned():
        try:helper.terminate_queue(signal.SIGTERM,None)
        except SystemExit:pass
    try:
        output=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'],text=True,timeout=15)
        free={int(a):int(b) for a,b in (line.split(',') for line in output.strip().splitlines())}
        write_new(queue/'GPU_ADMISSION.json',dict(free_mib=free,required_free_mib=20480,checked_at=now(),
                  unrelated_processes_signalled=False))
        require(set(free)=={0,1} and min(free.values())>=20480,'Need20GiB free on each assigned GPU')
        write_new(queue/'RECEIPT.json',dict(status='submitted',pid=os.getpid(),started_at=now(),**proof,
                    time_limits=LIMITS,assignments={'0':['rgb_z','gs_z'],'1':['rgb_constant','gs_constant']},goal_complete=False))
        run('teacher_assets',0);run('preflight',0)
        write_new(queue/'READY_FOR_VISUAL_REVIEW.json',dict(status='waiting_for_visual_review',**proof,
                  teacher_assets_sha256=sha(root/'TEACHER_ASSETS.json'),preflight_sha256=sha(root/'PREFLIGHT.json'),at=now()))
        print('READY_FOR_VISUAL_REVIEW',flush=True)
        gate_deadline=min(deadline,time.monotonic()+LIMITS['visual_gate'])
        while not (root/'TEACHER_VISUAL_ACCEPTANCE.json').exists():
            require(time.monotonic()<gate_deadline,'Bounded teacher visual review wait expired; no training started')
            time.sleep(min(5,max(.01,gate_deadline-time.monotonic())))
        gate=read(root/'TEACHER_VISUAL_ACCEPTANCE.json')
        require(gate.get('status')=='accepted' and gate.get('reviewed_frames')==[44,132]
                and gate.get('teacher_assets_sha256')==sha(root/'TEACHER_ASSETS.json')
                and gate.get('preflight_sha256')==sha(root/'PREFLIGHT.json'),'Teacher visual gate rejected/mismatched')
        def chain(gpu,arms):
            for arm in arms:
                run('train',gpu,arm);run('evaluate',gpu,arm)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(chain,0,('rgb_z','gs_z')),pool.submit(chain,1,('rgb_constant','gs_constant'))]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:stop_owned();raise
        run('summary',None);verify_sources(p)
        require(len(records)==11,'Incomplete planned child stages')
        write_new(queue/'DONE.json',dict(status='complete',**proof,summary_sha256=sha(root/'SUMMARY.json'),
            finished_at=now(),outcomes=records,formal_model_updates=800,discarded_preflight_updates=4,
            visual_acceptance='pending_final_model_review',goal_complete=False))
    except BaseException as error:
        stop_owned()
        write_new(queue/'FAILED.json',dict(status='failed',**proof,error=repr(error),traceback=traceback.format_exc(),
                  outcomes=records,at=now(),automatic_retry=False,unrelated_processes_signalled=False,goal_complete=False))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)

