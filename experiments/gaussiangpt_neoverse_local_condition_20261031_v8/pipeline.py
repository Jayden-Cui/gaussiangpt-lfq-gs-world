"""One source-bound four-arm submission; no polling, retries or checkpoint picking."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

sys.dont_write_bytecode=True
HELPER='../gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
ARMS=('base_z','local_z','base_constant','local_constant')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def timestamp():return datetime.now(timezone.utc).isoformat()

def verify(root,expected=None,cpu=True,gpu=True):
    manifest=read(root/'CODE_MANIFEST.json')
    required={'protocol.json','pipeline.py','worker.py','local_skip.py','compare.py','validate.py','witness_gpu.py','EXECUTION.md',HELPER}
    if not required.issubset(manifest):raise ValueError('Manifest missing executed inputs')
    for rel,digest in manifest.items():
        p=(root/rel).resolve()
        if Path(rel).is_absolute() or not p.is_relative_to(root.parent) or sha(p)!=digest:
            raise ValueError('Bound source changed: '+rel)
    cfg=read(root/'protocol.json')
    arms={name:dict(source_arm='constant_delta' if name.endswith('constant') else 'z_delta',local_skip_enabled=name.startswith('local_')) for name in ARMS}
    if (cfg['schema']!='gs_local_condition_v8' or cfg['arms']!=arms or cfg['delta_weight']!=1.
        or cfg['train']!=dict(source_step=400,total_steps=600,additional_steps=200,save_steps=[500,600],max_seconds=1800)
        or cfg['queue']!=dict(gpu_workers={'0':['base_z','local_constant'],'1':['base_constant','local_z']},train_process_seconds=1900,eval_process_seconds=2500,comparison_seconds=180,outer_seconds=9300)
        or cfg['automatic_retry'] is not False or cfg['goal_complete'] is not False):
        raise ValueError('Fixed experiment matrix or budgets changed')
    binding=dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),protocol_sha256=sha(root/'protocol.json'))
    if cpu:
        proof=read(root/'CPU_VALIDATION.json')
        if proof.get('status')!='passed' or proof.get('gpu_executed') is not False or any(proof.get(k)!=v for k,v in binding.items()):
            raise ValueError('Exact code has no passed CPU document witness')
        binding['cpu_validation_sha256']=sha(root/'CPU_VALIDATION.json')
    if gpu:
        proof=read(root/'GPU_WITNESS.json');preflight=read(root/'PREFLIGHT.json')
        if (proof.get('status')!='passed' or proof.get('process',{}).get('exit_code')!=0
            or proof.get('process',{}).get('timed_out') is not False
            or proof.get('preflight_sha256')!=sha(root/'PREFLIGHT.json')
            or preflight.get('status')!='passed'
            or any(proof.get(k)!=v for k,v in binding.items())
            or any(preflight.get(k)!=binding[k] for k in ('code_manifest_sha256','protocol_sha256'))):
            raise ValueError('Exact sources have no successful discarded native B4 preflight')
        binding['gpu_witness_sha256']=sha(root/'GPU_WITNESS.json')
        binding['preflight_sha256']=sha(root/'PREFLIGHT.json')
    if expected is not None and expected!=binding:raise ValueError('Admission changed')
    return cfg,binding

def check_train_result(root,arm):
    d=read(root/'arms'/arm/'TRAIN_DONE.json')
    step=d.get('completed_steps',d.get('steps'))
    if d.get('status')!='complete' or step!=600:
        raise RuntimeError('All four arms must actually finish600: '+arm)

def stop_owned_children(helper):
    # Reuse the helper's shared stop flag and own-group cleanup. Its post-Popen
    # check also kills a child whose launch raced with this stop request.
    try:helper.terminate_queue(signal.SIGTERM,None)
    except SystemExit:pass

def main(root):
    root=root.resolve();q=root/'queue';q.mkdir(exist_ok=False);outcomes=[];helper=None
    def startup_stop(signum,frame):raise TimeoutError('Queue stopped during admission')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,startup_stop)
        signal.alarm(9300)
        cfg,admission=verify(root)
        helper=load('_local_condition_owned_groups',root/HELPER)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,helper.terminate_queue)
        write(q/'RECEIPT.json',dict(pid=os.getpid(),started_at=timestamp(),admission=admission,only_own_new_session_groups_may_be_signalled=True,goal_complete=False))
        def worker(gpu,arms,mode):
            completed=[]
            try:
                for arm in arms:
                    if helper.STOP_REQUESTED.is_set():break
                    verify(root,admission)
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
                    cmd=[sys.executable,'-B',str(root/'worker.py'),'--root',str(root),'--mode',mode,'--arm',arm]
                    timeout=cfg['queue']['train_process_seconds' if mode=='train' else 'eval_process_seconds']
                    result=helper.run_child(cmd,env,q/f'{arm}_{mode}.log',timeout)
                    failed=result['exit_code']!=0 or result['timed_out']
                    if failed:stop_owned_children(helper)
                    result.update(arm=arm,gpu=gpu,mode=mode);write(q/f'{arm}_{mode}_PROCESS.json',result);completed.append(result)
                    if failed:break
                    if mode=='train':check_train_result(root,arm)
            except BaseException:
                stop_owned_children(helper);raise
            return completed
        for mode in ('train','eval'):
            write(q/'STATUS.json',dict(status=mode,pid=os.getpid(),selected_step=600,goal_complete=False))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,gpu,arms,mode) for gpu,arms in cfg['queue']['gpu_workers'].items()]
                current=[x for future in futures for x in future.result()]
            outcomes.extend(current)
            if len(current)!=4 or any(x['exit_code']!=0 or x['timed_out'] for x in current):
                raise RuntimeError('A bounded '+mode+' arm failed; no automatic retry')
            verify(root,admission)
        result=helper.run_child([sys.executable,'-B',str(root/'compare.py'),'--root',str(root)],os.environ.copy(),q/'comparison.log',cfg['queue']['comparison_seconds'])
        write(q/'COMPARISON_PROCESS.json',result)
        if result['exit_code']!=0 or result['timed_out']:raise RuntimeError('Comparison failed')
        verify(root,admission);summary=read(root/'SUMMARY.json')
        if (summary.get('schema')!='gs_local_condition_comparison_v1' or summary.get('engineering_status')!='complete'
            or summary.get('comparison_eligible') is not True or summary.get('selected_step')!=600 or summary.get('goal_complete') is not False):
            raise RuntimeError('No complete same-step comparison')
        write(q/'DONE.json',dict(status='complete',selected_step=600,finished_at=timestamp(),summary_sha256=sha(root/'SUMMARY.json'),outcomes=outcomes,goal_complete=False))
        write(q/'STATUS.json',dict(status='complete',goal_complete=False))
    except BaseException as e:
        if helper is not None:stop_owned_children(helper)
        write(q/'STATUS.json',dict(status='failed',error=repr(e),goal_complete=False))
        write(q/'FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),outcomes=outcomes,automatic_retry=False,goal_complete=False));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
