"""Conditional same-checkpoint-step evaluation; never starts or resumes training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def prefix_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def verify(root,pinned=None):
    for name in ('CODE_MANIFEST.json','POSTFIT_MANIFEST.json','EQUAL_STEP_MANIFEST.json'):
        for relative,expected in read(root/name).items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or sha(path)!=expected:
                raise ValueError('Immutable source mismatch: '+relative)
    cfg=read(root/'equal_step_code/PROTOCOL.json')
    if (sha(root/'CODE_MANIFEST.json')!=cfg['main_manifest_sha256'] or
            sha(root/'POSTFIT_MANIFEST.json')!=cfg['postfit_manifest_sha256']):
        raise ValueError('Different admitted main/postfit version')
    if pinned and any(sha(root/name)!=expected for name,expected in pinned.items()):
        raise ValueError('Queue-admitted source or CPU witness changed during waiting/execution')
    return cfg


def select_common_step(steps,available,planned=(50,100,150)):
    if set(steps)!=set(('frozen','partial','constant')) or set(available)!=set(steps):
        raise ValueError('All three arms required')
    if any(type(v) is not int or v<42 for v in steps.values()):
        raise ValueError('Insufficient complete-window training coverage')
    if len(set(steps.values()))==1:return None
    choices=[s for s in planned if s<=min(steps.values()) and all(s in available[a] for a in steps)]
    if not choices:raise ValueError('No common saved checkpoint after full-window coverage')
    return max(choices)


def wait_until_quiescent(root,output,cfg,shared):
    admitted={}
    for item in cfg['parents']:
        snapshot=shared.process_snapshot(item['pid'])
        if snapshot is not None and str(root/item['script']) not in snapshot['argv']:
            raise RuntimeError('Parent PID reused by another process')
        admitted[item['pid']]=snapshot
    write(output/'WAIT_IDENTITY.json',dict(parents=admitted,pid=os.getpid()))
    tracked=[str(root/name).encode() for name in (
        'code/native_experiment.py','analysis_code/future_information_probe.py','analysis_code/render_comparison.py')]
    deadline=time.monotonic()+cfg['wait_timeout_seconds']
    last=None
    known_groups=set()
    while True:
        alive=[]
        for item in cfg['parents']:
            current=shared.process_snapshot(item['pid']);original=admitted[item['pid']]
            if (current and str(root/item['script']) in current['argv'] and
                    (original is None or current['start_ticks']==original['start_ticks'])):
                alive.append(item['pid'])
        children=[]
        for arm in ('frozen','partial'):
            outcome=root/'diagnostics/postfit_queue_v1'/(arm+'_OUTCOME.json')
            if outcome.is_file():
                for job in read(outcome).get('jobs',[]):
                    if type(job.get('pid')) is int:known_groups.add(job['pid'])
        processes=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                argv=path.joinpath('cmdline').read_bytes().split(b'\0')
                stat=path.joinpath('stat').read_text();fields=stat[stat.rfind(')')+2:].split()
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if fields[0]=='Z':continue
            group=int(fields[2]);pid=int(path.name)
            native=(str(root).encode() in argv and any(script in argv for script in tracked))
            owned_media=any(arg.startswith(str(root/'diagnostics/render_comparison').encode()+b'/') for arg in argv)
            if native or owned_media:known_groups.add(group)
            processes.append((pid,group,native or owned_media))
        children=sorted(pid for pid,group,matched in processes if matched or group in known_groups)
        main_path=root/'PIPELINE_STATUS.json'
        post_path=root/'diagnostics/postfit_queue_v1/STATUS.json'
        main=read(main_path) if main_path.exists() else {}
        post=read(post_path) if post_path.exists() else {}
        if not alive and not children:
            if main.get('status') not in ('complete','failed') or post.get('status') not in (
                    'complete','completed_with_failures_or_skips','failed'):
                raise RuntimeError('Parent processes ended without explicit terminal evidence; no new GPU work')
            write(output/'PARENTS_TERMINAL.json',dict(main=main,postfit=post,main_status_sha256=sha(main_path),
                  postfit_status_sha256=sha(post_path),parents_exited=True,children=[]))
            return
        key=(tuple(alive),tuple(sorted(children)),main.get('status'),post.get('status'))
        if key!=last:
            write(output/'STATUS.json',dict(status='waiting_for_main_and_postfit',parent_pids=alive,
                  relevant_child_pids=sorted(children),pid=os.getpid()))
            last=key
        if time.monotonic()>=deadline:raise TimeoutError('Waiting budget exhausted; parent processes untouched')
        time.sleep(min(cfg['wait_interval_seconds'],max(.01,deadline-time.monotonic())))


def prepare_selection(root,output,cfg):
    from future_information_probe import load_completed_evidence
    evidence={arm:load_completed_evidence(root,arm) for arm in ('frozen','partial','constant')}
    steps={arm:item['train']['steps'] for arm,item in evidence.items()}
    available={arm:[s for s in cfg['checkpoint_steps'] if (root/arm/f'step_{s:04d}.pt').is_file()]
               for arm in evidence}
    selected=select_common_step(steps,available,tuple(cfg['checkpoint_steps']))
    if selected is None:
        original=read(root/'SUMMARY.json')
        if original.get('comparison_eligibility',{}).get('eligible') is not True:
            raise RuntimeError('Equal final steps but primary fairness invalid; checkpoint selection cannot repair it')
        write(output/'SKIPPED.json',dict(status='skipped_equal_final_steps',original_completed_steps=steps,
              primary_summary_sha256=sha(root/'SUMMARY.json'),reason='Primary comparison already uses equal final steps',
              gpu_evaluations=0,training_updates=0,goal_complete=False))
        return None
    arms={}
    for arm,item in evidence.items():
        plan=item['plan'];prefix=plan['starts'][:selected]
        if len(prefix)!=selected or {start for batch in prefix for start in batch}!=set(range(168)):
            raise ValueError('Selected checkpoint lacks actual full-window training coverage')
        checkpoint=root/arm/f'step_{selected:04d}.pt'
        arms[arm]=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                       completed_steps=item['train']['steps'],checkpoint_steps=selected,
                       initial_model_hash=plan['initial_model_hash'],plan_prefix_sha256=prefix_hash(prefix),
                       train_done_sha256=sha(root/arm/'TRAIN_DONE.json'),plan_sha256=sha(root/arm/'PLAN.json'))
    for field in ('initial_model_hash','plan_prefix_sha256'):
        if len({item[field] for item in arms.values()})!=1:raise ValueError('Three-arm source mismatch: '+field)
    selection=dict(schema='gs_native_equal_steps_selection_v1',status='selected',selected_step=selected,
                   selection_rule='Largest common existing periodic checkpoint; selected without quality metrics',
                   available_checkpoint_steps=available,original_completed_steps=steps,
                   main_manifest_sha256=sha(root/'CODE_MANIFEST.json'),protocol_sha256=sha(root/'protocol.json'),
                   postfit_manifest_sha256=sha(root/'POSTFIT_MANIFEST.json'),
                   equal_step_manifest_sha256=sha(root/'EQUAL_STEP_MANIFEST.json'),
                   initial_model_hash=arms['frozen']['initial_model_hash'],
                   plan_prefix_sha256=arms['frozen']['plan_prefix_sha256'],arms=arms,
                   training_updates=0,goal_complete=False)
    write(output/'SELECTION.json',selection)
    return selection


def worker(root,output,gpu,arms,selection,cfg,shared,pinned):
    outcomes=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
    for arm in arms:
        verify(root,pinned)
        command=[sys.executable,'-B',str(root/'equal_step_code/evaluate_checkpoint.py'),'--root',str(root),
                 '--arm',arm,'--step',str(selection['selected_step']),'--output',str(output/arm),
                 '--max-seconds',str(cfg['evaluation_seconds'])]
        result=shared.run_child(command,env,output/(arm+'_process.log'),cfg['process_timeout_seconds'])
        result.update(arm=arm,gpu=gpu);outcomes.append(result)
        write(output/(arm+'_PROCESS.json'),result)
        if result['exit_code']!=0 or result['timed_out']:break
    return outcomes


def main(root,output):
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root/'analysis_code'))
    import postfit_queue as shared
    try:
        cfg=verify(root)
        validation=read(root/'EQUAL_STEP_VALIDATION.json')
        if (validation.get('status')!='passed' or validation.get('gpu_executed') is not False
                or validation.get('equal_step_manifest_sha256')!=sha(root/'EQUAL_STEP_MANIFEST.json')
                or validation.get('main_manifest_sha256')!=sha(root/'CODE_MANIFEST.json')
                or validation.get('postfit_manifest_sha256')!=sha(root/'POSTFIT_MANIFEST.json')):
            raise RuntimeError('Exact equal-step code requires CPU document witness')
        pinned={name:sha(root/name) for name in ('CODE_MANIFEST.json','POSTFIT_MANIFEST.json',
                'EQUAL_STEP_MANIFEST.json','EQUAL_STEP_VALIDATION.json')}
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,shared.terminate_queue)
        signal.alarm(cfg['outer_timeout_seconds'])
        write(output/'QUEUE_RECEIPT.json',dict(pid=os.getpid(),status='submitted',protocol=cfg,
              equal_step_manifest_sha256=sha(root/'EQUAL_STEP_MANIFEST.json'),pinned_sha256=pinned,
              training_updates=0,goal_complete=False))
        wait_until_quiescent(root,output,cfg,shared)
        verify(root,pinned)
        selection=prepare_selection(root,output,cfg)
        if selection is None:
            write(output/'STATUS.json',dict(status='skipped_equal_final_steps',training_updates=0,goal_complete=False));return
        write(output/'STATUS.json',dict(status='evaluating_common_checkpoint',step=selection['selected_step'],pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(worker,root,output,int(gpu),arms,selection,cfg,shared,pinned) for gpu,arms in cfg['gpu_workers'].items()]
            outcomes=[value for job in jobs for value in job.result()]
        verify(root,pinned)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
        comparison=shared.run_child([sys.executable,'-B',str(root/'equal_step_code/compare_selected.py'),
            '--root',str(root),'--evaluation-root',str(output)],env,output/'comparison.log',180)
        success=len(outcomes)==3 and all(x['exit_code']==0 and not x['timed_out'] for x in outcomes) and comparison['exit_code']==0
        write(output/'QUEUE_DONE.json',dict(status='complete' if success else 'failed',outcomes=outcomes,
              comparison=comparison,training_updates=0,goal_complete=False))
        write(output/'STATUS.json',dict(status='complete' if success else 'failed',training_updates=0,goal_complete=False))
    except BaseException as error:
        write(output/'FAILED.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
              training_updates=0,goal_complete=False))
        write(output/'STATUS.json',dict(status='failed',error=repr(error),goal_complete=False));raise
    finally:signal.alarm(0)


def self_test():
    assert select_common_step(dict(frozen=134,partial=131,constant=150),
           dict(frozen=[50,100],partial=[50,100],constant=[50,100,150]))==100
    assert select_common_step(dict(frozen=80,partial=51,constant=150),
           dict(frozen=[50],partial=[50],constant=[50,100,150]))==50
    assert select_common_step(dict(frozen=150,partial=150,constant=150),
           dict(frozen=[50,100,150],partial=[50,100,150],constant=[50,100,150])) is None
    try:select_common_step(dict(frozen=60,partial=50,constant=70),dict(frozen=[50],partial=[],constant=[50]))
    except ValueError:pass
    else:raise AssertionError('Missing common checkpoint accepted')
    assert 'torch' not in sys.modules
    print('CPU_SELF_TEST_PASSED: common saved-step selection, equal-final skip, missing-checkpoint rejection; no GPU/files')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:
        if a.root is None or a.output is None:p.error('--root and --output required')
        root,output=a.root.resolve(),a.output.resolve()
        if output!=root/'diagnostics/equal_steps_v1':p.error('Only a new ROOT/diagnostics/equal_steps_v1 is allowed')
        if output.exists():p.error('Output already exists; no repeat, overwrite or resume')
        main(root,output)
