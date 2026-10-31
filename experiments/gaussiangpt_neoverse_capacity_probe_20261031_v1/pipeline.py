"""One bounded oracle submission, after the exact transfer queue exits."""
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
import time
import traceback

sys.dont_write_bytecode = True
HELPER_REL = '../gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
SNAPSHOT_REL = '../gaussiangpt_neoverse_tower14_transfer_20261031_v1/queue_prepare.py'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''): result.update(data)
    return result.hexdigest()

def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def now():
    return datetime.now(timezone.utc).isoformat()

def verify(root, expected=None, witness=True):
    manifest = read(root/'CODE_MANIFEST.json')
    if not {'protocol.json','probe.py','pipeline.py','test_probe.py','validate.py','EXECUTION.md',HELPER_REL,SNAPSHOT_REL}.issubset(manifest):
        raise ValueError('Manifest omits an executed or documented dependency')
    for rel, digest in manifest.items():
        p = (root/rel).resolve()
        if Path(rel).is_absolute() or not p.is_relative_to(root.parent) or sha(p) != digest:
            raise ValueError('Bound source changed: '+rel)
    cfg = read(root/'protocol.json')
    if (cfg['schema'] != 'gs_capacity_probe_protocol_v1' or cfg['starts'] != [35,123]
        or cfg['source_arm'] != 'z_delta' or cfg['selected_step'] != 400
        or cfg['automatic_retry'] is not False or cfg['goal_complete'] is not False
        or cfg['optimization'] != dict(steps=100,lr=.03,optimizer='Adam',checkpoints=[0,25,50,75,100])):
        raise ValueError('Fixed capacity diagnostic contract changed')
    q = cfg['queue']
    if (q['parent_pid'],q['parent_start_ticks'],q['wait_seconds'],q['wait_interval_seconds'],q['process_seconds'],q['outer_seconds'],q['gpu_workers']) != (431960,'38083128',6000,60,1550,7800,{'0':35,'1':123}):
        raise ValueError('Fixed queue identity/budget changed')
    bound = dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),protocol_sha256=sha(root/'protocol.json'))
    if witness:
        proof = read(root/'CPU_VALIDATION.json')
        if proof.get('status') != 'passed' or proof.get('gpu_executed') is not False or any(proof.get(k) != v for k,v in bound.items()):
            raise ValueError('No passed exact-source CPU document witness')
        bound['cpu_validation_sha256'] = sha(root/'CPU_VALIDATION.json')
    if expected is not None and bound != expected:
        raise ValueError('Admission changed')
    return cfg,bound

def expected_parent(cfg):
    p = cfg['queue']['parent_root']
    return p+'/pipeline.py',['--root',p]

def related(snapshot,cfg,parser):
    parsed = parser(snapshot)
    if parsed is None: return False
    script,args = parsed
    p = cfg['queue']['parent_root']
    if script not in (p+'/pipeline.py',p+'/evaluate_transfer.py',p+'/compare_transfer.py') or args.count('--root') != 1:
        return False
    i = args.index('--root')
    return i+1 < len(args) and args[i+1] == p

def classify_parent(snapshot,cfg,parser):
    if snapshot is None: return 'exited'
    q = cfg['queue']
    if snapshot.get('pid') != q['parent_pid'] or snapshot.get('stable',True) is not True:
        raise RuntimeError('Unstable parent identity')
    matches = parser(snapshot) == expected_parent(cfg)
    if str(snapshot.get('start_ticks')) != q['parent_start_ticks']:
        if matches or related(snapshot,cfg,parser): raise RuntimeError('Replacement transfer run; do not adopt')
        return 'pid_reused_unrelated'
    if not matches: raise RuntimeError('Exact parent argv changed')
    return 'alive'

def observe(cfg,helper,snapshots):
    pid = cfg['queue']['parent_pid']
    state = classify_parent(snapshots.stable_snapshot(helper,pid),cfg,snapshots.command_parts)
    workers=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==pid: continue
        preliminary = helper.process_snapshot(int(entry.name))
        if not related(preliminary,cfg,snapshots.command_parts): continue
        current = snapshots.stable_snapshot(helper,int(entry.name))
        if current is None: continue
        if snapshots.command_parts(current) == expected_parent(cfg):
            raise RuntimeError('Another transfer parent exists')
        if related(current,cfg,snapshots.command_parts):
            workers.append(dict(pid=current['pid'],start_ticks=current['start_ticks']))
    return dict(parent_identity=state,workers=workers,all_exited=state!='alive' and not workers)

def terminal(cfg):
    p = Path(cfg['queue']['parent_root'])/'queue'
    files=[x for x in (p/'DONE.json',p/'FAILED.json') if x.is_file()]
    if len(files)!=1: raise RuntimeError('Exactly one transfer terminal receipt is required')
    file=files[0];doc=read(file)
    if doc.get('status') != ('complete' if file.name=='DONE.json' else 'failed'):
        raise RuntimeError('Invalid transfer terminal receipt')
    return dict(path=str(file),sha256=sha(file),status=doc['status'],transfer_failure_does_not_prove_TRAIN_source_invalid=True)

def wait_dependency(root,cfg,helper,snapshots):
    begun=time.monotonic()
    deadline=begun+cfg['queue']['wait_seconds']
    while True:
        if helper.STOP_REQUESTED.is_set(): raise RuntimeError('Queue stopping')
        if time.monotonic()>=deadline: raise TimeoutError('Dependency wait expired; old processes untouched')
        observed=observe(cfg,helper,snapshots)
        write(root/'queue/STATUS.json',dict(status='waiting_transfer',pid=os.getpid(),checked_at=now(),**observed,goal_complete=False))
        if observed['all_exited']:
            receipt=terminal(cfg)
            if time.monotonic()>=deadline: raise TimeoutError('Dependency wait expired before admission')
            write(root/'queue/DEPENDENCY_TERMINAL.json',receipt)
            return receipt
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('Dependency wait expired; old processes untouched')
        helper.STOP_REQUESTED.wait(min(cfg['queue']['wait_interval_seconds'],remaining))

def main(root):
    root=root.resolve();out=root/'queue';out.mkdir(exist_ok=False)
    outcomes=[]
    def startup_stop(signum,frame):
        raise TimeoutError('Queue stopped during bounded admission')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM): signal.signal(sig,startup_stop)
        signal.alarm(7800)
        cfg,admission=verify(root)
        helper=load('_oracle_owned_groups',root/HELPER_REL)
        snapshots=load('_oracle_snapshot',root/SNAPSHOT_REL)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM): signal.signal(sig,helper.terminate_queue)
        write(out/'RECEIPT.json',dict(pid=os.getpid(),started_at=now(),admission=admission,parent_pid=cfg['queue']['parent_pid'],parent_start_ticks=cfg['queue']['parent_start_ticks'],goal_complete=False))
        dependency=wait_dependency(root,cfg,helper,snapshots)
        def unchanged():
            verify(root,admission)
            if not observe(cfg,helper,snapshots)['all_exited'] or terminal(cfg)!=dependency:
                raise RuntimeError('Dependency resumed or its receipt changed')
        def worker(gpu,start):
            unchanged()
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6')
            command=[sys.executable,'-B',str(root/'probe.py'),'--root',str(root),'--start',str(start)]
            result=helper.run_child(command,env,out/f'start_{start:04d}.log',cfg['queue']['process_seconds'])
            result.update(start=start,gpu=gpu)
            write(out/f'start_{start:04d}_PROCESS.json',result)
            return result
        write(out/'STATUS.json',dict(status='oracle_diagnostics',pid=os.getpid(),goal_complete=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(worker,gpu,start) for gpu,start in cfg['queue']['gpu_workers'].items()]
            outcomes=[j.result() for j in jobs]
        if any(x['exit_code']!=0 or x['timed_out'] for x in outcomes):
            raise RuntimeError('Oracle worker failed; no retry or budget reduction')
        unchanged()
        windows={}
        for start in cfg['starts']:
            p=root/'windows'/f'start_{start:04d}'/'SUMMARY.json';d=read(p)
            if (d.get('status')!='complete' or d.get('goal_complete') is not False
                or d.get('schema')!='gs_native_capacity_probe_window_v1'
                or d.get('selected_step')!=400 or d.get('source_arm')!=cfg['source_arm']
                or d.get('start')!=start or d.get('frames')!=list(range(start,start+10))
                or d.get('model_optimizer_updates')!=0 or d.get('parity_passed') is not True):
                raise RuntimeError('No complete diagnostic result')
            windows[str(start)]=dict(summary=str(p.relative_to(root)),sha256=sha(p),result=d)
        summary=dict(schema='gs_capacity_probe_comparison_v1',status='complete',selected_step=400,source_arm=cfg['source_arm'],windows=windows,admission=admission,dependency=dependency,model_training_updates=0,oracle_target_fitting=True,goal_complete=False,limitations=['Two deliberately selected TRAIN failures; no generalization or learned-Z claim','Target RGB enters oracle optimization; no deployable predictor was trained','Continuous64D action is a relaxation, not a legal LFQ action','Final step100 only; finite-budget failures do not prove impossibility'])
        write(root/'SUMMARY.json',summary)
        write(out/'DONE.json',dict(status='complete',finished_at=now(),summary_sha256=sha(root/'SUMMARY.json'),outcomes=outcomes,goal_complete=False))
        write(out/'STATUS.json',dict(status='complete',goal_complete=False))
    except BaseException as error:
        write(out/'FAILED.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),outcomes=outcomes,automatic_retry=False,goal_complete=False))
        raise
    finally: signal.alarm(0)

def self_test():
    cfg=read(Path(__file__).parent/'protocol.json')
    parser=lambda x:x.get('parsed') if x else None
    live=dict(pid=431960,start_ticks='38083128',parsed=expected_parent(cfg))
    assert classify_parent(live,cfg,parser)=='alive'
    assert classify_parent(None,cfg,parser)=='exited'
    assert classify_parent(dict(live,start_ticks='1',parsed=('unrelated',[])),cfg,parser)=='pid_reused_unrelated'
    for bad in (dict(live,start_ticks='1'),dict(live,parsed=('wrong',[]))):
        try: classify_parent(bad,cfg,parser)
        except RuntimeError: pass
        else: raise AssertionError('Changed parent adopted')
    assert related(live,cfg,parser)
    assert not related(dict(parsed=(expected_parent(cfg)[0],['--root','/other'])),cfg,parser)
    print('Parent identity/worker matching CPU fixtures passed; no proc/GPU access')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.root is None:parser.error('--root required')
    else:main(args.root)
