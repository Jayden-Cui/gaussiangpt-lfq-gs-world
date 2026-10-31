"""Bounded v9 queue: resource dependency is separate from fixed v7/v8 science."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
RUN = 'gaussiangpt_neoverse_spatial_action_20261031_v9'
BASE = PurePosixPath('/data/CJH/AGS-V6Direct/outputs')
RUNTIME = '/data/CJH/envs/gaussiangpt/bin/python'
HELPER = '../gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
ARMS = ('spatial_z', 'spatial_constant')
STATISTICS = '../gaussiangpt_neoverse_native_z_20261031_v6/code/compare_arms.py'
STATISTICS_SHA = '14fa085be07f8ab3f6ef79ec671362f062d6b2726f3968b02f8c4d95ff91c1b8'
REFERENCE_COMPARE = '../gaussiangpt_neoverse_local_condition_20261031_v8/compare.py'
REFERENCE_COMPARE_SHA = 'eea336f5ea955aa955dc7d7d37ac3a43b2dbbf8821d568e9a545083894b938fe'
REQUIRED = {'pipeline.py','test_pipeline.py','worker.py','test_worker.py','spatial_action.py',
            'test_spatial_action.py','compare.py','test_compare.py','validate.py','protocol.json','EXECUTION.md'}
QUEUE = dict(gpu_workers={'0':'spatial_z','1':'spatial_constant'},wait_seconds=7200,wait_interval_seconds=60,
             preflight_process_seconds=630,train_process_seconds=1900,eval_process_seconds=2500,
             comparison_seconds=300,outer_seconds=12600,minimum_free_gib=55)
DEPENDENCY = dict(root=str(BASE/'gaussiangpt_neoverse_multisource_codec_20261031_v1'),pid=439814,
                  start_ticks='38835617',manifest_sha256='b4958856fb50af2c55f4c2fce9eaa2a16c14ca9ec5f87436565a499052efe48a')
OUTCOME_LOCK = threading.Lock()
UNSPECIFIED = object()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def protocol_contract(cfg):
    import worker  # Standard-library definitions only; no runtime setup.
    worker.protocol_contract(cfg)
    if cfg.get('queue') != QUEUE or any(cfg.get('dependency', {}).get(k) != v for k,v in DEPENDENCY.items()):
        raise ValueError('Fixed v9 queue budget, assignment or resource dependency changed')
    if cfg.get('comparison_source_relative') != STATISTICS:
        raise ValueError('Pinned comparison source changed')


def external_helpers():
    import worker
    return {**worker.BOUND_HELPERS, HELPER:HELPER_SHA, STATISTICS:STATISTICS_SHA,
            REFERENCE_COMPARE:REFERENCE_COMPARE_SHA}


def verify(root, expected=None, cpu=True):
    """Only local code/protocol/document hashes; no remote data or Torch imports."""
    root=Path(root).resolve()
    manifest=read(root/'CODE_MANIFEST.json')
    pinned=external_helpers()
    # SUMMARY.md is a generated output, never inferred to be source code.
    required=REQUIRED | set(pinned) | {p.name for p in root.glob('*.py')}
    if not required.issubset(manifest) or any(manifest.get(k)!=v for k,v in pinned.items()):
        raise ValueError('Manifest missing executed sources or immutable v7/v8/helpers')
    for relative,digest in manifest.items():
        path=(root/relative).resolve()
        if (Path(relative).is_absolute() or PurePosixPath(relative).is_absolute() or '\\' in relative
                or not path.is_relative_to(root.parent) or not re.fullmatch('[0-9a-f]{64}',str(digest))
                or sha(path)!=digest):
            raise ValueError('Immutable source mismatch: '+relative)
    cfg=read(root/'protocol.json');protocol_contract(cfg)
    admission=dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),protocol_sha256=sha(root/'protocol.json'))
    if cpu:
        proof=read(root/'CPU_VALIDATION.json')
        if (proof.get('status')!='passed' or proof.get('gpu_executed') is not False
                or any(proof.get(k)!=v for k,v in admission.items())):
            raise ValueError('Exact source/protocol has no passed CPU_VALIDATION')
        admission['cpu_validation_sha256']=sha(root/'CPU_VALIDATION.json')
    if expected is not None and admission!=expected:raise ValueError('Immutable admission changed')
    return cfg,admission


def dependency_worker(snapshot, dependency):
    parts=command_parts(snapshot)
    if parts is None:return False
    script,argv=parts;root=dependency['root']
    if script not in {root+'/worker.py',root+'/compare.py',root+'/pipeline.py'} or argv.count('--root')!=1:
        return False
    index=argv.index('--root')
    if index+1==len(argv) or argv[index+1]!=root:return False
    # Include the parent's bounded CPU admission child; the exact main parent
    # remains subject to PID/start-ticks validation in observe().
    return script!=root+'/pipeline.py' or ('--mode' in argv and argv[argv.index('--mode')+1:] == ['source-admission'])


def terminal_contract(done, failed, dependency):
    if (done is None)==(failed is None):raise ValueError('Require one unique dependency DONE or FAILED receipt')
    kind,value=('DONE',done) if done is not None else ('FAILED',failed)
    if (not isinstance(value,dict) or value.get('status')!=('complete' if kind=='DONE' else 'failed')
            or value.get('admission',{}).get('code_manifest_sha256')!=dependency['manifest_sha256']):
        raise ValueError('Dependency terminal identity/status mismatch')
    return dict(parent_terminal=kind,resource_dependency_only=True,
                dependency_failure_does_not_change_fixed_scientific_sources=True)


def read_terminal(dependency):
    root=Path(dependency['root']);paths=[root/'queue/DONE.json',root/'queue/FAILED.json']
    if sha(root/'CODE_MANIFEST.json')!=dependency['manifest_sha256']:
        raise ValueError('Dependency manifest mismatch')
    for path in paths:
        if path.exists() and not path.is_file():raise ValueError('Malformed dependency terminal path')
    values=[read(path) if path.is_file() else None for path in paths]
    return dict(decision=terminal_contract(*values,dependency),
                hashes={str(path):sha(path) for path in paths if path.is_file()})


def bind_file(path, records, expected=UNSPECIFIED):
    path=Path(path).resolve(strict=True);actual=sha(path)
    if expected is not UNSPECIFIED and (not isinstance(expected,str) or actual!=expected):
        raise ValueError('Artifact hash mismatch: '+str(path))
    if path in records and records[path]!=actual:raise ValueError('Previously completed artifact changed: '+str(path))
    records[path]=actual
    return actual


def scientific_sources(root,cfg,records):
    """Small source receipts plus the two exact source400 checkpoint hashes."""
    source=Path(cfg['source_root']);ref=Path(cfg['reference_baselines']['root'])
    bind_file(source/'CODE_MANIFEST.json',records,cfg['source_manifest_sha256'])
    for name,key in [('CODE_MANIFEST.json','code_manifest_sha256'),('protocol.json','protocol_sha256'),
                     ('SUMMARY.json','summary_sha256')]:
        bind_file(ref/name,records,cfg['reference_baselines'][key])
    selection_path=source/'SELECTION.json';bind_file(selection_path,records);selection=read(selection_path)
    if (selection.get('status')!='selected' or selection.get('selected_step')!=400
            or selection.get('code_manifest_sha256')!=cfg['source_manifest_sha256']):
        raise ValueError('Science source must remain original selected v7 step400')
    arms={}
    for arm in ARMS:
        parent=cfg['arms'][arm]['source_arm'];row=selection['checkpoints'][parent];path=Path(row['path'])
        if not path.is_absolute() or not path.is_relative_to(source) or path.name!='step_0400.pt':
            raise ValueError('Science source checkpoint path changed')
        bind_file(path,records,row['sha256'])
        arms[parent]=dict(evaluation={'checkpoint_sha256':row['sha256']})
    return dict(root=root,protocol=cfg,context={'arms':arms})


def preflight_contract(root,cfg,admission,science,records):
    import worker
    evidence=dict(science,manifest_hash=admission['code_manifest_sha256'],protocol_hash=admission['protocol_sha256'])
    preflight_hash=worker.witness(evidence)  # Pure JSON, gradient and hash checks.
    proof=read(root/'PREFLIGHT.json')
    for arm,result in proof['arms'].items():
        required={'arm':arm,'source_arm':cfg['arms'][arm]['source_arm'],'source_steps':400,
                  'source_manifest_sha256':cfg['source_manifest_sha256'],'code_manifest_sha256':admission['code_manifest_sha256'],
                  'protocol_sha256':admission['protocol_sha256'],'spatial_action_enabled':True,
                  'optimizer_rng_restored_from_source':True,'reference_baselines':cfg['reference_baselines'],'goal_complete':False}
        require_fields(result,required,'preflight '+arm)
    bind_file(root/'PREFLIGHT.json',records,preflight_hash)
    bind_file(root/'preflight/RESULT.json',records,preflight_hash)
    return proof


def require_fields(value,expected,label):
    if any(type(value.get(k)) is not type(v) or value[k]!=v for k,v in expected.items()):
        raise ValueError('Incomplete/mismatched '+label)


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def train_contract(root,arm,cfg,admission,preflight,records):
    folder=root/'arms'/arm
    if (folder/'FAILED.json').exists():raise ValueError('Failed/partial training cannot proceed: '+arm)
    done=read(folder/'TRAIN_DONE.json')
    require_fields(done,dict(schema='gs_spatial_action_train_v1',status='complete',arm=arm,
        source_arm=cfg['arms'][arm]['source_arm'],source_steps=400,completed_steps=600,steps=600,additional_steps=200,
        source_checkpoint_sha256=preflight['arms'][arm]['source_checkpoint_sha256'],
        source_manifest_sha256=cfg['source_manifest_sha256'],code_manifest_sha256=admission['code_manifest_sha256'],
        protocol_sha256=admission['protocol_sha256'],reference_baselines=cfg['reference_baselines'],
        original_config_preserved=True,encoder_unchanged=True,spatial_action_enabled=True,
        optimizer_rng_restored_from_source=True,goal_complete=False),'actual source400->600 training '+arm)
    if done.get('partial') is True or done.get('actual_step',600)!=600 or done.get('last',{}).get('step')!=600:
        raise ValueError('Partial/in-flight checkpoint cannot pass as fixed600')
    source=read(folder/'SOURCE.json');plan=read(folder/'PLAN.json')
    require_fields(source,dict(schema='gs_spatial_action_source_v1',arm=arm,source_steps=400,
        source_checkpoint_sha256=done['source_checkpoint_sha256'],reference_baselines=cfg['reference_baselines']), 'SOURCE '+arm)
    for key in ('loaded_model_sha256','loaded_codec_sha256','original_optimizer_state_sha256','original_rng_state_sha256',
                'original_config','full_plan_sha256','continuation_plan_sha256','fixed_null_key_bias','output_std_sha256'):
        if source.get(key)!=preflight['arms'][arm].get(key) or done.get(key)!=source.get(key):
            raise ValueError('Training drift from exact restored preflight/source: '+key)
    require_fields(plan,dict(schema='gs_spatial_action_plan_v1',source_steps=400,intended_total_steps=600,
        continued_batch_indices=list(range(400,600)),full_plan_sha256=source['full_plan_sha256'],
        continuation_plan_sha256=source['continuation_plan_sha256']), 'fixed600 plan '+arm)
    import worker
    starts=plan.get('starts',[])
    if (len(starts)!=600 or any(not isinstance(row,list) or len(row)!=4
            or any(type(v) is not int or not 0<=v<168 for v in row) for row in starts)
            or worker.digest(starts)!=source['full_plan_sha256']
            or worker.digest(starts[400:])!=source['continuation_plan_sha256']):
        raise ValueError('Incomplete or changed original B4 continuation schedule')
    rows=jsonl(folder/'train.jsonl')
    if (len(rows)!=200 or [row.get('step') for row in rows]!=list(range(401,601))
            or any(row.get('plan_index')!=row['step']-1 or row.get('added_steps')!=row['step']-400
                   or row.get('starts')!=starts[row['step']-1] for row in rows)
            or rows[-1]!=done['last']):raise ValueError('Partial trace cannot authorize fixed600 evaluation')
    for filename,key in [('SOURCE.json','source_sha256'),('PLAN.json','plan_sha256'),('train.jsonl','train_jsonl_sha256')]:
        bind_file(folder/filename,records,done.get(key))
    if done.get('preflight_sha256')!=sha(root/'PREFLIGHT.json'):raise ValueError('Training used another preflight')
    if set(done.get('checkpoint_sha256',{}))!={'500','600'}:raise ValueError('Both real 500/600 checkpoints required')
    for step in (500,600):bind_file(folder/f'step_{step:04d}.pt',records,done['checkpoint_sha256'][str(step)])
    bind_file(folder/'TRAIN_DONE.json',records)
    return done


def evaluation_contract(root,arm,cfg,admission,train,records):
    folder=root/'evaluations'/arm
    if (folder/'FAILED.json').exists():raise ValueError('Evaluation failure receipt: '+arm)
    result=read(folder/'EVALUATION.json')
    require_fields(result,dict(schema='gs_spatial_action_evaluation_v1',status='complete',arm=arm,
        completed_steps=600,steps=600,additional_steps=200,selected_step=600,checkpoint_step=600,optimizer_updates=0,
        checkpoint_path=str(root/'arms'/arm/'step_0600.pt'),checkpoint_sha256=train['checkpoint_sha256']['600'],
        source_steps=400,source_checkpoint_sha256=train['source_checkpoint_sha256'],
        code_manifest_sha256=admission['code_manifest_sha256'],protocol_sha256=admission['protocol_sha256'],
        reference_baselines=cfg['reference_baselines'],train=train,windows=168,real_future_exposures=1512,
        unique_future_frames=176,physical_frames_with_some_reconstruction=177,
        all_initial_latent_points_retained=True,all_ten_frames=True,full_native_resolution=[252,448],goal_complete=False),
        'complete fixed600 evaluation '+arm)
    for filename,key in [('targets.jsonl','targets_sha256'),('initial_reconstruction.json','initial_reconstruction_sha256'),
                          ('actions.pt','actions_sha256'),('snapshots.json','snapshots_sha256')]:
        bind_file(folder/filename,records,result.get(key))
    rows=jsonl(folder/'targets.jsonl')
    if len(rows)!=1512 or {(row.get('start'),row.get('horizon')) for row in rows}!={(s,h) for s in range(168) for h in range(1,10)}:
        raise ValueError('Missing/duplicate complete168x9 evaluation exposures')
    initial=read(folder/'initial_reconstruction.json').get('rows',[])
    if len(initial)!=168 or {row.get('start') for row in initial}!=set(range(168)):
        raise ValueError('Missing complete168 initial reconstructions')
    snapshots=read(folder/'snapshots.json')['rows']
    mapping={row['source_frame']:(row['start'],row['horizon']) for row in snapshots}
    if len(snapshots)!=5 or set(mapping)!={0,44,88,132,176} or mapping[44]!=(35,9) or mapping[132]!=(123,9):
        raise ValueError('Missing fixed known-failure snapshots')
    for row in snapshots:
        if set(row['files'])!={'real','correct','shuffle84','blinded','adapted_held'}:
            raise ValueError('Missing fixed snapshot controls')
        for item in row['files'].values():
            path=Path(item['path']).resolve()
            if not path.is_relative_to(folder.resolve()/'snapshots'):raise ValueError('Snapshot path escaped this evaluation')
            bind_file(path,records,item['sha256'])
    bind_file(folder/'EVALUATION.json',records)
    return result


def comparison_contract(root,cfg):
    summary=read(root/'SUMMARY.json')
    require_fields(summary,dict(schema='gs_spatial_action_comparison_v1',engineering_status='complete',
        comparison_eligible=True,selected_step=600,goal_complete=False,new_arms=list(ARMS),
        reference_arms=['base_z','base_constant'],reference_root=cfg['reference_baselines']['root'],
        comparison_source_sha256=STATISTICS_SHA,reference_validator_sha256=REFERENCE_COMPARE_SHA),'comparison')
    if summary.get('comparison_eligibility',{}).get('eligible') is not True:
        raise ValueError('Comparison engineering evidence is incomplete')
    return summary


def jobs(root,stage):
    prefix=[RUNTIME,'-B','-u']
    if stage=='preflight':return [('preflight',0,prefix+[str(root/'worker.py'),'--root',str(root),'--mode','preflight'],630)]
    if stage in ('train','eval'):
        return [(stage+'_'+arm,gpu,prefix+[str(root/'worker.py'),'--root',str(root),'--mode',stage,'--arm',arm],
                 1900 if stage=='train' else 2500) for gpu,arm in enumerate(ARMS)]
    if stage=='compare':return [('compare',None,prefix+[str(root/'compare.py'),'--root',str(root)],300)]
    raise ValueError('Unknown stage: '+stage)


def load_helper(root):
    path = (root / HELPER).resolve(strict=True)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != HELPER_SHA:
        raise ValueError('Immutable process helper changed')
    spec = importlib.util.spec_from_file_location('_spatial_action_owned_groups', path)
    helper = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), 'exec'), helper.__dict__)
    return helper


def command_parts(snapshot):
    if not isinstance(snapshot, dict):
        return None
    argv = snapshot.get('argv')
    if (not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv)
            or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?', PurePosixPath(argv[0]).name)):
        return None
    index = 1
    while index < len(argv) and argv[index] in ('-B', '-u'):
        index += 1
    if index == len(argv) or argv[index].startswith('-'):
        return None
    return argv[index], argv[index + 1:]


def parent_command(snapshot, dependency):
    root = dependency['root']
    return command_parts(snapshot) == (root + '/pipeline.py', ['--root', root])


def classify_parent(snapshot, dependency):
    if snapshot is None:
        return 'exited'
    if (snapshot.get('pid') != dependency['pid'] or snapshot.get('stable', True) is not True
            or str(snapshot.get('start_ticks')) != dependency['start_ticks'] or not parent_command(snapshot, dependency)):
        raise RuntimeError('Dependency PID/start ticks/argv/root identity mismatch; no shared process signalled')
    return 'alive'


def stable_snapshot(helper, pid):
    first, second = helper.process_snapshot(pid), helper.process_snapshot(pid)
    if second is None:
        try:
            stat = (Path('/proc') / str(pid) / 'stat').read_text()
        except (FileNotFoundError, ProcessLookupError):
            return None
        if stat[stat.rfind(')') + 2:].split()[0] == 'Z':
            return None
        raise RuntimeError('Live dependency process identity unavailable; exit is not inferred')
    if first is None or first != second:
        raise RuntimeError('Dependency process identity raced during observation')
    return dict(second, stable=True)


def observe(helper, dependency):
    root = Path(dependency['root'])
    if sha(root / 'CODE_MANIFEST.json') != dependency['manifest_sha256']:
        raise RuntimeError('Exact dependency manifest changed')
    state = classify_parent(stable_snapshot(helper, dependency['pid']), dependency)
    workers = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        candidate = helper.process_snapshot(int(entry.name))
        if not (parent_command(candidate, dependency) or dependency_worker(candidate, dependency)):
            continue
        current = stable_snapshot(helper, int(entry.name))
        if current is None:
            continue
        if any(candidate[key] != current[key] for key in ('pid', 'start_ticks', 'argv')):
            raise RuntimeError('Dependency worker changed during observation')
        if parent_command(current, dependency):
            state = classify_parent(current, dependency)
        if dependency_worker(current, dependency):
            workers.append(current)
    return dict(parent_state=state, workers=workers, all_exited=state == 'exited' and not workers)


def wait_for_parent(output, cfg, helper):
    begun = time.monotonic()
    while True:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopped before dependency admission')
        observation = observe(helper, cfg['dependency'])
        write(output / 'STATUS.json', dict(status='waiting', checked_at=timestamp(),
              elapsed_seconds=time.monotonic() - begun, goal_complete=False, **observation))
        if observation['all_exited']:
            terminal = read_terminal(cfg['dependency'])
            write(output / 'DEPENDENCY_TERMINAL.json', dict(observation=observation, **terminal))
            return terminal
        remaining = cfg['queue']['wait_seconds'] - (time.monotonic() - begun)
        if remaining <= 0:
            raise TimeoutError('7200-second dependency wait expired; no shared process stopped or restarted')
        helper.STOP_REQUESTED.wait(min(cfg['queue']['wait_interval_seconds'], remaining))


def unchanged_dependency(helper, cfg, terminal):
    if not observe(helper, cfg['dependency'])['all_exited'] or read_terminal(cfg['dependency']) != terminal:
        raise RuntimeError('Dependency resumed or its exact completed receipt changed')


def stop_owned_children(helper):
    # The bound helper only signals groups it created using start_new_session.
    try:
        helper.terminate_queue(signal.SIGTERM, None)
    except SystemExit:
        pass


def run_job(job, output, helper, outcomes):
    name, gpu, command, timeout = job
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    row = dict(name=name, gpu=gpu, command=command, process_seconds=timeout)
    try:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping; no new child or retry')
        result = helper.run_child(command, env, output / (name + '.log'), timeout)
        row.update(result)
        if result.get('exit_code') != 0 or result.get('timed_out') is not False:
            raise RuntimeError(name + ' failed or timed out; no automatic retry')
        row['status'] = 'complete'
        return row
    except BaseException as error:
        row.update(status='failed', error=repr(error))
        stop_owned_children(helper)  # First failing thread immediately stops its sibling.
        raise
    finally:
        with OUTCOME_LOCK:
            outcomes.append(row)
            write(output / (name + '_PROCESS.json'), row)


def execute_stage(root, output, stage, helper, outcomes, before_stage):
    before_stage()
    if helper.STOP_REQUESTED.is_set():
        raise RuntimeError('Queue stopping before stage: ' + stage)
    write(output / 'STATUS.json', dict(status=stage, pid=os.getpid(), goal_complete=False))
    tasks = jobs(root, stage)
    if len(tasks) == 1:
        run_job(tasks[0], output, helper, outcomes)
    else:
        # Each run_job stops siblings before propagating; context waits only for
        # the bound helper to reap the owned children after that shared stop.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_job, task, output, helper, outcomes) for task in tasks]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop_owned_children(helper)
                raise
    before_stage()


def gpu_admission(output):
    result = subprocess.run(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15, check=True)
    parsed = {}
    for line in result.stdout.strip().splitlines():
        match = re.fullmatch(r'\s*([01])\s*,\s*(\d+)\s*', line)
        if match is None or int(match[1]) in parsed:
            raise ValueError('Unexpected two-GPU memory response')
        parsed[int(match[1])] = int(match[2])
    if set(parsed) != {0, 1}:
        raise ValueError('Both GPUs must be present in the single admission query')
    write(output / 'GPU_ADMISSION.json', dict(free_mib=parsed, required_mib=55 * 1024, queries=1))
    if any(value < 55 * 1024 for value in parsed.values()):
        raise RuntimeError('GPU0 or GPU1 has less than 55 GiB free; no waiting/retry')


def main(root):
    root=Path(root).resolve()
    if root.as_posix()!=(BASE/RUN).as_posix():raise ValueError('Require the dedicated v9 queue root')
    output=root/'queue';output.mkdir(exist_ok=False)
    helper,outcomes,admission,records,terminal=None,[],{},{},None
    def startup_stop(signum,frame):raise TimeoutError('Queue stopped during startup or exceeded12600 seconds')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,startup_stop)
        signal.alarm(12600)
        cfg,admission=verify(root)
        helper=load_helper(root)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,helper.terminate_queue)
        write(output/'RECEIPT.json',dict(status='waiting',pid=os.getpid(),started_at=timestamp(),
            admission=admission,dependency=cfg['dependency'],resource_dependency_only=True,
            scientific_source_root=cfg['source_root'],reference_baselines=cfg['reference_baselines'],
            only_own_new_session_groups_may_be_signalled=True,automatic_retry=False,goal_complete=False))
        terminal=wait_for_parent(output,cfg,helper)

        def before_stage():
            verify(root,admission)
            unchanged_dependency(helper,cfg,terminal)
            for path,digest in records.items():
                if sha(path)!=digest:raise RuntimeError('Frozen completed/source artifact changed: '+str(path))

        before_stage()
        science=scientific_sources(root,cfg,records)
        before_stage();gpu_admission(output)
        execute_stage(root,output,'preflight',helper,outcomes,before_stage)
        preflight=preflight_contract(root,cfg,admission,science,records)
        execute_stage(root,output,'train',helper,outcomes,before_stage)
        trains={arm:train_contract(root,arm,cfg,admission,preflight,records) for arm in ARMS}
        execute_stage(root,output,'eval',helper,outcomes,before_stage)
        for arm in ARMS:evaluation_contract(root,arm,cfg,admission,trains[arm],records)
        execute_stage(root,output,'compare',helper,outcomes,before_stage)
        summary=comparison_contract(root,cfg)
        before_stage()
        if helper.STOP_REQUESTED.is_set():raise RuntimeError('Queue stopped before complete terminal')
        done=dict(status='complete',engineering_status='complete',selected_step=600,
            scientific_status=summary['scientific_status'],summary_sha256=sha(root/'SUMMARY.json'),
            admission=admission,dependency_terminal=terminal,outcomes=outcomes,
            artifact_hashes={str(path):digest for path,digest in records.items()},
            automatic_retry=False,goal_complete=False,finished_at=timestamp())
        write(output/'STATUS.json',dict(status='complete',selected_step=600,goal_complete=False))
        write(output/'DONE.json',done)
    except BaseException as error:
        if helper is not None:stop_owned_children(helper)
        failed=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),admission=admission,
            dependency_terminal=terminal,outcomes=outcomes,artifact_hashes={str(path):digest for path,digest in records.items()},
            automatic_retry=False,old_processes_signalled=False,goal_complete=False,finished_at=timestamp())
        write(output/'FAILED.json',failed)
        write(output/'STATUS.json',dict(status='failed',error=repr(error),goal_complete=False))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    main(parser.parse_args().root)
