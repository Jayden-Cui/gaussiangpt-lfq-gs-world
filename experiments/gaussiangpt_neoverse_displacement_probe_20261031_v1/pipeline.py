"""Bounded two-window diagnostic, coexisting with unrelated GPU workloads.

Only process groups created by this queue may be signalled. Scientific failure
is a diagnostic result, never permission to retry, resume, downsample or train Z.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
import worker

RUN = 'gaussiangpt_neoverse_displacement_probe_20261031_v1'
BASE = '/data/CJH/AGS-V6Direct/outputs'
ROOT = BASE + '/' + RUN
RUNTIME = '/data/CJH/envs/gaussiangpt/bin/python'
HELPER_REL = 'gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
LIMITS = {'preflight': 210, 'run': 1830, 'replay': 330, 'outer': 3000}
LABELS = ('posterior_baseline', 'teacher_state_reference') + worker.BRANCHES
GATE_KEYS = ('future_motion_reduction_ge_10pct', 'h9_motion_reduction_ge_10pct',
             'future_full_nonworse', 'future_roi_nonworse')
REQUIRED = {'pipeline.py', 'test_pipeline.py', 'validate.py', 'worker.py', 'test_worker.py',
            'protocol.json', 'DESIGN.md', 'EXECUTION.md'}
LOCK = threading.Lock()
read, sha, require, write_new = worker.read, worker.sha, worker.require, worker.write_new


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def status(output, stage, **extra):
    temporary = output / 'STATUS.tmp'
    import json
    temporary.write_text(json.dumps(dict(status=stage, pid=os.getpid(), updated_at=timestamp(),
                                        goal_complete=False, **extra), indent=2), encoding='utf-8')
    temporary.replace(output / 'STATUS.json')


def helper_file(root):
    return Path(root).parent / HELPER_REL


def verify(root, expected=None, require_cpu=True):
    p, binding = worker.verify(root, cpu_witness=require_cpu)
    require(REQUIRED <= set(read(root / 'CODE_MANIFEST.json')), 'Manifest omits queue/tests/document witness')
    require(sha(helper_file(root)) == HELPER_SHA, 'Immutable owned-process helper changed')
    binding = dict(binding, process_helper_sha256=HELPER_SHA)
    if require_cpu:
        binding['cpu_validation_sha256'] = sha(root / 'CPU_VALIDATION.json')
    require(expected is None or expected == binding, 'Queue sources/CPU witness changed')
    return p, binding


def load_helper(root):
    path = helper_file(root); data = path.read_bytes()
    import hashlib
    require(hashlib.sha256(data).hexdigest() == HELPER_SHA, 'Owned-process helper changed before import')
    spec = importlib.util.spec_from_file_location('_displacement_owned_process_groups', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def gpu_admission(output):
    result = subprocess.run(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            check=True, timeout=15)
    devices = {}
    for line in result.stdout.strip().splitlines():
        match = re.fullmatch(r'\s*([01])\s*,\s*(\d+)\s*', line)
        require(match is not None and int(match[1]) not in devices, 'Malformed/duplicate GPU response')
        devices[int(match[1])] = int(match[2])
    require(set(devices) == {0, 1}, 'Both assigned GPUs must exist')
    evidence = dict(queried_at=timestamp(), queries=1, free_mib=devices, required_free_mib=20*1024,
                    independent_worker_allocator_cap_gib=16, unrelated_processes_signalled=False,
                    retry=False, measurement='NVML free memory at admission; not process peak memory')
    write_new(output / 'GPU_ADMISSION.json', evidence)
    require(all(value >= 20*1024 for value in devices.values()),
            'Less than 20 GiB free; fail once, leave shared workloads untouched')
    return evidence


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6',
               MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def stop_owned(helper):
    try:
        helper.terminate_queue(signal.SIGTERM, None)
    except SystemExit:
        pass


def command(root, mode, start=None):
    require(mode in ('preflight', 'run', 'replay'), 'Unknown worker mode')
    require((mode == 'preflight') == (start is None), 'Start required exactly for run/replay')
    require(start is None or start in worker.STARTS, 'Unplanned diagnostic window')
    argv = [RUNTIME, '-B', '-u', str(root / 'worker.py'), '--root', str(root), '--mode', mode]
    if start is not None:
        argv += ['--start', str(start)]
    return argv


def run_job(root, output, mode, start, gpu, helper, outcomes, deadline):
    name = mode if start is None else f'{mode}_{start:04d}'
    row = dict(name=name, mode=mode, start=start, gpu=gpu, maximum_process_seconds=LIMITS[mode])
    try:
        require(not helper.STOP_REQUESTED.is_set(), 'Queue stopped; no new child may start')
        seconds = min(LIMITS[mode], deadline-time.monotonic())
        require(seconds > 0, 'Total diagnostic budget expired')
        row.update(helper.run_child(command(root, mode, start), environment(gpu),
                                    output / (name + '.log'), seconds))
        require(row.get('exit_code') == 0 and row.get('timed_out') is False,
                f'{name} failed or timed out; no retry')
        row['status'] = 'complete'
        return row
    except BaseException as error:
        row.update(status='failed', error=repr(error))
        stop_owned(helper)
        raise
    finally:
        with LOCK:
            outcomes.append(row)
            write_new(output / (name + '_PROCESS.json'), row)


def parallel_stage(root, output, mode, helper, outcomes, deadline):
    require(mode in ('run', 'replay'), 'Only complete run/replay windows may be parallelized')
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_job, root, output, mode, start, gpu, helper, outcomes, deadline)
                   for gpu, start in enumerate(worker.STARTS)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop_owned(helper)
            raise


def fixed(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, label + ' mismatch')


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/non-numeric ' + label)


def common_contract(value, proof):
    for key in ('code_manifest_sha256', 'protocol_sha256'):
        fixed(value.get(key), proof[key], key)
    fixed(value.get('goal_complete'), False, 'goal completion claim')
    for key in ('gpu_peak_gib', 'gpu_reserved_peak_gib'):
        finite(value.get(key), key)
        require(0 < value[key] <= 16, 'Worker exceeded allocated/reserved 16 GiB cap')
    memory = value.get('memory_admission', {})
    fixed(memory.get('allocator_cap_gib'), 16, 'worker allocator cap')
    finite(memory.get('free_gib'), 'worker free memory')
    require(memory['free_gib'] >= 20, 'Worker began without 20 GiB free')
    require(memory.get('measurement_scope') == 'PyTorch allocated/reserved; not NVML whole-process peak',
            'Memory measurement scope changed')


def observation_contract(observation, start):
    rows = observation.get('per_horizon', [])
    require(len(rows) == 10 and [r.get('horizon') for r in rows] == list(range(10))
            and [r.get('frame') for r in rows] == list(range(start, start+10)), 'Incomplete ten-frame observation')
    for row in rows:
        require(type(row.get('gs_count')) is int and row['gs_count'] > 0, 'No native GS in observation')
        for key in ('mae', 'hand_activity_mae', 'psnr'):
            finite(row.get('metrics', {}).get(key), 'per-horizon '+key)
        if row['horizon']:
            finite(row.get('metrics', {}).get('motion_mae'), 'future motion error')
    mean = observation.get('mean_future9', {})
    for key in ('mae', 'hand_activity_mae', 'motion_mae', 'psnr'):
        finite(mean.get(key), 'future '+key)
        expected = math.fsum(row['metrics'][key] for row in rows[1:])/9
        require(math.isclose(mean[key], expected, rel_tol=1e-10, abs_tol=1e-10), 'Future mean changed: '+key)


def source_records_contract(records):
    require(isinstance(records, list) and len(records) == 354, 'Expected all 354 state/cache source files')
    paths = [r.get('path') for r in records]
    require(all(isinstance(p, str) for p in paths) and len(set(paths)) == 354, 'Duplicate/missing source file')
    require(all(type(r.get('bytes')) is int and r['bytes'] > 0
                and isinstance(r.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', r['sha256'])
                for r in records), 'Invalid source-file hash/size declaration')


def preflight_contract(root, p, proof):
    path = root / 'PREFLIGHT.json'; value = read(path)
    common_contract(value, proof)
    for key, expected in dict(schema='gs_displacement_preflight_v1', status='passed', gpu_witness=True,
                              source_frames=177, model_updates=0, oracle_updates=0).items():
        fixed(value.get(key), expected, 'preflight '+key)
    fixed(value.get('source_checkpoint_sha256'), p['source_hashes']['arms/spatial_z/step_0600.pt'], 'checkpoint')
    source_records_contract(value.get('source_files'))
    hashes = value.get('frozen_hashes', {})
    require(set(hashes) == {'model', 'codec'} and all(re.fullmatch('[0-9a-f]{64}', h) for h in hashes.values()),
            'Missing frozen model/codec hashes')
    windows = value.get('windows', [])
    require(len(windows) == 2 and {w.get('start') for w in windows} == set(worker.STARTS), 'Preflight window coverage')
    for window in windows:
        require(type(window.get('latent_rows')) is int and window['latent_rows'] > 0, 'No initial latent support')
        observation_contract(window['baseline'], window['start'])
        checks = window.get('actual_native_gradients', {})
        require(set(checks) == set(worker.BRANCHES), 'Preflight branch gradient checks incomplete')
        for branch, result in checks.items():
            for key, expected in dict(forward_frames=10, backward_future_frames=9, h0_frozen=True,
                                     optimizer_updates=0).items():
                fixed(result.get(key), expected, 'preflight gradient '+key)
            norms = result.get('gradient_norms', [])
            require(len(norms) == (9 if branch == 'xyz_only' else 1)
                    and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in norms),
                    'Preflight lacks finite nonzero gradients for every oracle variable')
    return value, sha(path)


def artifact_contract(directory, artifact_hashes):
    require(isinstance(artifact_hashes, dict) and artifact_hashes, 'Missing artifact hash map')
    required = {f'{label}/h{h:02d}{suffix}' for label in LABELS for h in range(10)
                for suffix in ('.pt', '_reference_rgb.pt', '_view0.png', '_view1.png')}
    required |= {'baseline_state.pt', 'bounded_features_state.pt'}
    required |= {f'{branch}{suffix}' for branch in worker.BRANCHES
                 for suffix in ('_variables.pt', '_trace.jsonl')}
    require(required <= set(artifact_hashes), 'Artifact map omits packets/views/variables/trace')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()
              and p.name != 'RUN_DONE.json'}
    require(actual == set(artifact_hashes), 'Artifacts added/omitted after run')
    for name, digest in artifact_hashes.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                and sha(worker.owned_path(directory, name)) == digest, 'Artifact hash/path changed: '+name)


def window_contract(root, start, p, proof, pre, pre_sha):
    directory = root / 'windows' / f'start_{start:04d}'
    require(not (directory / 'FAILED.json').exists(), 'Failed window cannot be accepted')
    path = directory / 'RUN_DONE.json'; value = read(path); common_contract(value, proof)
    for key, expected in dict(schema='gs_displacement_window_v1', status='complete_pending_independent_replay',
                              start=start, frames=list(range(start, start+10)), model_updates=0,
                              oracle_updates=200, all_initial_rows=True, resolution_hw=[252, 448],
                              preflight_sha256=pre_sha, frozen_hashes=pre['frozen_hashes'],
                              source_files=pre['source_files'], source_checkpoint_sha256=pre['source_checkpoint_sha256']).items():
        fixed(value.get(key), expected, 'window '+key)
    artifact_contract(directory, value.get('artifact_sha256'))
    observation_contract(value['baseline'], start); observation_contract(value['teacher_state_reference'], start)
    require(set(value.get('branches', {})) == set(worker.BRANCHES), 'Window branch coverage changed')
    for branch, result in value['branches'].items():
        fixed(result.get('steps'), 100, 'oracle steps'); fixed(result.get('legal_action_learning'), False, 'action claim')
        require(set(result.get('observations', {})) == {str(i) for i in worker.CHECKPOINTS}, 'Missing fixed observations')
        worker.parity([r['metrics'] for r in result['observations']['0']['per_horizon']],
                      [r['metrics'] for r in value['baseline']['per_horizon']], p['baseline_tolerances'])
        for observation in result['observations'].values():
            observation_contract(observation, start)
            reference = value['baseline']['per_horizon'][0]['metrics']
            for key, baseline in reference.items():
                actual = observation['per_horizon'][0]['metrics'].get(key)
                finite(actual, 'h0 '+key)
                tolerance = p['baseline_tolerances']['psnr' if 'psnr' in key else 'mae']
                require(abs(actual-baseline) <= tolerance, 'Frozen h0 reconstruction changed')
        recomputed = worker.numeric_gate(result['observations']['100'], value['baseline'])
        fixed(result.get('numeric_gate'), recomputed, 'Numeric gate from actual metrics')
        trace = [__import__('json').loads(line) for line in (directory / f'{branch}_trace.jsonl').read_text().splitlines()]
        require(len(trace) == 100 and [row.get('step') for row in trace] == list(range(1, 101)), 'Missing oracle updates')
        for row in trace:
            finite(row.get('loss_future_before_update'), 'oracle loss')
        if branch == 'xyz_only':
            fixed([r['gs_count'] for r in result['observations']['100']['per_horizon']],
                  [r['gs_count'] for r in value['baseline']['per_horizon']], 'XYZ GS row count')
        else:
            finite(result.get('audit', {}).get('max_abs_residual'), 'bounded feature residual')
            require(result['audit']['max_abs_residual'] <= 1.00001, 'Feature oracle exceeded original range')
    return value, sha(path)


def replay_contract(root, start, proof, run, run_sha):
    directory = root / 'replays' / f'start_{start:04d}'
    require(not (directory / 'FAILED.json').exists(), 'Failed replay cannot be accepted')
    path = directory / 'REPLAY_DONE.json'; value = read(path); common_contract(value, proof)
    for key, expected in dict(schema='gs_displacement_replay_v1', status='passed', start=start,
                              run_done_sha256=run_sha, model_loaded=False, optimizer_loaded=False,
                              z_required=False, displacement_required=False, views_per_frame=2).items():
        fixed(value.get(key), expected, 'replay '+key)
    rows = value.get('records', []); identities = [(r.get('label'), r.get('horizon')) for r in rows]
    require(len(rows) == 40 and len(set(identities)) == 40
            and set(identities) == {(label, h) for label in LABELS for h in range(10)}, 'Replay 4x10 coverage changed')
    for row in rows:
        label, h = row['label'], row['horizon']
        fixed(row.get('frame'), start+h, 'replay frame')
        fixed(row.get('packet_sha256'), run['artifact_sha256'][f'{label}/h{h:02d}.pt'], 'replay packet hash')
        observation = run[label] if label in ('baseline', 'teacher_state_reference') else (
            run['baseline'] if label == 'posterior_baseline' else run['branches'][label]['observations']['100'])
        fixed(row.get('gs_count'), observation['per_horizon'][h]['gs_count'], 'replay native GS count')
        for metric, limit in [('max_abs', 1e-6), ('mae', 1e-7)]:
            finite(row.get(metric), 'replay '+metric)
            require(0 <= row[metric] <= limit, 'Replay tolerance exceeded')
    return value, sha(path)


def build_summary(root, p, proof, pre, pre_sha):
    windows = []; rows = []
    for start in worker.STARTS:
        run, run_sha = window_contract(root, start, p, proof, pre, pre_sha)
        replay, replay_sha = replay_contract(root, start, proof, run, run_sha)
        windows.append(dict(start=start, run_done_sha256=run_sha, replay_done_sha256=replay_sha,
                            baseline=run['baseline'], teacher_state_reference=run['teacher_state_reference']))
        for branch in worker.BRANCHES:
            result = run['branches'][branch]; gate = result['numeric_gate']
            rows.append(dict(start=start, target_frame=start+9, branch=branch, fixed_steps=100,
                             numeric_passed=all(gate[key] for key in GATE_KEYS), numeric_gate=gate,
                             baseline_future9=run['baseline']['mean_future9'],
                             final_future9=result['observations']['100']['mean_future9'],
                             baseline_h9=run['baseline']['per_horizon'][9],
                             final_h9=result['observations']['100']['per_horizon'][9], audit=result['audit']))
    # Last read-only validation of all original state/cache files after all child processes.
    for record in pre['source_files']:
        path = Path(record['path'])
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Original data changed after probe')
    return dict(schema='gs_displacement_probe_summary_v1', status='complete', admission=proof,
                preflight_sha256=pre_sha, source_frames=177, bound_source_files=354, windows=windows,
                branch_window_results=rows, numeric_gates_passed=sum(row['numeric_passed'] for row in rows),
                numeric_gates_total=4, oracle_updates=400, model_updates=0, learned_action=False,
                legal_action_learning=False, goal_complete=False, visual_acceptance='pending',
                independent_native_replay_passed=True, model_free_native_replay=True,
                unique_window_frames=20, unique_optimized_future_frames=18,
                source_scope='two known TRAIN failure windows',
                scientific_scope='Target-fitting oracle only; unequal variable units/counts and learning rates; no learned Z, transfer or 3D correctness claim',
                memory_scope='PyTorch allocated/reserved peaks are not NVML whole-process peaks')


def summary_markdown(summary):
    lines = ['# 两个已知训练失败窗口的位移诊断', '',
             '已完成固定目标拟合及独立原生 GS 重放。模型更新数为 0；没有训练或证明学会动作 Z。人工视觉验收仍待进行。', '',
             '| 起始帧 | 分支 | 未来9帧运动MAE：基线→最终 | h9运动MAE：基线→最终 | 数值门槛 |',
             '|---:|---|---:|---:|---|']
    for row in summary['branch_window_results']:
        lines.append(f"| {row['start']} | {row['branch']} | {row['baseline_future9']['motion_mae']:.6f} → "
                     f"{row['final_future9']['motion_mae']:.6f} | {row['baseline_h9']['metrics']['motion_mae']:.6f} → "
                     f"{row['final_h9']['metrics']['motion_mae']:.6f} | {'通过' if row['numeric_passed'] else '未通过'} |")
    lines += ['', '两分支参数单位、数量和学习率不同；两个窗口不支持总体统计显著性。单视角拟合不证明三维正确，固定100步失败不证明不可达。',
              'PyTorch allocated/reserved 峰值只约束本探针，不是 NVML 全进程峰值；本队列没有终止或修改其他任务。', '']
    return '\n'.join(lines)


def assert_fresh(root):
    for name in ('queue', 'windows', 'replays', 'PREFLIGHT.json', 'SUMMARY.json', 'SUMMARY.md'):
        path = root / name
        require(not path.exists() and not path.is_symlink(), 'Existing probe attempt; no retry/overwrite/resume')


def main(root):
    root = Path(root).resolve(strict=True)
    require(root.as_posix() == ROOT and Path(__file__).resolve().parent == root, 'Use dedicated new probe root only')
    assert_fresh(root); output = root / 'queue'; output.mkdir(exist_ok=False)
    helper, outcomes, proof = None, [], {}
    deadline = time.monotonic()+LIMITS['outer']
    def startup_stop(signum, frame):
        raise TimeoutError('Probe interrupted during bounded startup')
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, startup_stop)
        signal.alarm(LIMITS['outer'])
        p, proof = verify(root); worker.verify_sources(p)
        helper = load_helper(root)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        write_new(output / 'RECEIPT.json', dict(status='submitted', pid=os.getpid(), started_at=timestamp(),
                  admission=proof, process_limits_seconds=LIMITS, assignments={'0':35, '1':123},
                  unrelated_processes_signalled=False, automatic_retry=False, model_updates=0, goal_complete=False))
        gpu_admission(output); status(output, 'preflight')
        run_job(root, output, 'preflight', None, 0, helper, outcomes, deadline)
        pre, pre_sha = preflight_contract(root, p, proof)
        verify(root, proof); worker.verify_sources(p)
        status(output, 'oracle_target_fitting')
        parallel_stage(root, output, 'run', helper, outcomes, deadline)
        verify(root, proof); worker.verify_sources(p)
        require(sha(root / 'PREFLIGHT.json') == pre_sha, 'Preflight changed during target fitting')
        for start in worker.STARTS:
            window_contract(root, start, p, proof, pre, pre_sha)
        status(output, 'independent_native_replay')
        parallel_stage(root, output, 'replay', helper, outcomes, deadline)
        status(output, 'cpu_summary')
        summary = build_summary(root, p, proof, pre, pre_sha)
        verify(root, proof); worker.verify_sources(p)
        require(sha(root / 'PREFLIGHT.json') == pre_sha, 'Preflight changed during independent replay')
        require(len(outcomes) == 5 and all(r.get('status') == 'complete' for r in outcomes), 'Incomplete process ledger')
        write_new(root / 'SUMMARY.json', summary)
        with (root / 'SUMMARY.md').open('x', encoding='utf-8') as stream:
            stream.write(summary_markdown(summary))
        verify(root, proof)
        write_new(output / 'DONE.json', dict(status='complete', finished_at=timestamp(), admission=proof,
                  preflight_sha256=pre_sha, summary_sha256=sha(root / 'SUMMARY.json'), outcomes=outcomes,
                  oracle_updates=400, model_updates=0, learned_action=False, visual_acceptance='pending',
                  goal_complete=False, unrelated_processes_signalled=False))
        status(output, 'complete')
    except BaseException as error:
        if helper is not None:
            stop_owned(helper)
        write_new(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                  finished_at=timestamp(), admission=proof, outcomes=outcomes, automatic_retry=False,
                  unrelated_processes_signalled=False, learned_action=False, goal_complete=False))
        status(output, 'failed', error=repr(error))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
