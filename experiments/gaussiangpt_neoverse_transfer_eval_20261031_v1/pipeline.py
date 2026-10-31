"""One bounded transfer-evaluation submission; all dependency waiting is program-owned."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
PREPARATION_NAME = 'gaussiangpt_neoverse_tower14_transfer_20261031_v1'
SOURCE_NAME = 'gaussiangpt_neoverse_native_z_20261031_v6'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
SNAPSHOT_SOURCE_SHA = '4f1c3724da2e6363c9c6966154e87ddb8c227ab5de121454b6ac792faaec22a7'


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
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def verify(root, expected=None, witness=True):
    manifest = read(root / 'CODE_MANIFEST.json')
    required = {'protocol.json', 'pipeline.py', 'evaluate_transfer.py', 'transfer_data.py',
                'compare_transfer.py', 'validate.py', 'EXECUTION.md'}
    if not required.issubset(manifest):
        raise ValueError('Transfer manifest omits executable inputs')
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root.parent) or sha(path) != digest:
            raise ValueError('Transfer source changed: ' + relative)
    cfg = read(root / 'protocol.json')
    queue = cfg['queue']
    if (cfg.get('schema') != 'gs_neoverse_cross_source_eval_v1'
            or cfg.get('training_performed') is not False or cfg.get('automatic_retry') is not False
            or queue != dict(parent_pid=431535, parent_start_ticks='37974645', parent_postfit_pid=430698,
                 parent_postfit_start_ticks='37943462', wait_seconds=18000, wait_interval_seconds=60,
                 outer_seconds=23300, process_seconds_per_arm=2500, comparison_seconds=180,
                 gpu_workers={'0': ['z_rgb', 'constant_delta'], '1': ['constant_rgb', 'z_delta']})):
        raise ValueError('Fixed queue identity, budgets or matrix changed')
    binding = dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'), protocol_sha256=sha(root / 'protocol.json'))
    if witness:
        proof = read(root / 'CPU_VALIDATION.json')
        if (proof.get('status') != 'passed' or proof.get('gpu_executed') is not False
                or proof.get('code_manifest_sha256') != binding['code_manifest_sha256']):
            raise ValueError('Exact transfer sources need a passed CPU document witness')
        binding['cpu_validation_sha256'] = sha(root / 'CPU_VALIDATION.json')
    if expected is not None and binding != expected:
        raise ValueError('Queue admission files changed')
    return cfg, binding


def expected_parent(cfg):
    prep = cfg['preparation_root']
    queue = cfg['queue']
    return prep + '/queue_prepare.py', ['--root', prep, '--parent-pid', str(queue['parent_postfit_pid']),
                                      '--parent-start-ticks', queue['parent_postfit_start_ticks']]


def flag_matches(arguments, name, value):
    if arguments.count(name) != 1:
        return False
    i = arguments.index(name)
    return i + 1 < len(arguments) and arguments[i + 1] == value


def related(snapshot, cfg, parser):
    parsed = parser(snapshot)
    if parsed is None:
        return False
    script, arguments = parsed
    prep, main = cfg['preparation_root'], cfg['main_root']
    if script in (prep + '/queue_prepare.py', prep + '/prepare_frozen_codec.py'):
        return flag_matches(arguments, '--root', prep)
    if script in tuple(main + '/' + s for s in ('pipeline.py', 'runner.py', 'compare.py',
            'postfit_code/queue_postfit.py', 'postfit_code/nearby_view_probe.py', 'postfit_code/render_six_panel.py')):
        return flag_matches(arguments, '--root', main)
    renderer = str(PurePosixPath(main).parent / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py')
    return script == renderer and flag_matches(arguments, '--output', prep + '/teacher/native448')


def classify_parent(snapshot, cfg, parser):
    if snapshot is None:
        return 'exited'
    q = cfg['queue']
    if snapshot.get('pid') != q['parent_pid'] or snapshot.get('stable', True) is not True:
        raise RuntimeError('Preparation parent identity read failed')
    match = parser(snapshot) == expected_parent(cfg)
    if str(snapshot.get('start_ticks')) != q['parent_start_ticks']:
        if match or related(snapshot, cfg, parser):
            raise RuntimeError('Preparation parent was replaced; never adopt a new run')
        return 'pid_reused_unrelated'
    if not match:
        raise RuntimeError('Preparation parent command changed')
    return 'alive'


def observe(cfg, helper, snapshots):
    parent = cfg['queue']['parent_pid']
    identity = classify_parent(snapshots.stable_snapshot(helper, parent), cfg, snapshots.command_parts)
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == parent:
            continue
        preliminary = helper.process_snapshot(int(entry.name))
        if not related(preliminary, cfg, snapshots.command_parts):
            continue
        current = snapshots.stable_snapshot(helper, int(entry.name))
        if current is None:
            continue
        if snapshots.command_parts(current) == expected_parent(cfg):
            raise RuntimeError('A second preparation parent exists')
        if related(current, cfg, snapshots.command_parts):
            matches.append(dict(pid=current['pid'], start_ticks=current['start_ticks'],
                                script=snapshots.command_parts(current)[0]))
    return dict(parent_identity=identity, workers=matches, all_exited=identity != 'alive' and not matches)


def terminal(cfg):
    prep, main = Path(cfg['preparation_root']), Path(cfg['main_root'])
    if (prep / 'queue/FAILED.json').exists() or (prep / 'FAILED.json').exists() or (main / 'queue/FAILED.json').exists():
        raise RuntimeError('Main or preparation failed; no transfer GPU work admitted')
    paths = [prep / 'queue/DONE.json', prep / 'SUMMARY.json', main / 'queue/DONE.json',
             main / 'SELECTION.json', main / 'SUMMARY.json']
    prepared, summary, main_done, selected, main_summary = [read(p) for p in paths]
    if (prepared.get('status') != 'complete' or prepared.get('summary_sha256') != sha(paths[1])
            or summary.get('status') != 'complete' or summary.get('preparation_contract_passed') is not True
            or summary.get('frames') != 177 or summary.get('codec_frozen') is not True
            or summary.get('training_performed') is not False
            or summary.get('input_binding_sha256') != cfg['preparation_binding_sha256']):
        raise RuntimeError('Complete frozen177-frame preparation required')
    if (main_done.get('status') != 'complete' or selected.get('status') != 'selected'
            or main_done.get('selected_step') != selected.get('selected_step')
            or main_summary.get('engineering_status') != 'complete'
            or main_summary.get('comparison_eligible') is not True
            or main_summary.get('checkpoint_selection') != selected):
        raise RuntimeError('Main common-step engineering terminal required')
    return dict(selected_step=selected['selected_step'], inputs={str(p): sha(p) for p in paths})


def wait_dependencies(root, cfg, helper, snapshots):
    begun = time.monotonic()
    while True:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping before dependencies finish')
        observed = observe(cfg, helper, snapshots)
        write(root / 'queue/STATUS.json', dict(status='waiting_dependencies', pid=os.getpid(),
              checked_at=timestamp(), elapsed_seconds=time.monotonic()-begun, **observed, goal_complete=False))
        if observed['all_exited']:
            result = terminal(cfg)
            write(root / 'queue/DEPENDENCIES_DONE.json', result)
            return result
        remaining = cfg['queue']['wait_seconds'] - (time.monotonic()-begun)
        if remaining <= 0:
            raise TimeoutError('Bounded dependency wait expired; no old process stopped or restarted')
        helper.STOP_REQUESTED.wait(min(cfg['queue']['wait_interval_seconds'], remaining))


def check_dependencies(cfg, helper, snapshots, admitted):
    if not observe(cfg, helper, snapshots)['all_exited'] or terminal(cfg) != admitted:
        raise RuntimeError('Dependency process or terminal evidence changed')


def main(root):
    root = root.resolve()
    output = root / 'queue'
    output.mkdir(exist_ok=False)
    outcomes = []
    def deadline(signum, frame):
        raise TimeoutError('Transfer queue admission exceeded its total budget')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(23300)
    try:
        cfg, admission = verify(root)
        helper_path = root.parent / SOURCE_NAME / 'analysis_code/postfit_queue.py'
        snapshot_path = root.parent / PREPARATION_NAME / 'queue_prepare.py'
        if sha(helper_path) != HELPER_SHA or sha(snapshot_path) != SNAPSHOT_SOURCE_SHA:
            raise RuntimeError('Pinned process helpers changed')
        helper = load('_transfer_owned_groups', helper_path)
        snapshots = load('_transfer_process_identity', snapshot_path)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        write(output / 'RECEIPT.json', dict(pid=os.getpid(), started_at=timestamp(), admission=admission,
              parent_pid=cfg['queue']['parent_pid'], parent_start_ticks=cfg['queue']['parent_start_ticks'],
              only_own_new_session_groups_may_be_signalled=True, goal_complete=False))
        admitted = wait_dependencies(root, cfg, helper, snapshots)
        verify(root, admission)

        def worker(gpu, arms):
            completed = []
            for arm in arms:
                verify(root, admission)
                check_dependencies(cfg, helper, snapshots, admitted)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
                command = [sys.executable, '-B', str(root / 'evaluate_transfer.py'), '--root', str(root),
                           '--arm', arm, '--max-seconds', str(cfg['max_seconds_per_arm'])]
                result = helper.run_child(command, env, output / (arm + '.log'), cfg['queue']['process_seconds_per_arm'])
                result.update(arm=arm, gpu=gpu)
                write(output / (arm + '_PROCESS.json'), result)
                completed.append(result)
                if result['exit_code'] != 0 or result['timed_out']:
                    break
            return completed

        write(output / 'STATUS.json', dict(status='evaluation', selected_step=admitted['selected_step'], goal_complete=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, gpu, arms) for gpu, arms in cfg['queue']['gpu_workers'].items()]
            outcomes = [result for future in futures for result in future.result()]
        if len(outcomes) != 4 or any(x['exit_code'] != 0 or x['timed_out'] for x in outcomes):
            raise RuntimeError('One or more transfer arms failed; no automatic retry')
        verify(root, admission)
        check_dependencies(cfg, helper, snapshots, admitted)
        result = helper.run_child([sys.executable, '-B', str(root / 'compare_transfer.py'), '--root', str(root)],
                                  os.environ.copy(), output / 'comparison.log', cfg['queue']['comparison_seconds'])
        write(output / 'COMPARISON_PROCESS.json', result)
        if result['exit_code'] != 0 or result['timed_out']:
            raise RuntimeError('Transfer comparison failed')
        verify(root, admission)
        check_dependencies(cfg, helper, snapshots, admitted)
        summary = read(root / 'SUMMARY.json')
        if (summary.get('schema') != 'gs_tower14_transfer_comparison_v1'
                or summary.get('engineering_status') != 'complete'
                or summary.get('comparison_eligible') is not True
                or summary.get('selected_step') != admitted['selected_step']
                or summary.get('goal_complete') is not False):
            raise RuntimeError('Transfer comparison has no matching complete engineering result')
        write(output / 'DONE.json', dict(status='complete', selected_step=admitted['selected_step'],
              summary_sha256=sha(root / 'SUMMARY.json'), admission=admission, outcomes=outcomes,
              training_performed=False, goal_complete=False))
        write(output / 'STATUS.json', dict(status='complete', goal_complete=False))
    except BaseException as error:
        signal.alarm(0)
        write(output / 'FAILED.json', dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
              outcomes=outcomes, old_processes_signalled=False, automatic_retry=False, goal_complete=False))
        raise
    finally:
        signal.alarm(0)


def self_test():
    cfg = read(Path(__file__).parent / 'protocol.json')
    def parser(snapshot):
        return snapshot.get('parsed') if snapshot else None
    live = dict(pid=431535, start_ticks='37974645', parsed=expected_parent(cfg))
    assert classify_parent(live, cfg, parser) == 'alive'
    assert classify_parent(None, cfg, parser) == 'exited'
    assert classify_parent(dict(live, start_ticks='1', parsed=('unrelated', [])), cfg, parser) == 'pid_reused_unrelated'
    try:
        classify_parent(dict(live, start_ticks='1'), cfg, parser)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Replacement parent was adopted')
    assert related(live, cfg, parser)
    assert not related(dict(parsed=('unrelated', [])), cfg, parser)
    renderer = str(PurePosixPath(cfg['main_root']).parent / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py')
    assert related(dict(parsed=(renderer, ['--output', cfg['preparation_root']+'/teacher/native448'])), cfg, parser)
    assert not related(dict(parsed=(renderer, ['--output', '/another/run'])), cfg, parser)
    assert not related(dict(parsed=(expected_parent(cfg)[0], ['--root', '/another/run'])), cfg, parser)
    print('CPU pipeline identity and source-specific worker matching passed; no proc/GPU/process launch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.root is None:
        parser.error('--root required')
    else:
        main(args.root)
