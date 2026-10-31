"""One bounded v7 postfit queue; wait in the program, never in an agent loop.

Only this queue's freshly created diagnostic process groups may be signalled.
The immutable main pipeline and shared jobs are observed, never controlled.
--self-test imports only the standard library and uses no /proc or signals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
MAIN_MANIFEST = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
PROCESS_HELPER = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
TIMING_HELPER = 'ad43827c53817edc18c02a62be6cae527dce0919cb89a702f8caf6b42c7ecd0f'
PARENT_PID, PARENT_START_TICKS = 429150, '37740478'
ARMS = ('z_rgb', 'constant_rgb', 'z_delta', 'constant_delta')
SOURCE_NAME = 'gaussiangpt_neoverse_native_z_20261031_v6'
ACCELERATION_NAME = 'gaussiangpt_neoverse_step_acceleration_20261031_v2'
JOBS = (('nearby_view', 'nearby_view_probe.py', 'nearby_view_v7', 'gs_v7_nearby_teacher_h9_v1'),
        ('render_comparison', 'render_six_panel.py', 'render_comparison_v7', 'gs_native_v7_six_panel_render_v1'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def write(path, value):
    """Atomic updates are confined to this queue's exclusive output directory."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def command_parts(snapshot):
    """Parse only direct Python script execution; shell/-c/substrings never match."""
    if not isinstance(snapshot, dict):
        return None
    argv = snapshot.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        return None
    executable = PurePosixPath(argv[0].replace('\\', '/')).name
    if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?', executable):
        return None
    index = 1
    while index < len(argv) and argv[index] in ('-B', '-u'):
        index += 1
    if index == len(argv) or argv[index].startswith('-'):
        return None
    return argv[index], argv[index + 1:]


def root_text(root):
    return str(root).replace('\\', '/').rstrip('/')


def matches_pipeline(snapshot, root):
    parsed = command_parts(snapshot)
    return bool(parsed and parsed == (root_text(root) + '/pipeline.py', ['--root', root_text(root)]))


def classify_parent(snapshot, root):
    """Pure classifier: never adopt a replacement PID as the admitted parent."""
    if snapshot is None:
        return 'exited'
    if snapshot.get('pid') != PARENT_PID or snapshot.get('stable', True) is not True:
        return 'identity_mismatch'
    if str(snapshot.get('start_ticks')) != PARENT_START_TICKS:
        return 'pid_reused'
    return 'alive' if matches_pipeline(snapshot, root) else 'identity_mismatch'


def matches_main_worker(snapshot, root):
    """Pure matcher for this root's runner.py/compare.py; never other roots."""
    parsed = command_parts(snapshot)
    if parsed is None:
        return False
    script, arguments = parsed
    if script not in (root_text(root) + '/runner.py', root_text(root) + '/compare.py'):
        return False
    if arguments.count('--root') != 1:
        return False
    position = arguments.index('--root')
    return position + 1 < len(arguments) and arguments[position + 1] == root_text(root)


def terminal_decision(done, summary, selection, failed=False):
    """Pure engineering gate. Scientific screen/hypothesis outcomes are irrelevant."""
    def reject(reason):
        return dict(eligible=False, reason=reason, selected_step=None)
    if failed:
        return reject('main_failure_receipt_present')
    if not all(isinstance(value, dict) for value in (done, summary, selection)):
        return reject('missing_terminal_documents')
    step = selection.get('selected_step')
    if (selection.get('schema') != 'gs_temporal_four_arm_selection_v1'
            or selection.get('status') != 'selected' or type(step) is not int or step not in (200, 300, 400)
            or selection.get('code_manifest_sha256') != MAIN_MANIFEST
            or set(selection.get('checkpoints', {})) != set(ARMS)):
        return reject('invalid_common_checkpoint_selection')
    if any(PurePosixPath(value.get('path', '')).name != f'step_{step:04d}.pt'
           for value in selection['checkpoints'].values()):
        return reject('selected_checkpoint_is_not_a_saved_step')
    if (done.get('status') != 'complete' or done.get('selected_step') != step
            or done.get('summary') != 'SUMMARY.json'):
        return reject('main_queue_not_complete')
    if (summary.get('schema') != 'gs_temporal_four_arm_comparison_v7'
            or summary.get('engineering_status') != 'complete'
            or summary.get('comparison_eligible') is not True
            or summary.get('code_manifest_sha256') != MAIN_MANIFEST
            or summary.get('selected_step') != step or summary.get('checkpoint_selection') != selection):
        return reject('main_engineering_summary_not_eligible')
    return dict(eligible=True, reason='complete_engineering_terminal', selected_step=step)


def protocol_contract(cfg):
    fixed = dict(schema='gs_native_v7_postfit_queue_protocol_v1', main_manifest_sha256=MAIN_MANIFEST,
        parent_pipeline_pid=PARENT_PID, parent_pipeline_start_ticks=PARENT_START_TICKS,
        parent_script='pipeline.py', wait_interval_seconds=60, wait_timeout_seconds=10000,
        queue_outer_timeout_seconds=11500, job_internal_seconds=600, job_process_seconds=650,
        gpu_loss_modes={'0': 'rgb', '1': 'delta'}, job_order=['nearby_view', 'render_comparison'],
        process_helper_sha256=PROCESS_HELPER, timing_helper_sha256=TIMING_HELPER,
        retry=False, goal_complete=False)
    if any(cfg.get(key) != value for key, value in fixed.items()):
        raise ValueError('Postfit fixed identity/budget/source/order contract changed')


def verify(root, expected=None, require_validation=True):
    """Verify exact manifests without importing main code or querying training."""
    source, acceleration = root.parent / SOURCE_NAME, root.parent / ACCELERATION_NAME
    allowed = (root, source, acceleration)
    if sha(root / 'CODE_MANIFEST.json') != MAIN_MANIFEST:
        raise ValueError('Wrong immutable v7 main manifest')
    postfit_hash = sha(root / 'POSTFIT_MANIFEST.json')
    manifests = {}
    for name in ('CODE_MANIFEST.json', 'POSTFIT_MANIFEST.json'):
        manifest = read(root / name)
        if not manifest:
            raise ValueError('Empty source manifest')
        for relative, digest in manifest.items():
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError('Manifest entries must be relative paths')
            path = (root / relative).resolve()
            if not any(path.is_relative_to(base) for base in allowed) or sha(path) != digest:
                raise ValueError('Immutable source hash mismatch: ' + name + ': ' + relative)
        manifests[name] = manifest
    required = {root / 'CODE_MANIFEST.json', source / 'analysis_code/postfit_queue.py',
                source / 'analysis_code/render_comparison.py'}
    required.update(path.resolve() for path in (root / 'postfit_code').iterdir()
                    if path.is_file() and path.suffix in ('.py', '.md', '.json'))
    named = {(root / relative).resolve() for relative in manifests['POSTFIT_MANIFEST.json']}
    if not required.issubset(named):
        raise ValueError('Postfit manifest omits current code/docs/protocol or reused helpers')
    for filename in ('queue_postfit.py', 'POSTFIT_PROTOCOL.json', 'EXECUTION.md', 'validate_postfit.py',
                     'render_six_panel.py', 'render_six_panel.md', 'nearby_view_probe.py', 'nearby_view_probe.md'):
        if root / 'postfit_code' / filename not in named:
            raise ValueError('Required postfit source is not bound: ' + filename)
    # The v6 code manifest is itself pinned by the v7 manifest. Check its entries
    # transitively as well, while allowing only that explicitly named sibling.
    for relative, digest in read(source / 'CODE_MANIFEST.json').items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or sha(path) != digest:
            raise ValueError('Pinned v6 constructor/renderer source changed: ' + relative)
    if (sha(source / 'analysis_code/postfit_queue.py') != PROCESS_HELPER
            or sha(source / 'analysis_code/render_comparison.py') != TIMING_HELPER):
        raise ValueError('Previously verified helper bytes changed')
    cfg = read(root / 'postfit_code/POSTFIT_PROTOCOL.json')
    protocol_contract(cfg)
    current = dict(main_manifest_sha256=MAIN_MANIFEST, postfit_manifest_sha256=postfit_hash,
                   postfit_protocol_sha256=sha(root / 'postfit_code/POSTFIT_PROTOCOL.json'))
    if require_validation:
        if (root / 'POSTFIT_VALIDATION_FAILED.json').exists():
            raise ValueError('CPU document witness contains a failure receipt')
        witness = read(root / 'POSTFIT_VALIDATION.json')
        if (witness.get('schema') != 'gs_v7_postfit_cpu_validation_v1' or witness.get('status') != 'passed'
                or witness.get('gpu_executed') is not False or witness.get('cpu_only') is not True
                or witness.get('main_manifest_sha256') != MAIN_MANIFEST
                or witness.get('postfit_manifest_sha256') != postfit_hash):
            raise ValueError('Exact postfit source has no successful CPU-only document witness')
        current['validation_sha256'] = sha(root / 'POSTFIT_VALIDATION.json')
    if expected is not None and current != expected:
        raise ValueError('Admission manifests/protocol/CPU witness changed after queue startup')
    return cfg, current


def main_workers(root, helper):
    workers, replacements = [], []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        snapshot = helper.process_snapshot(int(directory.name))
        if snapshot is None:
            continue
        if matches_main_worker(snapshot, root):
            workers.append(dict(pid=snapshot['pid'], start_ticks=snapshot['start_ticks'],
                                script=PurePosixPath(command_parts(snapshot)[0]).name))
        elif snapshot['pid'] != PARENT_PID and matches_pipeline(snapshot, root):
            replacements.append(dict(pid=snapshot['pid'], start_ticks=snapshot['start_ticks']))
    return sorted(workers, key=lambda item: item['pid']), replacements


def read_terminal(root):
    documents = {}
    paths = dict(done=root / 'queue/DONE.json', summary=root / 'SUMMARY.json', selection=root / 'SELECTION.json')
    for name, path in paths.items():
        documents[name] = read(path) if path.is_file() else None
    decision = terminal_decision(**documents, failed=(root / 'queue/FAILED.json').exists())
    if not decision['eligible']:
        raise RuntimeError('No postfit GPU admission: ' + decision['reason'])
    if documents['selection'].get('protocol_sha256') != sha(root / 'protocol.json'):
        raise RuntimeError('Main selection belongs to a different protocol')
    return dict(**documents, decision=decision,
                hashes={name: sha(path) for name, path in paths.items()})


def checked_parent_snapshot(helper):
    snapshot = helper.process_snapshot(PARENT_PID)
    if snapshot is not None:
        return snapshot
    # The reused helper maps PermissionError to None. For the explicitly
    # admitted PID only, distinguish exit/zombie from unreadable/live identity.
    directory = Path('/proc') / str(PARENT_PID)
    try:
        command = (directory / 'cmdline').read_bytes()
        stat = (directory / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # PermissionError deliberately propagates: it is not evidence of exit.
    fields = stat[stat.rfind(')') + 2:].split()
    if not fields or (command and fields[0] != 'Z'):
        raise RuntimeError('Admitted PID snapshot was unavailable while its process still exists')
    return None


def observe_main(root, helper):
    snapshot = checked_parent_snapshot(helper)
    identity = classify_parent(snapshot, root)
    if identity == 'identity_mismatch':
        raise RuntimeError('Admitted parent PID/start/command identity mismatch; no GPU admission')
    if identity == 'pid_reused' and matches_pipeline(snapshot, root):
        raise RuntimeError('Same-root pipeline was replaced under a reused PID; never adopt it')
    workers, replacements = main_workers(root, helper)
    if replacements:
        raise RuntimeError('A different same-root pipeline exists; no GPU admission and no process is signalled')
    return dict(parent_identity=identity, parent_alive=identity == 'alive',
                observed_parent_start_ticks=None if snapshot is None else snapshot['start_ticks'], workers=workers)


def wait_for_main(root, output, cfg, helper):
    begun = time.monotonic()
    while True:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping before main completion')
        observation = observe_main(root, helper)
        write(output / 'STATUS.json', dict(status='waiting_for_main', pid=os.getpid(), checked_at=timestamp(),
              waited_seconds=time.monotonic() - begun, **observation, goal_complete=False))
        if not observation['parent_alive'] and not observation['workers']:
            terminal = read_terminal(root)
            write(output / 'MAIN_TERMINAL.json', dict(status='eligible', observed_at=timestamp(),
                  observation=observation, decision=terminal['decision'], source_hashes=terminal['hashes'],
                  scientific_outcomes_do_not_gate=True, goal_complete=False))
            return terminal
        remaining = cfg['wait_timeout_seconds'] - (time.monotonic() - begun)
        if remaining <= 0:
            raise TimeoutError('Main-process wait exceeded10000 seconds; nothing restarted or stopped')
        # The submitted program owns this wait. A conversational status request
        # does not cancel/recreate the queue or require an agent to poll it.
        helper.STOP_REQUESTED.wait(min(cfg['wait_interval_seconds'], remaining))


def unchanged_terminal(root, terminal, helper):
    observation = observe_main(root, helper)
    if observation['parent_alive'] or observation['workers']:
        raise RuntimeError('Main work is active again; refusing to launch a postfit child')
    current = read_terminal(root)
    if current['hashes'] != terminal['hashes']:
        raise RuntimeError('Main terminal selection/summary/DONE changed during postfit')


def child_summary(root, destination, job, loss_mode, terminal):
    name, _, _, schema = job
    if (destination / 'FAILED.json').exists():
        raise ValueError('Postfit child wrote FAILED.json')
    path = destination / 'SUMMARY.json'
    summary = read(path)
    selected = terminal['selection']
    if (summary.get('schema') != schema or summary.get('status') != 'complete'
            or summary.get('loss_mode') != loss_mode or summary.get('selected_step') != selected['selected_step']
            or summary.get('optimizer_updates') != 0 or summary.get('goal_complete') is not False
            or summary.get('not_ar_rollout') is not True
            or summary.get('all_initial_latent_points_retained') is not True
            or summary.get('native_internal_hard_pruning_retained') is not True
            or not isinstance(summary.get('gpu_peak_gib'), (int, float))
            or not math.isfinite(summary['gpu_peak_gib']) or summary['gpu_peak_gib'] > 48):
        raise ValueError('Postfit child summary contract failed: ' + name)
    for arm in ('z_' + loss_mode, 'constant_' + loss_mode):
        if summary.get('checkpoints', {}).get(arm) != selected['checkpoints'][arm]:
            raise ValueError('Postfit child used a different checkpoint')
    inputs = summary.get('input_sha256', {})
    for file, key in ((root / 'CODE_MANIFEST.json', None), (root / 'SELECTION.json', 'selection'),
                      (root / 'SUMMARY.json', 'summary')):
        expected = MAIN_MANIFEST if key is None else terminal['hashes'][key]
        if inputs.get(str(file)) != expected:
            raise ValueError('Child does not bind the admitted main terminal input')
    if name == 'nearby_view':
        scope = summary.get('scope', {})
        if (scope.get('horizon') != 9 or scope.get('starts') != 168
                or scope.get('unique_physical_targets') != 168 or scope.get('target_range') != [9, 176]
                or scope.get('resolution_hw') != [252, 448]
                or summary.get('latent_window_decode_calls') != 672
                or summary.get('native_render_calls') != 1848):
            raise ValueError('Nearby-view child did not finish the fixed complete h9 scope')
    else:
        if (summary.get('parity_pass') is not True or summary.get('physical_frames') != 177
                or summary.get('unique_physical_frames') != 177
                or summary.get('panel_native_size_wh') != [448, 252]
                or summary.get('video_size_wh') != [1344, 612]
                or summary.get('latent_window_decode_calls') != 80 or summary.get('native_render_calls') != 885):
            raise ValueError('Six-panel child did not finish native-size177/parity replay')
    artifacts = summary.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Child has no artifact hashes')
    for relative, digest in artifacts.items():
        file = (destination / relative).resolve()
        if not file.is_relative_to(destination) or sha(file) != digest:
            raise ValueError('Child artifact hash mismatch: ' + relative)
    return dict(summary_path=str(path), summary_sha256=sha(path), selected_step=summary['selected_step'],
                artifact_sha256=artifacts, status='complete')


def run_loss_mode(root, output, loss_mode, gpu, cfg, admission, terminal, helper):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
    outcomes = []
    for job in JOBS:
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopping before next child')
        verify(root, admission)
        unchanged_terminal(root, terminal, helper)
        name, script, directory, _ = job
        destination = root / 'diagnostics' / directory / loss_mode
        if destination.exists():
            raise RuntimeError('Output already exists; no overwrite/retry: ' + str(destination))
        command = [sys.executable, '-B', str(root / 'postfit_code' / script), '--root', str(root),
                   '--loss-mode', loss_mode, '--output', str(destination),
                   '--max-seconds', str(cfg['job_internal_seconds'])]
        receipt = output / f'{loss_mode}_{name}_PROCESS.json'
        try:
            outcome = helper.run_child(command, env, output / f'{loss_mode}_{name}.log', cfg['job_process_seconds'])
        except Exception as error:
            outcome = dict(command=command, pid=None, exit_code=None, timed_out=False,
                           process_error=repr(error), status='failed')
        outcome.update(loss_mode=loss_mode, gpu=gpu, job=name, output=str(destination),
                       postfit_manifest_sha256=admission['postfit_manifest_sha256'],
                       main_terminal_hashes=terminal['hashes'], goal_complete=False)
        write(receipt, outcome)  # Preserve even a nonzero/timeout outcome before validation.
        if outcome['exit_code'] != 0 or outcome['timed_out']:
            return dict(status='failed', loss_mode=loss_mode, jobs=outcomes + [outcome],
                        reason='child_process_failed_or_timed_out', next_job_skipped=True)
        try:
            outcome['verified_result'] = child_summary(root, destination, job, loss_mode, terminal)
            verify(root, admission)
        except Exception as error:
            outcome.update(result_validation_error=repr(error), status='failed')
            write(receipt, outcome)
            return dict(status='failed', loss_mode=loss_mode, jobs=outcomes + [outcome],
                        reason='child_result_validation_failed', next_job_skipped=True)
        outcome['status'] = 'complete'
        write(receipt, outcome)
        outcomes.append(outcome)
    return dict(status='complete', loss_mode=loss_mode, gpu=gpu, jobs=outcomes)


def main(root):
    root = root.resolve()
    output = root / 'queue_postfit'
    output.mkdir(parents=False, exist_ok=False)
    helper, outcomes = None, []

    def admission_deadline(signum, frame):
        raise TimeoutError('Postfit queue exceeded11500 seconds during source admission')

    try:
        # Start the whole-queue budget before source admission. Before the
        # trusted helper is loaded, no child exists and this handler kills none.
        signal.signal(signal.SIGALRM, admission_deadline)
        signal.alarm(11500)
        cfg, admission = verify(root)
        helper_path = root.parent / SOURCE_NAME / 'analysis_code/postfit_queue.py'
        spec = importlib.util.spec_from_file_location('_v6_owned_child_groups', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        # Changing the handler preserves the already-running outer timer.
        write(output / 'RECEIPT.json', dict(schema='gs_native_v7_postfit_queue_receipt_v1', status='waiting',
              pid=os.getpid(), started_at=timestamp(), admission=admission, protocol=cfg,
              expected_parent=dict(pid=PARENT_PID, start_ticks=PARENT_START_TICKS,
                  script=str(root / 'pipeline.py'), arguments=['--root', str(root)]),
              only_own_new_child_groups_may_be_signalled=True, goal_complete=False))
        terminal = wait_for_main(root, output, cfg, helper)
        verify(root, admission)  # Required recheck immediately after waiting.
        unchanged_terminal(root, terminal, helper)
        write(output / 'STATUS.json', dict(status='postfit_running', pid=os.getpid(),
              started_at=timestamp(), selected_step=terminal['decision']['selected_step'], goal_complete=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_loss_mode, root, output, mode, int(gpu), cfg, admission, terminal, helper)
                       for gpu, mode in cfg['gpu_loss_modes'].items()]
            for future in futures:
                outcomes.append(future.result())
        verify(root, admission)
        unchanged_terminal(root, terminal, helper)
        if len(outcomes) != 2 or any(item['status'] != 'complete' for item in outcomes):
            raise RuntimeError('One or more bounded postfit jobs failed; no retry or success receipt')
        result = dict(schema='gs_native_v7_postfit_queue_done_v1', status='complete', finished_at=timestamp(),
              selected_step=terminal['decision']['selected_step'], admission=admission,
              main_terminal_hashes=terminal['hashes'], outcomes=outcomes, optimizer_updates=0,
              scientific_outcomes_do_not_gate=True, goal_complete=False,
              interpretation='Complete diagnostics and replay only; not a scientific success or AR claim')
        write(output / 'STATUS.json', dict(status='complete', finished_at=timestamp(), goal_complete=False))
        write(output / 'DONE.json', result)
    except BaseException as error:
        signal.alarm(0)
        write(output / 'FAILED.json', dict(schema='gs_native_v7_postfit_queue_failure_v1', status='failed',
              error=repr(error), traceback=traceback.format_exc(), outcomes=outcomes,
              finished_at=timestamp(), automatic_retry=False, main_processes_signalled=False, goal_complete=False))
        write(output / 'STATUS.json', dict(status='failed', error=repr(error), finished_at=timestamp(), goal_complete=False))
        raise
    finally:
        signal.alarm(0)


def cpu_self_test():
    import copy
    root = '/fixture/v7'
    live = dict(pid=PARENT_PID, start_ticks=PARENT_START_TICKS,
                argv=['/env/bin/python', '-B', root + '/pipeline.py', '--root', root])
    assert classify_parent(live, root) == 'alive'
    assert classify_parent(None, root) == 'exited'
    assert classify_parent(dict(live, start_ticks='37740479'), root) == 'pid_reused'
    assert classify_parent(dict(live, argv=['/env/bin/python', '-c', root + '/pipeline.py']), root) == 'identity_mismatch'
    assert classify_parent(dict(live, argv=live['argv'] + ['--unexpected']), root) == 'identity_mismatch'
    assert classify_parent(dict(live, pid=PARENT_PID + 1), root) == 'identity_mismatch'
    assert classify_parent(dict(live, stable=False), root) == 'identity_mismatch'
    assert classify_parent(dict(live, argv=live['argv'][:-1] + [root + '_other']), root) == 'identity_mismatch'
    worker = dict(pid=3, start_ticks='4', argv=['python3.10', '-B', root + '/runner.py',
                  '--root', root, '--mode', 'eval', '--arm', 'z_rgb', '--checkpoint-step', '200'])
    assert matches_main_worker(worker, root)
    assert matches_main_worker(dict(worker, argv=['python', root + '/compare.py', '--root', root]), root)
    assert not matches_main_worker(dict(worker, argv=['sh', '-c', ' '.join(worker['argv'])]), root)
    assert not matches_main_worker(dict(worker, argv=['python', root + '/runner.py', '--root', root + 'x']), root)
    assert not matches_main_worker(dict(worker, argv=worker['argv'] + ['--root', root]), root)
    selection = dict(schema='gs_temporal_four_arm_selection_v1', status='selected', selected_step=200,
        code_manifest_sha256=MAIN_MANIFEST,
        checkpoints={arm: dict(path=root + '/' + arm + '/step_0200.pt', sha256='fixture') for arm in ARMS})
    done = dict(status='complete', selected_step=200, summary='SUMMARY.json')
    summary = dict(schema='gs_temporal_four_arm_comparison_v7', engineering_status='complete',
        comparison_eligible=True, selected_step=200, code_manifest_sha256=MAIN_MANIFEST,
        checkpoint_selection=selection, z_contribution_screens={'rgb': {'passed': False}, 'delta': {'passed': False}},
        delta_hypothesis_supported=False)
    assert terminal_decision(done, summary, selection)['eligible'] is True
    assert terminal_decision(done, summary, selection, failed=True)['eligible'] is False
    assert terminal_decision(None, summary, selection)['eligible'] is False
    for key, value in [('engineering_status', 'failed'), ('comparison_eligible', False),
                       ('selected_step', 300), ('code_manifest_sha256', 'wrong')]:
        assert not terminal_decision(done, dict(summary, **{key: value}), selection)['eligible']
    assert not terminal_decision(dict(done, status='running'), summary, selection)['eligible']
    fake = copy.deepcopy(selection)
    fake['checkpoints']['z_rgb']['path'] = root + '/z_rgb/final.pt'
    assert not terminal_decision(done, dict(summary, checkpoint_selection=fake), fake)['eligible']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print('CPU_SELF_TEST_PASSED: exact PID/start/argv, PID reuse, root-specific workers, '
          'engineering terminal gate, scientific-screen failure accepted; no proc/signals/GPU/files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
    elif args.root is None:
        parser.error('--root is required')
    else:
        main(args.root)
