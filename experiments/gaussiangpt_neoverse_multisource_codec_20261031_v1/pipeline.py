"""One bounded, fail-closed multisource codec queue; no retries or old-run writes."""
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
RUN = 'gaussiangpt_neoverse_multisource_codec_20261031_v1'
BASE = PurePosixPath('/data/CJH/AGS-V6Direct/outputs')
RUNTIME = '/data/CJH/envs/gaussiangpt/bin/python'
HELPER = '../gaussiangpt_neoverse_native_z_20261031_v6/analysis_code/postfit_queue.py'
HELPER_SHA = 'ae6606ce7666b77c8b797c523a08467438c9bc41bd2e15526bcf2b44fba666f2'
PREPARE_HELPER = '../gaussiangpt_neoverse_tower14_transfer_20261031_v1/prepare_frozen_codec.py'
PREPARE_HELPER_SHA = '3f26fdc51885eb7e8b2185446e891dd5dbda30d5be83cd4900b7db401e1e6c85'
EXTERNAL_HELPERS = {HELPER: HELPER_SHA, PREPARE_HELPER: PREPARE_HELPER_SHA}
ARMS = ('decoder_only', 'full_codec')
REQUIRED = {'pipeline.py', 'test_pipeline.py', 'sources.py', 'test_sources.py', 'worker.py',
            'test_worker.py', 'compare.py', 'test_compare.py', 'validate.py', 'protocol.json',
            'EXECUTION.md', HELPER, PREPARE_HELPER}
QUEUE = dict(wait_seconds=15000, wait_interval_seconds=60, source_admission_seconds=300,
             preflight_process_seconds=930, baseline_process_seconds=1830, train_process_seconds=1830,
             eval_process_seconds=1830, comparison_seconds=300, outer_seconds=22400, runtime=RUNTIME,
             gpu_workers={'0': 'decoder_only', '1': 'full_codec'},
             job_order=['wait_for_exact_tower10_terminal', 'bind_all_sources_cpu',
                        'discarded_two_arm_preflight_gpu0', 'shared_baseline_gpu0',
                        'two_train_parallel', 'two_evaluations_parallel', 'compare_cpu'])
OUTCOME_LOCK = threading.Lock()


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
    from sources import validate_protocol
    validate_protocol(cfg)
    if cfg.get('queue') != QUEUE or cfg.get('automatic_retry') is not False or cfg.get('goal_complete') is not False:
        raise ValueError('Fixed queue timeouts/order/runtime or no-retry scope changed')
    expected = {'cpu_threads': 6, 'peak_limit_gib': 48, 'minimum_free_gib': 55,
                'gpu_assignment': {'decoder_only': 0, 'full_codec': 1}}
    if cfg.get('resources') != expected:
        raise ValueError('Fixed two-GPU assignment or admission limits changed')
    for section, values in {'preflight': {'max_seconds': 900, 'updates_per_arm': 2,
            'force_target_sequence': [True, False], 'frames_per_clip': [0, 44], 'discard_parameters': True},
            'train': {'steps': 300, 'batch_size': 2, 'one_frame_per_train_clip_each_step': True,
            'force_target_steps': 240, 'no_force_steps': 60, 'save_steps': [150, 300], 'max_seconds': 1800},
            'evaluation': {'max_seconds': 1800, 'baseline_once': True, 'frames_per_clip': 177,
            'clips': 3, 'records_per_model': 531, 'save_reencoded_states': True, 'target_forcing': False}}.items():
        for key, value in values.items():
            if type(cfg.get(section, {}).get(key)) is not type(value) or cfg[section][key] != value:
                raise ValueError('Fixed queue phase contract changed: ' + section + '.' + key)


def verify(root, expected=None, cpu=True):
    """Hash local code/protocol/proof only; cpu=False never accesses remote data."""
    root = Path(root).resolve()
    manifest = read(root / 'CODE_MANIFEST.json')
    required = REQUIRED | {p.name for p in root.glob('*.py')} | {p.name for p in root.glob('*.md')}
    if not required.issubset(manifest) or any(manifest.get(path) != digest for path, digest in EXTERNAL_HELPERS.items()):
        raise ValueError('Manifest omits queue sources/tests/protocol/docs or the exact process helper')
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if (not re.fullmatch('[0-9a-f]{64}', str(digest)) or PurePosixPath(relative).is_absolute()
                or Path(relative).is_absolute() or '\\' in relative
                or (relative not in EXTERNAL_HELPERS and not path.is_relative_to(root))
                or not path.is_relative_to(root.parent) or sha(path) != digest):
            raise ValueError('Immutable source mismatch: ' + relative)
    cfg = read(root / 'protocol.json')
    protocol_contract(cfg)
    admission = dict(code_manifest_sha256=sha(root / 'CODE_MANIFEST.json'),
                     protocol_sha256=sha(root / 'protocol.json'))
    if cpu:
        proof = read(root / 'CPU_VALIDATION.json')
        if (proof.get('status') != 'passed' or proof.get('gpu_executed') is not False
                or any(proof.get(key) != value for key, value in admission.items())):
            raise ValueError('Exact queue manifest/protocol has no passed CPU_VALIDATION')
        admission['cpu_verification_sha256'] = sha(root / 'CPU_VALIDATION.json')
    if expected is not None and admission != expected:
        raise ValueError('Admitted code/protocol/CPU verification changed')
    return cfg, admission


def load_helper(root):
    path = (root / HELPER).resolve(strict=True)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != HELPER_SHA:
        raise ValueError('Immutable process helper changed')
    spec = importlib.util.spec_from_file_location('_multisource_owned_groups', path)
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


def dependency_worker(snapshot, dependency):
    parts = command_parts(snapshot)
    if parts is None:
        return False
    script, arguments = parts
    root = dependency['root']
    if script == root + '/prepare.py':
        flag, destination = '--root', root
    elif script == str(BASE / 'neoverse_tower177_fps30_20261029_v1/render_neoverse.py'):
        flag, destination = '--output', root + '/teacher/buildingblock_tower_10_176'
    else:
        return False
    if arguments.count(flag) != 1:
        return False
    index = arguments.index(flag)
    return index + 1 < len(arguments) and arguments[index + 1] == destination


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


def read_terminal(dependency):
    from sources import verify_tower10
    records = {}
    summary = verify_tower10(Path(dependency['root']), records)
    return dict(parent_terminal='DONE', summary_sha256=sha(Path(dependency['root']) / 'SUMMARY.json'),
                preparation_contract_passed=summary['preparation_contract_passed'],
                hashes={str(path): row['sha256'] for path, row in records.items()})


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
            raise TimeoutError('15000-second dependency wait expired; no shared process stopped or restarted')
        helper.STOP_REQUESTED.wait(min(cfg['queue']['wait_interval_seconds'], remaining))


def unchanged_dependency(helper, cfg, terminal):
    if not observe(helper, cfg['dependency'])['all_exited'] or read_terminal(cfg['dependency']) != terminal:
        raise RuntimeError('Dependency resumed or its exact completed receipt changed')


def source_admission(root):
    """Runs as one owned CPU child under the queue's 300-second process limit."""
    from sources import load_verified
    _, admission = verify(root)
    output = root / 'queue/SOURCE_ADMISSION.json'
    if output.exists():
        raise FileExistsError('Source admission already exists; no retry')
    begun = time.monotonic()
    bundle = load_verified(root, import_modules=False)
    elapsed = time.monotonic() - begun
    if elapsed >= 300:
        raise TimeoutError('Source admission exceeded 300 seconds')
    write(output, dict(status='passed', gpu_executed=False, binding_sha256=bundle.binding_sha256,
          cache_records=531, files=len(bundle.records), admission=admission, elapsed_seconds=elapsed,
          initial_checkpoint_sha256=bundle.binding['initial_codec_sha256'], goal_complete=False))


def source_contract(root, admission):
    receipt = read(root / 'queue/SOURCE_ADMISSION.json')
    if (receipt.get('status') != 'passed' or receipt.get('gpu_executed') is not False
            or receipt.get('cache_records') != 531 or receipt.get('admission') != admission
            or receipt.get('binding_sha256') != sha(root / 'DATA_BINDING.json')):
        raise ValueError('Full source admission is absent or not bound to this exact queue')
    binding = read(root / 'DATA_BINDING.json')
    if (binding.get('schema') != 'gs_multisource_codec_data_binding_v1'
            or binding.get('frames_per_clip') != 177 or binding.get('source_points_per_frame') != 112896
            or binding.get('historical_latents_reused') is not False):
        raise ValueError('Invalid complete-frame data binding')
    return dict(source_admission_sha256=sha(root / 'queue/SOURCE_ADMISSION.json'),
                data_binding_sha256=sha(root / 'DATA_BINDING.json'))


def expected_evidence(root, binding_sha256):
    from sources import CODEC_SHA, balanced_plan
    plan = balanced_plan(20261031, 300)
    return {'binding_sha256': binding_sha256, 'initial_checkpoint_sha256': CODEC_SHA,
            'protocol_sha256': sha(root / 'protocol.json'), 'worker_sha256': sha(root / 'worker.py'),
            'sources_sha256': sha(root / 'sources.py'),
            'plan_sha256': hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest(),
            'inherited_codec_steps': 3000, 'additional_codec_updates': 300, 'seed': 20261031,
            'batch_frames': 2, 'forcing_updates': 240, 'unforced_updates': 60,
            'goal_complete': False, 'scope': 'Codec reconstruction comparison only; no action/transfer conclusion'}


def preflight_contract(root, evidence):
    gate = read(root / 'PREFLIGHT.json')
    arms = gate.get('arms', {})
    if (gate.get('status') != 'passed' or set(arms) != set(ARMS) or gate.get('source') != evidence
            or gate.get('initial_arms_identical') is not True or gate.get('goal_complete') is not False):
        raise ValueError('Exact complete two-arm preflight is required')
    for arm in ARMS:
        value = arms[arm]
        if (not isinstance(value, dict) or value.get('passed') is not True
                or value.get('discarded_updates') != 2 or value.get('production_updates') != 0
                or not re.fullmatch('[0-9a-f]{64}', str(value.get('initial_model_sha256')))):
            raise ValueError('Preflight must discard two B2 updates per arm with zero production updates')
    if arms[ARMS[0]]['initial_model_sha256'] != arms[ARMS[1]]['initial_model_sha256']:
        raise ValueError('Preflight arms did not initialize identically')
    return sha(root / 'PREFLIGHT.json')


def train_contract(root, arm, evidence):
    folder = root / 'arms' / arm
    if (folder / 'FAILURE.json').exists():
        raise ValueError('Training arm has a failure receipt: ' + arm)
    done = read(folder / 'TRAIN_DONE.json')
    fixed = {'status': 'complete', 'arm': arm, 'source': evidence, 'steps': 300,
             'inherited_codec_steps': 3000, 'additional_codec_updates': 300, 'frames_seen': 354,
             'first_epoch_coverage_at_step': 177, 'no_target_forcing_steps': 60,
             'fresh_optimizer_verified': True, 'goal_complete': False}
    if any(type(done.get(k)) is not type(v) or done[k] != v for k, v in fixed.items()):
        raise ValueError('Training arm did not complete exact 300 updates/354-frame coverage: ' + arm)
    preflight = read(root / 'PREFLIGHT.json')
    if done.get('initial_model_sha256') != preflight['arms'][arm]['initial_model_sha256']:
        raise ValueError('Production training initial model differs from discarded preflight: ' + arm)
    if done.get('frozen_encoder_verified') is not (arm == 'decoder_only'):
        raise ValueError('Training arm encoder freeze policy differs: ' + arm)
    if arm == 'decoder_only' and done.get('initial_encoder_sha256') != done.get('final_encoder_sha256'):
        raise ValueError('Decoder-only arm changed its frozen encoder: ' + arm)
    for filename, key in [('step_0300.pt', 'checkpoint_sha256'), ('step_0150.pt', 'checkpoint_0150_sha256'),
                          ('trace.jsonl', 'trace_sha256')]:
        if sha(folder / filename) != done.get(key):
            raise ValueError('Training artifact hash mismatch: ' + arm + '/' + filename)
    return sha(folder / 'TRAIN_DONE.json')


def evaluation_contract(root, label, evidence):
    folder = root / 'evaluations' / label
    if (folder / 'FAILURE.json').exists():
        raise ValueError('Evaluation has a failure receipt: ' + label)
    done = read(folder / 'EVALUATION.json')
    fixed = {'status': 'evaluated', 'arm': label, 'frames': 531, 'frames_per_clip': 177,
             'additional_codec_updates': 0 if label == 'baseline' else 300, 'inherited_codec_steps': 3000,
             'source': evidence, 'every_packet_saved_replay_passed': True, 'all_frames_freshly_encoded': True,
             'decoder_eval': True, 'decoder_target_key': None, 'goal_complete': False}
    if any(type(done.get(k)) is not type(v) or done[k] != v for k, v in fixed.items()):
        raise ValueError('Evaluation must contain all 531 freshly encoded frames: ' + label)
    expected_checkpoint = (evidence['initial_checkpoint_sha256'] if label == 'baseline'
                           else read(root / 'arms' / label / 'TRAIN_DONE.json')['checkpoint_sha256'])
    if done.get('checkpoint_sha256') != expected_checkpoint:
        raise ValueError('Evaluation used another checkpoint: ' + label)
    for filename, key in [('targets.jsonl', 'targets_sha256'), ('state_manifest.jsonl', 'state_manifest_sha256'),
                          ('screenshot_manifest.jsonl', 'screenshot_manifest_sha256')]:
        if sha(folder / filename) != done.get(key):
            raise ValueError('Evaluation artifact hash mismatch: ' + label + '/' + filename)
    return sha(folder / 'EVALUATION.json')


def comparison_contract(root):
    summary = read(root / 'SUMMARY.json')
    if (summary.get('schema') != 'gs_multisource_codec_comparison_v1'
            or summary.get('engineering_status') != 'complete'
            or summary.get('comparison_eligibility', {}).get('eligible') is not True
            or summary.get('goal_complete') is not False or summary.get('production_codec_selected', 'missing') is not None
            or summary.get('action_z_trained') is not False or summary.get('state_lfq_enabled') is not False):
        raise ValueError('Comparison did not confirm complete engineering evidence within codec-only scope')
    return summary


def jobs(root, stage):
    prefix = [RUNTIME, '-B', '-u']
    if stage == 'source_admission':
        return [('source_admission', None, prefix + [str(root / 'pipeline.py'), '--root', str(root), '--mode', 'source-admission'], 300)]
    if stage in ('preflight', 'baseline'):
        return [(stage, 0, prefix + [str(root / 'worker.py'), '--root', str(root), '--mode', stage],
                 930 if stage == 'preflight' else 1830)]
    if stage in ('train', 'eval'):
        return [(stage + '_' + arm, gpu, prefix + [str(root / 'worker.py'), '--root', str(root),
                 '--mode', stage, '--arm', arm], 1830) for gpu, arm in enumerate(ARMS)]
    if stage == 'compare':
        return [('compare', None, prefix + [str(root / 'compare.py'), '--root', str(root)], 300)]
    raise ValueError('Unknown queue stage: ' + stage)


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
    root = Path(root).resolve()
    if root.as_posix() != (BASE / RUN).as_posix():
        raise ValueError('Require the dedicated remote queue root')
    output = root / 'queue'
    output.mkdir(exist_ok=False)
    helper, outcomes, admission, source_hashes, artifact_hashes = None, [], {}, {}, {}

    def startup_stop(signum, frame):
        raise TimeoutError('Queue stopped before helper admission or exceeded 22400 seconds')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, startup_stop)
        signal.alarm(22400)
        cfg, admission = verify(root)
        helper = load_helper(root)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(sig, helper.terminate_queue)
        write(output / 'RECEIPT.json', dict(status='waiting', pid=os.getpid(), started_at=timestamp(),
              admission=admission, dependency=cfg['dependency'], automatic_retry=False,
              only_own_new_session_groups_may_be_signalled=True, goal_complete=False))
        terminal = wait_for_parent(output, cfg, helper)

        def before_stage():
            verify(root, admission)
            unchanged_dependency(helper, cfg, terminal)
            if source_hashes and source_contract(root, admission) != source_hashes:
                raise RuntimeError('Frozen source binding/admission changed')
            for path, digest in artifact_hashes.items():
                if sha(path) != digest:
                    raise RuntimeError('Completed earlier phase receipt changed: ' + str(path))

        execute_stage(root, output, 'source_admission', helper, outcomes, before_stage)
        source_hashes = source_contract(root, admission)
        evidence = expected_evidence(root, source_hashes['data_binding_sha256'])
        before_stage()
        gpu_admission(output)
        execute_stage(root, output, 'preflight', helper, outcomes, before_stage)
        artifact_hashes[root / 'PREFLIGHT.json'] = preflight_contract(root, evidence)
        execute_stage(root, output, 'baseline', helper, outcomes, before_stage)
        artifact_hashes[root / 'evaluations/baseline/EVALUATION.json'] = evaluation_contract(root, 'baseline', evidence)
        execute_stage(root, output, 'train', helper, outcomes, before_stage)
        for arm in ARMS:
            artifact_hashes[root / 'arms' / arm / 'TRAIN_DONE.json'] = train_contract(root, arm, evidence)
        execute_stage(root, output, 'eval', helper, outcomes, before_stage)
        for arm in ARMS:
            artifact_hashes[root / 'evaluations' / arm / 'EVALUATION.json'] = evaluation_contract(root, arm, evidence)
        execute_stage(root, output, 'compare', helper, outcomes, before_stage)
        summary = comparison_contract(root)
        before_stage()
        if helper.STOP_REQUESTED.is_set():
            raise RuntimeError('Queue stopped before final complete receipt')
        done = dict(status='complete', engineering_status=summary['engineering_status'],
                    scientific_status=summary.get('scientific_status'), summary_sha256=sha(root / 'SUMMARY.json'),
                    admission=admission, source_hashes=source_hashes, outcomes=outcomes,
                    artifact_hashes={str(path): digest for path, digest in artifact_hashes.items()},
                    goal_complete=False, production_codec_selected=None, automatic_retry=False,
                    finished_at=timestamp())
        write(output / 'STATUS.json', dict(status='complete', engineering_status='complete', goal_complete=False))
        write(output / 'DONE.json', done)
    except BaseException as error:
        if helper is not None:
            stop_owned_children(helper)
        failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                       finished_at=timestamp(), admission=admission, source_hashes=source_hashes,
                       outcomes=outcomes, automatic_retry=False, old_processes_signalled=False, goal_complete=False)
        write(output / 'FAILED.json', failure)
        write(output / 'STATUS.json', dict(status='failed', error=repr(error), goal_complete=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', choices=('queue', 'source-admission'), default='queue')
    args = parser.parse_args()
    if args.mode == 'source-admission':
        source_admission(args.root.resolve())
    else:
        main(args.root)
