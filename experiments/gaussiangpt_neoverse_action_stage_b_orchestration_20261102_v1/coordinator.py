"""Conditionally coordinate stage-B training, evaluation, and native replay.

This program does not decide whether stage B is scientifically justified.
It requires an already completed human/agent visual review of stage A, and
the worker independently checks that review and all model/data provenance.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PROTOCOL = B_ROOT / 'B_PREDECLARED_PROTOCOL.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def verify_code(root, expected):
    root = Path(root)
    if sha(root / 'CODE.sha256') != expected:
        raise RuntimeError('Code manifest changed: ' + str(root))
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(root / name) != digest:
            raise RuntimeError('Bound source changed: ' + name)


def validate_config(config, draft):
    """Allow only the reviewed hash and draft status to change after A."""
    import copy
    candidate = copy.deepcopy(config)
    review_entry = candidate.get('stage_a_review', {})
    digest = review_entry.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError('A real stage-A review hash is required')
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise RuntimeError('Invalid review SHA-256') from error
    if candidate.get('status') != 'ready_after_stage_a_review':
        raise RuntimeError('Config is still a draft')
    review_entry['sha256'] = None
    candidate['status'] = draft['status']
    if candidate != draft:
        raise RuntimeError('The predeclared stage-B budget or source changed')
    return digest


def validate_review(config):
    entry = config['stage_a_review']
    path = Path(entry['path'])
    if not path.is_absolute() or not path.is_file() or sha(path) != entry['sha256']:
        raise RuntimeError('Stage-A review is missing or changed')
    review = read(path)
    starts = review.get('reviewed_starts', [])
    if (review.get('status') != 'reviewed' or review.get('decision') != 'stage_B'
            or not {35, 123}.issubset(set(starts)) or len(set(starts)) < 8):
        raise RuntimeError('Stage B requires the reviewed A branch and eight windows')
    evaluation = review['evaluation']
    path = Path(evaluation['path'])
    if not path.is_absolute() or sha(path) != evaluation['sha256']:
        raise RuntimeError('Reviewed stage-A evaluation is missing or changed')
    value = read(path)
    if (value.get('status') != 'complete' or value.get('window_predictions') != 1512
            or value.get('correction', {}).get('status') != 'applied'
            or value.get('source', {}).get('sweep') != 20):
        raise RuntimeError('Review must bind the corrected final A evaluation')
    return {'review_sha256': entry['sha256'], 'evaluation_sha256': evaluation['sha256']}


def execute_plan(train, evaluate_and_replay):
    """Train once, evaluate 5/10 on distinct GPUs, then evaluate 20."""
    train()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate_and_replay, 5, 0),
                   pool.submit(evaluate_and_replay, 10, 1)]
        # Observe both outcomes; a failure forbids the final evaluation.
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError('Intermediate evaluation/replay failed: ' + '; '.join(errors))
    evaluate_and_replay(20, 0)


def verify_training(folder, config):
    done = read(folder / 'TRAIN_DONE.json')
    if (done.get('status') != 'complete' or done.get('updates') != 7560
            or done.get('sweeps') != 20 or done.get('code_shape') != [168, 9, 16]
            or done.get('codec_and_initial_encoder_frozen') is not True
            or done.get('source', {}).get('config_sha256') != sha(folder.parent / 'CONFIG.json')
            or done.get('source', {}).get('stage_a_review', {}).get('review_sha256')
            != config['stage_a_review']['sha256']):
        raise RuntimeError('Training did not finish the fixed stage-B protocol')
    for sweep in (5, 10, 20):
        metadata = read(folder / f'sweep_{sweep:02d}.json')
        if (metadata.get('status') != 'complete' or metadata.get('sweep') != sweep
                or sha(folder / f'sweep_{sweep:02d}.pt') != metadata.get('sha256')):
            raise RuntimeError('Missing or changed paired checkpoint ' + str(sweep))
    return done


def verify_result(folder, sweep, replay_required):
    output = folder / f'sweep_{sweep:02d}' / 'free_z'
    evaluation = output / 'evaluation'
    status = read(output / 'STATUS.json')
    metrics = read(output / 'STAGE_METRICS.json')
    result = read(evaluation / 'EVALUATION.json')
    if (status.get('status') != 'complete' or metrics.get('status') != 'complete'
            or metrics.get('sweep') != sweep or result.get('status') != 'complete'
            or result.get('window_predictions') != 1512 or result.get('unique_future_frames') != 176
            or result.get('source', {}).get('sweep') != sweep
            or result.get('decoder_future_geometry_input') is not False
            or sha(evaluation / 'PACKETS.json') != result.get('packets_sha256')):
        raise RuntimeError('Incomplete or mismatched stage-B evaluation ' + str(sweep))
    replay_info = None
    if replay_required:
        replay = read(evaluation / 'REPLAY.json')
        if (replay.get('status') != 'passed' or replay.get('packets') != 104
                or replay.get('views') != 208 or replay.get('encoder_or_codec_loaded') is not False
                or replay.get('packets_sha256') != result['packets_sha256']):
            raise RuntimeError('Independent native replay failed for sweep ' + str(sweep))
        replay_info = {'sha256': sha(evaluation / 'REPLAY.json'), 'packets': replay['packets'],
                       'views': replay['views'], 'max_mae': replay['max_mae']}
    return {'sweep': sweep, 'evaluation_path': str(evaluation / 'EVALUATION.json'),
            'evaluation_sha256': sha(evaluation / 'EVALUATION.json'),
            'metrics_sha256': sha(output / 'STAGE_METRICS.json'),
            'correct_motion_mae': metrics['metrics']['unique_frame_equal']['correct']['motion_mae'],
            'anchor_motion_mae': metrics['anchor_motion_mae'],
            'original_two_window_anchor_motion_mae': metrics['original_two_window_anchor_motion_mae'],
            'preliminary_known_anchor_proxy_pass': metrics['preliminary_known_anchor_proxy_pass'],
            'replay': replay_info}


def coordinate(config_path, run_root):
    config_path = Path(config_path).resolve(strict=True)
    run_root = Path(run_root)
    if (not run_root.is_absolute() or run_root.parent.resolve(strict=True) != OUTPUTS.resolve(strict=True)
            or run_root.exists()):
        raise RuntimeError('Run must be a new direct child of the server outputs directory')
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Coordinator CPU witness is missing or changed')
    verify_code(ROOT, witness['code_sha256'])
    draft = read(PROTOCOL)
    config = read(config_path)
    validate_config(config, draft)
    review_binding = validate_review(config)
    worker_witness = read(B_ROOT / 'CPU_TESTS.json')
    if worker_witness.get('status') != 'passed':
        raise RuntimeError('Stage-B core CPU witness failed')
    verify_code(B_ROOT, worker_witness['code_sha256'])
    config_digest = sha(config_path)
    run_root.mkdir()
    train_folder = run_root / 'training'
    began = time.monotonic()
    phase = 'initializing'
    try:
        write(run_root / 'CONFIG.json', config)
        write(run_root / 'BINDING.json', {
            'config_sha256': config_digest,
            'config_snapshot_sha256': sha(run_root / 'CONFIG.json'),
            'predeclared_protocol_sha256': sha(PROTOCOL),
            'coordinator_code_sha256': sha(ROOT / 'CODE.sha256'),
            'stage_b_code_sha256': worker_witness['code_sha256'],
            'worker_sha256': sha(B_ROOT / 'b_worker_v3.py'),
            'evaluator_sha256': sha(B_ROOT / 'b_evaluate_v2.py'),
            'review': review_binding, 'quantized': False})

        def child(name, gpu, timeout, command):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='6',
                       MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', PYTHONDONTWRITEBYTECODE='1')
            with (run_root / (name + '.log')).open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-B'] + command, cwd=B_ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
                write(run_root / (name + '_PROCESS.json'),
                      {'pid': proc.pid, 'gpu': gpu, 'command': command, 'timeout_seconds': timeout})
                try:
                    exit_code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise RuntimeError(name + ' exceeded its fixed deadline')
            if exit_code != 0:
                raise RuntimeError(name + ' failed with exit code ' + str(exit_code))

        def train():
            nonlocal phase
            phase = 'training'
            write(run_root / 'STATUS.json', {'status': 'running', 'phase': phase}, replace=True)
            child('train', 0, config['max_seconds'] + 60,
                  [str(B_ROOT / 'b_worker_v3.py'), '--config', str(run_root / 'CONFIG.json'),
                   '--out', str(train_folder)])
            verify_training(train_folder, config)
            phase = 'evaluation_and_replay'
            write(run_root / 'STATUS.json', {'status': 'running', 'phase': phase}, replace=True)

        def evaluate_and_replay(sweep, gpu):
            child(f'eval_{sweep:02d}', gpu, 10890,
                  [str(B_ROOT / 'b_evaluate_v2.py'), 'evaluate', '--run-root', str(train_folder),
                   '--sweep', str(sweep)])
            verify_result(train_folder, sweep, False)
            child(f'replay_{sweep:02d}', gpu, 1230,
                  [str(B_ROOT / 'b_evaluate_v2.py'), 'replay', '--run-root', str(train_folder),
                   '--sweep', str(sweep)])
            verify_result(train_folder, sweep, True)

        def tracked_eval(sweep, gpu):
            nonlocal phase
            phase = 'evaluation_and_replay'
            evaluate_and_replay(sweep, gpu)

        execute_plan(train, tracked_eval)
        phase = 'summarizing'
        results = [verify_result(train_folder, sweep, True) for sweep in (5, 10, 20)]
        phase = 'preparing_visual_review_assets'
        write(run_root / 'STATUS.json', {'status': 'running', 'phase': phase}, replace=True)
        from review_assets_b import create
        create(train_folder / 'sweep_20' / 'free_z' / 'evaluation', run_root / 'REVIEW_ASSETS')
        write(run_root / 'SUMMARY.json', {
            'status': 'complete', 'stage': 'B', 'training_sha256': sha(train_folder / 'TRAIN_DONE.json'),
            'results': results, 'primary_checkpoint': 20,
            'visual_review_assets_sha256': sha(run_root / 'REVIEW_ASSETS' / 'CONTACTS.json'),
            'manual_dynamic_review_required': True, 'action_codec_complete': False,
            'scope': 'Free-code teacher fit on one training video; not trained posterior or cross-video generalization',
            'elapsed_seconds': time.monotonic() - began})
        write(run_root / 'STATUS.json', {'status': 'complete', 'phase': 'awaiting_dynamic_review',
                                        'action_codec_complete': False}, replace=True)
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(run_root / 'FAILED.json', failure)
        write(run_root / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    coordinate(args.config, args.run_root)
