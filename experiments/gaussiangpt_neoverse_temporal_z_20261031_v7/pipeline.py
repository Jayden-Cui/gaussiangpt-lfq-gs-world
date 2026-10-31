"""One bounded submission: four continuations, common-step evaluation, CPU comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def verify(root, expected=None):
    digest = sha(root/'CODE_MANIFEST.json')
    if expected is not None and digest != expected:
        raise RuntimeError('New trial manifest changed during queue')
    for name, value in read(root/'CODE_MANIFEST.json').items():
        if sha(root/name) != value:
            raise RuntimeError('Immutable input changed: ' + name)
    return digest


def choose_common(root, cfg):
    arms = cfg['arms']
    done = {a: read(root/a/'TRAIN_DONE.json') for a in arms}
    if any(x.get('status') != 'complete' for x in done.values()):
        raise RuntimeError('Every training process must finish successfully before selection')
    eligible = [s for s in cfg['train']['save_steps']
                if all(int(done[a]['steps']) >= s and (root/a/f'step_{s:04d}.pt').is_file()
                       for a in arms)]
    if not eligible:
        raise RuntimeError('No common predeclared saved step; do not relabel final checkpoints')
    step = max(eligible)
    for arm in arms:
        if done[arm].get('checkpoint_sha256', {}).get(str(step)) != sha(root/arm/f'step_{step:04d}.pt'):
            raise RuntimeError('Saved checkpoint differs from training receipt: '+arm)
    plans = {a: read(root/a/'PLAN.json')['starts'][:step] for a in arms}
    if any(p != next(iter(plans.values())) for p in plans.values()):
        raise RuntimeError('Completed sampling plans differ')
    result = dict(schema='gs_temporal_four_arm_selection_v1', status='selected',
                  selected_step=step, original_completed_steps={a:x['steps'] for a,x in done.items()},
                  selection_rule='Largest common predeclared saved step, without quality selection',
                  code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),
                  protocol_sha256=sha(root/'protocol.json'),
                  plan_prefix_sha256=hashlib.sha256(json.dumps(next(iter(plans.values())),
                      separators=(',', ':')).encode()).hexdigest(),
                  checkpoints={a:dict(path=str(root/a/f'step_{step:04d}.pt'),
                        sha256=sha(root/a/f'step_{step:04d}.pt')) for a in arms}, goal_complete=False)
    write(root/'SELECTION.json', result)
    return step


def main(root):
    manifest = verify(root)
    cfg = read(root/'protocol.json')
    preflight = read(root/'PREFLIGHT.json')
    if preflight.get('status') != 'passed' or preflight.get('code_manifest_sha256') != manifest:
        raise RuntimeError('Exact-code real B4 preflight required')
    helper_path = Path(cfg['source_root'])/'analysis_code/postfit_queue.py'
    if sha(helper_path) != cfg['process_helper_sha256']:
        raise RuntimeError('Previously tested process-group helper changed')
    spec = importlib.util.spec_from_file_location('bounded_child', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, helper.terminate_queue)
    signal.alarm(cfg['queue']['outer_seconds'])
    (root/'queue').mkdir(exist_ok=False)
    receipt = dict(pid=os.getpid(), started_unix=time.time(), code_manifest_sha256=manifest,
                   gpu_workers=cfg['queue']['gpu_workers'], goal_complete=False)
    write(root/'queue/RECEIPT.json', receipt)

    def worker(gpu, arms, mode, selected=None):
        outcomes = []
        for arm in arms:
            if helper.STOP_REQUESTED.is_set():
                raise RuntimeError('Queue stop requested')
            verify(root, manifest)
            command = [sys.executable, '-B', str(root/'runner.py'), '--root', str(root),
                       '--mode', mode, '--arm', arm]
            if selected is not None:
                command += ['--checkpoint-step', str(selected)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
            outcome = helper.run_child(command, env, root/'queue'/f'{arm}_{mode}.log',
                                        cfg['queue'][f'{mode}_process_seconds'])
            outcome.update(arm=arm, mode=mode, gpu=gpu)
            write(root/'queue'/f'{arm}_{mode}_PROCESS.json', outcome)
            outcomes.append(outcome)
            if outcome['exit_code'] != 0 or outcome['timed_out']:
                break
        return outcomes

    for mode in ('train', 'eval'):
        step = None if mode == 'train' else choose_common(root, cfg)
        write(root/'queue/STATUS.json', dict(status=mode, pid=os.getpid(), selected_step=step,
                                            timestamp=time.time(), goal_complete=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, int(gpu), arms, mode, step)
                       for gpu, arms in cfg['queue']['gpu_workers'].items()]
            outcomes = [x for f in futures for x in f.result()]
        if (len(outcomes) != len(cfg['arms']) or
                any(x['exit_code'] != 0 or x['timed_out'] for x in outcomes)):
            raise RuntimeError(mode+' phase did not complete all four arms; no automatic retry')
    verify(root, manifest)
    outcome = helper.run_child([sys.executable, '-B', str(root/'compare.py'), '--root', str(root)],
                              os.environ.copy(), root/'queue/comparison.log', 180)
    if outcome['exit_code'] != 0 or outcome['timed_out']:
        raise RuntimeError('Terminal comparison failed')
    write(root/'queue/STATUS.json', dict(status='complete', pid=os.getpid(),
          finished_unix=time.time(), selected_step=step, goal_complete=False))
    write(root/'queue/DONE.json', dict(status='complete', selected_step=step,
          summary='SUMMARY.json', goal_complete=False))
    signal.alarm(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root.resolve()
    try:
        main(root)
    except BaseException as error:
        write(root/'queue/FAILED.json', dict(status='failed', error=repr(error),
              traceback=traceback.format_exc(), goal_complete=False))
        raise
