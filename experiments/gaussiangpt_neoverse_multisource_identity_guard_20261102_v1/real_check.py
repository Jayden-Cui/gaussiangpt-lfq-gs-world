"""CPU preflight on the pinned three-source metadata and guarded adapter."""
import json
from pathlib import Path
import subprocess
import sys

import torch
import guarded_store as guard

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or guard.sha(ROOT / name) != digest:
            raise RuntimeError('Identity-guard source changed: ' + name)
    cpu = guard.read(ROOT / 'CPU_TESTS.json')
    if cpu.get('status') != 'passed' or cpu.get('code_sha256') != guard.sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Identity-guard CPU witness differs')
    unit = subprocess.run([sys.executable, '-B', str(ROOT / 'tests.py')], cwd=ROOT,
                          capture_output=True, text=True, check=True, timeout=60)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    store = guard.GuardedSourceStore()
    report = store.identity_evidence
    if len(report['rows']) != 3 or not report['source_video_hashes_distinct']:
        raise RuntimeError('Unexpected verified source identities')
    try:
        store.authorize(guard.EVAL[0], 'train')
    except PermissionError:
        pass
    else:
        raise RuntimeError('Evaluation source was admitted for training')
    train_store = guard.make_window_store(guard.TRAIN[0], 'train', base=None, codec_std=torch.ones(64),
        hand_roi_xyxy=[[0, 0, 1, 1]], device='cpu')
    eval_store = guard.make_window_store(guard.EVAL[0], 'eval', base=None, codec_std=torch.ones(64),
        hand_roi_xyxy=[[0, 0, 1, 1]], device='cpu')
    if not isinstance(train_store.source, guard.GuardedSourceStore) or not isinstance(eval_store.source, guard.GuardedSourceStore):
        raise RuntimeError('Window factory did not retain the guarded loader')
    try:
        guard.make_window_store(guard.EVAL[0], 'train', base=None, codec_std=torch.ones(64),
            hand_roi_xyxy=[[0, 0, 1, 1]], device='cpu')
    except PermissionError:
        pass
    else:
        raise RuntimeError('Window factory admitted evaluation frames for training')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    output = dict(status='passed', code_sha256=guard.sha(ROOT / 'CODE.sha256'),
        unit_test_stdout=unit.stdout.strip(), identities=report,
        original_eval_training_rejection_preserved=True, guarded_window_factory_uses_guard=True,
        eval_train_window_factory_rejected=True, gpu_used=False, model_training_performed=False,
        native_video_rehashed=False,
        scope='Three historically bound source metadata; no new action training or clean holdout claim')
    with (ROOT / 'REAL_CHECK.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: real bound source identities and guarded train/eval window construction; CPU only')


if __name__ == '__main__':
    main()
