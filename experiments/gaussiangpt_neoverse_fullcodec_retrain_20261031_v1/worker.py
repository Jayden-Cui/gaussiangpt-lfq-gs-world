"""Independent full-codec retraining; frozen full-pool implementation stays read-only.

Import is CPU-safe. CUDA work is only reachable through main(--root).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time

RUN = 'gaussiangpt_neoverse_fullcodec_retrain_20261031_v1'
OLD_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1'
OLD_MANIFEST = '01505377362bf47c89ae23aeaa5de6ca0c5663fe9119e35f6c01c03524351376'
INITIAL_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
PLAN_SHAS = ['75c9bd9f8e42c062df5a31407fc9a6875f87baf51bccda94b387e13fdc36906e',
             '6676c4ef3d6954214505d526ef18200a6299633cc971ca39de89988a485e66c3']
FRAMES, EPOCH_UPDATES, STEPS, GIB = 8850, 2213, 4426, 2**30


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(p):
    expected = {
        ('schema',): 'fullcodec_retrain_v1', ('run',): RUN,
        ('source', 'old_root'): OLD_ROOT, ('source', 'old_code_manifest_sha256'): OLD_MANIFEST,
        ('source', 'initial_checkpoint_sha256'): INITIAL_SHA,
        ('train', 'steps'): STEPS, ('train', 'epochs'): 2, ('train', 'frames_per_epoch'): FRAMES,
        ('train', 'updates_per_epoch'): EPOCH_UPDATES, ('train', 'batch_size'): 4,
        ('train', 'epoch_seeds'): [20261031, 20261032], ('train', 'force_target_epochs'): [True, False],
        ('train', 'learning_rate'): 0.0001, ('train', 'betas'): [0.9, 0.95],
        ('train', 'weight_decay'): 0, ('train', 'grad_clip'): 1,
        ('train', 'max_seconds'): 7200, ('resources', 'gpu_physical_id'): 1,
        ('resources', 'cpu_threads'): 6, ('resources', 'peak_limit_gib'): 12,
        ('resources', 'minimum_free_gib'): 16, ('resources', 'minimum_disk_free_gib'): 8,
        ('execution', 'outer_seconds'): 8500,
        ('data', 'training_clips'): 50, ('data', 'source_groups'): 19,
        ('data', 'frames_per_clip'): 177, ('data', 'height'): 252, ('data', 'width'): 448,
        ('data', 'input_gaussians_per_frame'): 112896, ('data', 'frame_stride'): 1,
        ('model', 'latent_width'): 64, ('model', 'state_quantization'): False,
        ('evaluation_in_this_run',): False, ('automatic_LDM_replacement',): False,
        ('action_Z_training',): False,
    }
    for keys, value in expected.items():
        actual = p
        for key in keys:
            actual = actual[key]
        require(type(actual) is type(value) and actual == value, 'Protocol drift: ' + '.'.join(keys))
    require(all(type(v) is bool for v in p['train']['force_target_epochs']), 'Forcing flags must be bool')


def verify_manifest(root, expected_sha=None):
    root = Path(root)
    digest = sha(root / 'CODE_MANIFEST.json')
    require(expected_sha is None or digest == expected_sha, 'Wrong frozen source manifest')
    manifest = read(root / 'CODE_MANIFEST.json')
    require(isinstance(manifest, dict) and 'protocol.json' in manifest, 'Protocol must be manifest-bound')
    require({p.name for p in root.glob('*.py')} <= set(manifest), 'Unbound executable in run root')
    for name, pinned in manifest.items():
        require(Path(name).name == name and sha(root / name) == pinned, 'Manifest mismatch: ' + name)
    return manifest, digest


def verify_root(root):
    manifest, digest = verify_manifest(root)
    require({'worker.py', 'test_worker.py', 'WORKER_NOTES.md'} <= set(manifest), 'Worker files must be bound')
    proof = read(root / 'CPU_VALIDATION.json')
    require(proof.get('status') == 'passed' and proof.get('gpu_executed') is False
            and proof.get('code_manifest_sha256') == digest, 'Fresh exact CPU validation required')
    p = read(root / 'protocol.json')
    validate_protocol(p)
    return p, dict(run=RUN, code_manifest_sha256=digest, protocol_sha256=sha(root / 'protocol.json'),
                   cpu_validation_sha256=sha(root / 'CPU_VALIDATION.json'))


def load_old(root):
    """Bind before import; prevent a similarly named pool_index from another run."""
    root = Path(root).resolve()
    manifest, _ = verify_manifest(root, OLD_MANIFEST)
    name = 'pool_index'
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == root / 'pool_index.py', 'Foreign pool_index loaded')
    else:
        spec = importlib.util.spec_from_file_location(name, root / 'pool_index.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile((root / 'pool_index.py').read_bytes(), str(root / 'pool_index.py'), 'exec'), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    source = (root / 'worker.py').read_bytes()
    require(hashlib.sha256(source).hexdigest() == manifest['worker.py'], 'Old worker changed before import')
    spec = importlib.util.spec_from_file_location('_retrain_bound_fullpool', root / 'worker.py')
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(root / 'worker.py'), 'exec'), module.__dict__)
    return module


def check_admission(gpu_index, free_mib, total_mib, disk_free_bytes, visible_devices):
    require(visible_devices == '1' and gpu_index == 1, 'Expose only assigned physical GPU1')
    require(all(math.isfinite(v) and v >= 0 for v in (free_mib, total_mib, disk_free_bytes)), 'Invalid resource reading')
    require(total_mib >= free_mib >= 16 * 1024, 'GPU1 requires at least 16 GiB free')
    require(disk_free_bytes >= 8 * GIB, 'At least 8 GiB disk space required; do not delete old data')


def check_limits(elapsed, allocated, reserved, process_bytes):
    require(all(math.isfinite(v) and v >= 0 for v in (elapsed, allocated, reserved, process_bytes)), 'Invalid guard sample')
    require(max(allocated, reserved, process_bytes) <= 12 * GIB, '12 GiB GPU guard exceeded; no truncation')
    if elapsed > 7200:
        raise TimeoutError('Worker exceeded 7200 seconds')


def admit(root):
    text = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=index,uuid,memory.free,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [line.split(',') for line in text.strip().splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 4, 'Expected one GPU1 resource reading')
    index, uuid, free, total = [part.strip() for part in rows[0]]
    disk = shutil.disk_usage(root).free
    check_admission(int(index), float(free), float(total), disk, os.environ.get('CUDA_VISIBLE_DEVICES'))
    return dict(gpu_physical_id=int(index), gpu_uuid=uuid, free_mib=float(free), total_mib=float(total),
                disk_free_bytes=disk, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], pid=os.getpid())


class Guard:
    def __init__(self, torch, started, gpu_uuid):
        self.torch, self.started, self.gpu_uuid = torch, started, gpu_uuid
        self.last_query = float('-inf')
        self.peak_process_bytes = 0

    def check(self):
        self.torch.cuda.synchronize()
        now = time.monotonic()
        if now - self.last_query >= 10:
            output = subprocess.check_output(['nvidia-smi', '--id=1',
                '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=15)
            used = 0
            for line in output.strip().splitlines():
                fields = [v.strip() for v in line.split(',')]
                require(len(fields) == 3, 'Malformed GPU process-memory reading')
                if fields[0] == self.gpu_uuid and fields[1] == str(os.getpid()):
                    used += int(fields[2]) * 2**20
            self.peak_process_bytes = max(self.peak_process_bytes, used)
            self.last_query = now
        check_limits(now - self.started, self.torch.cuda.max_memory_allocated(),
                     self.torch.cuda.max_memory_reserved(), self.peak_process_bytes)

    def evidence(self):
        return dict(elapsed_seconds=time.monotonic() - self.started, max_seconds=7200,
                    peak_allocated_gib=self.torch.cuda.max_memory_allocated() / GIB,
                    peak_reserved_gib=self.torch.cuda.max_memory_reserved() / GIB,
                    sampled_process_peak_gib=self.peak_process_bytes / GIB,
                    process_sample_interval_seconds=10, process_peak_is_continuous_measurement=False)


def capture_rng(*, include_cuda):
    """Only tensors and safe builtins: torch.load(weights_only=True) can read it."""
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if include_cuda else []
    require(not include_cuda or len(cuda) == 1, 'RNG capture requires exactly one assigned CUDA device')
    return dict(python=random.getstate(), numpy=dict(algorithm=numpy_state[0], keys=numpy_state[1].tolist(),
                position=int(numpy_state[2]), has_gauss=int(numpy_state[3]), cached_gaussian=float(numpy_state[4])),
                torch_cpu=torch.get_rng_state(), torch_cuda=cuda, cuda_captured=include_cuda)


def restore_rng(state, *, require_cuda):
    """Testable state restore primitive; no production resume CLI is enabled."""
    import numpy as np
    import torch
    require(state['cuda_captured'] is require_cuda, 'RNG device provenance differs')
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['algorithm'], np.asarray(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if require_cuda:
        require(len(state['torch_cuda']) == torch.cuda.device_count() == 1, 'CUDA RNG device count differs')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def add_coverage(seen, batch, expected):
    ids = [(r.clip, r.frame) for r in batch]
    require(len(ids) in (2, 4) and len(set(ids)) == len(ids), 'Invalid full-pool batch')
    require(set(ids) <= expected and seen.isdisjoint(ids), 'Duplicate or foreign frame exposure')
    seen.update(ids)


def checkpoint_cursor(step, seen, expected, plan_shas):
    require(type(step) is int and step in (EPOCH_UPDATES, STEPS), 'Only complete epoch checkpoints allowed')
    require(len(expected) == FRAMES and len(seen) == 2 and plan_shas == PLAN_SHAS, 'Coverage/plan binding differs')
    epoch = step // EPOCH_UPDATES
    require(all(seen[i] == expected for i in range(epoch)), 'Incomplete epoch coverage')
    require(epoch == 2 or not seen[1], 'Unexpected second-epoch exposure at first checkpoint')
    return dict(completed_updates=step, epoch_index=epoch, next_batch_index=0,
                next_global_step=step + 1 if epoch < 2 else None,
                next_force_target=False if epoch < 2 else None,
                epoch_plan_sha256=list(plan_shas), epoch_frames_seen=[len(s) for s in seen],
                total_frame_exposures=sum(len(s) for s in seen), training_complete=epoch == 2,
                cursor_convention='zero-based next epoch and batch; epoch_index=2 means exhausted')


def assert_finite_state(model, optimizer, completed=None):
    import torch
    for name, value in model.state_dict().items():
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite model: ' + name)
    params = [p for group in optimizer.param_groups for p in group['params']]
    require(len(optimizer.state) == len(params), 'Optimizer state is incomplete')
    for parameter in params:
        state = optimizer.state[parameter]
        require({'step', 'exp_avg', 'exp_avg_sq'} <= set(state), 'Incomplete AdamW parameter state')
        require(completed is None or int(state['step']) == completed, 'AdamW update count differs')
        require(state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape, 'AdamW shape mismatch')
        require(all(not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all()) for v in state.values()),
                'Nonfinite AdamW state')


def main(root):
    root = Path(root).resolve()
    require(root.name == RUN and root == Path(__file__).resolve().parent, 'Dedicated new run root required')
    p, new_source = verify_root(root)
    require(not any((root / n).exists() for n in ('STARTED.json', 'DONE.json', 'FAILED.json', 'arms')),
            'Run already attempted; never overwrite evidence')
    with (root / 'STARTED.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(status='initializing', source=new_source, pid=os.getpid(), gpu_executed=False), stream, indent=2)
    started = time.monotonic()
    folder = root / 'arms' / 'full_codec'
    folder.mkdir(parents=True)
    model = optimizer = guard = old = None
    completed = 0
    evidence = dict(new_source=new_source, old_source=None)
    def interrupted(signum, frame):
        raise RuntimeError('Owned worker interrupted: signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.alarm(7200)
    try:
        resource = admit(root)
        write(root / 'RESOURCE_ADMISSION.json', resource)
        import numpy as np
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One actual CUDA GPU required')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., 10 * GIB / torch.cuda.get_device_properties(0).total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
        guard = Guard(torch, started, resource['gpu_uuid'])
        guard.check()
        old = load_old(Path(p['source']['old_root']))
        x = old.load_runtime(Path(p['source']['old_root']))
        require(Path(sys.executable).resolve() == Path(x.p['queue']['runtime']).resolve(),
                'Use the same bound Python runtime as the old preflight')
        preflight = old.gate(x)
        require(x.evidence['code_manifest_sha256'] == OLD_MANIFEST
                and x.evidence['initial_checkpoint_sha256'] == INITIAL_SHA
                and x.evidence['epoch_plan_sha256'] == PLAN_SHAS, 'Frozen training source differs')
        plans = old.make_epochs(x.index)
        require([old.pool.plan_summary(x.index, plan)['plan_sha256'] for plan in plans] == PLAN_SHAS, 'Plan mismatch')
        evidence.update(old_source=x.evidence, old_preflight_sha256=sha(x.root / 'PREFLIGHT.json'),
                        old_preflight_reused_read_only=True, new_preflight_measurement=False,
                        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                            numpy=str(np.__version__), cuda=torch.version.cuda,
                            gpu_name=torch.cuda.get_device_name(0), gpu_uuid=resource['gpu_uuid'],
                            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                            cudnn_deterministic=torch.backends.cudnn.deterministic,
                            cudnn_benchmark=torch.backends.cudnn.benchmark))
        write(folder / 'SOURCE.json', evidence)
        model = old.legacy.initial_model(x.old)
        initial = old.legacy.digest_snapshot(old.legacy.snapshot(model))
        require(initial == preflight['arms']['full_codec']['initial_model_sha256'], 'Initial weights differ from old preflight')
        old.legacy.configure_arm(model, 'full_codec')
        initial_encoder = old.legacy.snapshot(model, encoder_only=True, parameters_only=True)
        optimizer = old.legacy.fresh_optimizer(model)
        require(not optimizer.state, 'Restart must use fresh AdamW, not step2213 state')
        vgg = x.old.perceptual.Vgg19(x.old.codec_root / 'torch_cache').cuda()
        expected = {(r.clip, r.frame) for r in x.index}
        seen = [set(), set()]
        nonzero_groups_seen = set()
        required_gradient_groups = {old.legacy.group_name(name) for name, parameter in model.named_parameters()
                                    if parameter.requires_grad}
        write(folder / 'PLAN.json', dict(epochs=x.plan_summaries, epoch_seeds=[20261031, 20261032],
                                      forcing=[True, False], source=evidence))
        guard.check()
        with (folder / 'trace.jsonl').open('x', encoding='utf-8', buffering=1) as trace:
            for epoch, plan in enumerate(plans):
                for batch_index, batch in enumerate(plan):
                    step = completed + 1
                    row = old.accumulated_update(model, optimizer, batch,
                        lambda ref: old.pool.load_frame(ref, device='cuda'), x.old.train.terms, vgg,
                        old.forcing_for_step(step), guard=guard.check)
                    completed = step
                    nonzero_groups_seen.update(name for name, report in row['gradient_groups'].items()
                                               if report['nonzero_grad'] > 0)
                    add_coverage(seen[epoch], batch, expected)
                    old.legacy.assert_training_modes(model, 'full_codec')
                    row.update(step=step, epoch=epoch + 1, batch_index=batch_index, arm='full_codec',
                               epoch_frames_seen=len(seen[epoch]), **guard.evidence())
                    trace.write(json.dumps(row, allow_nan=False) + '\n')
                    if step == 1 or step % 100 == 0:
                        print('FULLCODEC_RETRAIN', step, row['loss'], flush=True)
                cursor = checkpoint_cursor(completed, seen, expected, x.evidence['epoch_plan_sha256'])
                assert_finite_state(model, optimizer, completed)
                if completed == STEPS:
                    require(required_gradient_groups <= nonzero_groups_seen, 'Missing nonzero gradients in parameter group')
                    require(old.legacy.snapshot(model, encoder_only=True, parameters_only=True) != initial_encoder,
                            'Encoder parameters did not update')
                    old.recheck(x)
                    require(verify_root(root)[1] == new_source, 'New source binding changed')
                guard.check()
                old.legacy.atomic_save(dict(schema='fullcodec_retrain_checkpoint_v1', model=model.state_dict(),
                    optimizer=optimizer.state_dict(), arm='full_codec', step=completed, source=evidence,
                    initial_model_sha256=initial, initial_checkpoint_sha256=INITIAL_SHA, inherited_codec_steps=3000,
                    rng=capture_rng(include_cuda=True), sampling=cursor,
                    optimizer_parameter_names=[name for name, parameter in model.named_parameters() if parameter.requires_grad],
                    encoder_sha256=old.legacy.digest_snapshot(old.legacy.snapshot(model, encoder_only=True)),
                    failed=False, resumable=False, recovery_state_complete=True, exact_resume_verified=False,
                    eligible_for_final_evaluation=completed == STEPS, requires_successful_done_receipt=True),
                    folder / f'step_{completed:04d}.pt')
        cursor = checkpoint_cursor(completed, seen, expected, PLAN_SHAS)
        guard.check()
        done = dict(status='complete', training_complete=True, source=evidence, arm='full_codec',
                    steps=completed, inherited_codec_steps=3000, additional_codec_updates=completed,
                    initial_model_sha256=initial, initial_checkpoint_sha256=INITIAL_SHA, sampling=cursor,
                    forcing_updates=EPOCH_UPDATES, unforced_updates=EPOCH_UPDATES,
                    finite_nonzero_gradient_groups=sorted(nonzero_groups_seen),
                    fresh_optimizer_verified=True, checkpoint_sha256=sha(folder / 'step_4426.pt'),
                    first_epoch_checkpoint_sha256=sha(folder / 'step_2213.pt'), trace_sha256=sha(folder / 'trace.jsonl'),
                    plan_sha256=sha(folder / 'PLAN.json'), gpu_executed=True, **guard.evidence(),
                    codec_accepted=False, evaluation_complete=False, action_Z_trained=False,
                    automatic_LDM_codec_replacement=False, goal_complete=False)
        write(folder / 'TRAIN_DONE.json', done)
        write(root / 'DONE.json', done)
        print('FULLCODEC_RETRAIN_TRAINING_COMPLETE', flush=True)
    except BaseException as error:
        signal.alarm(0)
        failure = dict(status='failed', training_complete=False, source=evidence, error=repr(error),
                       last_completed_step=completed, eligible_for_final_evaluation=False, resumable=False,
                       goal_complete=False, failed_checkpoint_saved=False)
        if model is not None and old is not None:
            try:
                old.legacy.atomic_save(dict(schema='fullcodec_retrain_failed_v1', model=model.state_dict(),
                    optimizer=optimizer.state_dict() if optimizer is not None else {}, source=evidence,
                    last_completed_step=completed, uncommitted_update_may_have_run=True,
                    failed=True, eligible_for_final_evaluation=False, resumable=False), folder / 'FAILED_DO_NOT_EVALUATE.pt')
                failure['failed_checkpoint_saved'] = True
            except BaseException as save_error:
                failure['failed_checkpoint_error'] = repr(save_error)
        write(folder / 'FAILURE.json', failure)
        write(root / 'FAILED.json', failure)
        raise
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
