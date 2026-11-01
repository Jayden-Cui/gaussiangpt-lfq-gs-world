"""Resume the frozen full177 evaluator from completed checkpoints; no optimizer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, replace=False):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)


def load_file(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError('Explicit module origin differs')
    return module


def setup():
    for line in (ROOT / 'CODE.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Resume source changed: ' + name)
    spec = read(ROOT / 'resume_protocol.json')
    parent = Path(spec['parent_root'])
    for name, key in [('protocol.json', 'parent_protocol_sha256'),
                      ('CODE.sha256', 'parent_code_manifest_sha256')]:
        if sha(parent / name) != spec[key]:
            raise RuntimeError('Training source binding changed: ' + name)
    if sha(ROOT / 'protocol.json') != spec['parent_protocol_sha256']:
        raise RuntimeError('Evaluation protocol differs from trained protocol')
    sys.path.insert(0, str(parent))
    io = load_file('io_utils', parent / 'io_utils.py')
    p = io.verify()
    return spec, parent, p


def trained_record(spec, parent, mode):
    binding = spec['training'][mode]
    receipt = parent / mode / 'TRAIN_DONE.json'
    checkpoint = parent / mode / 'step_2000.pt'
    if sha(receipt) != binding['train_done_sha256']:
        raise RuntimeError('Training receipt changed')
    done = read(receipt)
    if (done['status'] != 'complete' or done['steps'] != 2000
            or done['checkpoint_replay'] != 'passed'
            or not done['codec_and_state_encoder_frozen']):
        raise RuntimeError('Training has not passed its completion checks')
    if (sha(checkpoint) != binding['checkpoint_sha256']
            or done['checkpoint_sha256'] != binding['checkpoint_sha256']):
        raise RuntimeError('Final checkpoint changed')
    if done['source']['protocol_sha256'] != spec['parent_protocol_sha256']:
        raise RuntimeError('Checkpoint protocol differs')
    return done, checkpoint


def model_modules(parent):
    import full_model
    import data
    import runtime
    for name in ['core', 'support_decoder', 'runtime', 'posterior', 'full_model', 'data', 'io_utils']:
        if Path(sys.modules[name].__file__).resolve() != parent / (name + '.py'):
            raise RuntimeError('Model dependency origin differs: ' + name)
    return full_model, data, runtime


def restore(spec, parent, p, mode, device, modules):
    import torch
    full_model, _, runtime = modules
    done, path = trained_record(spec, parent, mode)
    decoder, posterior = full_model.build_models(p, 'cpu', train_decoder=False)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['schema'] != 'joint_full177_continuous_v1'
            or checkpoint['step'] != 2000 or checkpoint['mode'] != mode
            or checkpoint['source'] != done['source']):
        raise RuntimeError('Checkpoint metadata mismatch')
    decoder.load_state_dict(checkpoint['decoder'], strict=True)
    posterior.load_state_dict(checkpoint['posterior'], strict=True)
    if (runtime.digest(decoder) != done['decoder_digest']
            or runtime.digest(posterior) != done['posterior_digest']
            or full_model.frozen_digest(decoder) != done['source']['frozen_component_digest']):
        raise RuntimeError('Restored tensor digest differs')
    del checkpoint
    decoder.requires_grad_(False).eval()
    posterior.requires_grad_(False).eval()
    return decoder.to(device), posterior.to(device), done


def regression():
    import torch
    spec, parent, p = setup()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    modules = model_modules(parent)
    rows = []
    for mode in p['modes']:
        decoder, posterior, done = restore(spec, parent, p, mode, 'cpu', modules)
        ordinary = importlib.util.find_spec('evaluate').origin
        evaluator = load_file('_full177_bound_evaluator', parent / 'evaluate.py')
        if not callable(evaluator.evaluate):
            raise RuntimeError('Pinned evaluator entry missing')
        if Path(ordinary).resolve() == parent / 'evaluate.py':
            raise RuntimeError('Regression did not reproduce the original search-path collision')
        rows.append(dict(mode=mode, ordinary_import_origin=ordinary,
                         explicit_import_origin=evaluator.__file__,
                         decoder_digest=done['decoder_digest'],
                         posterior_digest=done['posterior_digest'], restored_checkpoint=True))
        del decoder, posterior
    for name in ['replay', 'coordinator']:
        module = load_file('_full177_bound_' + name, parent / (name + '.py'))
        if Path(module.__file__).resolve() != parent / (name + '.py'):
            raise RuntimeError('Utility module origin mismatch')
    write(ROOT / 'REGRESSION.json', dict(status='passed', rows=rows,
          code_manifest_sha256=sha(ROOT / 'CODE.sha256'),
          scope='CPU checkpoint integrity and actual import-collision regression; no GPU rendering'))
    print('PASS: both completed checkpoints restored; colliding import reproduced; explicit evaluator selected', flush=True)


def worker(mode):
    import torch
    spec, parent, p = setup()
    if mode not in p['modes']:
        raise ValueError('Unknown arm')
    out = ROOT / mode
    out.mkdir()
    began = time.monotonic()
    write(out / 'STARTED.json', dict(pid=os.getpid(), mode=mode, phase='evaluation_resume'))
    write(out / 'STATUS.json', dict(status='running', phase='restoring_checkpoint', completed_updates=2000))
    try:
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU')
        free, total = torch.cuda.mem_get_info()
        if free < p['minimum_free_gpu_gib'] * 2**30:
            raise RuntimeError('GPU memory unavailable')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib'] * 2**30 / total)
        modules = model_modules(parent)
        decoder, posterior, done = restore(spec, parent, p, mode, 'cuda', modules)
        full_model, data, runtime = modules
        cache = parent / 'cache' / 'MANIFEST.json'
        if sha(cache) != done['source']['cache_manifest_sha256']:
            raise RuntimeError('Training cache manifest changed')
        store = data.WindowStore(p)
        if store.manifest['frozen_model_digest'] != done['source']['frozen_component_digest']:
            raise RuntimeError('Frozen cache encoder differs')
        from runtime_bridge import BoundNativeBridge
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        evaluator = load_file('_full177_bound_evaluator', parent / 'evaluate.py')
        source = dict(done['source'], evaluation_resume=dict(
            parent_root=str(parent), checkpoint_sha256=done['checkpoint_sha256'],
            decoder_digest=done['decoder_digest'], posterior_digest=done['posterior_digest'],
            train_done_sha256=spec['training'][mode]['train_done_sha256'],
            resume_code_sha256=sha(ROOT / 'CODE.sha256'),
            evaluator_sha256=sha(parent / 'evaluate.py'), evaluator_path=evaluator.__file__))
        (out / 'TRAIN_DONE.json').write_bytes((parent / mode / 'TRAIN_DONE.json').read_bytes())
        write(out / 'PROVENANCE.json', source)
        write(out / 'STATUS.json', dict(status='running', phase='evaluation', completed_updates=2000), replace=True)
        evaluator.evaluate(decoder, posterior, bridge, store, p, out, source, mode)
        bridge.assert_frozen()
        if (runtime.digest(decoder) != done['decoder_digest']
                or runtime.digest(posterior) != done['posterior_digest']):
            raise RuntimeError('Evaluation modified model tensors')
        write(out / 'STATUS.json', dict(status='complete', phase='evaluation', completed_updates=2000,
              native_replay='pending', unchanged_model_tensors=True, elapsed_seconds=time.monotonic()-began), replace=True)
    except BaseException as error:
        failure = dict(status='failed', phase='evaluation_resume', error=repr(error), traceback=traceback.format_exc())
        write(out / 'FAILED.json', failure)
        write(out / 'STATUS.json', failure, replace=True)
        raise


def replay(mode):
    spec, parent, p = setup()
    trained_record(spec, parent, mode)
    module = load_file('_full177_bound_replay', parent / 'replay.py')
    module.ROOT = ROOT
    module.main(mode)


def child(mode, phase, timeout):
    p = read(ROOT / 'protocol.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p['gpu_assignment'][mode]),
               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / f'{mode}_{phase}.log').open('x') as log:
        job = subprocess.Popen([sys.executable, '-B', str(ROOT / 'resume.py'), phase, '--mode', mode],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(ROOT / f'{mode}_PROCESS.json', dict(pid=job.pid, mode=mode, phase=phase), replace=True)
        try:
            code = job.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            job.terminate()
            try:
                job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()
            return dict(status='failed', mode=mode, phase=phase, error='deadline')
    return dict(status='complete' if code == 0 else 'failed', mode=mode, phase=phase, exit_code=code)


def coordinate():
    began = time.monotonic()
    try:
        spec, parent, p = setup()
        check = read(ROOT / 'REGRESSION.json')
        if check['status'] != 'passed' or check['code_manifest_sha256'] != sha(ROOT / 'CODE.sha256'):
            raise RuntimeError('Import regression witness stale')
        available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')).split()[1]) * 1024
        cache = read(parent / 'cache' / 'MANIFEST.json')
        cache_bytes = sum(v['bytes'] for v in cache['windows'].values()) + cache['frames']['bytes']
        if available < max(p['minimum_host_available_gib'] * 2**30, 2*cache_bytes + 8*2**30):
            raise RuntimeError('Insufficient host RAM')
        if shutil.disk_usage(ROOT).free < p['minimum_disk_free_gib'] * 2**30:
            raise RuntimeError('Insufficient disk space')
        write(ROOT / 'STARTED.json', dict(pid=os.getpid(), phase='evaluation_resume',
              training_parent=str(parent), resume_code_sha256=sha(ROOT / 'CODE.sha256')))
        write(ROOT / 'STATUS.json', dict(status='running', phase='evaluation_resume', steps_per_arm=2000))

        def arm(mode):
            result = child(mode, 'worker', p['evaluation_max_seconds'] + 300)
            if result['status'] != 'complete':
                return result
            result = child(mode, 'replay', p['replay_max_seconds'] + 30)
            if result['status'] == 'complete':
                write(ROOT / mode / 'STATUS.json', dict(status='complete', phase='evaluation_and_replay',
                      completed_updates=2000, native_replay='passed'), replace=True)
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(arm, mode) for mode in p['modes']]
            results = [future.result() for future in futures]
        if any(result['status'] != 'complete' for result in results):
            raise RuntimeError('Evaluation branch failed: ' + str(results))
        summary_module = load_file('_full177_bound_coordinator', parent / 'coordinator.py')
        summary_module.ROOT = ROOT
        summary_module.summarize(p)
        summary = read(ROOT / 'SUMMARY.json')
        summary['training_parent'] = str(parent)
        summary['evaluation_only_resume'] = True
        summary['additional_training_updates'] = 0
        write(ROOT / 'SUMMARY.json', summary, replace=True)
        write(ROOT / 'STATUS.json', dict(status='complete', steps_per_arm=2000, native_replay='passed',
              additional_training_updates=0, action_success='pending visual review',
              elapsed_seconds=time.monotonic()-began), replace=True)
    except BaseException as error:
        failure = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
        write(ROOT / 'FAILED.json', failure)
        write(ROOT / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['regression', 'worker', 'replay', 'coordinate'])
    parser.add_argument('--mode', choices=['posterior', 'initial_only'])
    args = parser.parse_args()
    if args.phase in ['worker', 'replay'] and not args.mode:
        parser.error('--mode required')
    if args.phase == 'regression':
        regression()
    elif args.phase == 'worker':
        worker(args.mode)
    elif args.phase == 'replay':
        replay(args.mode)
    else:
        coordinate()
