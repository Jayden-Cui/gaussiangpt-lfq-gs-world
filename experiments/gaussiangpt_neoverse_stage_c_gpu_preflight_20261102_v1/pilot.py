"""Discarded GPU gradient/memory probe for C0/C1/C2 with synthetic targets.

This is engineering preflight, not formal posterior training or evaluation.
The real stage-B teacher has not yet been selected.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
STAGE_A = OUTPUTS / 'gaussiangpt_neoverse_action_stage_a_20261102_v1'
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
VARIANTS = OUTPUTS / 'gaussiangpt_neoverse_posterior_variants_20261102_v2'
sys.path.insert(0, str(B_ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def verify_manifest(root, expected):
    if sha(root / 'CODE.sha256') != expected:
        raise RuntimeError('Source manifest changed: ' + str(root))
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(root / name) != digest:
            raise RuntimeError('Source file changed: ' + name)


def check_codes(codes):
    if (not isinstance(codes, torch.Tensor) or codes.dtype != torch.float32
            or tuple(codes.shape) != (168, 9, 16) or not bool(torch.isfinite(codes).all())):
        raise RuntimeError('Expected finite synthetic [168,9,16] codes')
    return codes


def grad_norm(parameters):
    values = [x.grad.detach().float().square().sum() for x in parameters if x.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def required_gradients(model, variant):
    values = {'queries': grad_norm(model.posterior.queries.parameters()),
              'projection': grad_norm(model.posterior.project_in.parameters())}
    if variant != 'C0':
        values['encoder_input'] = grad_norm(model.encoder.input.parameters())
        values['encoder_backbone'] = grad_norm(model.encoder.encoder_backbone.parameters())
    if variant == 'C2':
        values['time_projection'] = grad_norm(model.time_projection.parameters())
        values['delta_projection'] = grad_norm(model.delta_projection.parameters())
    if not all(math.isfinite(x) and x > 0 for x in values.values()):
        raise RuntimeError('Required posterior gradient is missing: ' + str(values))
    return values


def run(variant):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if variant not in ('C0', 'C1', 'C2'):
        raise ValueError('Unknown posterior variant')
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed':
        raise RuntimeError('Pilot CPU tests are missing')
    verify_manifest(ROOT, witness['code_sha256'])
    for source_root in (B_ROOT, VARIANTS):
        source_witness = read(source_root / 'CPU_TESTS.json')
        if source_witness.get('status') != 'passed':
            raise RuntimeError('Bound source CPU tests are missing')
        verify_manifest(source_root, source_witness['code_sha256'])
    training_core_witness = read(VARIANTS / 'TRAINING_CORE_WITNESS.json')
    if (training_core_witness.get('status') != 'passed'
            or training_core_witness['training_core_sha256'] != sha(VARIANTS / 'training_core.py')):
        raise RuntimeError('Posterior training core witness changed')
    from gpu_runtime import setup_gpu_once
    gpu_test = read(B_ROOT / 'GPU_RUNTIME_TEST.json')
    if gpu_test.get('status') != 'passed' or gpu_test.get('runtime_sha256') != sha(B_ROOT / 'gpu_runtime.py'):
        raise RuntimeError('Single GPU setup witness changed')
    import importlib.util
    spec = importlib.util.spec_from_file_location('_c_gpu_stage_a_context', STAGE_A / 'stage_context.py')
    context = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = context
    spec.loader.exec_module(context)
    c, helper, original, p, paths, receipt, evaluation = context.setup()
    decoder, modules = context.load_decoder(c, helper, original, p, paths, receipt, 'cpu')
    fm, _, runtime = modules
    unused, posterior = fm.build_models(p, 'cpu', train_decoder=False)
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    posterior.load_state_dict(checkpoint['posterior'], strict=True)
    posterior.eval()
    del unused, checkpoint
    if runtime.digest(posterior) != receipt['posterior_digest']:
        raise RuntimeError('Original 9560-step Q-Former binding differs')
    store = context.dataset(modules, p, c)
    variants = context.explicit('_c_gpu_preflight_variants', VARIANTS / 'posterior_variants.py')
    core = context.explicit('_c_gpu_preflight_training_core', VARIANTS / 'training_core.py')
    torch.manual_seed(20261102)
    model = variants.ContinuousPosterior(decoder.base, posterior, variant,
        include_time=(variant == 'C2'), include_change=(variant == 'C2'), checkpoint_prefixes=True)
    if runtime.digest(model.posterior) != receipt['posterior_digest']:
        raise RuntimeError('Q-Former initialization changed')
    # This perturbation only drives a nonzero test gradient. It is not a teacher.
    synthetic_codes = check_codes(context.load_codes(paths, evaluation)['correct'].clone() + 0.05)
    code_scale = core.teacher_scale(synthetic_codes)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_digest = runtime.digest(model)
    decoder_digest = runtime.digest(decoder)
    out = ROOT / variant
    out.mkdir()
    phase = 'gpu_initialization'
    write(out / 'STATUS.json', {'status': 'running', 'phase': phase, 'variant': variant})
    try:
        setup_gpu_once(p)
        torch.cuda.reset_peak_memory_stats()
        decoder = decoder.cuda()
        model = model.cuda().train()
        from runtime_bridge import BoundNativeBridge
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        if any(x.requires_grad for x in decoder.parameters()) or any(x.requires_grad for x in bridge.parameters()):
            raise RuntimeError('Decoder or native GS codec is not frozen')
        std = store.frames['std'].cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
        batch = [{'start': s, 'horizon': 9} for s in (0, 35, 123, 167)]
        torch.cuda.synchronize()
        started = time.monotonic()
        phase = 'synthetic_distillation_step'
        distill = core.step(model, decoder, bridge, store, std, optimizer, batch,
                            synthetic_codes, code_scale, p, 'distill')
        torch.cuda.synchronize()
        distill_seconds = time.monotonic() - started
        distill_gradients = required_gradients(model, variant)
        phase = 'synthetic_joint_rgb_step'
        write(out / 'STATUS.json', {'status': 'running', 'phase': phase, 'variant': variant}, replace=True)
        started = time.monotonic()
        joint = core.step(model, decoder, bridge, store, std, optimizer, batch,
                          synthetic_codes, code_scale, p, 'joint', probe_rgb=True)
        torch.cuda.synchronize()
        joint_seconds = time.monotonic() - started
        gradients = required_gradients(model, variant)
        if any(x.grad is not None for x in decoder.parameters()) or runtime.digest(decoder) != decoder_digest:
            raise RuntimeError('Frozen decoder changed or acquired a gradient')
        bridge.assert_frozen()
        model.load_state_dict(initial, strict=True)
        if runtime.digest(model) != initial_digest:
            raise RuntimeError('Synthetic updates were not discarded')
        del optimizer
        result = {'status': 'passed', 'variant': variant, 'logical_batch_size': 4,
            'windows': [0, 35, 123, 167], 'horizon': 9, 'synthetic_code_offset': 0.05,
            'distill_seconds': distill_seconds, 'joint_seconds': joint_seconds,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'distill_gradients': distill_gradients, 'joint_gradients': gradients,
            'rgb_only_gradients': [row['rgb_only_gradients'] for row in joint['windows']],
            'synthetic_updates_discarded': True, 'decoder_and_codec_frozen': True,
            'model_initial_digest': initial_digest, 'decoder_digest': decoder_digest,
            'posterior_initial_digest': receipt['posterior_digest'],
            'cache_manifest_sha256': sha(paths['cache_manifest']),
            'pilot_code_sha256': sha(ROOT / 'CODE.sha256'),
            'scope': 'Engineering GPU preflight using a perturbed baseline code as synthetic target; no teacher or quality claim'}
        write(out / 'GPU_PREFLIGHT.json', result)
        write(out / 'STATUS.json', {'status': 'complete', 'phase': 'gpu_preflight_passed', 'variant': variant},
              replace=True)
        print('PASS:', variant, 'peak allocated GiB', round(result['peak_allocated_gib'], 3),
              'joint seconds', round(joint_seconds, 2))
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'variant': variant, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(out / 'FAILED.json', failure)
        write(out / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('C0', 'C1', 'C2'), required=True)
    run(parser.parse_args().variant)
