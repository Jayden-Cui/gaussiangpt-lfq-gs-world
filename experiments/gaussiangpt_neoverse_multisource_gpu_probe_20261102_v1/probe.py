"""Real TRAIN-source radius-6 GPU plumbing probe; no model update or evaluation-set use."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
DATA_ROOT = OUTPUTS / 'gaussiangpt_neoverse_multisource_action_data_20261102_v1'
STAGE_A = OUTPUTS / 'gaussiangpt_neoverse_action_stage_a_20261102_v1'
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
sys.path.insert(0, str(DATA_ROOT))
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


def validate_window(window, clip, start):
    if (window.get('clip') != clip or window.get('start') != start
            or window.get('source_frames') != list(range(start, start + 10))
            or len(window.get('states', [])) != 10 or len(window['targets']['indices']) != 9):
        raise RuntimeError('Unexpected source, adjacent frames or supervision coverage')
    for state in window['states']:
        if (state['coordinates'].ndim != 2 or state['coordinates'].shape[1] != 3
                or tuple(state['features'].shape) != (len(state['coordinates']), 64)):
            raise RuntimeError('Unexpected native continuous GS state shape')


def image_pair(target, prediction):
    values = []
    for tensor in (target, prediction):
        value = tensor.detach().float().cpu().numpy()
        if value.shape != (252, 448, 3) or not np.isfinite(value).all():
            raise RuntimeError('Unexpected source or rendered image')
        values.append(np.rint(value.clip(0, 1) * 255).astype(np.uint8))
    return Image.fromarray(np.concatenate(values, axis=1), mode='RGB')


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed' or witness.get('code_sha256') != sha(ROOT / 'CODE.sha256'):
        raise RuntimeError('Probe CPU tests are missing or changed')
    for line in (ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(ROOT / name) != digest:
            raise RuntimeError('Probe source changed: ' + name)
    source_witness = read(DATA_ROOT / 'CPU_TESTS.json')
    if source_witness.get('status') != 'passed' or source_witness.get('code_sha256') != sha(DATA_ROOT / 'CODE.sha256'):
        raise RuntimeError('Source loader witness changed')
    for line in (DATA_ROOT / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(DATA_ROOT / name) != digest:
            raise RuntimeError('Source loader changed: ' + name)
    adapter_test = read(DATA_ROOT / 'ADAPTER_TEST.json')
    metric_test = read(DATA_ROOT / 'METRIC_TEST.json')
    if (adapter_test.get('status') != 'passed' or adapter_test.get('adapter_sha256') != sha(DATA_ROOT / 'window_adapter.py')
            or metric_test.get('status') != 'passed' or metric_test.get('metrics_sha256') != sha(DATA_ROOT / 'cross_source_metrics.py')):
        raise RuntimeError('Source adapter or unmasked-view metric witness changed')
    training_status = read(B_RUN / 'training/STATUS.json')
    if training_status.get('status') != 'running' or training_status.get('completed_updates', 999999) >= 900:
        raise RuntimeError('Insufficient separation from the scheduled B checkpoint GPU preview')
    from source_store import TRAIN, EVAL
    from window_adapter import MultiSourceWindowStore
    from cross_source_metrics import image_metrics_unmasked_teacher
    from gpu_runtime import setup_gpu_once
    gpu_test = read(B_ROOT / 'GPU_RUNTIME_TEST.json')
    if gpu_test.get('status') != 'passed' or gpu_test.get('runtime_sha256') != sha(B_ROOT / 'gpu_runtime.py'):
        raise RuntimeError('One-time GPU setup witness changed')
    clip, start = TRAIN[1], 26
    if clip in EVAL:
        raise RuntimeError('Probe must use a training source only')
    spec = importlib.util.spec_from_file_location('_multisource_gpu_stage_a_context', STAGE_A / 'stage_context.py')
    context = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = context
    spec.loader.exec_module(context)
    c, helper, original, p, paths, receipt, evaluation = context.setup()
    decoder, modules = context.load_decoder(c, helper, original, p, paths, receipt, 'cpu')
    fm, _, runtime = modules
    unused, posterior = fm.build_models(p, 'cpu', train_decoder=False)
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
    posterior.load_state_dict(checkpoint['posterior'], strict=True)
    posterior.requires_grad_(False).eval()
    del unused, checkpoint
    if runtime.digest(posterior) != receipt['posterior_digest']:
        raise RuntimeError('Original Q-Former binding differs')
    original_store = context.dataset(modules, p, c)
    std = original_store.frames['std'].detach().clone()
    decoder_digest, posterior_digest = runtime.digest(decoder), runtime.digest(posterior)
    output = ROOT / 'result'
    output.mkdir()
    phase = 'gpu_initialization'
    write(output / 'STATUS.json', {'status': 'running', 'phase': phase})
    try:
        setup_gpu_once(p)
        torch.cuda.reset_peak_memory_stats()
        decoder = decoder.cuda().eval()
        posterior = posterior.cuda().eval()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import state_loss, select_state, support_metrics
        from runtime import render
        from core import image_losses
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        adapter = MultiSourceWindowStore(clip, 'train', decoder.base, std, p['hand_roi_xyxy'],
                                         radius=6, max_candidates=25000, need_cached_memories=True, device='cuda')
        phase = 'load_radius6_source_window'
        began = time.monotonic()
        window = adapter.get(start)
        validate_window(window, clip, start)
        audit = read(DATA_ROOT / 'GLOBAL_RADIUS6_AUDIT.json')
        expected_nodes = audit['sources'][clip]['per_start_counts'][start]
        if len(window['condition'].coordinates) != expected_nodes or expected_nodes > 25000:
            raise RuntimeError('Candidate count differs from prior source-only geometry audit')
        sample = adapter.sample(start, 9)
        if 'mask' in sample:
            raise RuntimeError('Unexpected fabricated reference-view mask')
        with torch.no_grad():
            original_codes = posterior([window['memories']])['codes'][0]
        if (tuple(original_codes.shape) != (9, 16) or original_codes.dtype != torch.float32
                or not bool(torch.isfinite(original_codes).all())):
            raise RuntimeError('Unexpected continuous posterior output')
        codes = original_codes.detach().clone().requires_grad_(True)
        phase = 'render_and_gradient_probe'
        prediction = decoder(window['condition'], codes)
        latent, terms = state_loss(prediction, window['targets'], std.cuda())
        state = select_state(window['condition'].coordinates, prediction['features'][8], prediction['logits'][8],
                             p['max_predicted_latent_nodes'])
        images, gs, sh = render(bridge, state, sample['record'], views=(0, 1))
        rgb, rgb_terms = image_losses(images[0], sample['record']['real'], window['roi'], sample['motion'])
        rgb_grad, = torch.autograd.grad(rgb, codes, retain_graph=True)
        total_grad, = torch.autograd.grad(rgb + 0.1 * latent, codes)
        rgb_norm, total_norm = float(rgb_grad.norm()), float(total_grad.norm())
        if not all(math.isfinite(v) and v > 0 for v in (rgb_norm, total_norm)):
            raise RuntimeError('RGB or combined loss did not reach the continuous code')
        metrics = image_metrics_unmasked_teacher(images.detach(), window, sample)
        support = support_metrics(prediction['logits'][8], window['targets']['occupancy'][8])
        image_pair(sample['record']['real'], images[0]).save(output / 'source_vs_posterior_h09.png')
        torch.cuda.synchronize()
        if (runtime.digest(decoder) != decoder_digest or runtime.digest(posterior) != posterior_digest
                or any(x.grad is not None for x in decoder.parameters())
                or any(x.grad is not None for x in posterior.parameters())):
            raise RuntimeError('Frozen decoder or posterior changed')
        bridge.assert_frozen()
        result = {'status': 'passed', 'clip': clip, 'split': 'TRAIN', 'start': start,
            'source_frames': list(range(start, start + 10)), 'horizon': 9,
            'frame_stride': 1, 'candidate_radius': 6, 'candidate_nodes': expected_nodes,
            'posterior_code_shape': list(original_codes.shape), 'predicted_nodes': len(state['coordinates']),
            'predicted_gaussians': len(gs), 'native_view_mask': 'not available; second-view metric is full image',
            'metrics': metrics, 'support': support, 'rgb_to_code_grad_norm': rgb_norm,
            'combined_to_code_grad_norm': total_norm, 'latent_loss': float(latent.detach()),
            'occupancy_loss': float(terms['occupancy'].detach()), 'feature_loss': float(terms['feature'].detach()),
            'motion_mae_from_loss': float(rgb_terms['motion_mae'].detach()),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
            'elapsed_seconds': time.monotonic() - began,
            'train_only_codec_scale': True, 'no_eval_source_read': True, 'no_optimizer_update': True,
            'decoder_and_posterior_frozen': True, 'decoder_future_geometry_input': False,
            'source_image_sha256': sha(output / 'source_vs_posterior_h09.png'),
            'source_loader_sha256': source_witness['code_sha256'],
            'adapter_sha256': sha(DATA_ROOT / 'window_adapter.py'),
            'decoder_digest': decoder_digest, 'posterior_digest': posterior_digest,
            'scope': 'Single TRAIN-source engineering probe with frozen 9560-step posterior; not stage-B/C training, generalization or teacher acceptance'}
        write(output / 'GPU_PROBE.json', result)
        write(output / 'STATUS.json', {'status': 'complete', 'phase': 'gpu_probe_passed'}, replace=True)
        print('PASS: second TRAIN source radius-6 render and RGB-to-code gradient')
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(output / 'FAILED.json', failure)
        write(output / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
