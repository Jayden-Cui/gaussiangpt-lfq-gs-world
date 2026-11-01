"""Read-only comparison of two native decoders on identical predicted GS states."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
STAGE_A = OUTPUTS / 'gaussiangpt_neoverse_action_stage_a_20261102_v1'
CORRECTION = OUTPUTS / 'gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v3'
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
CODEC_AUDIT = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_reuse_audit_20261102_v1'
ADAPTED = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_20261031_v1/arms/decoder_only/step_0300.pt'
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
        raise RuntimeError('Source manifest changed')
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(root / name) != digest:
            raise RuntimeError('Source file changed: ' + name)


def checked_state(path, expected_source):
    value = torch.load(path, map_location='cpu', weights_only=True)
    coordinates, features = value['coordinates'], value['features']
    if (value.get('source') != expected_source or coordinates.ndim != 2 or coordinates.shape[1] != 3
            or coordinates.dtype != torch.int32 or features.dtype != torch.float32
            or tuple(features.shape) != (len(coordinates), 64) or not bool(torch.isfinite(features).all())):
        raise RuntimeError('Saved predicted state or its provenance differs')
    return {'coordinates': coordinates, 'features': features}


def mean_metric(rows, arm, metric):
    values = [row['metrics'][arm]['metrics'][metric] for row in rows]
    if len(values) != 18 or not all(math.isfinite(x) for x in values):
        raise RuntimeError('Expected 18 finite paired future-frame metrics')
    return sum(values) / len(values)


def rgb(tensor):
    value = tensor.detach().float().cpu().numpy()
    if value.shape != (252, 448, 3) or not np.isfinite(value).all():
        raise RuntimeError('Unexpected source or decoded image')
    return np.rint(value.clip(0, 1) * 255).astype(np.uint8)


def image_triplet(target, baseline, adapted):
    return Image.fromarray(np.concatenate([rgb(target), rgb(baseline), rgb(adapted)], axis=1), mode='RGB')


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    witness = read(ROOT / 'CPU_TESTS.json')
    if witness.get('status') != 'passed':
        raise RuntimeError('Decoder swap CPU tests missing')
    verify_manifest(ROOT, witness['code_sha256'])
    codec_witness = read(CODEC_AUDIT / 'CPU_TESTS.json')
    if codec_witness.get('status') != 'passed':
        raise RuntimeError('Encoder identity audit witness missing')
    verify_manifest(CODEC_AUDIT, codec_witness['code_sha256'])
    codec_audit = read(CODEC_AUDIT / 'CODEC_REUSE_AUDIT.json')
    if (codec_audit.get('status') != 'complete'
            or codec_audit['models']['decoder_only']['changed_encoder_tensors']
            or sha(ADAPTED) != codec_audit['models']['decoder_only']['checkpoint_sha256']):
        raise RuntimeError('Decoder-only checkpoint or encoder identity differs')
    evaluation_path = CORRECTION / 'sweep_20/EVALUATION.json'
    evaluation = read(evaluation_path)
    if (evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
            or evaluation.get('correction', {}).get('status') != 'applied'
            or evaluation.get('source', {}).get('sweep') != 20):
        raise RuntimeError('Final corrected stage-A evaluation is required')
    rows_reference = {(r['start'], r['horizon']): r for r in evaluation['rows'] if r['horizon'] > 0}
    if not {(s, h) for s in (35, 123) for h in range(1, 10)}.issubset(rows_reference):
        raise RuntimeError('Anchor reference rows are incomplete')
    training = read(B_RUN / 'training/STATUS.json')
    if training.get('status') != 'running' or training.get('completed_updates', 999999) >= 1600:
        raise RuntimeError('Insufficient separation from the stage-B GPU checkpoint preview')
    spec = importlib.util.spec_from_file_location('_swap_stage_a_context', STAGE_A / 'stage_context.py')
    context = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = context
    spec.loader.exec_module(context)
    c, helper, original, p, paths, receipt, previous = context.setup()
    if (evaluation['source']['decoder_digest'] != c['decoder_tensor_digest']
            or evaluation['source']['codec_sha256'] != receipt['source']['codec_sha256']):
        raise RuntimeError('Stage-A state decoder binding differs')
    formal = Path(p['formal_root']).resolve()
    if sha(formal / 'runtime_bridge.py') != p['dependencies']['runtime_bridge.py']:
        raise RuntimeError('Native bridge source changed')
    sys.path.insert(0, str(formal))
    modules = helper.model_modules(original)
    store = context.dataset(modules, p, c)
    from gpu_runtime import setup_gpu_once
    gpu_witness = read(B_ROOT / 'GPU_RUNTIME_TEST.json')
    if gpu_witness.get('status') != 'passed' or gpu_witness.get('runtime_sha256') != sha(B_ROOT / 'gpu_runtime.py'):
        raise RuntimeError('GPU setup witness changed')
    states_root = STAGE_A / 'sweep_20/free_z/evaluation'
    state_files = {(s, h): states_root / f's{s:03d}_h{h:02d}_correct_state.pt'
                   for s in (35, 123) for h in range(1, 10)}
    if any(not path.is_file() for path in state_files.values()):
        raise RuntimeError('A predicted-state artifact is missing')
    out = ROOT / 'result'
    out.mkdir()
    phase = 'gpu_initialization'
    write(out / 'STATUS.json', {'status': 'running', 'phase': phase})
    try:
        setup_gpu_once(p)
        torch.cuda.reset_peak_memory_stats()
        import runtime_bridge
        if Path(runtime_bridge.__file__).resolve() != formal / 'runtime_bridge.py':
            raise RuntimeError('Native bridge imported from an unexpected path')
        from runtime import render
        from data import evaluate_metrics
        bridge = runtime_bridge.BoundNativeBridge(partial=False, device='cuda').eval()
        phase = 'baseline_codec_render'
        began = time.monotonic()
        cached_images = {}
        rows = []
        state_hashes = {}
        with torch.no_grad():
            for start in (35, 123):
                window = store.get(start)
                sample = store.sample(start, 0)
                image, _, _ = render(bridge, window['states'][0], sample['record'], views=(0, 1))
                cached_images[(start, 0)] = image[0].detach().cpu()
                for h in range(1, 10):
                    path = state_files[(start, h)]
                    state = checked_state(path, evaluation['source'])
                    state_hashes[f'{start}:{h}'] = sha(path)
                    state = {key: value.cuda() for key, value in state.items()}
                    sample = store.sample(start, h)
                    image, _, _ = render(bridge, state, sample['record'], views=(0, 1))
                    metrics = evaluate_metrics(image, window, sample, p)
                    expected = rows_reference[(start, h)]['metrics']['correct']['metrics']
                    for key in ('mae', 'motion_mae'):
                        if abs(metrics['metrics'][key] - expected[key]) > 1e-5:
                            raise RuntimeError('Saved predicted state does not reproduce original codec metric')
                    cached_images[(start, h)] = image[0].detach().cpu()
                    rows.append({'start': start, 'horizon': h, 'frame': start + h,
                                 'metrics': {'baseline': metrics}})
                del window
        bridge.assert_frozen()
        phase = 'bind_decoder_only_codec'
        checkpoint = torch.load(ADAPTED, map_location='cpu', weights_only=True)
        if (checkpoint.get('arm') != 'decoder_only' or checkpoint.get('step') != 300
                or checkpoint.get('failed') is not False or not checkpoint.get('eligible_for_final_evaluation')):
            raise RuntimeError('Adapted checkpoint is not the completed decoder-only arm')
        # Restore floating state to checkpoint precision before applying the new
        # weights, then rebuild the audited float64 decode adapter in this process.
        precision_type = type(bridge.precision)
        bridge.codec.float()
        bridge.codec.load_state_dict(checkpoint['model'], strict=True)
        bridge.codec.requires_grad_(False).eval()
        sys.path.insert(0, str(CODEC_AUDIT))
        from audit import encoder_snapshot, snapshot_digest
        encoder_digest = snapshot_digest(encoder_snapshot(bridge.codec.state_dict()))
        if encoder_digest != codec_audit['models']['baseline']['encoder_sha256']:
            raise RuntimeError('Swapped codec no longer has the identical encoder')
        bridge.precision = precision_type(bridge.codec, 'float64')
        bridge._frozen = bridge._frozen_digests()
        bridge.assert_frozen()
        del checkpoint
        phase = 'decoder_only_codec_render'
        with torch.no_grad():
            for start in (35, 123):
                window = store.get(start)
                folder = out / f's{start:03d}'
                folder.mkdir()
                sample = store.sample(start, 0)
                adapted_image, _, _ = render(bridge, window['states'][0], sample['record'], views=(0, 1))
                image_triplet(sample['record']['real'], cached_images[(start, 0)], adapted_image[0]).save(folder / 'frame_00.png')
                for h in range(1, 10):
                    path = state_files[(start, h)]
                    if sha(path) != state_hashes[f'{start}:{h}']:
                        raise RuntimeError('Predicted state changed during decoder comparison')
                    state = {key: value.cuda() for key, value in checked_state(path, evaluation['source']).items()}
                    sample = store.sample(start, h)
                    image, _, _ = render(bridge, state, sample['record'], views=(0, 1))
                    metrics = evaluate_metrics(image, window, sample, p)
                    row = next(x for x in rows if x['start'] == start and x['horizon'] == h)
                    row['metrics']['decoder_only'] = metrics
                    image_triplet(sample['record']['real'], cached_images[(start, h)], image[0]).save(folder / f'frame_{h:02d}.png')
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-i',
                                str(folder / 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                str(out / f's{start:03d}_source_original_decoder_only_5fps.mp4')], check=True, timeout=120)
                del window
        bridge.assert_frozen()
        torch.cuda.synchronize()
        result = {'status': 'complete', 'windows': [35, 123], 'future_predictions': 18,
            'original_codec_motion_mae': mean_metric(rows, 'baseline', 'motion_mae'),
            'decoder_only_codec_motion_mae': mean_metric(rows, 'decoder_only', 'motion_mae'),
            'original_codec_image_mae': mean_metric(rows, 'baseline', 'mae'),
            'decoder_only_codec_image_mae': mean_metric(rows, 'decoder_only', 'mae'),
            'rows': rows, 'same_predicted_state_bytes': True, 'state_sha256': state_hashes,
            'same_camera_and_target': True, 'original_codec_reproduced_stage_a_metrics': True,
            'decoder_only_encoder_digest': encoder_digest, 'decoder_only_checkpoint_sha256': sha(ADAPTED),
            'stage_a_corrected_evaluation_sha256': sha(evaluation_path),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'elapsed_seconds': time.monotonic() - began, 'optimizer_updates': 0,
            'native_replay': 'not performed for decoder-only swap',
            'visual_verdict': 'pending',
            'scope': 'Two known TRAIN-video windows and 18 fixed stage-A free-code predictions; no B/C update, source transfer or AR claim'}
        write(out / 'SWAP_AUDIT.json', result)
        write(out / 'STATUS.json', {'status': 'complete', 'phase': 'awaiting_visual_review'}, replace=True)
        print('PASS: original codec metrics reproduced; two decoders rendered identical predicted states')
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(out / 'FAILED.json', failure)
        write(out / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
