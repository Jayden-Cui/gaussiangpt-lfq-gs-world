"""Compare predicted state and true-future codec on the same TRAIN frame/camera."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
DATA = OUTPUTS / 'gaussiangpt_neoverse_multisource_action_data_20261102_v1'
PROBE = OUTPUTS / 'gaussiangpt_neoverse_multisource_gpu_probe_20261102_v1'
MULTISOURCE_CODEC = OUTPUTS / 'gaussiangpt_neoverse_multisource_codec_20261031_v1/evaluations/baseline'
STAGE_A = OUTPUTS / 'gaussiangpt_neoverse_action_stage_a_20261102_v1'
B_ROOT = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN = OUTPUTS / 'gaussiangpt_neoverse_action_stage_b_20261102_v1'
sys.path.insert(0, str(DATA))
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


def find_frame(path, clip, frame):
    result = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('clip') == clip and row.get('frame') == frame:
                result.append(row)
    if len(result) != 1 or result[0].get('split') != 'TRAIN':
        raise RuntimeError('Expected exactly one bound training-source frame')
    return result[0]


def verify_manifest(root, expected):
    if sha(root / 'CODE.sha256') != expected:
        raise RuntimeError('Source manifest changed')
    for line in (root / 'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or sha(root / name) != digest:
            raise RuntimeError('Source file changed: ' + name)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    test = read(ROOT / 'CPU_TESTS.json')
    if test.get('status') != 'passed':
        raise RuntimeError('Reference CPU test missing')
    verify_manifest(ROOT, test['code_sha256'])
    previous_test = read(PROBE / 'CPU_TESTS.json')
    if previous_test.get('status') != 'passed':
        raise RuntimeError('Preceding GPU probe source witness missing')
    verify_manifest(PROBE, previous_test['code_sha256'])
    previous = read(PROBE / 'result/GPU_PROBE.json')
    prior_image = PROBE / 'result/source_vs_posterior_h09.png'
    if (previous.get('status') != 'passed' or previous.get('split') != 'TRAIN'
            or previous.get('start') != 26 or previous.get('horizon') != 9
            or sha(prior_image) != previous.get('source_image_sha256')):
        raise RuntimeError('Preceding train-source diagnostic changed')
    from source_store import SourceStore, TRAIN, EVAL, CODEC_SHA
    from cross_source_metrics import image_metrics_unmasked_teacher
    from gpu_runtime import setup_gpu_once
    clip, frame = TRAIN[1], 35
    if clip in EVAL or previous['clip'] != clip:
        raise RuntimeError('Reference must use the same training-source frame')
    store = SourceStore()
    historic = read(MULTISOURCE_CODEC / 'EVALUATION.json')
    targets = MULTISOURCE_CODEC / 'targets.jsonl'
    if (historic.get('status') != 'evaluated' or historic.get('checkpoint_sha256') != CODEC_SHA
            or sha(targets) != historic.get('targets_sha256')):
        raise RuntimeError('Historic multi-source codec evaluation changed')
    old_row = find_frame(targets, clip, frame)
    spec = importlib.util.spec_from_file_location('_cross_source_reference_context', STAGE_A / 'stage_context.py')
    context = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = context
    spec.loader.exec_module(context)
    c, helper, original, p, paths, receipt, evaluation = context.setup()
    if receipt['source']['codec_sha256'] != CODEC_SHA:
        raise RuntimeError('Native codec checkpoint differs')
    formal_root = Path(p['formal_root']).resolve()
    bridge_path = formal_root / 'runtime_bridge.py'
    if sha(bridge_path) != p['dependencies']['runtime_bridge.py']:
        raise RuntimeError('Native bridge source changed')
    sys.path.insert(0, str(formal_root))
    training = read(B_RUN / 'training/STATUS.json')
    if training.get('status') != 'running' or training.get('completed_updates', 999999) >= 1800:
        raise RuntimeError('Insufficient separation from the scheduled checkpoint preview')
    output = ROOT / 'result'
    output.mkdir()
    phase = 'gpu_initialization'
    write(output / 'STATUS.json', {'status': 'running', 'phase': phase})
    try:
        setup_gpu_once(p)
        torch.cuda.reset_peak_memory_stats()
        import runtime_bridge
        if Path(runtime_bridge.__file__).resolve() != bridge_path:
            raise RuntimeError('Native bridge imported from the wrong source')
        BoundNativeBridge = runtime_bridge.BoundNativeBridge
        from runtime import render
        from PIL import Image
        sys.path.insert(0, str(PROBE))
        from probe import image_pair
        bridge = BoundNativeBridge(partial=False, device='cuda').eval()
        began = time.monotonic()
        initial = store.load_frame(clip, 26, purpose='train', device='cuda')
        target = store.load_frame(clip, frame, purpose='train', device='cuda')
        phase = 'true_future_state_codec_render'
        with torch.no_grad():
            images, gs, sh = render(bridge, target['state'], target['sample']['record'], views=(0, 1))
            roi = torch.zeros((252, 448), dtype=torch.bool, device='cuda')
            for x0, y0, x1, y1 in p['hand_roi_xyxy']:
                roi[y0:y1, x0:x1] = True
            sample = target['sample']
            sample['motion'] = ((sample['record']['real'] - initial['sample']['record']['real']).abs().mean(-1) / .1).clamp(0, 1)
            metrics = image_metrics_unmasked_teacher(images, {'roi': roi}, sample)
            error = abs(metrics['metrics']['mae'] - old_row['codec_real']['mae'])
            if error > 1e-5:
                raise RuntimeError('True-state codec does not reproduce historic full-image reference')
            current_pair = image_pair(sample['record']['real'], images[0])
            current_pair.save(output / 'source_vs_actual_codec_h09.png')
            with Image.open(prior_image) as old:
                if old.size != (896, 252) or current_pair.size != old.size:
                    raise RuntimeError('Source/prediction image shape differs')
                if old.crop((0, 0, 448, 252)).tobytes() != current_pair.crop((0, 0, 448, 252)).tobytes():
                    raise RuntimeError('Predicted and codec references use different source images')
                panel = Image.new('RGB', (1344, 252))
                panel.paste(old, (0, 0))
                panel.paste(current_pair.crop((448, 0, 896, 252)), (896, 0))
                panel.save(output / 'source_posterior_actual_codec_h09.png')
        torch.cuda.synchronize()
        bridge.assert_frozen()
        result = {'status': 'complete', 'clip': clip, 'split': 'TRAIN', 'initial_frame': 26,
            'target_frame': frame, 'same_source_and_camera': True,
            'posterior_prediction': previous['metrics'], 'actual_target_codec': metrics,
            'historic_codec_real_mae': old_row['codec_real']['mae'], 'historic_codec_real_psnr': old_row['codec_real']['psnr'],
            'historic_full_mae_abs_difference': error, 'codec_target_gaussians': len(gs),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'elapsed_seconds': time.monotonic() - began, 'no_eval_source_read': True, 'no_optimizer_update': True,
            'three_panel_sha256': sha(output / 'source_posterior_actual_codec_h09.png'),
            'historic_targets_sha256': sha(targets), 'prior_probe_sha256': sha(PROBE / 'result/GPU_PROBE.json'),
            'codec_sha256': CODEC_SHA, 'scope': 'One TRAIN-source frame; target codec uses true future GS and is an offline reference'}
        write(output / 'REFERENCE.json', result)
        write(output / 'STATUS.json', {'status': 'complete', 'phase': 'reference_ready'}, replace=True)
        print('PASS: same-camera true-future codec matches historic reference; no optimizer update')
    except BaseException as error:
        failure = {'status': 'failed', 'phase': phase, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'no_automatic_retry': True}
        write(output / 'FAILED.json', failure)
        write(output / 'STATUS.json', failure, replace=True)
        raise


if __name__ == '__main__':
    main()
