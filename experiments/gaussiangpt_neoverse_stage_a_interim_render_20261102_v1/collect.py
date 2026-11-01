"""Collect valid intermediate A metrics, excluding the known-bad S0 control."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
A = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
CONTROLS = ('correct', 'baseline_qformer', 'actual_target', 'swapped_window_codes')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    outcomes = {}
    for sweep in (5, 10):
        output = A / f'sweep_{sweep:02d}' / 'free_z'
        folder = output / 'evaluation'
        metrics_path = output / 'STAGE_METRICS.json'
        eval_path = folder / 'EVALUATION.json'
        replay_path = folder / 'REPLAY.json'
        merged_path = A / f'sweep_{sweep:02d}' / 'merged_codes.pt'
        metrics, evaluation, replay = map(read, (metrics_path, eval_path, replay_path))
        if (metrics.get('status') != 'complete' or metrics.get('sweep') != sweep
                or evaluation.get('status') != 'complete' or evaluation.get('window_predictions') != 1512
                or evaluation.get('unique_future_frames') != 176
                or evaluation.get('source', {}).get('sweep') != sweep
                or evaluation.get('source', {}).get('merged_codes_sha256') != sha(merged_path)
                or evaluation.get('decoder_future_geometry_input') is not False
                or replay.get('status') != 'passed' or replay.get('packets') != 104
                or replay.get('views') != 208 or replay.get('packets_sha256') != evaluation.get('packets_sha256')):
            raise RuntimeError('Incomplete or mismatched intermediate evaluation: ' + str(sweep))
        future = [r for r in evaluation['rows'] if r['horizon'] > 0]
        if len(future) != 1512 or {(r['start'], r['horizon']) for r in future} != {
                (s, h) for s in range(168) for h in range(1, 10)}:
            raise RuntimeError('Intermediate window/horizon coverage differs')
        controls = metrics['metrics']['unique_frame_equal']
        if any(name not in controls for name in CONTROLS):
            raise RuntimeError('A valid control is missing')
        free = controls['correct']['motion_mae']
        baseline = controls['baseline_qformer']['motion_mae']
        if abs((1 - free / baseline) - metrics['global_motion_mae_gain_vs_baseline']) > 1e-9:
            raise RuntimeError('Reported motion-MAE gain does not match raw means')
        outcomes[str(sweep)] = {
            'metrics_unique_frame_equal': {name: controls[name] for name in CONTROLS},
            'motion_mae_gain_vs_original_qformer': metrics['global_motion_mae_gain_vs_baseline'],
            'by_horizon_window_equal': {h: {name: values[name] for name in CONTROLS}
                                        for h, values in metrics['by_horizon'].items()},
            'dynamic_anchors': metrics['dynamic_anchors'],
            'anchor_motion_mae': metrics['anchor_motion_mae'],
            'baseline_anchor_motion_mae': metrics['baseline_anchor_motion_mae'],
            'preliminary_known_anchor_proxy_pass': metrics['preliminary_known_anchor_proxy_pass'],
            'native_replay': {k: replay[k] for k in ('status', 'packets', 'views', 'max_mae')},
            'source': {'metrics_sha256': sha(metrics_path), 'evaluation_sha256': sha(eval_path),
                       'replay_sha256': sha(replay_path), 'merged_codes_sha256': sha(merged_path)}}
    result = {
        'status': 'intermediate_complete', 'sweeps': outcomes,
        'scope': '168 overlapping windows from one TRAIN video; 176 distinct future frames equally weighted',
        'optimized_free_code_scope': 'Codes are optimized against true future observations; not a trained posterior or online prediction',
        'excluded_control': 'initial_only: original stage-A evaluator reused correct-code predictions; a separate repair is pending',
        'color_proxy_scope': 'HSV/ROI proxy, not ground-truth object tracking or 3D pose',
        'final_sweep_20_and_visual_decision': 'pending',
        'collector_sha256': sha(ROOT / 'collect.py')}
    with (ROOT / 'INTERIM_RENDER.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS: valid 5/10-sweep results collected; incorrect initial-only control excluded')


if __name__ == '__main__':
    main()
