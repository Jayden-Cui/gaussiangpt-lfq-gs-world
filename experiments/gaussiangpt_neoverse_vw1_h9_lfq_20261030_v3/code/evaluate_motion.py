"""Evaluate all H9 posterior bundles from a shared initial continuous GS state.

The 176 overlapping windows have 1548 real target exposures and 36 padding
positions. Metrics exclude padding. The 176-frame video uses non-overlapping
blocks with a real initial-state reset every nine targets; it is not an AR rollout.
"""
import argparse
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from contracts import verify_sources
from motion_common import (BASE, CLIPS, H, W, frozen_codec, load_target, metrics,
                           read, region_metrics, render_points, sha, ssim, write)
from qmotion import build_model, select_support
from sequence_data import load_record, load_window


HORIZON = 9
WINDOWS = 176
FPS = 29.97
CONTROLS = ('held_initial', 'frozen_target', 'correct_motion',
            'shuffled_motion', 'reversed_motion')
PSNR_KEYS = ('psnr', 'workspace_roi_psnr', 'object_psnr', 'hand_activity_psnr')
HEADER = 48


def image_metrics(image, target, boxes):
    result = {**metrics(image, target), **region_metrics(image, target, boxes),
              'ssim': ssim(image, target)}
    return {key: float(value) for key, value in result.items()}


@torch.no_grad()
def decode_prediction(codec, prediction, max_tokens):
    # No target indices enter support selection or the frozen decoder.
    coordinates, features, keep = select_support(prediction, max_tokens=max_tokens)
    points = codec.decode_state(coordinates, features, 4)
    return points, coordinates, features, keep


def mean_metrics(rows, key):
    assert rows, 'Cannot aggregate an empty horizon'
    result = {name: float(np.mean([row[key][name] for row in rows]))
              for name in rows[0][key]}
    assert all(math.isfinite(value) for value in result.values())
    return result


def aggregate(rows):
    return {key: mean_metrics(rows, key) for key in CONTROLS}


def quality_tests(summary, gates, distinct_bundles):
    drops = {key: summary['frozen_target'][key] - summary['correct_motion'][key]
             for key in PSNR_KEYS}
    gains = {key: summary['correct_motion'][key] - summary['held_initial'][key]
             for key in PSNR_KEYS}
    shuffle_drop = (summary['correct_motion']['hand_activity_psnr'] -
                    summary['shuffled_motion']['hand_activity_psnr'])
    tests = {
        'full_and_region_drop': max(drops.values()) <= gates['max_full_and_region_drop_vs_frozen_target_db'],
        'full_gain': gains['psnr'] >= gates['min_full_gain_vs_held_initial_db'],
        'hand_gain': gains['hand_activity_psnr'] >= gates['min_hand_gain_vs_held_initial_db'],
        'shuffle_control': shuffle_drop >= gates['min_hand_drop_when_motion_shuffled_db'],
        'distinct_motion_bundles': distinct_bundles >= gates['min_distinct_motion_id_tuples'],
    }
    return tests, drops, gains, shuffle_drop


def make_panel(images, start, end, horizon, dt):
    labels = ('Real target', 'Held initial @ target camera', 'Frozen target codec',
              'Correct H9 posterior IDs', 'Cross-window IDs (shift 85)', 'Time-reversed H9 IDs')
    panel = Image.new('RGB', (W * 6, H + HEADER), '#16202c')
    draw = ImageDraw.Draw(panel)
    for column, (label, image) in enumerate(zip(labels, images)):
        array = (image.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        assert array.shape == (H, W, 3)
        panel.paste(Image.fromarray(array), (column * W, HEADER))
        draw.text((column * W + 6, 2), label, fill='white')
        draw.text((column * W + 6, 17),
                  f'initial {start:03d} -> {end:03d} | h={horizon} | {dt:.4f}s', fill='#bbcddd')
        draw.text((column * W + 6, 32),
                  'TRAIN posterior | true reset / 9 targets', fill='#ffd18a')
    return panel


class Video:
    def __init__(self, path, fps=FPS):
        self.path = path
        self.frames = 0
        self.error = path.with_suffix('.ffmpeg.log').open('wb')
        try:
            self.process = subprocess.Popen([
                'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{W * 6}x{H + HEADER}',
                '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-threads', '2', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE, stderr=self.error)
        except Exception:
            self.error.close()
            raise

    def write(self, image):
        self.process.stdin.write(np.asarray(image).tobytes())
        self.frames += 1

    def close(self):
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                result = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
                raise
            if result:
                raise RuntimeError(f'ffmpeg failed with exit code {result}: {self.path}')
        finally:
            self.error.close()


def check_time(started, timeout):
    if time.monotonic() - started > timeout:
        raise TimeoutError(f'Evaluation exceeded its {timeout}s budget; no retry or truncation')


@torch.no_grad()
def evaluate(root, arm):
    started = time.monotonic()
    verify_sources(root)
    protocol = read(root / 'protocol.json')
    config = next(item for item in protocol['arms'] if item['name'] == arm)
    assert config['horizon'] == HORIZON and config['motion_tokens'] == 1
    assert config['quantizer'] == 'lfq'
    # This protocol is the single fixed NeoVerse clip, not a variable-length dataset.
    assert len(CLIPS) == 1, 'Expected the registered single 177-state clip'
    resource = protocol.get('resource', {})
    torch.set_num_threads(resource.get('cpu_threads', 6))
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(resource.get('allocator_fraction', .60))
    timeout = protocol.get('time_limits_seconds', {}).get('evaluation', 9000)
    folder = root / arm
    output = folder / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    training = read(folder / 'train_done.json')
    checkpoint = folder / 'motion_final.pt'
    checkpoint_sha = sha(checkpoint)
    assert checkpoint_sha == training['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['protocol_sha256'] == sha(root / 'protocol.json')
    model = build_model(config).cuda().eval()
    model.load_state_dict(saved['model'])
    del saved
    codec = frozen_codec(root)
    regions = read(BASE / 'protocol.json')['evaluation']['region_boxes']
    binding = {
        'motion_checkpoint_sha256': checkpoint_sha,
        'state_checkpoint_sha256': protocol['source_checkpoint_sha256'],
        'protocol_sha256': sha(root / 'protocol.json'),
        'source_binding_sha256': sha(root / 'source_binding.json'),
    }
    window_keys = [(clip, start) for clip in CLIPS for start in range(WINDOWS)]
    expected_exposures = sum(min(HORIZON, WINDOWS - start) for _, start in window_keys)
    assert expected_exposures == 1548
    max_tokens = protocol['support']['max_predicted_tokens']
    zs = {}
    # Encode each full window independently; all h share its initial state.
    # Four windows per call matches the training LFQ batch interface.
    for offset in range(0, len(window_keys), 4):
        check_time(started, timeout)
        keys = window_keys[offset:offset + 4]
        windows = [load_window(root, clip, start) for clip, start in keys]
        encoded = model.encode_windows(windows)
        assert encoded['indices'].shape == (len(keys), HORIZON)
        for index, key in enumerate(keys):
            zs[key] = {name: encoded[name][index:index + 1].detach().cpu()
                       for name in ('z_prequant', 'z_quantized', 'indices')}
            assert zs[key]['z_prequant'].shape == (1, HORIZON, 64)
            assert zs[key]['z_quantized'].shape == (1, HORIZON, 64)
        del windows, encoded
        if offset % 16 == 0:
            print('ENCODE', arm, offset + len(keys), '/', len(window_keys), flush=True)
            write(folder / 'eval_progress.json', {'phase': 'encode', 'windows': len(zs),
                  'total_windows': len(window_keys), 'elapsed_seconds': time.monotonic() - started})
    valid_mask = torch.tensor([[[start + h <= WINDOWS for h in range(1, HORIZON + 1)]
                              for start in range(WINDOWS)]], dtype=torch.bool)
    sequence_ids = torch.stack([zs[(CLIPS[0], start)]['indices'][0] for start in range(WINDOWS)])[None]
    assert tuple(sequence_ids.shape) == (1, 176, 9)
    assert int(valid_mask.sum()) == expected_exposures
    torch.save({
        'schema': 'ar-action-ids-vw1-h9-native64-lfq-v3', 'ids': sequence_ids,
        'flattened_ids': sequence_ids.flatten(1), 'valid_future_mask': valid_mask,
        'window_starts': torch.arange(WINDOWS), 'clips': list(CLIPS), 'binding': binding,
        'fps': FPS, 'vocabulary': model.vocabulary, 'horizon': HORIZON, 'queries_per_horizon': 1,
        'semantics': 'One nine-ID posterior future bundle per window, all relative to its initial state; '
                     'overlapping targets repeat across windows. Not independent adjacent differences or trained AR predictions.',
    }, output / 'ar_motion_z.pt')
    torch.save({
        'schema': 'motion-z-diagnostic-vw1-h9-v3', 'binding': binding,
        'valid_future_mask': valid_mask,
        'windows': [{'clip': clip, 'start': start, **zs[(clip, start)]}
                    for clip, start in window_keys],
        'semantics': 'Continuous pre/post-LFQ features are diagnostics, not inference packet side channels.',
    }, output / 'motion_z.pt')
    rows, manifest, video_manifest = [], [], []
    replay_report = None
    # A rerun must replace the row log, never append duplicate target exposures.
    with (output / 'targets.jsonl').open('w', encoding='utf-8') as log:
        for clip in CLIPS:
            clipdir = output / clip
            clipdir.mkdir(parents=True, exist_ok=True)
            video = Video(clipdir / 'comparison.mp4')
            video_rows = []
            try:
                for start in range(WINDOWS):
                    check_time(started, timeout)
                    record = load_record(root, clip, start, codec)
                    assert len(record['window']) == HORIZON + 1
                    assert record['valid'] == [start + h <= WINDOWS for h in range(1, HORIZON + 1)]
                    valid_horizons = [h for h in range(HORIZON) if record['valid'][h]]
                    targets = {h: load_target(clip, start + h + 1) for h in valid_horizons}
                    initial, condition = record['initial'], record['condition']
                    ids = zs[(clip, start)]['indices'].cuda().long()
                    wrong_start = (start + 85) % WINDOWS
                    wrong_ids = zs[(clip, wrong_start)]['indices'].cuda().long()
                    reverse_ids = ids.flip(1)
                    packet = model.packet(initial, ids, binding=binding, metadata={
                        'clip': clip, 'start': start, 'horizon': HORIZON,
                        'normalization': record['window'][0]['normalization'],
                        'semantics': 'Shared initial state plus nine posterior IDs; no intermediate true states.',
                    })
                    packet_path = clipdir / f'{start:03d}_motion.pt'
                    torch.save(packet, packet_path)
                    manifest.append({'clip': clip, 'start': start, 'packet': packet_path.name,
                                     'motion_packet_sha256': sha(packet_path),
                                     'serialized_packet_bytes': packet_path.stat().st_size,
                                     'valid_horizons': [h + 1 for h in valid_horizons]})
                    predictions = model.decode_ids(ids, [condition])[0]
                    assert len(predictions) == HORIZON
                    replay = None
                    if replay_report is None:
                        replay = model.replay(torch.load(packet_path, weights_only=True, map_location='cpu'),
                                              codec, binding=binding, device='cuda')
                        assert len(replay) == HORIZON
                    held_points = codec.decode_state(initial.coordinates, initial.features, 4)
                    block_selected = start % HORIZON == 0
                    panels, local_rows, roundtrip_horizons = {}, {}, []
                    for h in valid_horizons:
                        prediction, target = predictions[h], targets[h]
                        points, coordinates, features, keep = decode_prediction(codec, prediction, max_tokens)
                        correct = render_points(points, target)
                        nearby = render_points(points, target, 1)
                        held = render_points(held_points, target)
                        frozen = record['window'][h + 1]['reconstruction'][0]
                        label_indices = record['labels'][h]
                        label_mask = torch.zeros_like(keep)
                        label_mask[label_indices] = True
                        tp = int((keep & label_mask).sum())
                        assert len(label_indices) > 0
                        elapsed = float(record['window'][h + 1]['pts'] - record['window'][0]['pts'])
                        local_rows[h] = {
                            'clip': clip, 'start': start, 'end': start + h + 1, 'horizon': h + 1,
                            'valid_target': True, 'dt': elapsed, 'adjacent_dt': float(initial.dt),
                            'held_initial': image_metrics(held, target['real'], regions[clip]),
                            'frozen_target': image_metrics(frozen, target['real'], regions[clip]),
                            'correct_motion': image_metrics(correct, target['real'], regions[clip]),
                            'correct_vs_frozen_target': {k: float(v) for k, v in metrics(correct, frozen).items()},
                            'pseudo_view_teacher': {k: float(v) for k, v in metrics(nearby, target['targets'][1]).items()},
                            'motion_ids': ids[0].cpu().tolist(), 'horizon_motion_id': int(ids[0, h]),
                            'shuffled_source_start': wrong_start, 'shuffled_motion_ids': wrong_ids[0].cpu().tolist(),
                            'reversed_motion_ids': reverse_ids[0].cpu().tolist(),
                            'shuffle_bundle_changed': not torch.equal(ids, wrong_ids),
                            'shuffle_horizon_id_changed': bool(ids[0, h] != wrong_ids[0, h]),
                            'reverse_bundle_changed': not torch.equal(ids, reverse_ids),
                            'reverse_horizon_id_changed': bool(ids[0, h] != reverse_ids[0, h]),
                            'predicted_tokens': len(coordinates), 'candidate_count': len(keep),
                            'occupancy_precision': tp / len(coordinates),
                            'occupancy_recall': tp / len(label_indices),
                            'serialized_packet_bytes': packet_path.stat().st_size,
                            'ideal_motion_bits_per_bundle': HORIZON * math.log2(model.vocabulary),
                            'ideal_motion_bits_per_horizon': math.log2(model.vocabulary),
                        }
                        if block_selected:
                            panels[h] = [target['real'].detach().cpu(), held.detach().cpu(),
                                         frozen.detach().cpu(), correct.detach().cpu(), None, None]
                        if replay is not None:
                            restored = replay[h]
                            assert torch.equal(restored['coordinates'], prediction['coordinates'])
                            feature_error = float((restored['features'] - prediction['features']).abs().max())
                            occupancy_error = float((restored['occupancy_logits'] - prediction['occupancy_logits']).abs().max())
                            rpoints, rcoords, rfeatures, rkeep = decode_prediction(codec, restored, max_tokens)
                            state_exact = (torch.equal(rcoords, coordinates) and torch.equal(rkeep, keep)
                                           and torch.allclose(rfeatures, features, atol=1e-5, rtol=0))
                            rimage = render_points(rpoints, target)
                            render_check = {k: float(v) for k, v in metrics(rimage, correct).items()}
                            assert feature_error < 1e-5 and occupancy_error < 1e-5 and state_exact
                            assert render_check['psnr'] >= 60 and render_check['mae'] <= 1e-5
                            roundtrip_horizons.append({'horizon': h + 1, 'passed': True,
                                'feature_max_abs': feature_error, 'occupancy_max_abs': occupancy_error,
                                'state_coordinates_and_features_equal_within_tolerance': state_exact,
                                'render': render_check, 'render_max_abs': float((rimage - correct).abs().max())})
                            del restored, rpoints, rcoords, rfeatures, rkeep, rimage
                        del prediction, points, coordinates, features, keep, correct, nearby, held, frozen, label_mask
                    if replay is not None:
                        assert len(roundtrip_horizons) == HORIZON
                        replay_report = {'passed': True, 'horizons_checked': HORIZON,
                                         'packet': str(packet_path.relative_to(output)),
                                         'per_horizon': roundtrip_horizons}
                    del predictions, replay, held_points
                    # Decode the entire intervened bundle from the same initial-only
                    # condition. Never substitute a true intermediate state at h>1.
                    for control, intervened_ids, column in (
                            ('shuffled_motion', wrong_ids, 4), ('reversed_motion', reverse_ids, 5)):
                        predictions = model.decode_ids(intervened_ids, [condition])[0]
                        assert len(predictions) == HORIZON
                        for h in valid_horizons:
                            points, coordinates, features, keep = decode_prediction(codec, predictions[h], max_tokens)
                            rendered = render_points(points, targets[h])
                            local_rows[h][control] = image_metrics(rendered, targets[h]['real'], regions[clip])
                            local_rows[h][control + '_predicted_tokens'] = len(coordinates)
                            if block_selected:
                                panels[h][column] = rendered.detach().cpu()
                            del points, coordinates, features, keep, rendered
                        del predictions
                    for h in valid_horizons:
                        row = local_rows[h]
                        rows.append(row)
                        log.write(json.dumps(row, allow_nan=False) + '\n')
                        if block_selected:
                            panel = make_panel(panels[h], start, row['end'], h + 1, row['dt'])
                            video.write(panel)
                            video_rows.append({'video_frame': video.frames - 1, 'initial_frame': start,
                                               'target_frame': row['end'], 'horizon': h + 1})
                            if h == 0 and start % 27 == 0:
                                panel.save(clipdir / f'comparison_{start:03d}.png')
                    log.flush()
                    write(folder / 'eval_progress.json', {
                        'phase': 'render', 'windows': len(manifest), 'total_windows': len(window_keys),
                        'real_target_exposures': len(rows), 'expected_real_target_exposures': expected_exposures,
                        'elapsed_seconds': time.monotonic() - started})
                    print('EVAL', arm, clip, start, 'real_target_exposures', len(rows), flush=True)
                    del record, initial, condition, ids, wrong_ids, reverse_ids, targets, packet, panels, local_rows
            finally:
                video.close()
            assert video.frames == WINDOWS
            assert [item['target_frame'] for item in video_rows] == list(range(1, WINDOWS + 1))
            video_manifest.append({'clip': clip, 'path': f'{clip}/comparison.mp4', 'frames': video.frames,
                'fps': FPS, 'columns': ['real_target', *CONTROLS], 'mapping': video_rows,
                'initial_state_resets': list(range(0, WINDOWS, HORIZON)),
                'semantics': '176 targets shown exactly once, from starts 0,9,...,171; '
                             'real initial state reset each block of up to nine targets. No long-horizon AR claim.'})
    assert len(rows) == expected_exposures and len(manifest) == len(window_keys)
    assert len({(r['clip'], r['start'], r['horizon']) for r in rows}) == expected_exposures
    summary = aggregate(rows)
    summary['per_clip'] = {clip: aggregate([r for r in rows if r['clip'] == clip]) for clip in CLIPS}
    per_horizon = {}
    for horizon in range(1, HORIZON + 1):
        selected = [row for row in rows if row['horizon'] == horizon]
        assert len(selected) == WINDOWS - horizon + 1
        per_horizon[f'h{horizon}'] = {
            'real_target_exposures': len(selected), 'metrics': aggregate(selected),
            'distinct_ids': len({row['horizon_motion_id'] for row in selected}),
            'shuffle_horizon_id_changed_fraction': float(np.mean([row['shuffle_horizon_id_changed'] for row in selected])),
            'reverse_horizon_id_changed_fraction': float(np.mean([row['reverse_horizon_id_changed'] for row in selected])),
            'occupancy_precision': float(np.mean([row['occupancy_precision'] for row in selected])),
            'occupancy_recall': float(np.mean([row['occupancy_recall'] for row in selected])),
        }
    tuples = Counter(tuple(zs[key]['indices'][0].tolist()) for key in window_keys)
    complete_tuples = Counter(tuple(zs[(clip, start)]['indices'][0].tolist())
                              for clip, start in window_keys if start + HORIZON <= WINDOWS)
    flattened = Counter(sequence_ids.flatten().tolist())
    gates = protocol['evaluation']['pass']
    tests, drops, gains, shuffle_drop = quality_tests(summary, gates, len(complete_tuples))
    for report in per_horizon.values():
        horizon_tests, hdrop, hgain, hshuffle = quality_tests(report['metrics'], gates, len(complete_tuples))
        horizon_tests.pop('distinct_motion_bundles')
        report.update(quality_gates=horizon_tests, quality_passed=all(horizon_tests.values()),
                      drop_vs_frozen_target_db=hdrop, gain_vs_held_initial_db=hgain, shuffle_hand_drop_db=hshuffle)
    engineering = {
        'trained_causality': read(folder / 'causality.json')['status'] == 'passed',
        'all_windows': len(manifest) == WINDOWS,
        'all_real_target_exposures': len(rows) == expected_exposures,
        'all_horizons_evaluated': all(f'h{h}' in per_horizon for h in range(1, HORIZON + 1)),
        'padding_excluded_from_metrics': all(row['valid_target'] for row in rows),
        'packet_roundtrip_all_horizons': bool(replay_report and replay_report['passed']),
        'all_training_windows_seen': bool(training['all_windows_seen']),
        'training_without_target_support_occurred': training['no_target_forcing_render_steps'] > 0,
        'ar_action_ids_shape': tuple(sequence_ids.shape) == (1, 176, 9),
        'video_all176_targets_once': all(v['frames'] == 176 for v in video_manifest),
    }
    limitations = list(protocol.get('limitations', [])) + [
        'All image metrics are oracle posterior TRAIN reconstruction, not AR prediction or generalization.',
        'Overall metrics average 1548 real (window,horizon) exposures: repeated target frames have repeated weight; padding is excluded.',
        'The six-column video resets from a real initial state every nine targets; it is not a 176-step generated rollout.',
        'Time reversal is a post-hoc bundle intervention: a slot may retain its ID (h5 always does), '
        'while the decoder can still change through preceding token slots.',
        'The GS decoder uses causal temporal padding as an adaptation; this does not assert that the official VideoWorld1 decoder is causal.',
    ]
    result = {
        'status': 'complete', 'arm': arm, 'horizon': HORIZON, 'motion_tokens': 1, 'queries_per_horizon': 1,
        'ids_per_window': HORIZON, 'quantizer': 'lfq', 'vector_width': 64, 'vocabulary': model.vocabulary,
        'windows': len(manifest), 'real_target_exposures': len(rows), 'unique_real_target_frames': WINDOWS,
        'padded_target_positions_excluded': int((~valid_mask).sum()), 'complete_real_windows': WINDOWS - HORIZON + 1,
        'aggregation': 'Arithmetic mean over valid (window,start+h) exposures; target frames repeat across overlapping windows.',
        'training_steps': training['steps'], 'batch_size': training['batch_size'],
        'window_exposures': training['window_exposures'],
        'future_target_exposures': training['future_target_exposures'],
        'real_future_target_exposures': training['real_future_target_exposures'],
        'elapsed_seconds': time.monotonic() - started, 'binding': binding,
        'metrics': summary, 'per_horizon': per_horizon,
        'drop_vs_frozen_target_db': drops, 'gain_vs_held_initial_db': gains,
        'shuffle_hand_drop_db': shuffle_drop,
        'reverse_hand_drop_db': summary['correct_motion']['hand_activity_psnr'] - summary['reversed_motion']['hand_activity_psnr'],
        'unique_motion_tuples': len(tuples), 'unique_complete_real_motion_tuples': len(complete_tuples),
        'distinct_bundle_gate_population': '168 windows with nine real future targets; padded windows excluded',
        'unique_motion_codes': len(flattened),
        'per_horizon_distinct_ids_valid': [per_horizon[f'h{h}']['distinct_ids'] for h in range(1, HORIZON + 1)],
        'per_horizon_distinct_ids_including_padding': [len(set(sequence_ids[0, :, h].tolist())) for h in range(HORIZON)],
        'top_motion_tuples': [{'ids': list(key), 'count': value} for key, value in tuples.most_common(20)],
        'shuffle_changed_fraction': float(np.mean([not torch.equal(zs[(clip, start)]['indices'],
                                        zs[(clip, (start + 85) % WINDOWS)]['indices']) for clip, start in window_keys])),
        'reverse_changed_fraction': float(np.mean([not torch.equal(zs[key]['indices'], zs[key]['indices'].flip(1)) for key in window_keys])),
        'occupancy_precision': float(np.mean([row['occupancy_precision'] for row in rows])),
        'occupancy_recall': float(np.mean([row['occupancy_recall'] for row in rows])),
        'roundtrip': replay_report, 'engineering_gates': engineering, 'engineering_passed': all(engineering.values()),
        'quality_gates': tests, 'quality_passed': all(tests.values()),
        'all_horizon_quality_passed': all(report['quality_passed'] for report in per_horizon.values()),
        'acceptance_passed': all(engineering.values()) and all(tests.values()),
        'video': {'frames': WINDOWS, 'fps': FPS, 'manifest': 'evaluation/video_manifest.json',
                  'reset_interval': HORIZON, 'long_horizon_autoregressive': False},
        'limitations': limitations,
    }
    write(output / 'manifest.json', manifest)
    write(output / 'video_manifest.json', video_manifest)
    write(folder / 'evaluation.json', result)
    print('EVAL_COMPLETE', json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    evaluate(args.root, args.arm)
