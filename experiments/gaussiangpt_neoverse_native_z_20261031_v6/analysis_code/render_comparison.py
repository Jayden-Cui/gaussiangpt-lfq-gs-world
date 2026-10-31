"""Completed-v6 four-panel posterior reconstruction, not an AR rollout.

No training or main-code edits. All 177 physical frames appear once. Every
panel remains 448x252; labels occupy added space. The source's hash-bound
nominal time step determines the exact rational video rate (currently 2997/100).
Run --self-test for standard-library-only checks without torch, files, or GPU.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

FRAME_COUNT = 177
WINDOW_COUNT = 168
WIDTH, HEIGHT = 448, 252
HEADER_HEIGHT, PANEL_HEADER_HEIGHT = 52, 28
VIDEO_SIZE = (WIDTH * 2, HEADER_HEIGHT + 2 * (PANEL_HEADER_HEIGHT + HEIGHT))
SNAPSHOTS = (0, 44, 88, 132, 176)
MAX_SECONDS = 600
ACTION_SEMANTICS = ('Posterior labels or explicitly marked constant-control IDs; '
                    'not a flattened physical adjacent-action stream or complete AR sample')
CONTROLS = ('adapted_held', 'correct', 'shuffle84')
METRICS = ('psnr', 'hand_activity_psnr', 'mae', 'hand_activity_mae', 'motion_mae')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def frame_mapping():
    """The fixed requested physical-frame map; no repeated or dropped tail."""
    rows, previous = [], None
    for frame in range(FRAME_COUNT):
        start = 0 if frame == 0 else min(((frame - 1) // 9) * 9, 167)
        horizon = frame - start
        rows.append(dict(source_frame=frame, start=start, horizon=horizon,
                         source_initial_frame=start, correct_bundle_start=start,
                         shuffle84_bundle_start=(start + 84) % WINDOW_COUNT,
                         reset=previous != start,
                         reset_reason='first_frame' if frame == 0 else
                         ('new_true_initial_window' if previous != start else None)))
        previous = start
    return rows


def rational_fps(nominal_delta):
    if (type(nominal_delta) not in (int, float) or not math.isfinite(nominal_delta)
            or nominal_delta <= 0):
        raise ValueError('Invalid bound source nominal_delta_seconds')
    fps = Fraction(1.0 / nominal_delta).limit_denominator(1_000_000)
    if not math.isclose(float(1 / fps), nominal_delta, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Source nominal time step has no sufficiently exact rational rate')
    return fps


def validate_times(pts, fps):
    """Preserve cached source PTS; constant-rate export is allowed only if equal."""
    if len(pts) != FRAME_COUNT or not isinstance(fps, Fraction) or fps <= 0:
        raise ValueError('Expected 177 source timestamps and a positive rational rate')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in pts):
        raise ValueError('Nonfinite/invalid source PTS')
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError('Repeated or decreasing source PTS')
    presentation = [value - pts[0] for value in pts]
    max_error = max(abs(value - float(Fraction(frame, 1) / fps))
                    for frame, value in enumerate(presentation))
    if max_error > 1e-7:
        raise ValueError('Source PTS do not match the bound constant rate; no resampling is allowed')
    return dict(source_pts_origin=pts[0], source_pts_last=pts[-1],
                presentation_pts=presentation, pts_max_alignment_error_seconds=max_error,
                fps_numerator=fps.numerator, fps_denominator=fps.denominator,
                ffmpeg_framerate=f'{fps.numerator}/{fps.denominator}', fps=float(fps),
                source_endpoint_span_seconds=presentation[-1],
                encoded_duration_seconds=float(Fraction(FRAME_COUNT, 1) / fps),
                last_frame_duration_seconds=float(1 / fps),
                pts_scope='Hash-bound cached nominal PTS; original MP4 container PTS not independently verified')


def artifact_names(arm, fps):
    if arm not in ('partial', 'frozen'):
        raise ValueError('Only completed posterior partial/frozen arms are supported')
    stem = f'{arm}_posterior_reset_177f_fps{fps.numerator}_{fps.denominator}'
    return dict(video=stem + '.mp4',
                snapshots={frame: f'{stem}_frame_{frame:03d}.png' for frame in SNAPSHOTS})


def valid_output_path(root, output):
    allowed = (root / 'diagnostics' / 'render_comparison').resolve()
    return output != allowed and output.is_relative_to(allowed)


def encoding_progress(text):
    blocks, current = [], {}
    for line in text.splitlines():
        key, separator, value = line.partition('=')
        if not separator:
            continue
        current[key.strip()] = value.strip()
        if key.strip() == 'progress':
            blocks.append(current)
            current = {}
    if not blocks or blocks[-1].get('progress') != 'end':
        raise ValueError('ffmpeg did not report a completed stream')
    final = blocks[-1]
    if int(final.get('frame', '-1')) != FRAME_COUNT:
        raise ValueError('ffmpeg frame count is not 177')
    return final


def aggregate(rows):
    def means(observations):
        return {control: {key: statistics.fmean(row['metrics'][control][key] for row in observations)
                          for key in METRICS} for control in CONTROLS}
    return dict(all177_physical_frames=means(rows), future176_physical_frames=means(rows[1:]),
                frame0=rows[0]['metrics'],
                unit='Each physical frame occurs once, using the declared periodic true-initial reset mapping')


def compose_panel(images, row, arm, fps, Image, ImageDraw, font):
    canvas = Image.new('RGB', VIDEO_SIZE, (18, 20, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), 'POSTERIOR RECONSTRUCTION | periodic TRUE-initial reset | NOT AR ROLLOUT',
              font=font, fill=(255, 220, 130))
    flag = ' | RESET' if row['reset'] else ''
    draw.text((8, 25), f'{arm} | frame {row["source_frame"]:03d}/176 | initial {row["start"]:03d}'
              f' | h={row["horizon"]} | source PTS {row["source_pts"]:.6f}s | {float(fps):g} fps{flag}',
              font=font, fill=(235, 235, 235))
    labels = ('REAL RGB', 'HOLD INITIAL | same adapted codec',
              'CORRECT posterior ID bundle', f'SHUFFLE84 | bundle start {row["shuffle84_bundle_start"]:03d} | same h')
    for index, (image, label) in enumerate(zip(images, labels)):
        if image.size != (WIDTH, HEIGHT):
            raise ValueError('A panel was resized or has the wrong native dimensions')
        x = (index % 2) * WIDTH
        y = HEADER_HEIGHT + (index // 2) * (PANEL_HEADER_HEIGHT + HEIGHT)
        draw.text((x + 8, y + 6), label, font=font, fill=(230, 235, 242))
        canvas.paste(image, (x, y + PANEL_HEADER_HEIGHT))
    return canvas


def run(root, arm, output, max_seconds=MAX_SECONDS):
    begun = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    stage, completed_frames, process = 'verify_completed_inputs', 0, None
    streams = []
    previous_alarm = None

    def remaining():
        seconds = max_seconds - (time.monotonic() - begun)
        if seconds <= 0:
            raise TimeoutError(f'{max_seconds:g}-second comparison budget exhausted')
        return seconds

    def deadline(signum, frame):
        raise TimeoutError(f'{max_seconds:g}-second comparison budget exhausted (SIGALRM)')

    try:
        if hasattr(signal, 'setitimer'):
            previous_alarm = signal.signal(signal.SIGALRM, deadline)
            signal.setitimer(signal.ITIMER_REAL, max_seconds)
        # This helper imports only the standard library and validates all 1512
        # future records plus 168 h0 records. Bind its exact content as well.
        from future_information_probe import verify_main_manifest, load_completed_evidence, check_rerender
        helper_path = Path(__file__).with_name('future_information_probe.py')
        helper_hash = sha(helper_path)
        main_manifest_hash = verify_main_manifest(root)
        evidence = load_completed_evidence(root, arm)
        evaluation = evidence['evaluation']
        if (evaluation.get('scope') != 'single_clip_TRAIN_posterior_reconstruction'
                or evaluation.get('windows') != 168 or evaluation.get('real_future_exposures') != 1512
                or evaluation.get('unique_future_frames') != 176
                or evaluation.get('physical_frames_with_some_reconstruction') != 177
                or evidence['plan'].get('partial_native') is not (arm == 'partial')):
            raise ValueError('Completed evaluation/arm semantics differ from the declared v6 task')
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise FileNotFoundError('System ffmpeg is required; no installation is attempted')
        version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True,
                                 timeout=min(5., remaining()), check=True).stdout.splitlines()[0]
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default()
        # Heavy imports and GPU setup happen only after terminal artifacts pass.
        sys.path.insert(0, str(root / 'code'))
        import torch
        import native_experiment as core

        actions = torch.load(evidence['directory'] / 'actions.pt', map_location='cpu', weights_only=True)
        ids = actions.get('ids')
        if (actions.get('schema') != 'common_initial_h9_bundle_v1'
                or actions.get('code_mode') != 'posterior'
                or actions.get('constant_id') is not None
                or actions.get('checkpoint_sha256') != evidence['final_hash']
                or actions.get('starts') != list(range(168))
                or actions.get('horizons') != list(range(1, 10))
                or actions.get('all_windows_complete') is not True
                or actions.get('source_frames') != FRAME_COUNT or actions.get('frame_gap') != 1
                or actions.get('semantics') != ACTION_SEMANTICS
                or not torch.is_tensor(ids) or ids.shape != (168, 9)
                or ids.dtype not in (torch.int32, torch.int64)
                or not bool(((ids >= 0) & (ids < 65536)).all())):
            raise ValueError('Invalid checkpoint-bound posterior actions.pt export')
        id_list = ids.tolist()
        for (start, horizon), reference in evidence['correct'].items():
            if reference.get('motion_ids') != id_list[start]:
                raise ValueError('Exported IDs differ from those used by completed evaluation')

        # The old motion protocol is a hashed dependency of the immutable native
        # binding. Its nominal step, not a rounded display FPS, defines timing.
        binding_path = core.V4_ROOT / 'source_binding.json'
        binding = json.loads(binding_path.read_text())
        source_protocol_path = core.MOTION_ROOT / 'protocol.json'
        dependency = next((row for row in binding['dependencies']
                           if row['path'] == str(source_protocol_path)), None)
        if dependency is None or sha(source_protocol_path) != dependency['sha256']:
            raise ValueError('Source timing protocol is absent from or differs from native binding')
        source_protocol = json.loads(source_protocol_path.read_text())
        fps = rational_fps(source_protocol['nominal_delta_seconds'])
        extra_hashes = {str(binding_path): sha(binding_path), str(source_protocol_path): sha(source_protocol_path)}
        names = artifact_names(arm, fps)
        remaining()
        stage = 'load_native_data_and_model'
        cfg = core.setup(root)
        if (cfg['data']['frames'] != FRAME_COUNT or cfg['data']['resolution_hw'] != [HEIGHT, WIDTH]
                or cfg['data']['frame_gap'] != 1):
            raise ValueError('Main data contract changed')
        torch.cuda.reset_peak_memory_stats()
        data = core.Data()
        if len(data.states) != FRAME_COUNT or len(data.records) != FRAME_COUNT:
            raise ValueError('Data loader omitted or added physical frames')
        times = validate_times([float(state['pts']) for state in data.states], fps)
        for frame, state in enumerate(data.states):
            if state.get('frame_index') != frame:
                raise ValueError('Loaded frame identity is not the original physical frame')
            if 'source_pts' in state and not math.isclose(float(state['source_pts']), float(state['pts']),
                                                         rel_tol=0, abs_tol=1e-7):
                raise ValueError('Stored source_pts and pts differ; do not silently replace timing')
        model, bridge, optimizer = core.build(cfg, data, arm)
        del optimizer  # No zero_grad/backward/step is ever called.
        for action_key, binding_key in [('cache_manifest_sha256', 'cache_manifest_sha256'),
                                        ('state_source_sha256', 'state_checkpoint_sha256')]:
            if actions.get(action_key) != bridge.binding[binding_key]:
                raise ValueError('Actions and native source binding disagree: ' + action_key)
        checkpoint = torch.load(evidence['directory'] / 'final.pt', map_location='cpu', weights_only=False)
        if (checkpoint.get('config') != cfg or checkpoint.get('completed_steps') != evidence['train']['steps']
                or checkpoint.get('plan') != evidence['plan'].get('starts')):
            raise ValueError('Checkpoint config/steps/plan differ from completed evidence')
        model.load_state_dict(checkpoint['model'], strict=True)
        bridge.codec.load_state_dict(checkpoint['codec'], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        bridge.eval().requires_grad_(False)
        if model.code_mode != 'posterior' or bridge.codec.ae.decoder.training:
            raise ValueError('Expected an eval-mode posterior decoder without target forcing')
        if (bridge.codec.protocol['max_decoder_candidates'] != cfg['resources']['max_decoder_candidates']
                or bridge.codec.protocol['max_decoded_gaussians'] != cfg['resources']['max_decoded_gaussians']
                or len(bridge.codec._guards) != len(bridge.codec.ae.decoder.ups) + 1):
            raise ValueError('Native support guards differ from the fixed experiment')
        remaining()

        video_path = output / names['video']
        command = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'warning',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
                   f'{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}', '-framerate', times['ffmpeg_framerate'],
                   '-i', 'pipe:0', '-an', '-frames:v', str(FRAME_COUNT), '-c:v', 'libx264',
                   '-preset', 'fast', '-threads', '6', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-progress', 'pipe:1', str(video_path)]
        stderr = (output / 'ffmpeg.log').open('x', encoding='utf-8')
        progress = (output / 'encoding_progress.txt').open('x', encoding='utf-8')
        streams.extend([stderr, progress])
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=progress, stderr=stderr)
        mapping = frame_mapping()
        rows, current_start, cached = [], None, None
        decode_calls = 0
        stage = 'render_and_encode177'
        with torch.no_grad(), (output / 'frame_mapping_metrics.jsonl').open('x', encoding='utf-8') as log:
            for index, base in enumerate(mapping):
                remaining()
                row = dict(base, source_pts=float(data.states[index]['pts']),
                           presentation_pts=times['presentation_pts'][index],
                           initial_source_pts=float(data.states[base['start']]['pts']),
                           clip=core.CLIP, conditioning_uses_true_initial=True,
                           correct_ids=id_list[base['start']],
                           shuffle84_ids=id_list[base['shuffle84_bundle_start']])
                start, horizon = row['start'], row['horizon']
                if start != current_start:
                    condition = [core.make_condition(data.states[start])]
                    cached = {name: model.decode_ids(ids[bundle:bundle+1].to('cuda'), condition)[0]
                              for name, bundle in [('correct', start), ('shuffle84', row['shuffle84_bundle_start'])]}
                    decode_calls += 2
                    for sequence in cached.values():
                        if len(sequence) != 10:
                            raise RuntimeError('ID replay omitted a learned time slot')
                        for prediction in sequence:
                            if (not torch.equal(prediction['coordinates'], data.states[start]['coordinates'])
                                    or prediction['features'].shape != data.states[start]['features'].shape):
                                raise RuntimeError('ID replay changed or dropped initial latent coordinates')
                    current_start = start
                real = data.records[index]['real']
                if (real.shape != (HEIGHT, WIDTH, 3) or not bool(torch.isfinite(real).all())
                        or bool((real < 0).any()) or bool((real > 1).any())):
                    raise ValueError('Invalid native real RGB; no resize/normalization fallback')
                tensors, metrics, gs_counts = [real], {}, {}
                for control, prediction in [('adapted_held', data.states[start]),
                                             ('correct', cached['correct'][horizon]),
                                             ('shuffle84', cached['shuffle84'][horizon])]:
                    remaining()
                    image = bridge.render(prediction, data.records[index])
                    metrics[control] = data.metrics(image, start, horizon)
                    if any(not math.isfinite(metrics[control][key]) for key in METRICS):
                        raise ValueError('Nonfinite replay metric')
                    gs_counts[control] = bridge.last_count
                    tensors.append(image)
                reference = evidence['initial'][start] if horizon == 0 else evidence['correct'][(start, horizon)]
                deltas = {control: {key: metrics[control][key] - reference[control][key]
                                    for key in METRICS if key in reference[control]}
                          for control in CONTROLS if control in reference}
                for control in CONTROLS:
                    if control in reference:
                        check_rerender(metrics[control], reference[control])
                row.update(metrics=metrics, decoded_gaussians=gs_counts,
                           replay_minus_completed_evaluation=deltas)
                panels = [Image.fromarray(t.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
                          for t in tensors]
                canvas = compose_panel(panels, row, arm, fps, Image, ImageDraw, font)
                if index in SNAPSHOTS:
                    with (output / names['snapshots'][index]).open('xb') as image_file:
                        canvas.save(image_file, format='PNG')
                remaining()
                process.stdin.write(canvas.tobytes())
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                rows.append(row)
                completed_frames += 1
                if torch.cuda.max_memory_allocated()/2**30 > cfg['resources']['peak_limit_gib']:
                    raise RuntimeError('GPU allocation exceeds the fixed resource limit')
                if index % 44 == 0:
                    print('RENDER_COMPARISON', arm, completed_frames, '/177', flush=True)
                del tensors, panels, canvas, image
        stage = 'finalize_video_and_recheck_hashes'
        process.stdin.close()
        exit_code = process.wait(timeout=remaining())
        process = None
        for stream in streams:
            stream.close()
        streams.clear()
        if exit_code != 0:
            raise RuntimeError(f'ffmpeg failed with exit code {exit_code}; partial output retained')
        progress_result = encoding_progress((output / 'encoding_progress.txt').read_text())
        if completed_frames != FRAME_COUNT or [row['source_frame'] for row in rows] != list(range(FRAME_COUNT)):
            raise RuntimeError('Physical-frame coverage is incomplete')
        if decode_calls != 2*len({row['start'] for row in mapping}):
            raise RuntimeError('Unexpected repeated window decoding')
        # ffprobe is optional; encoder progress always confirms the frame count.
        # If installed, additionally inspect actual MP4 dimensions/rate/count.
        ffprobe = shutil.which('ffprobe')
        video_probe = None
        if ffprobe:
            probe = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                    '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration',
                                    '-of', 'json', str(video_path)], capture_output=True, text=True,
                                   timeout=min(15., remaining()), check=True)
            video_probe = json.loads(probe.stdout)['streams'][0]
            if (video_probe['width'] != VIDEO_SIZE[0] or video_probe['height'] != VIDEO_SIZE[1]
                    or int(video_probe['nb_frames']) != FRAME_COUNT
                    or Fraction(video_probe['avg_frame_rate']) != fps):
                raise RuntimeError('Encoded MP4 geometry/frame count/rate does not match the declared mapping')
        remaining()
        if verify_main_manifest(root) != main_manifest_hash or sha(helper_path) != helper_hash:
            raise RuntimeError('Source code changed during comparison')
        for name, expected in evidence['input_hashes'].items():
            if sha(evidence['directory'] / name) != expected:
                raise RuntimeError('Completed arm artifact changed during comparison: ' + name)
        for path, expected in extra_hashes.items():
            if sha(path) != expected:
                raise RuntimeError('Native timing source changed during comparison: ' + path)
        remaining()
        artifacts = {names['video']: sha(video_path),
                     **{name: sha(output/name) for name in names['snapshots'].values()},
                     'frame_mapping_metrics.jsonl': sha(output/'frame_mapping_metrics.jsonl')}
        summary = dict(status='complete', arm=arm, optimizer_updates=0,
                       semantics='Periodic true-initial reset posterior window reconstruction; NOT AR rollout',
                       physical_frames=FRAME_COUNT, unique_physical_frames=FRAME_COUNT,
                       completed_steps=evidence['train']['steps'],
                       window_starts_used=sorted({row['start'] for row in mapping}),
                       resets=[row['source_frame'] for row in mapping if row['reset']],
                       latent_window_decode_calls=decode_calls, native_render_calls=FRAME_COUNT*3,
                       panel_native_size_wh=[WIDTH, HEIGHT], video_size_wh=list(VIDEO_SIZE),
                       panel_order=['real_rgb', 'adapted_held', 'correct', 'shuffle84'],
                       time_contract=times, metrics=aggregate(rows),
                       checkpoint_sha256=evidence['final_hash'], actions_sha256=evidence['input_hashes']['actions.pt'],
                       main_code_manifest_sha256=main_manifest_hash, input_hashes=evidence['input_hashes'],
                       timing_source_hashes=extra_hashes, native_source_binding=bridge.binding,
                       script_sha256=sha(Path(__file__)), verification_helper_sha256=helper_hash,
                       ffmpeg_version=version, ffmpeg_command=command,
                       encoder_final_progress=progress_result, ffprobe=video_probe,
                       artifact_sha256=artifacts, elapsed_seconds=time.monotonic()-begun,
                       gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30, max_seconds=max_seconds,
                       limitations=['One admitted TRAIN clip and posterior codes, not future prediction or AR',
                                    'True initial state is reset at each declared window; tail starts at167,h5',
                                    'Original native hard pruning/guards remain, all initial latent inputs retained',
                                    'Metrics are recomputed on float RGB before video compression; replay deltas are logged',
                                    'Encoded duration includes one final frame duration beyond first-to-last PTS span'])
        write_new(output / 'SUMMARY.json', summary)
        text = ('# 原生 GS 四格后验重建对比\n\n'
                f'臂 `{arm}`，checkpoint 完成 {summary["completed_steps"]} 步。'
                '**这是定期重新给真实初态的后验窗口重建，不是 AR rollout。**\n\n'
                '四格依次为真实 RGB、同一适配 codec 的保持初态、正确 ID、跨起点 shuffle84 整束 ID；'
                '每格 448×252，文字置于额外页眉，没有缩放画面。\n\n'
                f'全部177个物理帧各出现一次，源帧率 {fps.numerator}/{fps.denominator} = {float(fps):g} fps；'
                f'首末 PTS 跨度 {times["source_endpoint_span_seconds"]:.9f}s，'
                f'视频177帧时长 {times["encoded_duration_seconds"]:.9f}s。'
                '播放 PTS 等于源 PTS 减去首帧 PTS；缓存为名义 PTS，未宣称校验原 MP4 容器时间戳。\n\n'
                '`frame_mapping_metrics.jsonl` 记录每帧 source/start/h/PTS/reset、两束 IDs、'
                '原分辨率浮点指标、GS 数量和相对已完成评测的指标差；`SUMMARY.json` 绑定所有输入与输出 hash。'
                'frame172 起采用最后完整窗口 start167、h5…9，无复制或丢弃尾帧。\n\n'
                f'视频：`{names["video"]}`。截图为 frame0/44/88/132/176。'
                f'运行耗时 {summary["elapsed_seconds"]:.2f}s；无训练或参数更新。\n')
        with (output/'SUMMARY.md').open('x', encoding='utf-8') as stream:
            stream.write(text)
        print(json.dumps(dict(status='complete', arm=arm, video=str(video_path), frames=FRAME_COUNT)), flush=True)
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass  # Preserve the original failure; the outer job owns final process-group cleanup.
        for stream in streams:
            stream.close()
        write_new(output/'FAILED.json', dict(status='failed', arm=arm, stage=stage,
                  error=repr(error), traceback=failure_traceback, physical_frames_written=completed_frames,
                  elapsed_seconds=time.monotonic()-begun, max_seconds=max_seconds, optimizer_updates=0,
                  partial_outputs_retained=True))
        raise
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)


def cpu_self_test():
    mapping = frame_mapping()
    assert len(mapping) == FRAME_COUNT and [r['source_frame'] for r in mapping] == list(range(FRAME_COUNT))
    assert (mapping[0]['start'], mapping[0]['horizon']) == (0, 0)
    assert (mapping[171]['start'], mapping[171]['horizon']) == (162, 9)
    assert (mapping[172]['start'], mapping[172]['horizon']) == (167, 5)
    assert (mapping[176]['start'], mapping[176]['horizon']) == (167, 9)
    assert all(r['start'] + r['horizon'] == r['source_frame'] and 0 <= r['start'] <= 167
               and 1 <= r['horizon'] <= 9 for r in mapping[1:])
    assert sum(r['reset'] for r in mapping) == 20
    assert all(r['shuffle84_bundle_start'] == (r['start']+84)%168 for r in mapping)
    fps = rational_fps(1/29.97)
    assert fps == Fraction(2997, 100)
    pts = [frame/29.97 for frame in range(FRAME_COUNT)]
    times = validate_times(pts, fps)
    assert times['presentation_pts'] == pts and times['ffmpeg_framerate'] == '2997/100'
    assert math.isclose(times['encoded_duration_seconds'] - times['source_endpoint_span_seconds'], 1/29.97)
    validate_times([value+7.5 for value in pts], fps)
    for invalid, rate in [(pts, Fraction(30)), (pts[:-1], fps), (pts[:-1]+[pts[-2]], fps),
                           (pts[:-1]+[float('nan')], fps)]:
        try:
            validate_times(invalid, rate)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/mismatched source time was accepted')
    names, other = artifact_names('partial', fps), artifact_names('frozen', fps)
    assert names['video'] != other['video'] and names['video'].endswith('.mp4')
    assert set(names['snapshots']) == set(SNAPSHOTS)
    assert all(name.endswith('.png') and name != names['video'] for name in names['snapshots'].values())
    root = Path.cwd().resolve()
    assert valid_output_path(root, root/'diagnostics/render_comparison/new_attempt')
    assert not valid_output_path(root, root/'diagnostics/render_comparison')
    assert not valid_output_path(root, root/'partial')
    assert encoding_progress('frame=20\nprogress=continue\nframe=177\nprogress=end\n')['frame'] == '177'
    try:
        encoding_progress('frame=176\nprogress=end\n')
    except ValueError:
        pass
    else:
        raise AssertionError('Short video passed verification')
    assert 'torch' not in sys.modules and 'PIL' not in sys.modules
    print('CPU_SELF_TEST_PASSED: 177 physical frames, final tail, 20 resets, source2997/100 PTS, '
          '30fps mismatch rejected, unique MP4/PNG names, output containment, encoder177; no torch/GPU/files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--arm', choices=('partial', 'frozen'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
        return
    if args.root is None or args.arm is None or args.output is None:
        parser.error('--root, --arm and --output are required')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        parser.error('--max-seconds must be positive and no greater than600')
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir() or not valid_output_path(root, output):
        parser.error('Output must be a new directory below ROOT/diagnostics/render_comparison')
    if output.exists():
        parser.error('Output already exists; refusing to overwrite or resume')
    run(root, args.arm, output, args.max_seconds)


if __name__ == '__main__':
    main()
