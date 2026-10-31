"""Full quality evaluation with explicit, unchanged replay acceptance thresholds.

Importing this file never imports Torch or MinkowskiEngine. A threshold failure is
data; malformed/nonfinite data, source drift and runtime errors remain fatal.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import inference_profile as inference

RUN = 'gaussiangpt_neoverse_codec_complete_eval_20261031_v1'
OLD_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1'
OLD_MANIFEST = '01505377362bf47c89ae23aeaa5de6ca0c5663fe9119e35f6c01c03524351376'
DECODER_SHA = 'dcdcd0be5f999b09086bbc31c91aea62898600c123a962791845314898d6eef3'
FULL_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullcodec_retrain_20261031_v1'
FULL_MANIFEST = '479e75989ca7c5cde716dfc9d842a5f703a735837ac1a11b8faad15de638034b'
INITIAL_SHA = 'a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
METRICS = {'mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr'}
ARMS = ('baseline', 'decoder_only', 'full_codec')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(p):
    inference.validate_profile(p['inference'])
    expected = {
        ('schema',): 'codec_complete_evaluation_v1',
        ('source', 'old_root'): OLD_ROOT, ('source', 'old_manifest_sha256'): OLD_MANIFEST,
        ('source', 'decoder_checkpoint_sha256'): DECODER_SHA,
        ('source', 'full_retrain_root'): FULL_ROOT, ('source', 'full_retrain_manifest_sha256'): FULL_MANIFEST,
        ('evaluation', 'frames_per_model'): 9027, ('evaluation', 'frames_per_clip'): 177,
        ('evaluation', 'snapshot_frames'): [0, 44, 88, 132, 176], ('evaluation', 'max_seconds'): 7200,
        ('evaluation', 'replay_mae_max'): 1e-5, ('evaluation', 'replay_psnr_min'): 60,
        ('evaluation', 'continue_on_replay_threshold_failure'): True,
        ('resources', 'cpu_threads'): 6, ('resources', 'peak_limit_gib'): 12,
    }
    for keys, value in expected.items():
        actual = p
        for key in keys:
            actual = actual[key]
        require(actual == value and (type(actual) is bool if type(value) is bool else type(actual) is not bool),
                'Protocol drift: ' + '.'.join(keys))


def verify_manifest(root, pinned=None):
    root = Path(root)
    digest = sha(root / 'CODE_MANIFEST.json')
    require(pinned is None or digest == pinned, 'Source manifest SHA differs')
    manifest = read(root / 'CODE_MANIFEST.json')
    require(isinstance(manifest, dict) and 'protocol.json' in manifest, 'Manifest must bind protocol')
    require({p.name for p in root.glob('*.py')} <= set(manifest), 'Unbound executable')
    for name, expected in manifest.items():
        require(Path(name).name == name and sha(root / name) == expected, 'Source file changed: ' + name)
    return manifest, digest


def verified_source(root, *, pinned=None, run=None):
    _, digest = verify_manifest(root, pinned)
    proof = read(root / 'CPU_VALIDATION.json')
    require(proof.get('status') == 'passed' and proof.get('gpu_executed') is False
            and proof.get('code_manifest_sha256') == digest, 'Exact passing CPU witness required')
    source = dict(code_manifest_sha256=digest, protocol_sha256=sha(root / 'protocol.json'),
                  cpu_validation_sha256=sha(root / 'CPU_VALIDATION.json'))
    if run is not None:
        source['run'] = run
    return source


def verify_root(root):
    source = verified_source(root, run=RUN)
    p = read(root / 'protocol.json')
    validate_protocol(p)
    return p, source


def load_old(root):
    verify_manifest(root, OLD_MANIFEST)
    sys.dont_write_bytecode = True
    if 'pool_index' in sys.modules:
        require(Path(sys.modules['pool_index'].__file__).resolve() == (root / 'pool_index.py').resolve(),
                'Foreign pool_index already loaded')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_complete_eval_bound_fullpool', root / 'worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_metrics(metrics):
    require(isinstance(metrics, dict) and set(metrics) == METRICS, 'Unexpected metric structure')
    require(all(type(value) in (float, int) and math.isfinite(value) for value in metrics.values()),
            'Nonfinite or malformed metric')
    require(metrics['mae'] >= 0 and metrics['workspace_roi_mae'] >= 0, 'Negative image error')
    return metrics


def replay_status(replay):
    require(isinstance(replay, dict) and set(replay) == {'real_view', 'nearby_view'}, 'Both replay views required')
    failed = []
    for view, values in replay.items():
        finite_metrics(values)
        if values['mae'] > 1e-5 or values['psnr'] < 60:
            failed.append(view)
    return dict(replay_passed=not failed, failed_views=sorted(failed),
                replay_mae_max=1e-5, replay_psnr_min=60)


def final_replay_summary(rows, expected):
    require(len(expected) == 9027 and len(rows) == 9027
            and {(r['clip'], r['frame']) for r in rows} == expected, 'Incomplete/duplicate 9027-frame evaluation')
    failed = []
    for row in rows:
        status = replay_status(row['replay'])
        require(row['replay_passed'] is status['replay_passed'], 'Replay flag differs from recorded metrics')
        if not status['replay_passed']:
            failed.append(dict(clip=row['clip'], frame=row['frame'], failed_views=status['failed_views']))
    return dict(replay_passed_all=not failed, replay_failed_frames=len(failed), replay_failed_frame_ids=failed,
                replay_acceptance=dict(status='failed' if failed else 'passed', passed=not failed,
                                       mae_max=1e-5, psnr_min=60, views_per_frame=2),
                codec_accepted=False, codec_acceptance_status='pending_separate_review')


def validate_full_done(done, train_done, training_source, old_source, *, has_failure):
    require(not has_failure, 'Failed retraining cannot be evaluated')
    require(isinstance(done, dict) and done == train_done, 'Root DONE and TRAIN_DONE must match')
    require(done.get('status') == 'complete' and done.get('training_complete') is True
            and done.get('steps') == 4426 and done.get('arm') == 'full_codec'
            and done.get('additional_codec_updates') == 4426, 'Complete 4426-update retraining required')
    source = done.get('source', {})
    require(source.get('new_source') == training_source and source.get('old_source') == old_source,
            'Retraining source binding differs')
    cursor = done.get('sampling', {})
    require(cursor.get('training_complete') is True and cursor.get('completed_updates') == 4426
            and cursor.get('epoch_frames_seen') == [8850, 8850]
            and cursor.get('total_frame_exposures') == 17700
            and cursor.get('epoch_index') == 2 and cursor.get('next_global_step') is None
            and cursor.get('next_batch_index') == 0
            and cursor.get('epoch_plan_sha256') == old_source.get('epoch_plan_sha256'), 'Retraining coverage incomplete')
    digest = done.get('checkpoint_sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
            'Invalid final checkpoint digest')
    return digest


def validate_full_checkpoint(saved, done, digest):
    require(saved.get('schema') == 'fullcodec_retrain_checkpoint_v1'
            and saved.get('arm') == 'full_codec' and saved.get('step') == 4426
            and saved.get('failed') is False and saved.get('eligible_for_final_evaluation') is True
            and saved.get('requires_successful_done_receipt') is True, 'Ineligible retrained checkpoint')
    require(saved.get('source') == done['source'] and saved.get('sampling') == done['sampling']
            and saved.get('initial_checkpoint_sha256') == done.get('initial_checkpoint_sha256')
            and digest == done['checkpoint_sha256'], 'Retrained checkpoint provenance differs')


def full_receipts(root):
    folder = root / 'arms' / 'full_codec'
    require(not any(path.exists() for path in (root / 'FAILED.json', folder / 'FAILURE.json',
                                              folder / 'FAILED_DO_NOT_EVALUATE.pt')), 'Failed retraining run')
    require((root / 'DONE.json').is_file() and (folder / 'TRAIN_DONE.json').is_file(),
            'Retraining is not complete; queue must wait, worker will not poll')
    source = verified_source(root, pinned=FULL_MANIFEST, run=Path(FULL_ROOT).name)
    return read(root / 'DONE.json'), read(folder / 'TRAIN_DONE.json'), source


def evaluation_model(torch, old, x, arm, full_records):
    if arm == 'baseline':
        require(sha(x.old.checkpoint) == INITIAL_SHA, 'Original codec3000 checkpoint changed')
        model = old.legacy.initial_model(x.old).eval()
        return model, x.old.checkpoint, INITIAL_SHA, dict(kind='original_codec3000', source=x.evidence), 0
    if arm == 'decoder_only':
        model, path, step = old.evaluation_model(x, arm)
        require(step == 4426 and sha(path) == DECODER_SHA, 'Unexpected old decoder-only checkpoint')
        return model, path, DECODER_SHA, dict(kind='old_fullpool', source=x.evidence,
            train_done_sha256=sha(x.root / 'arms' / arm / 'TRAIN_DONE.json')), 4426
    done, arm_done, source = full_records
    digest = validate_full_done(done, arm_done, source, x.evidence, has_failure=False)
    path = Path(FULL_ROOT) / 'arms' / 'full_codec' / 'step_4426.pt'
    require(sha(path) == digest, 'Retrained final checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    validate_full_checkpoint(saved, done, digest)
    old.legacy.seed_all()
    model = x.old.codec.load_codec(x.old.codec_root, None, device='cpu')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    return model.cuda().eval(), path, digest, dict(kind='independent_fullcodec_retrain', source=done['source'],
        done_sha256=sha(Path(FULL_ROOT) / 'DONE.json'),
        train_done_sha256=sha(path.parent / 'TRAIN_DONE.json')), 4426


class Guard:
    def __init__(self, torch, seconds, started):
        self.torch, self.seconds, self.started = torch, seconds, started
        self.last_query, self.peak_process_mib = float('-inf'), 0

    def check(self):
        t = self.torch
        t.cuda.synchronize()
        require(max(t.cuda.max_memory_allocated(), t.cuda.max_memory_reserved()) <= 12 * 2**30,
                '12 GiB Torch memory guard exceeded')
        now = time.monotonic()
        if now - self.last_query >= 10:
            text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True, timeout=10)
            own = [int(line.split(',')[1]) for line in text.splitlines()
                   if line.split(',')[0].strip() == str(os.getpid())]
            require(len(own) == 1, 'One actual process GPU memory reading required')
            self.peak_process_mib = max(self.peak_process_mib, own[0])
            self.last_query = now
        require(self.peak_process_mib <= 12 * 1024, '12 GiB process GPU memory guard exceeded')
        if now - self.started > self.seconds:
            raise TimeoutError('Evaluation exceeded 7200 seconds')

    def evidence(self):
        return dict(elapsed_seconds=time.monotonic() - self.started, max_seconds=self.seconds,
                    peak_allocated_gib=self.torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=self.torch.cuda.max_memory_reserved() / 2**30,
                    sampled_process_peak_mib=self.peak_process_mib, process_sample_interval_seconds=10,
                    process_peak_is_continuous_measurement=False)


def validate_image(torch, image):
    require(isinstance(image, torch.Tensor) and tuple(image.shape) == (252, 448, 3)
            and bool(torch.isfinite(image).all()), 'Nonfinite or malformed 448x252 RGB render')


def gs_count(torch, points):
    """Observe tensors returned by the bound codec; preserve original enum keys."""
    require(isinstance(points, dict) and points, 'Empty/malformed decoded GS')
    counts = []
    for value in points.values():
        require(isinstance(value, torch.Tensor) and value.ndim >= 1, 'Malformed decoded GS tensor')
        require(bool(torch.isfinite(value).all()), 'Nonfinite decoded GS')
        counts.append(len(value))
    require(len(set(counts)) == 1 and counts[0] > 0, 'Inconsistent/empty decoded GS rows')
    return counts[0]


def encode_replay(torch, legacy, model, payload, render, metadata, path, *, ME, adapter, scope_record):
    """Float32 state, float64 decode, float32 render; two distinct default managers."""
    model.eval()
    require(not model.ae.decoder.training, 'Evaluation decoder must be eval')
    require(not torch.is_grad_enabled(), 'Sparse-manager lifecycle requires no_grad inference')
    require(not torch.is_autocast_enabled('cuda'), 'Decoder precision evaluation forbids CUDA autocast')

    def direct_phase(register_manager):
        encoded = model.encode_state(payload['representatives'], payload['set_statistics'])
        register_manager(encoded['sparse'].coordinate_manager)
        packet = legacy.state_packet(encoded, metadata)
        prepared = adapter.prepare_sparse(encoded['sparse'], ME)
        direct, _ = model.ae.decode(prepared, target_key=None)
        direct = adapter.to_render_points(model._world(direct))
        metadata['decoded_gaussians'] = gs_count(torch, direct)
        image, nearby = render(direct, payload, 0), render(direct, payload, 1)
        validate_image(torch, image); validate_image(torch, nearby)
        require(image.grad_fn is None and nearby.grad_fn is None, 'Render retains autograd graph')
        # Only dense renders and cloned CPU packet escape; sparse locals die on return.
        return image, nearby, packet

    def save_load(direct_result):
        packet = direct_result[2]
        legacy.atomic_save(packet, path)
        replay = torch.load(path, map_location='cpu', weights_only=True)
        require(replay['metadata'] == metadata and replay['schema'] == packet['schema']
                and replay['stride'] == packet['stride'] and replay['grid'] == packet['grid']
                and torch.equal(replay['coordinates'], packet['coordinates'])
                and torch.equal(replay['features'], packet['features']), 'Saved state differs from fresh packet')
        return replay

    def replay_phase(replay, release_original):
        device = payload['representatives'].device
        # Bound decode sequence with the explicit precision adapter at its two boundaries.
        z = model._sparse(replay['coordinates'].to(device), replay['features'].to(device), replay['stride'])
        release_original(z.coordinate_manager)
        prepared = adapter.prepare_sparse(z, ME)
        reconstructed, _ = model.ae.decode(prepared, target_key=None)
        reconstructed = adapter.to_render_points(model._world(reconstructed))
        replay_count = gs_count(torch, reconstructed)
        replay_image, replay_nearby = render(reconstructed, payload, 0), render(reconstructed, payload, 1)
        validate_image(torch, replay_image); validate_image(torch, replay_nearby)
        return replay_image, replay_nearby, replay_count

    direct, rebuilt = inference.independent_replay(torch, ME, direct_phase, replay_phase, save_load, scope_record)
    image, nearby, packet = direct
    replay_image, replay_nearby, replay_count = rebuilt
    metrics = dict(real_view=legacy.image_metrics(image, replay_image),
                   nearby_view=legacy.image_metrics(nearby, replay_nearby))
    status = replay_status(metrics)
    status['replay_decoded_gaussians'] = replay_count
    status['decoded_count_difference'] = replay_count - metadata['decoded_gaussians']
    status['inference_scopes'] = scope_record
    status['cold_manager_replay_verified'] = True
    return image, nearby, packet, metrics, status


def main(root, arm):
    root = Path(root).resolve()
    require(root.name == RUN and root == Path(__file__).resolve().parent and arm in ARMS, 'Dedicated evaluation root/arm required')
    p, new_source = verify_root(root)
    folder = root / 'evaluations' / arm
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, context, guard = [], {}, None
    evidence = dict(evaluation_source=new_source, arm=arm, old_source=None)
    write(folder / 'STARTED.json', dict(status='initializing', source=evidence, pid=os.getpid(), gpu_executed=False))
    def interrupted(signum, frame):
        raise RuntimeError('Owned evaluation interrupted: signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.alarm(7200)
    try:
        full_records = full_receipts(Path(FULL_ROOT)) if arm == 'full_codec' else None
        probe, inference_binding = inference.load_diagnostic(p['inference'])
        import torch
        import numpy as np
        from PIL import Image
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one assigned CUDA GPU')
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(min(1., 10 * 2**30 / torch.cuda.get_device_properties(0).total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
        guard = Guard(torch, 7200, started)
        old = load_old(Path(OLD_ROOT))
        x = old.load_runtime(Path(OLD_ROOT)); old.gate(x)
        import MinkowskiEngine as ME
        inference.verify_actual_sources(probe, ME, inference_binding)
        inference.entry_state(ME)
        evidence['old_source'] = x.evidence
        model, checkpoint, checkpoint_sha, training, step = evaluation_model(torch, old, x, arm, full_records)
        all_state = old.legacy.snapshot(model)
        checkpoint_model_sha = old.legacy.digest_snapshot(all_state)
        encoder_sha = old.legacy.digest_snapshot({k: v for k, v in all_state.items() if not old.legacy.in_decoder(k)})
        checkpoint_decoder_sha = old.legacy.digest_snapshot({k: v for k, v in all_state.items() if old.legacy.in_decoder(k)})
        del all_state
        precision = inference.load_precision_helper(p['inference'])
        adapter = precision.DecoderPrecision(model, 'float64')
        conversion = inference.validate_conversion(adapter.report)
        all_state = old.legacy.snapshot(model)
        model_sha = old.legacy.digest_snapshot(all_state)
        require(old.legacy.digest_snapshot({k: v for k, v in all_state.items() if not old.legacy.in_decoder(k)}) == encoder_sha,
                'Precision adapter changed encoder digest')
        decoder_sha = old.legacy.digest_snapshot({k: v for k, v in all_state.items() if old.legacy.in_decoder(k)})
        del all_state
        evidence.update(training_source=training, checkpoint_sha256=checkpoint_sha,
                        encoder_sha256=encoder_sha, decoder_sha256=decoder_sha, model_state_sha256=model_sha,
                        checkpoint_model_state_sha256=checkpoint_model_sha, checkpoint_decoder_sha256=checkpoint_decoder_sha,
                        precision_conversion=conversion, encoder_state_dtype='float32', decoder_precision='float64',
                        renderer_gs_dtype='float32',
                        old_preflight_sha256=sha(x.root / 'PREFLIGHT.json'), new_preflight_measurement=False,
                        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        sparse_decode_setting=p['inference']['profile'], deterministic_configuration_changed=False,
                        inference_binding=inference_binding, inference_profile=p['inference'], cold_manager_replay_required=True)
        write(folder / 'SOURCE.json', evidence)
        clips = [*sorted(old.pool.EXPECTED), old.EVAL_CLIP]
        with torch.no_grad(), (folder / 'targets.jsonl').open('x', encoding='utf-8', buffering=1) as log, \
             (folder / 'state_manifest.jsonl').open('x', encoding='utf-8', buffering=1) as manifest, \
             (folder / 'screenshot_manifest.jsonl').open('x', encoding='utf-8', buffering=1) as shots:
            for clip in clips:
                state_dir = folder / 'states' / clip; state_dir.mkdir(parents=True)
                picture_dir = folder / 'screenshots' / clip; picture_dir.mkdir(parents=True)
                for frame in range(177):
                    guard.check(); context = dict(clip=clip, frame=frame)
                    payload = (old.pool.load_frame(x.by_id[clip, frame], device='cuda') if clip in old.pool.EXPECTED
                               else x.sources.load_frame(x.old.sources[clip], frame, device='cuda'))
                    metadata = dict(**context, arm=arm, split='TRAIN' if clip in old.pool.EXPECTED else 'EVAL-only',
                        source_group=old.pool.EXPECTED.get(clip, 'buildingblock_tower_14'), normalization=payload['normalization'],
                        pts=payload['pts'], source_cache_sha256=payload['_source_cache_sha256'],
                        checkpoint_sha256=checkpoint_sha, encoder_sha256=encoder_sha, decoder_sha256=decoder_sha,
                        source=x.evidence, evaluation_source=new_source, training_source=training,
                        additional_codec_updates=step, inherited_codec_steps=3000, decoder_target_key=None, fresh_encode=True,
                        inference_profile=p['inference'], precision_conversion=conversion,
                        checkpoint_model_state_sha256=checkpoint_model_sha, model_state_sha256=model_sha,
                        encoder_state_dtype='float32', decoder_precision='float64', renderer_gs_dtype='float32')
                    path = state_dir / f'{frame:03d}.pt'
                    scope_record = {}
                    context['inference_scopes'] = scope_record
                    image, nearby, packet, replay, status = encode_replay(torch, old.legacy, model, payload,
                        x.old.train.render_prediction, metadata, path, ME=ME, adapter=adapter, scope_record=scope_record)
                    teacher = payload['targets'][0].clamp(0, 1)
                    for target in (payload['real'], payload['targets'][0], payload['targets'][1]):
                        validate_image(torch, target)
                    row = dict(clip=clip, frame=frame, split=metadata['split'], source_group=metadata['source_group'], pts=payload['pts'],
                        codec_real=finite_metrics(old.legacy.image_metrics(image, payload['real'])),
                        teacher_real=finite_metrics(old.legacy.image_metrics(teacher, payload['real'])),
                        codec_teacher=finite_metrics(old.legacy.image_metrics(image, teacher)),
                        pseudo_view_teacher=finite_metrics(old.legacy.image_metrics(nearby, payload['targets'][1].clamp(0, 1))),
                        latent_positions=len(packet['coordinates']), latent_channels=64, input_gaussians=int(payload['n_input']),
                        input_voxels=len(payload['representatives']), decoded_gaussians=metadata['decoded_gaussians'],
                        replay=replay, checkpoint_sha256=checkpoint_sha, encoder_sha256=encoder_sha,
                        decoder_sha256=decoder_sha, additional_codec_updates=step, inference_profile=p['inference'],
                        precision_conversion=conversion, encoder_state_dtype='float32', decoder_precision='float64',
                        renderer_gs_dtype='float32', checkpoint_model_state_sha256=checkpoint_model_sha,
                        model_state_sha256=model_sha, **status)
                    rows.append(row); log.write(json.dumps(row, allow_nan=False) + '\n')
                    manifest.write(json.dumps(dict(**metadata, state_path=path.relative_to(folder).as_posix(),
                        state_sha256=sha(path), state_bytes=path.stat().st_size, shape=list(packet['features'].shape),
                        saved_packet_replay=replay, **status), allow_nan=False) + '\n')
                    if frame in p['evaluation']['snapshot_frames']:
                        for kind, picture in [('real', payload['real']), ('teacher', teacher), ('codec', image),
                                               ('nearby', nearby), ('nearby_teacher', payload['targets'][1])]:
                            target = picture_dir / f'{frame:03d}_{kind}.png'
                            Image.fromarray((picture.detach().cpu().numpy().clip(0, 1)*255).round().astype(np.uint8)).save(target)
                            shots.write(json.dumps(dict(clip=clip, frame=frame, kind=kind, path=target.relative_to(folder).as_posix(),
                                bytes=target.stat().st_size, sha256=sha(target))) + '\n')
                    del payload, image, nearby, packet, teacher
                    guard.check()
                print('COMPLETE_CODEC_EVAL', arm, clip, len(rows), flush=True)
        acceptance = final_replay_summary(rows, {(c, f) for c in clips for f in range(177)})
        old.recheck(x)
        inference.entry_state(ME)
        inference.verify_actual_sources(probe, ME, inference_binding)
        require(inference.load_diagnostic(p['inference'])[1] == inference_binding, 'Diagnostic inference binding changed')
        require(verify_root(root)[1] == new_source and sha(checkpoint) == checkpoint_sha, 'Evaluation/checkpoint binding changed')
        require(old.legacy.digest_snapshot(old.legacy.snapshot(model)) == model_sha, 'Evaluation changed model state')
        precision_verification = adapter.assert_unchanged()
        inference.validate_conversion(conversion, precision_verification)
        if arm == 'full_codec':
            current = full_receipts(Path(FULL_ROOT))
            require(current == full_records, 'Retraining completion evidence changed')
        guard.check()
        def average(selected):
            return {k: {m: float(np.mean([r[k][m] for r in selected])) for m in selected[0][k]}
                    for k in ('codec_real', 'teacher_real', 'codec_teacher', 'pseudo_view_teacher')}
        write(folder / 'EVALUATION.json', dict(status='evaluated', quality_evaluation_complete=True,
            arm=arm, frames=9027, train_frames=8850, exposed_eval_only_frames=177,
            additional_codec_updates=step, checkpoint_sha256=checkpoint_sha, encoder_sha256=encoder_sha,
            decoder_sha256=decoder_sha, source=evidence, **acceptance,
            inference_profile=p['inference'], inference_binding=inference_binding,
            checkpoint_model_state_sha256=checkpoint_model_sha, model_state_sha256=model_sha,
            precision_conversion=conversion, precision_state_verification=precision_verification,
            encoder_state_dtype='float32', decoder_precision='float64', renderer_gs_dtype='float32',
            cold_manager_replay_verified_all=True, independent_manager_frames=9027,
            summary_by_clip={c: average([r for r in rows if r['clip'] == c]) for c in clips},
            summary_train_frames=average([r for r in rows if r['split'] == 'TRAIN']),
            summary_exposed_eval=average([r for r in rows if r['split'] == 'EVAL-only']),
            every_packet_saved_replay_passed=acceptance['replay_passed_all'],
            all_frames_freshly_encoded=True, saved_latent_state_replay=True, independent_native_gs_replay=False,
            decoder_target_key=None, roi_xyxy=list(old.legacy.ROI),
            targets_sha256=sha(folder / 'targets.jsonl'), state_manifest_sha256=sha(folder / 'state_manifest.jsonl'),
            screenshot_manifest_sha256=sha(folder / 'screenshot_manifest.jsonl'),
            continue_on_replay_threshold_failure=True, action_Z_trained=False,
            automatic_LDM_codec_replacement=False, goal_complete=False, **guard.evidence()))
    except BaseException as error:
        signal.alarm(0)
        resource_evidence = None
        if guard is not None:
            try:
                resource_evidence = guard.evidence()
            except BaseException as evidence_error:
                resource_evidence = dict(unavailable=True, error=repr(evidence_error))
        write(folder / 'FAILURE.json', dict(status='failed', error=repr(error), source=evidence, context=context,
            frames_complete=len(rows), quality_evaluation_complete=False, codec_accepted=False,
            valid_complete_evaluation=False, goal_complete=False,
            resource_evidence=resource_evidence))
        raise
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=ARMS)
    args = parser.parse_args()
    main(args.root, args.arm)
