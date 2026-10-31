"""CPU-only, complete-coverage descriptive comparison; never load tensor packets."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import worker as w
import inference_profile as inference

GROUPS = ('codec_real', 'teacher_real', 'codec_teacher', 'pseudo_view_teacher')
METRICS = ('mae', 'psnr', 'workspace_roi_mae', 'workspace_roi_psnr')
PICTURES = ('real', 'teacher', 'codec', 'nearby', 'nearby_teacher')
SNAPSHOTS = (0, 44, 88, 132, 176)


def load_bound_compare(root):
    """Reuse immutable CPU provenance audit, never its replay gate or statistics."""
    w.verify_manifest(root, w.OLD_MANIFEST)
    sys.dont_write_bytecode = True
    if 'pool_index' in sys.modules:
        w.require(Path(sys.modules['pool_index'].__file__).resolve() == (root / 'pool_index.py').resolve(),
                  'Foreign pool_index already imported')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_complete_eval_old_cpu_compare', root / 'compare.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    w.require('torch' not in sys.modules and 'MinkowskiEngine' not in sys.modules, 'CPU comparison imported tensor runtime')
    return old


def index_frames(rows, expected):
    result = {}
    for row in rows:
        key = (row.get('clip'), row.get('frame'))
        w.require(type(row.get('frame')) is int and key in expected and key not in result, 'Duplicate or foreign frame')
        result[key] = row
    w.require(set(result) == expected and len(result) == len(expected), 'Incomplete frame coverage')
    return result


def mean_metrics(rows):
    w.require(bool(rows), 'Cannot average empty group')
    return {group: {metric: math.fsum(r[group][metric] for r in rows) / len(rows)
                    for metric in METRICS} for group in GROUPS}


def aggregates(by_clip, train_groups, eval_clip):
    w.require(set(by_clip) == set(train_groups) | {eval_clip} and eval_clip not in train_groups,
              'Aggregate membership differs')
    groups = {group: mean_metrics([by_clip[clip] for clip in train_groups if train_groups[clip] == group])
              for group in sorted(set(train_groups.values()))}
    return dict(train_clip_equal=mean_metrics([by_clip[clip] for clip in train_groups]),
                train_source_group_equal=mean_metrics(list(groups.values())),
                by_train_source_group=groups, exposed_eval_only=by_clip[eval_clip],
                group_clip_counts={group: sum(g == group for g in train_groups.values()) for group in groups})


def validate_pairing(a, b, expected, teacher_check):
    w.require(set(a) == expected == set(b), 'Paired comparison has incomplete coverage')
    for key in expected:
        x, y = a[key], b[key]
        w.require(x['split'] == y['split'] and x['source_group'] == y['source_group']
                  and math.isclose(x['pts'], y['pts'], abs_tol=1e-9, rel_tol=0), 'Paired role/source/PTS differs')
        teacher_check(x, y)


def descriptive_differences(evaluations):
    output = {}
    for candidate, reference in (('decoder_only', 'baseline'), ('full_codec', 'baseline'), ('full_codec', 'decoder_only')):
        a, b = evaluations[candidate], evaluations[reference]
        def difference(left, right):
            return {g: {m: left[g][m] - right[g][m] for m in METRICS} for g in GROUPS}
        output[candidate + '_vs_' + reference] = dict(
            per_clip_reference_minus_candidate={clip: difference(b['per_clip'][clip], a['per_clip'][clip]) for clip in a['per_clip']},
            aggregate_reference_minus_candidate={scope: difference(b['aggregates'][scope], a['aggregates'][scope])
                for scope in ('train_clip_equal', 'train_source_group_equal', 'exposed_eval_only')},
            by_source_group_reference_minus_candidate={group: difference(b['aggregates']['by_train_source_group'][group],
                 a['aggregates']['by_train_source_group'][group]) for group in a['aggregates']['by_train_source_group']},
            mae_positive_is_improvement=True, psnr_negative_is_improvement=True,
            confidence_intervals_computed=False, automatic_quality_gate=False)
    return output


def training_contract(old, contract):
    old_root, full_root = Path(w.OLD_ROOT), Path(w.FULL_ROOT)
    decoder = old.load_training(old_root, 'decoder_only', contract)
    w.require(decoder['checkpoint_sha256'] == w.DECODER_SHA, 'Wrong completed decoder checkpoint')
    old.check_hash(Path(w.read(old_root / 'protocol.json')['source']['checkpoint_path']), w.INITIAL_SHA)
    done, train_done, full_source = w.full_receipts(full_root)
    digest = w.validate_full_done(done, train_done, full_source, contract['source'], has_failure=False)
    folder = full_root / 'arms' / 'full_codec'
    old.check_hash(folder / 'step_4426.pt', digest)
    for path, key in (('step_2213.pt', 'first_epoch_checkpoint_sha256'), ('trace.jsonl', 'trace_sha256'), ('PLAN.json', 'plan_sha256')):
        old.check_hash(folder / path, done[key])
    full_trace = old.rows(folder / 'trace.jsonl')
    w.require(len(full_trace) == 4426, 'Incomplete retraining trace')
    for index, (row, batch) in enumerate(zip(full_trace, [*contract['plans'][0], *contract['plans'][1]]), 1):
        old.fields(row, dict(step=index, arm='full_codec', epoch=1 if index <= 2213 else 2,
                            force_target=index <= 2213, actual_batch_size=len(batch)), 'Retraining trace')
        w.require([(r['clip'], r['frame']) for r in row['frames']] == [(r.clip, r.frame) for r in batch],
                  'Retraining trace differs from admitted sampling plan')
    return {
        'baseline': dict(step=0, checkpoint=w.INITIAL_SHA, encoder=old.INITIAL_ENCODER_SHA,
            training=dict(kind='original_codec3000', source=contract['source'])),
        'decoder_only': dict(step=4426, checkpoint=w.DECODER_SHA, encoder=decoder['final_encoder_sha256'],
            training=dict(kind='old_fullpool', source=contract['source'],
                train_done_sha256=w.sha(old_root / 'arms' / 'decoder_only' / 'TRAIN_DONE.json'))),
        'full_codec': dict(step=4426, checkpoint=digest, encoder=None,
            training=dict(kind='independent_fullcodec_retrain', source=done['source'],
                done_sha256=w.sha(full_root / 'DONE.json'), train_done_sha256=w.sha(folder / 'TRAIN_DONE.json'))),
    }


def validate_replay_fields(row, replay, old):
    status = w.replay_status(replay)
    old.fields(row, status, 'Replay status')
    w.require(type(row.get('replay_decoded_gaussians')) is int and row['replay_decoded_gaussians'] > 0,
              'Missing replay decoded count')
    w.require(type(row.get('decoded_count_difference')) is int
              and row['decoded_count_difference'] == row['replay_decoded_gaussians'] - row['decoded_gaussians'],
              'Replay decoded count difference differs')
    return status


def artifact_path(folder, relative):
    path = folder / relative
    w.require(path.resolve().is_relative_to(folder.resolve()), 'Artifact escapes evaluation folder')
    current = path
    while current != folder:
        w.require(not current.is_symlink(), 'Artifact path contains symlink')
        current = current.parent
    return path


def precision_fields(e, evidence, *, baseline_initial=None):
    conversion = inference.validate_conversion(evidence.get('precision_conversion'), e.get('precision_state_verification', {}))
    for name in ('checkpoint_model_state_sha256', 'checkpoint_decoder_sha256', 'model_state_sha256'):
        w.require(inference.valid_sha(evidence.get(name)), 'Missing checkpoint/runtime model digest')
    w.require(evidence['checkpoint_model_state_sha256'] != evidence['model_state_sha256'],
              'Runtime precision change must not be conflated with original checkpoint state')
    if baseline_initial is not None:
        w.require(evidence['checkpoint_model_state_sha256'] == baseline_initial, 'Original baseline checkpoint model differs')
    values = dict(precision_conversion=conversion, checkpoint_model_state_sha256=evidence['checkpoint_model_state_sha256'],
        model_state_sha256=evidence['model_state_sha256'], encoder_state_dtype='float32', decoder_precision='float64',
        renderer_gs_dtype='float32')
    inference.fields(e, values, 'Evaluation precision')
    inference.fields(evidence, values, 'Source precision')
    return values


def load_evaluation(root, arm, old, contract, training, source, inference_binding):
    folder = root / 'evaluations' / arm
    w.require(folder.resolve().is_relative_to(root) and not folder.is_symlink(), 'Evaluation folder is external')
    w.require(not (folder / 'FAILURE.json').exists(), 'Evaluation failure receipt exists: ' + arm)
    evaluation_sha = w.sha(folder / 'EVALUATION.json')
    source_sha = w.sha(folder / 'SOURCE.json')
    e = w.read(folder / 'EVALUATION.json')
    t = training[arm]
    old.fields(e, dict(status='evaluated', arm=arm, quality_evaluation_complete=True, frames=9027,
        train_frames=8850, exposed_eval_only_frames=177, additional_codec_updates=t['step'],
        checkpoint_sha256=t['checkpoint'], all_frames_freshly_encoded=True, saved_latent_state_replay=True,
        independent_native_gs_replay=False, decoder_target_key=None, continue_on_replay_threshold_failure=True,
        codec_accepted=False, codec_acceptance_status='pending_separate_review',
        inference_profile=inference_binding['profile'], inference_binding=inference_binding,
        cold_manager_replay_verified_all=True, independent_manager_frames=9027,
        action_Z_trained=False, automatic_LDM_codec_replacement=False, goal_complete=False), 'Evaluation')
    evidence = w.read(folder / 'SOURCE.json')
    old.fields(e, dict(source=evidence), 'Evaluation source receipt')
    old.fields(evidence, dict(evaluation_source=source, arm=arm, old_source=contract['source'],
        training_source=t['training'], checkpoint_sha256=t['checkpoint'], encoder_sha256=e['encoder_sha256'],
        decoder_sha256=e['decoder_sha256'], old_preflight_sha256=w.sha(Path(w.OLD_ROOT) / 'PREFLIGHT.json'),
        new_preflight_measurement=False, inference_profile=inference_binding['profile'],
        inference_binding=inference_binding, sparse_decode_setting=inference_binding['profile']['profile'],
        deterministic_configuration_changed=False, cold_manager_replay_required=True), 'Evaluation provenance')
    precision = precision_fields(e, evidence, baseline_initial=contract['initial_model_sha256'] if arm == 'baseline' else None)
    for name in ('encoder_sha256', 'decoder_sha256', 'model_state_sha256'):
        w.require(old.pool.valid_sha(evidence.get(name)), 'Missing model/encoder/decoder digest')
    if t['encoder'] is not None:
        w.require(e['encoder_sha256'] == t['encoder'], 'Baseline/frozen decoder encoder differs')
    else:
        w.require(e['encoder_sha256'] != old.INITIAL_ENCODER_SHA, 'Full-codec encoder unexpectedly unchanged')
    for name in ('targets', 'state_manifest', 'screenshot_manifest'):
        old.check_hash(folder / (name + '.jsonl'), e[name + '_sha256'])
    targets = index_frames(old.rows(folder / 'targets.jsonl'), old.KEYS)
    packets = index_frames(old.rows(folder / 'state_manifest.jsonl'), old.KEYS)
    failed = []
    for key, row in targets.items():
        clip, frame = key
        split = 'TRAIN' if clip in old.pool.EXPECTED else 'EVAL-only'
        group = old.pool.EXPECTED.get(clip, 'buildingblock_tower_14')
        bound = dict(split=split, source_group=group, additional_codec_updates=t['step'],
                     checkpoint_sha256=t['checkpoint'], encoder_sha256=e['encoder_sha256'], decoder_sha256=e['decoder_sha256'])
        old.fields(row, bound, 'Target row')
        old.fields(row, dict(inference_profile=inference_binding['profile'], cold_manager_replay_verified=True, **precision), 'Target inference')
        inference.validate_frame_scopes(row.get('inference_scopes'))
        w.require(type(row.get('pts')) in (float, int) and math.isfinite(row['pts'])
                  and math.isclose(row['pts'], frame / 29.97, abs_tol=1e-9, rel_tol=0), 'Frame PTS differs')
        for metric_group in GROUPS:
            w.finite_metrics(row.get(metric_group))
        old.fields(row, dict(latent_channels=64, input_gaussians=112896), 'Frame shape')
        for name, maximum in (('input_voxels', 80000), ('latent_positions', 80000), ('decoded_gaussians', 2400000)):
            w.require(type(row.get(name)) is int and 0 < row[name] <= maximum, 'Invalid frame count ' + name)
        replay = validate_replay_fields(row, row['replay'], old)
        packet = packets[key]
        old.fields(packet, dict(**bound, arm=arm, pts=row['pts'], normalization=contract['normals'][clip],
            source_cache_sha256=contract['cache'][key], source=contract['source'], evaluation_source=source,
            training_source=t['training'], inherited_codec_steps=3000, decoder_target_key=None, fresh_encode=True,
            inference_profile=inference_binding['profile'], inference_scopes=row['inference_scopes'], cold_manager_replay_verified=True,
            **precision,
            shape=[row['latent_positions'], 64], decoded_gaussians=row['decoded_gaussians'],
            state_path=f'states/{clip}/{frame:03d}.pt', saved_packet_replay=row['replay'],
            replay_decoded_gaussians=row['replay_decoded_gaussians'], decoded_count_difference=row['decoded_count_difference'],
            **replay), 'State packet')
        old.check_hash(artifact_path(folder, packet['state_path']), packet['state_sha256'], packet['state_bytes'])
        if not replay['replay_passed']:
            failed.append(dict(clip=clip, frame=frame, pts=row['pts'], split=split, source_group=group,
                failed_views=replay['failed_views'], replay=row['replay'], decoded_count_difference=row['decoded_count_difference']))
    expected_pictures = {(c, f, k) for c in old.CLIPS for f in SNAPSHOTS for k in PICTURES}
    seen = set()
    for row in old.rows(folder / 'screenshot_manifest.jsonl'):
        key = row.get('clip'), row.get('frame'), row.get('kind')
        w.require(type(row.get('frame')) is int and key in expected_pictures and key not in seen, 'Duplicate/foreign screenshot')
        seen.add(key)
        relative = f'screenshots/{key[0]}/{key[1]:03d}_{key[2]}.png'
        w.require(row.get('path') == relative, 'Screenshot path/identity differs')
        path = artifact_path(folder, relative)
        old.check_hash(path, row.get('sha256'), row.get('bytes'))
        with path.open('rb') as stream:
            header = stream.read(24)
        w.require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
                  and struct.unpack('>II', header[16:24]) == (448, 252), 'PNG dimensions differ')
    w.require(seen == expected_pictures and len(seen) == 1275, 'Missing screenshot coverage')
    acceptance = w.final_replay_summary(list(targets.values()), old.KEYS)
    old.fields(e, acceptance, 'Final replay summary')
    old.fields(e, dict(every_packet_saved_replay_passed=not failed), 'Saved replay result')
    per_clip = {clip: mean_metrics([targets[clip, f] for f in range(177)]) for clip in old.CLIPS}
    w.require(set(e['summary_by_clip']) == set(old.CLIPS), 'Summary clip set differs')
    for clip, actual in per_clip.items():
        old.compare_means(actual, e['summary_by_clip'][clip], arm + '/' + clip)
    aggregate = aggregates(per_clip, old.pool.EXPECTED, old.EVAL_CLIP)
    old.compare_means(aggregate['train_clip_equal'], e['summary_train_frames'], arm + '/TRAIN')
    old.compare_means(aggregate['exposed_eval_only'], e['summary_exposed_eval'], arm + '/exposed14')
    # Recheck manifests after all artifacts; concurrent modifications cannot silently pass.
    for name in ('targets', 'state_manifest', 'screenshot_manifest'):
        old.check_hash(folder / (name + '.jsonl'), e[name + '_sha256'])
    old.check_hash(folder / 'EVALUATION.json', evaluation_sha)
    old.check_hash(folder / 'SOURCE.json', source_sha)
    return dict(receipt_sha256=evaluation_sha, source_sha256=source_sha,
        checkpoint_sha256=t['checkpoint'], encoder_sha256=e['encoder_sha256'], decoder_sha256=e['decoder_sha256'],
        additional_codec_updates=t['step'], per_clip=per_clip, aggregates=aggregate,
        inference_profile=inference_binding['profile'], cold_manager_replay_verified_frames=len(targets),
        **precision, precision_state_verification=e['precision_state_verification'],
        replay_failed_frames=len(failed), replay_failed_frame_details=failed, replay_passed_all=not failed,
        codec_accepted=False, artifact_validation=dict(state_files_checked=len(packets), screenshot_files_checked=len(seen),
            manifest_hashes_checked=3, actual_artifact_bytes_and_sha_checked=True, tensor_deserialization=False),
        _rows=targets)


def main(root):
    root = Path(root).resolve()
    w.require(root.name == w.RUN and root == Path(__file__).resolve().parent, 'Dedicated evaluation root required')
    p, source = w.verify_root(root)
    _, inference_binding = inference.load_diagnostic(p['inference'])
    for name in ('SUMMARY.json', 'SUMMARY.md'):
        w.require(not (root / name).exists(), 'Comparison already attempted; no overwrite')
    old = load_bound_compare(Path(w.OLD_ROOT))
    for item in inference_binding['installed_minkowski_sources'].values():
        old.check_hash(item['path'], item['sha256'])
    contract = old.load_contract(Path(w.OLD_ROOT))
    training = training_contract(old, contract)
    evaluations = {arm: load_evaluation(root, arm, old, contract, training, source, inference_binding) for arm in w.ARMS}
    for arm in ('decoder_only', 'full_codec'):
        validate_pairing(evaluations['baseline']['_rows'], evaluations[arm]['_rows'], old.KEYS, old.teacher_alignment)
    w.require(w.verify_root(root)[1] == source, 'Evaluation source changed during comparison')
    w.verify_manifest(Path(w.OLD_ROOT), w.OLD_MANIFEST)
    # Recheck training completion and checkpoint bindings after the long artifact audit.
    w.require(training_contract(old, contract) == training, 'Training provenance changed during comparison')
    w.require(inference.load_diagnostic(p['inference'])[1] == inference_binding, 'Diagnostic inference binding changed')
    result = dict(status='complete', quality_evaluation_complete=True, source=source,
        models={arm: {key: value for key, value in result.items() if key != '_rows'} for arm, result in evaluations.items()},
        comparisons=descriptive_differences(evaluations), train_clips=50, train_source_groups=19,
        train_frames_per_model=8850, exposed_eval_only_frames_per_model=177, frames_per_model=9027,
        confidence_intervals_computed=False, automatic_quality_gate=False, cross_clip_physical_frame_unique_count=None,
        replay_passed_all=all(e['replay_passed_all'] for e in evaluations.values()),
        codec_accepted=False, codec_acceptance_status='pending_separate_visual_and_replay_review',
        action_Z_trained=False, automatic_LDM_codec_replacement=False, goal_complete=False,
        tensor_deserialization=False, independent_native_gs_replay=False,
        inference_profile=p['inference'], inference_binding=inference_binding,
        cold_manager_replay_verified_all=True, diagnostic_is_identical_evaluation_repetition=False,
        checkpoint_selection='Original codec3000 step0 and fixed additional step4426; exposed tower14 is never selection data',
        inference='Descriptive statistics only; overlapping clips are not independent population samples',
        teacher_numerical_tolerance=dict(mae_absolute=2e-6, mae_relative=1e-5, psnr_absolute=.003, psnr_relative=1e-6))
    text = ['# 完整 codec 质量评测与重放记录', '',
        '三臂各9027帧质量评测完整；质量评测完整、纯状态重放通过与总codec验收分别报告。codec尚未验收，等待另行视觉与重放审查。', '',
        '|模型|TRAIN 50片段等权 PSNR|TRAIN 19来源组等权 PSNR|来源组等权 MAE|暴露14 PSNR|重放失败帧|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in w.ARMS:
        item = evaluations[arm]; a = item['aggregates']
        text.append(f"|{arm}|{a['train_clip_equal']['codec_real']['psnr']:.4f}|{a['train_source_group_equal']['codec_real']['psnr']:.4f}|{a['train_source_group_equal']['codec_real']['mae']:.6f}|{a['exposed_eval_only']['codec_real']['psnr']:.4f}|{item['replay_failed_frames']}|")
    text += ['', 'SUMMARY.json 保留每片段、每来源组以及三种汇总范围的全图/ROI MAE与PSNR；包含codec对真实图、teacher对真实图、codec对teacher与伪视角对teacher指标。',
             '每个重放失败帧的clip、frame、PTS、失败视角与实测值保存在各模型 replay_failed_frame_details。原阈值为全图MAE≤1e-5且PSNR≥60，两个视角都必须满足。',
             '差值统一为参考模型减候选模型：MAE为正表示候选改善，PSNR为负表示候选改善。仅作描述统计，不计算置信区间或自动质量判定。',
             '50个片段存在重叠，19来源组等权与片段等权分别报告；已暴露14单列，不能视为未见验证集。',
             '三臂每臂实际核验9027个state文件和1275个PNG的bytes及SHA。重放是双视角latent重放，不是独立native GS重放。动作Z未训练，不能据此判定LDM成功。']
    text += ['三臂统一使用v7固定float64 decoder/head/multiplier，encoder和保存state保持float32，world GS转回float32后交给原渲染器。ME始终False/SEPARATE/globalNone，两次manager对象必须不同。冷manager隔离证据与图像重放阈值结果独立报告。',
             '原checkpoint模型摘要记录于精度转换前，运行模型摘要另列；castback_exact和encoder_exact明确验证原值可精确还原及encoder未变，运行state再作完整一致性检查。',
             '准入v7是带剪枝观察钩子的两帧精度诊断；本完整评测不挂这些钩子，不能冒充同条件重复。它自己逐帧测量fresh→disk重放，float32对照与历史不同输入结果不替代float64 fresh-state准入。']
    with (root / 'SUMMARY.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text) + '\n')
    with (root / 'SUMMARY.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', summary_sha256=w.sha(root / 'SUMMARY.json'), codec_accepted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
