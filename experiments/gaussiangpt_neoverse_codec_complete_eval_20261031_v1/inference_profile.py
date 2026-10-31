"""CPU-readable admission for decoder float64 with unchanged default ME state."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

DIAGNOSTIC_ROOT = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_replay_probe_20261031_v7'
MANIFEST_SHA = 'd45b63505b25bf3121dfad420200fbb42af3fd175eb20bab76654661d7b23261'
PROBE_SHA = '6f28cfbafa5ce75eec293cf08383edca6d16ce0bcc0c02d0fa758ad1cebf3dc4'
PRECISION_SHA = '2b6d4513afed3bc1d5f6d31d4a076a341c49a23ba500a49f257b65319609d7e8'
SELECTED = ['ae.decoder', 'ae.feature_heads', 'ae.gaussian_multiplier']
FIXED = dict(profile='decoder_float64_default_me_independent_replay_v1',
    diagnostic_root=DIAGNOSTIC_ROOT, diagnostic_manifest_sha256=MANIFEST_SHA, diagnostic_probe_sha256=PROBE_SHA,
    precision_helper_sha256=PRECISION_SHA, decoder_precision='float64', selected_modules=SELECTED,
    encoder_state_dtype='float32', renderer_gs_dtype='float32', minkowski_deterministic=False,
    sparse_tensor_operation_mode='SEPARATE_COORDINATE_MANAGER',
    scope='independent_fresh_encode_and_disk_rebuild_managers', pruning_capture_hooks=False,
    diagnostic_pruning_capture_hooks=True, cold_manager_replay_required=True)
RUNTIME = dict(minkowski_deterministic=False, sparse_tensor_operation_mode='SEPARATE_COORDINATE_MANAGER',
               global_manager_python_id=None)
CASES = {('buildingblock_tower_15_176', 113), ('buildingblock_tower_15_176', 114)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def fields(actual, expected, context):
    require(isinstance(actual, dict), context + ': expected object')
    for key, value in expected.items():
        require(type(actual.get(key)) is type(value) and actual[key] == value, context + '.' + key + ' differs')


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_profile(profile):
    fields(profile, FIXED, 'Inference profile')
    require(set(profile) == set(FIXED) | {'diagnostic_result_sha256'}, 'Unexpected inference profile fields')
    value = profile['diagnostic_result_sha256']
    require(valid_sha(value) and value != '0' * 64, 'Actual diagnostic RESULT SHA must be pinned before evaluation')
    return profile


def validate_conversion(report, verification=None):
    fields(report, dict(schema='decoder_precision_adapter_v1', decoder_precision='float64', selected_modules=SELECTED,
        original_floating_state_float32=True, castback_exact=True, encoder_exact=True, nonselected_state_exact=True),
        'Precision conversion')
    require(all(valid_sha(report.get(k)) for k in ('original_state_digest', 'mode_state_digest')),
            'Precision state digests missing')
    require(report['original_state_digest'] != report['mode_state_digest'], 'Runtime dtype conversion was not recorded')
    original = report.get('original_dtype_counts', {})
    selected = report.get('selected_mode_dtype_counts', {})
    current = report.get('mode_dtype_counts', {})
    require('torch.float64' not in original and 'torch.float32' not in selected, 'Unexpected floating dtype scope')
    for counts, dtype in ((original, 'torch.float32'), (selected, 'torch.float64'),
                          (current, 'torch.float32'), (current, 'torch.float64')):
        item = counts.get(dtype, {})
        require(type(item.get('tensors')) is int and item['tensors'] > 0
                and type(item.get('elements')) is int and item['elements'] > 0, 'Missing actual precision dtype counts')
    if verification is not None:
        fields(verification, dict(unchanged=True, decoder_precision='float64', state_digest=report['mode_state_digest'],
            mode_state_digest=report['mode_state_digest']), 'Precision state verification')
        require(type(verification.get('checked_tensors')) is int and verification['checked_tensors'] > 0,
                'No tensors checked after evaluation')
    return report


def entry_state(ME):
    require(ME.global_coordinate_manager() is None, 'Frame boundary has a live global manager')
    require(ME.is_deterministic() is False, 'ME deterministic flag must remain False')
    mode = ME.sparse_tensor_operation_mode()
    require(mode == ME.SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER,
            'ME mode must remain SEPARATE_COORDINATE_MANAGER')
    return dict(RUNTIME)


def validate_frame_scopes(record):
    fields(record, dict(cold_manager_replay_verified=True, manager_identity_distinct=True,
        direct_sparse_locals_released_before_replay=True, original_manager_reference_released_before_replay_decode=True,
        pruning_capture_hooks=False, global_state_mutated=False), 'Independent replay managers')
    identities = []
    for phase in ('direct', 'replay'):
        item = record.get(phase, {})
        fields(item, dict(body_completed=True, body_error=None, before=RUNTIME, after=RUNTIME), phase + ' runtime')
        identity = item.get('manager_python_id')
        require(type(identity) is int and identity > 0, 'Actual manager identity missing')
        identities.append(identity)
    require(identities[0] != identities[1], 'Direct and replay manager identities must differ')
    return record


def validate_diagnostic(result, source, probe):
    fields(result, dict(status='completed', completed=True, gpu_executed=True, source=source,
        formal_updates=0, checkpoint_sha256=probe.CHECKPOINT_SHA, thresholds=dict(mae_max=1e-5, psnr_min=60),
        inference_runtime=RUNTIME), 'Diagnostic result')
    fields(result.get('original_deterministic_flag_restore'), dict(restored=True), 'Diagnostic flag restore')
    results = result.get('decoder_precision_results')
    require(isinstance(results, list) and len(results) == 2
            and [r.get('decoder_precision') for r in results] == ['float32', 'float64'], 'Both precision results required')
    chosen = results[1]
    fields(chosen, dict(checkpoint_sha256=probe.CHECKPOINT_SHA,
        checkpoint_model_state_sha256=result['original_checkpoint_model_state_sha256'],
        inference_runtime_before=RUNTIME, inference_runtime_after=RUNTIME), 'Diagnostic float64 mode')
    validate_conversion(chosen.get('conversion'), chosen.get('state_verification', {}))
    fields(chosen.get('replay_summaries', {}).get('fresh_same_state'), dict(passed=True), 'Diagnostic float64 fresh replay')
    cases = chosen.get('cases', [])
    require(len(cases) == 2 and {(c.get('clip'), c.get('frame')) for c in cases} == CASES,
            'Diagnostic must complete both admitted cases')
    for case in cases:
        fields(case.get('replay_summaries', {}).get('fresh_same_state'), dict(passed=True), 'Case fresh replay')
        historical = case.get('replay_summaries', {}).get('historical_state_comparison')
        require(isinstance(historical, dict) and type(historical.get('passed')) is bool,
                'Historical-state comparison must be retained separately')
    installed = result.get('installed_minkowski_sources', {})
    require(set(installed) == set(probe.MINKOWSKI_SOURCES), 'Diagnostic installed ME source set differs')
    for name, digest in probe.MINKOWSKI_SOURCES.items():
        fields(installed[name], dict(sha256=digest), 'Diagnostic ME source ' + name)
    return chosen


def import_bound(path, name, digest):
    import worker as w
    require(w.sha(path) == digest, 'Bound inference implementation changed: ' + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_diagnostic(profile):
    """CPU-only gate; does not import Torch/ME or require full-evaluation results."""
    import worker as w
    validate_profile(profile)
    root = Path(profile['diagnostic_root'])
    require(not (root / 'FAILED.json').exists(), 'Diagnostic failed; inference profile is not eligible')
    source = w.verified_source(root, pinned=MANIFEST_SHA)
    require(w.sha(root / 'precision.py') == PRECISION_SHA, 'Precision helper changed')
    require(w.sha(root / 'RESULT.json') == profile['diagnostic_result_sha256'], 'Diagnostic RESULT SHA differs')
    sys.dont_write_bytecode = True
    probe = import_bound(root / 'probe.py', '_bound_eval_precision_probe', PROBE_SHA)
    p, probe_source = probe.verify_root(root)
    require(probe_source == source and p.get('capture_pruning_logits') is True, 'Diagnostic provenance/hooks differ')
    result = w.read(root / 'RESULT.json')
    chosen = validate_diagnostic(result, source, probe)
    require(w.sha(root / 'RESULT.json') == profile['diagnostic_result_sha256'], 'Diagnostic result changed while reading')
    return probe, dict(profile=deepcopy(profile), diagnostic_source=source,
        diagnostic_result_sha256=profile['diagnostic_result_sha256'], installed_minkowski_sources=result['installed_minkowski_sources'],
        diagnostic_float64_replay_summaries=deepcopy(chosen['replay_summaries']),
        diagnostic_conversion=deepcopy(chosen['conversion']), diagnostic_state_verification=deepcopy(chosen['state_verification']),
        precision_helper_sha256=PRECISION_SHA, evaluation_scope=profile['scope'],
        pruning_capture_hooks=False, diagnostic_pruning_capture_hooks=True,
        diagnostic_is_identical_evaluation_repetition=False, cold_manager_replay_required=True)


def load_precision_helper(profile):
    validate_profile(profile)
    return import_bound(Path(profile['diagnostic_root']) / 'precision.py', '_bound_eval_decoder_precision', PRECISION_SHA)


def verify_actual_sources(probe, ME, binding):
    actual = probe.verify_minkowski_sources(ME)
    require(actual == binding['installed_minkowski_sources'], 'Actual ME sources differ from admitted diagnostic')
    return actual


def independent_replay(torch, ME, direct_call, replay_call, save_load_call, record):
    """Observe two default managers; never set or clear process-global ME state."""
    retained = []
    record.update(global_state_mutated=False, pruning_capture_hooks=False)

    def register_direct(manager):
        entry_state(ME)
        require(not retained and manager is not None, 'One direct manager required')
        retained.append(manager)
        record['direct']['manager_python_id'] = id(manager)

    def release_original(manager):
        entry_state(ME)
        require(len(retained) == 1 and manager is not None and manager is not retained[0],
                'Disk replay must use a distinct new manager')
        record['replay']['manager_python_id'] = id(manager)
        record['manager_identity_distinct'] = True
        retained.clear()
        record['original_manager_reference_released_before_replay_decode'] = True

    def phase(name, call):
        receipt = record[name] = dict(before=entry_state(ME), body_completed=False, body_error=None)
        try:
            result = call()
            torch.cuda.synchronize(0)
            receipt['body_completed'] = True
            return result
        except BaseException as error:
            receipt['body_error'] = repr(error)
            raise
        finally:
            receipt['after'] = entry_state(ME)

    try:
        direct = phase('direct', lambda: direct_call(register_direct))
        require(len(retained) == 1, 'Direct phase did not record manager identity')
        record['direct_sparse_locals_released_before_replay'] = True
        disk_packet = save_load_call(direct)
        replay = phase('replay', lambda: replay_call(disk_packet, release_original))
        require(not retained, 'Replay did not release old manager before decode')
        record['cold_manager_replay_verified'] = True
        validate_frame_scopes(record)
        return direct, replay
    finally:
        retained.clear()
