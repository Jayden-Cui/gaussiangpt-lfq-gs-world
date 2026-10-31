"""Bound v7 four-arm continuation and common-checkpoint evaluation.

The immutable v6 implementation supplies data, codec and model constructors.
Imports that can initialize CUDA occur only after provenance verification.
No resume/retry, overwrite, source editing, or synthetic completed steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

sys.dont_write_bytecode = True
ARMS = ('z_rgb', 'constant_rgb', 'z_delta', 'constant_delta')
CONTROLS = ('correct', 'shuffle42', 'shuffle84', 'shuffle126', 'adapted_held', 'held_initial', 'frozen_target')
ALL_CONTROLS = CONTROLS + ('blinded',)
LIVE_CONTROLS = CONTROLS[:5] + ('blinded',)
DELTA_KEYS = ('delta_mae', 'hand_activity_delta_mae', 'motion_delta_mae')
NULL_KEY_BIAS_POLICY = 'qformer_softmax_key_bias_fixed_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def plan_hash(starts):
    return hashlib.sha256(json.dumps(starts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_plan(seed, steps):
    rng = random.Random(seed + 17)
    starts = []
    while len(starts) < steps * 4:
        epoch = list(range(168)); rng.shuffle(epoch); starts.extend(epoch)
    return [starts[i:i+4] for i in range(0, steps*4, 4)]


def verify_manifest(root, allowed_roots):
    manifest = read(root/'CODE_MANIFEST.json')
    if not isinstance(manifest, dict) or 'protocol.json' not in manifest:
        raise ValueError('Manifest must bind protocol.json')
    for relative, expected in manifest.items():
        path = (root/relative).resolve()
        if not any(path.is_relative_to(base) for base in allowed_roots) or sha(path) != expected:
            raise ValueError('Manifest source mismatch: '+str(path))
    return sha(root/'CODE_MANIFEST.json')


def verify_benchmark(root, protocol):
    acceleration = Path(protocol['acceleration_root']).resolve()
    acceptance_path = root/'BENCHMARK_ACCEPTED.json'
    acceptance = read(acceptance_path)
    summary_path = Path(acceptance['summary_path']).resolve()
    staged_path = Path(acceptance['staged_step_path']).resolve()
    if (summary_path != acceleration/'diagnostics/attempt1/SUMMARY.json'
            or staged_path != acceleration/'staged_step.py'
            or sha(summary_path) != acceptance['summary_sha256']
            or sha(staged_path) != acceptance['staged_step_sha256']
            or acceptance.get('minimum_speedup') != 1.2):
        raise ValueError('Benchmark acceptance does not bind the admitted files')
    summary = read(summary_path)
    fixed_bias = summary.get('fixed_null_key_bias',{})
    if (summary.get('schema') != 'gs_native_step_acceleration_benchmark_v1'
            or summary.get('status') != 'complete'
            or any(summary.get(key) is not True for key in
                   ('adoption_eligible','numerical_pass','integrity_pass','resource_pass'))
            or not math.isfinite(summary.get('median_speedup', float('nan')))
            or summary['median_speedup'] < 1.2
            or summary.get('measured_repeats_per_mode') != 2
            or summary.get('starts') != [0,42,84,126]
            or summary.get('checkpoint_completed_steps') != 100
            or fixed_bias.get('policy') != NULL_KEY_BIAS_POLICY
            or fixed_bias.get('count') != 27 or fixed_bias.get('scalar_count') != 864
            or fixed_bias.get('in_optimizer_count') != 27 or fixed_bias.get('unchanged') is not True
            or len(set(fixed_bias.get('parameter_names',[]))) != 27
            or not all(isinstance(fixed_bias.get(k),str) and len(fixed_bias[k])==64
                       for k in ('values_sha256','optimizer_state_sha256'))):
        raise ValueError('Actual benchmark did not pass the required numerical/resource/speed checks')
    source = Path(protocol['source_root']).resolve()
    checkpoint = protocol['source_checkpoints']['partial']
    if (summary.get('baseline_main_manifest_sha256') != sha(source/'CODE_MANIFEST.json')
            or summary.get('baseline_protocol_sha256') != sha(source/'protocol.json')
            or Path(summary.get('checkpoint_path','')).resolve() != Path(checkpoint['path']).resolve()
            or summary.get('checkpoint_sha256') != checkpoint['sha256']):
        raise ValueError('Benchmark baseline differs from this continuation source')
    candidates = summary.get('candidate_sources_sha256', {})
    for name in ('benchmark.py','staged_step.py','test_staged_step.py','README.md',
                 'null_key_bias.py','test_null_key_bias.py'):
        if candidates.get(name) != sha(acceleration/name):
            raise ValueError('Benchmark candidate source changed: '+name)
    declared = protocol['acceleration_source_sha256']
    if not isinstance(declared, dict) or not {'staged_step.py','null_key_bias.py'}.issubset(declared):
        raise ValueError('Protocol must bind acceleration_source_sha256 file/hash dictionary')
    for relative, digest in declared.items():
        path = (acceleration/relative).resolve()
        if not path.is_relative_to(acceleration) or sha(path) != digest:
            raise ValueError('Acceleration source binding changed')
    return dict(acceptance_sha256=sha(acceptance_path),summary_path=str(summary_path),
                summary_sha256=sha(summary_path),staged_step_sha256=sha(staged_path),
                null_key_bias_sha256=sha(acceleration/'null_key_bias.py'),
                fixed_null_key_bias=fixed_bias,
                median_speedup=summary['median_speedup'])


def load_verified(root):
    root = Path(root).resolve()
    protocol = read(root/'protocol.json')
    source = Path(protocol['source_root']).resolve()
    acceleration = Path(protocol['acceleration_root']).resolve()
    main_hash = verify_manifest(root, (root,source,acceleration))
    if not {'runner.py','temporal_step.py','protocol.json','BENCHMARK_ACCEPTED.json'}.issubset(read(root/'CODE_MANIFEST.json')):
        raise ValueError('New manifest does not bind every executed local module and benchmark acceptance')
    if sha(source/'CODE_MANIFEST.json') != protocol['source_manifest_sha256']:
        raise ValueError('Pinned v6 manifest changed')
    verify_manifest(source, (source,))
    if sha(source/'protocol.json') != protocol['source_protocol_sha256']:
        raise ValueError('Pinned source protocol changed')
    blind_helper = source/'analysis_code/future_information_probe.py'
    if (sha(blind_helper) != protocol['future_information_probe_sha256']
            or blind_helper not in {(root/name).resolve() for name in read(root/'CODE_MANIFEST.json')}):
        raise ValueError('The previously verified future-blinding helper is not source-bound')
    source_cfg = read(source/'protocol.json')
    stability = protocol.get('numerical_stability',{})
    if (stability.get('implementation_policy') != NULL_KEY_BIAS_POLICY
            or stability.get('policy') != 'freeze_exact_27_softmax_key_projection_biases_preserve_values_and_optimizer_state'
            or stability.get('tolerances_unchanged') is not True):
        raise ValueError('The explicit unchanged-value/optimizer-state null-bias policy is required')
    expected_arms = {'z_rgb':('partial',0.),'constant_rgb':('constant',0.),
                     'z_delta':('partial',1.),'constant_delta':('constant',1.)}
    if set(protocol['arms']) != set(expected_arms):
        raise ValueError('Expected the four declared factorial arms')
    for arm, (parent,weight) in expected_arms.items():
        if (protocol['arms'][arm]['source_arm'] != parent
                or protocol['arms'][arm]['delta_weight'] != weight):
            raise ValueError('Arm source or delta weight differs from the fixed design')
    if (protocol['train']['total_steps'] != 400 or protocol['train']['max_seconds'] != 1800
            or protocol['train']['save_steps'] != [200,300,400]
            or protocol['preflight']['max_seconds'] != 600
            or protocol['evaluation']['max_seconds'] != 2400):
        raise ValueError('Unexpected fixed continuation/evaluation budget')
    if (source_cfg['data']['frames'] != 177 or source_cfg['data']['window_starts'] != 168
            or source_cfg['data']['frames_per_window'] != 10 or source_cfg['train']['batch_size'] != 4):
        raise ValueError('The native complete-window B4 contract changed')
    plan = make_plan(source_cfg['seed'],400)
    source_records = {}
    for arm in ('partial','constant'):
        bound = protocol['source_checkpoints'][arm]
        path = Path(bound['path']).resolve()
        if path != source/arm/'step_0100.pt' or sha(path) != bound['sha256']:
            raise ValueError('Source must be the exact hash-bound original step100 checkpoint')
        old_plan = read(source/arm/'PLAN.json')
        old_done = read(source/arm/'TRAIN_DONE.json')
        if (old_done.get('status') != 'complete' or old_done.get('steps',0) < 100
                or old_done.get('full_window_coverage') is not True
                or old_done.get('target_frames_seen') != list(range(177))
                or old_plan.get('starts') != plan[:len(old_plan.get('starts',[]))]
                or len(old_plan.get('starts',[])) < 100):
            raise ValueError('Original completed source/seeded plan evidence is invalid')
        source_records[arm] = dict(path=str(path),sha256=bound['sha256'],
            plan=old_plan,plan_sha256=sha(source/arm/'PLAN.json'),
            train_done_sha256=sha(source/arm/'TRAIN_DONE.json'))
    if (source_records['partial']['plan']['initial_model_hash'] != source_records['constant']['plan']['initial_model_hash']
            or source_records['partial']['plan']['starts'][:100] != source_records['constant']['plan']['starts'][:100]):
        raise ValueError('Source arms do not share original initialization/first100 sampling batches')
    benchmark = verify_benchmark(root,protocol)
    return dict(root=root,protocol=protocol,source=source,acceleration=acceleration,
                source_cfg=source_cfg,plan=plan,source_records=source_records,
                code_manifest_sha256=main_hash,protocol_sha256=sha(root/'protocol.json'),benchmark=benchmark)


def setup_runtime(evidence):
    sys.path.insert(0,str(evidence['source']/'code'))
    sys.path.insert(0,str(evidence['acceleration']))
    sys.path.insert(0,str(evidence['root']))
    import torch
    import native_experiment as core
    import staged_step
    import temporal_step
    # The source setup verifies its immutable recipe and applies native guards.
    cfg = core.setup(evidence['source'])
    if cfg != evidence['source_cfg']:
        raise ValueError('Source config changed after verification')
    data = core.Data()
    return torch,core,staged_step,temporal_step,cfg,data


def restore_source(evidence,arm,core,torch,cfg,data):
    source_arm = evidence['protocol']['arms'][arm]['source_arm']
    source = evidence['source_records'][source_arm]
    saved = torch.load(source['path'],map_location='cpu',weights_only=False)
    if (saved.get('completed_steps') != 100 or saved.get('config') != cfg
            or saved.get('plan') != source['plan']['starts']):
        raise ValueError('Source checkpoint internal step/config/plan mismatch')
    model,bridge,optimizer = core.build(cfg,data,source_arm)
    model.load_state_dict(saved['model'],strict=True)
    bridge.codec.load_state_dict(saved['codec'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    core.np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
    # Apply after the complete restore: requires_grad is not serialized, and
    # changing parameter-group membership would invalidate optimizer restore.
    from null_key_bias import POLICY,freeze_null_key_bias
    if POLICY != NULL_KEY_BIAS_POLICY: raise RuntimeError('Unexpected null-key-bias implementation')
    guard = freeze_null_key_bias(model,optimizer)
    null_key_bias_state(model,optimizer,guard)
    model.train();bridge.train()
    del saved
    return model,bridge,optimizer,guard


def null_key_bias_state(model,optimizer,guard):
    from null_key_bias import assert_null_key_bias_unchanged
    state = assert_null_key_bias_unchanged(model,optimizer,guard)
    expected_members = 0 if model.code_mode=='constant' else 27
    if (state.get('policy')!=NULL_KEY_BIAS_POLICY or state.get('count')!=27
            or state.get('scalar_count')!=864 or state.get('unchanged') is not True
            or state.get('in_optimizer_count')!=expected_members):
        raise RuntimeError('Null-key-bias guard violated policy or original optimizer membership')
    return state


def ensure_eval_codec(bridge):
    if bridge.codec.training or bridge.codec.ae.decoder.training:
        raise RuntimeError('Native codec must remain eval with frozen BN statistics')


def source_record(evidence,arm,core,model,bridge,optimizer,guard,data):
    source_arm = evidence['protocol']['arms'][arm]['source_arm']
    source = evidence['source_records'][source_arm]
    return dict(schema='gs_native_v7_continuation_source_v1',arm=arm,source_arm=source_arm,
        source_checkpoint_path=source['path'],source_checkpoint_sha256=source['sha256'],source_steps=100,
        source_plan_sha256=source['plan_sha256'],source_train_done_sha256=source['train_done_sha256'],
        original_initial_model_hash=source['plan']['initial_model_hash'],
        loaded_model_hash=core.tensor_hash(model),loaded_codec_hash=core.tensor_hash(bridge.codec),
        frozen_encoder_sha256=core.tensor_hash(bridge.codec.ae.encoder),
        first100_plan_sha256=plan_hash(evidence['plan'][:100]),
        source_manifest_sha256=evidence['protocol']['source_manifest_sha256'],
        source_protocol_sha256=sha(evidence['source']/'protocol.json'),
        code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256'],
        benchmark=evidence['benchmark'],source_binding=bridge.binding,
        fixed_null_key_bias=null_key_bias_state(model,optimizer,guard),
        null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
        frozen_reference_sha256=data.reference_hash,delta_weight=evidence['protocol']['arms'][arm]['delta_weight'],
        restoration='Full model/codec/optimizer and torch/CUDA/numpy/Python RNG restored; no optimizer reset')


def check_witness(evidence):
    witness = read(evidence['root']/'PREFLIGHT.json')
    if (witness.get('status') != 'passed' or witness.get('code_manifest_sha256') != evidence['code_manifest_sha256']
            or witness.get('protocol_sha256') != evidence['protocol_sha256']
            or witness.get('benchmark_summary_sha256') != evidence['benchmark']['summary_sha256']
            or witness.get('null_key_bias_source_sha256') != evidence['benchmark']['null_key_bias_sha256']
            or set(witness.get('arms',{})) != {'z_delta','constant_delta'}):
        raise ValueError('Current exact-source B4 delta preflight has not passed')
    for arm, result in witness['arms'].items():
        source_arm = evidence['protocol']['arms'][arm]['source_arm']
        fixed = result.get('fixed_null_key_bias',{})
        if (result.get('source_checkpoint_sha256') != evidence['source_records'][source_arm]['sha256']
                or fixed.get('policy')!=NULL_KEY_BIAS_POLICY or fixed.get('unchanged') is not True
                or fixed.get('count')!=27 or fixed.get('in_optimizer_count')!=(0 if source_arm=='constant' else 27)
                or result.get('frozen_encoder_unchanged') is not True
                or not all(result.get('native_groups_updated',{}).get(k) is True
                           for k in ('native_out','native_heads','native_multiplier'))):
            raise ValueError('Preflight did not witness the admitted source or native updates')
    return sha(evidence['root']/'PREFLIGHT.json')


def preflight(evidence,out):
    begun = time.monotonic()
    torch,core,staged,temporal,cfg,data = setup_runtime(evidence)
    result = {}
    for arm in ('z_delta','constant_delta'):
        if time.monotonic()-begun >= 600: raise TimeoutError('Preflight exceeded600 seconds')
        model,bridge,optimizer,guard = restore_source(evidence,arm,core,torch,cfg,data)
        ensure_eval_codec(bridge)
        encoder_before = core.tensor_hash(bridge.codec.ae.encoder)
        groups = dict(native_out=bridge.codec.ae.decoder.out,native_heads=bridge.codec.ae.feature_heads,
                      native_multiplier=bridge.codec.ae.gaussian_multiplier)
        before = {name:core.tensor_hash(module) for name,module in groups.items()}
        torch.cuda.reset_peak_memory_stats()
        measured = temporal.step_native(model,bridge,optimizer,data,[0,42,84,126],cfg,delta_weight=1.)
        torch.cuda.synchronize();ensure_eval_codec(bridge)
        fixed = null_key_bias_state(model,optimizer,guard)
        values = [value for row in measured['rgb_quantized_grad_per_sample_h'] for value in row]
        feature_values = [value for row in measured['feature_grad_norm_per_sample_h'] for value in row]
        if (not all(math.isfinite(v) for v in values+feature_values)
                or len(values)!=36 or min(values)<=0 or len(feature_values)!=40 or min(feature_values)<=0
                or min(measured['initial_rgb_grad_norm_per_sample'])<=0):
            raise RuntimeError('Missing RGB gradient to every B4/horizon/initial feature')
        if arm=='z_delta' and min(measured['rgb_query_grad_per_h'] or [0])<=0:
            raise RuntimeError('Posterior RGB gradient did not reach all9 queries')
        if arm=='constant_delta' and measured['auxiliary'] != 0:
            raise RuntimeError('Constant control unexpectedly received posterior auxiliary loss')
        changed = {name:core.tensor_hash(module)!=before[name] for name,module in groups.items()}
        if (not all(changed.values()) or any(measured['gradient_norms'][name]<=0 for name in groups)
                or core.tensor_hash(bridge.codec.ae.encoder)!=encoder_before
                or torch.cuda.max_memory_allocated()/2**30>48):
            raise RuntimeError('Native update/frozen-encoder/memory witness failed')
        source_arm = evidence['protocol']['arms'][arm]['source_arm']
        result[arm] = dict(source_checkpoint_sha256=evidence['source_records'][source_arm]['sha256'],
            optimizer_updates=1,starts=[0,42,84,126],frames_per_window=10,
            fixed_null_key_bias=fixed,
            frozen_encoder_unchanged=True,native_groups_updated=changed,measurement=measured)
        del model,bridge,optimizer,guard
        torch.cuda.empty_cache()
    if time.monotonic()-begun >= 600: raise TimeoutError('Preflight exceeded600 seconds')
    load_verified(evidence['root'])
    receipt = dict(schema='gs_native_v7_b4_preflight_v1',status='passed',arms=result,
        code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256'],
        benchmark_summary_sha256=evidence['benchmark']['summary_sha256'],elapsed_seconds=time.monotonic()-begun,
        null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
        numerical_stability_policy=NULL_KEY_BIAS_POLICY,
        trial_optimizer_updates=2,production_optimizer_updates=0,parameters_discarded=True,goal_complete=False)
    write_new(out/'RESULT.json',receipt)
    write_new(evidence['root']/'PREFLIGHT.json',receipt)


def checkpoint_save(path,evidence,arm,core,torch,model,bridge,optimizer,guard,steps,cfg):
    source_arm = evidence['protocol']['arms'][arm]['source_arm']
    fixed = null_key_bias_state(model,optimizer,guard)
    saved = dict(model=model.state_dict(),codec=bridge.codec.state_dict(),optimizer=optimizer.state_dict(),
        completed_steps=steps,plan=evidence['plan'],config=cfg,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
        numpy_rng=core.np.random.get_state(),python_rng=random.getstate(),
        continuation=dict(schema='gs_native_v7_checkpoint_v1',arm=arm,source_arm=source_arm,source_steps=100,
            intended_total_steps=400,source_recipe_config_preserved=True,
            fixed_null_key_bias=fixed,null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
            source_checkpoint_sha256=evidence['source_records'][source_arm]['sha256'],
            code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256'],
            delta_weight=evidence['protocol']['arms'][arm]['delta_weight']))
    with path.open('xb') as stream: torch.save(saved,stream)


def train(evidence,out,arm):
    witness_hash = check_witness(evidence)
    begun = time.monotonic()
    torch,core,staged,temporal,cfg,data = setup_runtime(evidence)
    model,bridge,optimizer,guard = restore_source(evidence,arm,core,torch,cfg,data)
    source = source_record(evidence,arm,core,model,bridge,optimizer,guard,data)
    write_new(out/'SOURCE.json',source)
    write_new(out/'PLAN.json',dict(schema='gs_native_v7_plan_v1',starts=evidence['plan'],source_steps=100,
        intended_total_steps=400,continued_batch_indices=list(range(100,400)),
        original_initial_model_hash=source['original_initial_model_hash'],
        first100_plan_sha256=source['first100_plan_sha256'],full_plan_sha256=plan_hash(evidence['plan']),
        code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256']))
    delta_weight = evidence['protocol']['arms'][arm]['delta_weight']
    completed,last = 100,{}
    checkpoint_hashes = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (out/'train.jsonl').open('x',encoding='utf-8') as log:
        for index in range(100,400):
            if time.monotonic()-started >= 1800: break
            ensure_eval_codec(bridge)
            starts = evidence['plan'][index]
            last = (temporal.step_native(model,bridge,optimizer,data,starts,cfg,delta_weight=1.)
                    if delta_weight else staged.step_native(model,bridge,optimizer,data,starts,cfg))
            completed = index+1
            last.update(step=completed,added_steps=completed-100,starts=starts,
                        delta_weight=delta_weight,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(last,allow_nan=False)+'\n');log.flush()
            if completed%10==0: print(arm,'STEP',completed,'L1',last['full_l1'],flush=True)
            if completed in (200,300,400):
                checkpoint_save(out/f'step_{completed:04d}.pt',evidence,arm,core,torch,model,bridge,optimizer,guard,completed,cfg)
                checkpoint_hashes[str(completed)] = sha(out/f'step_{completed:04d}.pt')
    training_seconds = time.monotonic()-started
    ensure_eval_codec(bridge)
    if core.tensor_hash(bridge.codec.ae.encoder)!=source['frozen_encoder_sha256']:
        raise RuntimeError('Frozen native encoder changed during continuation')
    checkpoint_save(out/'final.pt',evidence,arm,core,torch,model,bridge,optimizer,guard,completed,cfg)
    load_verified(evidence['root'])
    fixed = null_key_bias_state(model,optimizer,guard)
    visited = {s for batch in evidence['plan'][100:completed] for s in batch}
    write_new(out/'TRAIN_DONE.json',dict(schema='gs_native_v7_train_done_v1',status='complete',arm=arm,
        steps=completed,completed_steps=completed,source_steps=100,added_steps=completed-100,planned_steps=400,
        stop_reason='step_budget' if completed==400 else 'wall_budget',
        training_seconds=training_seconds,elapsed_seconds=time.monotonic()-begun,
        wall_budget_seconds=1800,budget_check='Before each update; an in-flight update/save may finish after the boundary',
        windows_visited=len(visited),full_window_coverage=visited==set(range(168)),
        target_frames_seen=sorted({s+h for s in visited for h in range(10)}),
        encoder_unchanged=True,final_path=str(out/'final.pt'),final_sha256=sha(out/'final.pt'),
        fixed_null_key_bias=fixed,null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
        saved_steps=[step for step in (200,300,400) if step<=completed],checkpoint_sha256=checkpoint_hashes,
        source_sha256=sha(out/'SOURCE.json'),plan_sha256=sha(out/'PLAN.json'),
        preflight_sha256=witness_hash,code_manifest_sha256=evidence['code_manifest_sha256'],
        protocol_sha256=evidence['protocol_sha256'],last=last,goal_complete=False))


def delta_metrics(image,initial_image,data,start,h):
    if h<=0: raise ValueError('Delta metrics exclude h0')
    error = ((image-initial_image)-(data.records[start+h]['real']-data.records[start]['real'])).abs().mean(-1)
    weight = data.motion_weight(start,h)
    return dict(delta_mae=float(error.mean()),hand_activity_delta_mae=float(error[data.roi].mean()),
                motion_delta_mae=float((error*weight).sum()/weight.sum().clamp_min(1e-6)))


def aggregate_delta(rows,control):
    if control not in LIVE_CONTROLS: return None
    def mean(selected):
        return {key:sum(row['delta'][control][key] for row in selected)/len(selected) for key in DELTA_KEYS}
    groups = {}
    for row in rows: groups.setdefault(row['end'],[]).append(row)
    frame = [mean(group) for group in groups.values()]
    return dict(exposure_mean=mean(rows),
        frame_uniform={key:sum(value[key] for value in frame)/len(frame) for key in DELTA_KEYS},
        per_horizon={str(h):mean([row for row in rows if row['horizon']==h]) for h in range(1,10)})


def validate_targets(rows):
    keys = [(row['start'],row['horizon']) for row in rows]
    if (len(rows)!=1512 or set(keys)!={(s,h) for s in range(168) for h in range(1,10)}
            or any(row['end']!=row['start']+row['horizon'] for row in rows)):
        raise RuntimeError('Evaluation did not cover all168x9 targets exactly once')
    for row in rows:
        for control in ALL_CONTROLS:
            if not all(math.isfinite(value) for value in row[control].values()):
                raise RuntimeError('Nonfinite image metric')
            delta = row['delta'][control]
            if control in LIVE_CONTROLS:
                if set(delta)!=set(DELTA_KEYS) or not all(math.isfinite(v) for v in delta.values()):
                    raise RuntimeError('Incomplete/nonfinite temporal metrics')
            elif delta is not None:
                raise RuntimeError('Uncomputed old reference delta must be null')


def evaluate(evidence,out,arm,step):
    begun = time.monotonic()
    def check():
        if time.monotonic()-begun >= 2400: raise TimeoutError('Evaluation exceeded2400 seconds')
    if step not in (200,300,400): raise ValueError('Evaluation requires a saved common200/300/400 checkpoint')
    directory = evidence['root']/arm
    done = read(directory/'TRAIN_DONE.json')
    checkpoint = directory/f'step_{step:04d}.pt'
    source = read(directory/'SOURCE.json')
    if (done.get('status')!='complete' or done.get('completed_steps',0)<step
            or done.get('source_sha256')!=sha(directory/'SOURCE.json')
            or done.get('plan_sha256')!=sha(directory/'PLAN.json')
            or done.get('code_manifest_sha256')!=evidence['code_manifest_sha256']
            or done.get('protocol_sha256')!=evidence['protocol_sha256']):
        raise ValueError('No matching successful continuation evidence')
    checkpoint_hash = sha(checkpoint)
    if done.get('checkpoint_sha256',{}).get(str(step)) != checkpoint_hash:
        raise ValueError('Selected checkpoint bytes differ from the training completion receipt')
    torch,core,staged,temporal,cfg,data = setup_runtime(evidence)
    import importlib.util
    helper_path = evidence['source']/'analysis_code/future_information_probe.py'
    helper_spec = importlib.util.spec_from_file_location('_v6_future_information_helpers',helper_path)
    blind_helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(blind_helper)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    metadata = saved.get('continuation',{})
    source_arm = evidence['protocol']['arms'][arm]['source_arm']
    if (saved.get('completed_steps')!=step or saved.get('config')!=cfg or saved.get('plan')!=evidence['plan']
            or metadata.get('schema')!='gs_native_v7_checkpoint_v1' or metadata.get('arm')!=arm
            or metadata.get('source_arm')!=source_arm or metadata.get('source_steps')!=100
            or metadata.get('source_checkpoint_sha256')!=evidence['source_records'][source_arm]['sha256']
            or metadata.get('code_manifest_sha256')!=evidence['code_manifest_sha256']
            or metadata.get('protocol_sha256')!=evidence['protocol_sha256']
            or metadata.get('fixed_null_key_bias')!=source.get('fixed_null_key_bias')
            or done.get('fixed_null_key_bias')!=source.get('fixed_null_key_bias')
            or metadata.get('null_key_bias_source_sha256')!=evidence['benchmark']['null_key_bias_sha256']
            or metadata.get('fixed_null_key_bias',{}).get('policy')!=NULL_KEY_BIAS_POLICY
            or metadata.get('delta_weight')!=evidence['protocol']['arms'][arm]['delta_weight']):
        raise ValueError('Selected checkpoint internal source/step/config binding mismatch')
    model,bridge,optimizer = core.build(cfg,data,source_arm);del optimizer
    model.load_state_dict(saved['model'],strict=True);bridge.codec.load_state_dict(saved['codec'],strict=True);del saved
    model.eval().requires_grad_(False);bridge.eval().requires_grad_(False);ensure_eval_codec(bridge)
    if core.tensor_hash(bridge.codec.ae.encoder)!=source['frozen_encoder_sha256']:
        raise RuntimeError('Checkpoint changed the frozen encoder')
    torch.cuda.reset_peak_memory_stats()
    rows,initial_rows = [],[]
    with torch.no_grad():
        ids = []
        for start in range(0,168,4):
            check();ids.append(model.encode_windows([data.window(s) for s in range(start,start+4)])['indices'])
        ids = torch.cat(ids,0)
        constant = model.code_mode=='constant'
        if ids.shape!=(168,9) or (constant and not bool((ids==model.constant_id).all())):
            raise RuntimeError('Invalid complete-window ID bundle')
        if constant:
            blinded_ids = ids
        else:
            batches = []
            for start in range(0,168,4):
                check()
                windows = [blind_helper.blind_window(data.window(s)) for s in range(start,start+4)]
                batches.append(model.encode_windows(windows)['indices'])
            blinded_ids = torch.cat(batches,0)
        id_changes = blind_helper.id_statistics(ids.cpu().tolist(),blinded_ids.cpu().tolist())
        with (out/'targets.jsonl').open('x',encoding='utf-8') as log:
            for start in range(168):
                check();condition=[core.make_condition(data.states[start])]
                predictions={'correct':model.decode_ids(ids[start:start+1],condition)[0]}
                if not constant:
                    for offset in (42,84,126):
                        index=(start+offset)%168
                        predictions[f'shuffle{offset}']=model.decode_ids(ids[index:index+1],condition)[0]
                    predictions['blinded']=model.decode_ids(blinded_ids[start:start+1],condition)[0]
                if any(len(frames)!=10 for frames in predictions.values()): raise RuntimeError('Incomplete decoder10 outputs')
                if any(not torch.equal(frame['coordinates'],data.states[start]['coordinates'])
                       for frames in predictions.values() for frame in frames):
                    raise RuntimeError('Initial latent support changed, including h0')
                initial_images = {}
                initial_row = dict(start=start,end=start,horizon=0)
                for control,frames in predictions.items():
                    initial_images[control]=bridge.render(frames[0],data.records[start])
                    initial_row[control]=data.metrics(initial_images[control],start,0)
                if constant:
                    for name in ('shuffle42','shuffle84','shuffle126','blinded'):
                        initial_images[name]=initial_images['correct'];initial_row[name]=dict(initial_row['correct'])
                initial_images['adapted_held']=bridge.render(data.states[start],data.records[start])
                initial_row['adapted_held']=data.metrics(initial_images['adapted_held'],start,0)
                initial_row.update(held_initial=None,frozen_target=None)
                initial_rows.append(initial_row)
                for h in range(1,10):
                    check()
                    row=dict(start=start,end=start+h,horizon=h,motion_ids=ids[start].cpu().tolist(),
                             blinded_motion_ids=blinded_ids[start].cpu().tolist(),
                             blinded_horizon_id_changed=bool(ids[start,h-1]!=blinded_ids[start,h-1]),
                             **data.reference[(start,h)],delta={'held_initial':None,'frozen_target':None})
                    for control,frames in predictions.items():
                        if not torch.equal(frames[h]['coordinates'],data.states[start]['coordinates']):
                            raise RuntimeError('Initial latent support changed')
                        image=bridge.render(frames[h],data.records[start+h])
                        row[control]=data.metrics(image,start,h)
                        row['delta'][control]=delta_metrics(image,initial_images[control],data,start,h)
                        del image
                    if constant:
                        for name in ('shuffle42','shuffle84','shuffle126','blinded'):
                            row[name]=dict(row['correct']);row['delta'][name]=dict(row['delta']['correct'])
                    image=bridge.render(data.states[start],data.records[start+h])
                    row['adapted_held']=data.metrics(image,start,h)
                    row['delta']['adapted_held']=delta_metrics(image,initial_images['adapted_held'],data,start,h)
                    del image
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                    if torch.cuda.max_memory_allocated()/2**30>48: raise RuntimeError('Original48GiB peak guard exceeded')
                log.flush();del predictions,initial_images
                if start%21==0: print(arm,'EVAL',start+1,'/168',flush=True)
        validate_targets(rows)
        if len(initial_rows)!=168 or {row['start'] for row in initial_rows}!=set(range(168)):
            raise RuntimeError('Incomplete h0 reconstruction set')
        actions = dict(schema='gs_native_v7_common_checkpoint_actions_v1',ids=ids.cpu(),starts=list(range(168)),
            blinded_ids=blinded_ids.cpu(),
            blinding='All10 posterior states use S0 coordinates/features with original nominal times; same current S0 decoded',
            future_information_probe_sha256=evidence['protocol']['future_information_probe_sha256'],
            horizons=list(range(1,10)),source_frames=177,frame_gap=1,all_windows_complete=True,
            arm=arm,selected_step=step,checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_hash,
            source_checkpoint_sha256=evidence['source_records'][source_arm]['sha256'],
            numerical_stability_policy=NULL_KEY_BIAS_POLICY,
            null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
            code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256'],
            cache_manifest_sha256=bridge.binding['cache_manifest_sha256'],
            code_mode=model.code_mode,constant_id=model.constant_id if constant else None,
            semantics='Whole nine-code posterior bundles at each start, or marked constant control; not adjacent independent AR actions')
        with (out/'actions.pt').open('xb') as stream: torch.save(actions,stream)
        write_new(out/'initial_reconstruction.json',dict(rows=initial_rows,unique_initial_frames=168,
            metrics={control:core.mean_rows(initial_rows,control,['psnr','hand_activity_psnr','mae','hand_activity_mae'])
                     for control in LIVE_CONTROLS},
            uncomputed_controls=['held_initial','frozen_target']))
        entropy,distinct = [],[]
        for h in range(9):
            _,counts=ids[:,h].unique(return_counts=True);prob=counts.float()/168
            entropy.append(float(-(prob*prob.log()).sum()));distinct.append(len(counts))
        check();load_verified(evidence['root'])
        if sha(checkpoint)!=checkpoint_hash: raise RuntimeError('Evaluation checkpoint changed')
        write_new(out/'EVALUATION.json',dict(schema='gs_native_v7_common_checkpoint_evaluation_v1',status='complete',
            scope='single_clip_TRAIN_common_checkpoint_posterior_reconstruction',arm=arm,selected_step=step,checkpoint_step=step,
            checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_hash,optimizer_updates=0,goal_complete=False,
            original_completed_steps=done['completed_steps'],source_steps=100,
            code_manifest_sha256=evidence['code_manifest_sha256'],protocol_sha256=evidence['protocol_sha256'],
            source_checkpoint_sha256=evidence['source_records'][source_arm]['sha256'],
            fixed_null_key_bias=metadata['fixed_null_key_bias'],
            null_key_bias_source_sha256=evidence['benchmark']['null_key_bias_sha256'],
            source_record_sha256=sha(directory/'SOURCE.json'),plan_sha256=sha(directory/'PLAN.json'),
            train_done_sha256=sha(directory/'TRAIN_DONE.json'),source_binding=bridge.binding,
            windows=168,real_future_exposures=1512,unique_future_frames=176,physical_frames_with_some_reconstruction=177,
            metrics={control:core.aggregate(rows,control) for control in ALL_CONTROLS},
            delta_metrics={control:aggregate_delta(rows,control) for control in ALL_CONTROLS},
            blinded_id_statistics=id_changes,
            future_information_probe_sha256=evidence['protocol']['future_information_probe_sha256'],
            blinded_scope='Distribution-shift ablation; cannot independently prove action semantics; no unquantized future reaches decoder',
            delta_definition='(pred_h-pred_0)-(real_h-real_0), each control uses its own predicted h0; camera/lighting included; not 3D or semantic-motion error',
            uncomputed_delta_controls=['held_initial','frozen_target'],
            constant_shuffle_render_reused=constant,
            constant_blinded_render_reused=constant,
            constant_reuse_reason='All168x9 IDs equal; same current initial and codec produce identical control predictions' if constant else None,
            conditional_entropy_nats=sum(entropy)/9,per_h_entropy_nats=entropy,per_h_distinct_ids=distinct,
            distinct_bundles=len(torch.unique(ids,dim=0)),frozen_reference_sha256=data.reference_hash,
            initial_reconstruction='initial_reconstruction.json',
            targets_sha256=sha(out/'targets.jsonl'),initial_reconstruction_sha256=sha(out/'initial_reconstruction.json'),
            actions_sha256=sha(out/'actions.pt'),elapsed_seconds=time.monotonic()-begun,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=('preflight','train','eval'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--checkpoint-step',type=int)
    args=parser.parse_args();root=args.root.resolve()
    if args.mode!='preflight' and args.arm is None: parser.error('--arm is required for train/eval')
    if args.mode=='preflight' and args.arm is not None: parser.error('Preflight witnesses both delta arms together')
    if args.mode=='eval' and args.checkpoint_step is None: parser.error('--checkpoint-step is required for eval')
    if args.mode!='eval' and args.checkpoint_step is not None: parser.error('--checkpoint-step is only valid for eval')
    out=(root/'preflight' if args.mode=='preflight' else root/args.arm if args.mode=='train'
         else root/'evaluations'/f'step_{args.checkpoint_step:04d}'/args.arm)
    if out.exists(): parser.error('Output exists: no overwrite, retry or resume')
    out.mkdir(parents=True,exist_ok=False)
    try:
        evidence=load_verified(root)
        if args.mode=='preflight': preflight(evidence,out)
        elif args.mode=='train': train(evidence,out,args.arm)
        else: evaluate(evidence,out,args.arm,args.checkpoint_step)
    except BaseException as error:
        write_new(out/'FAILED.json',dict(status='failed',mode=args.mode,arm=args.arm,
            checkpoint_step=args.checkpoint_step,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
