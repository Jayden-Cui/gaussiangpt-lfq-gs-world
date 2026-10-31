"""Fixed400->600 local-condition experiment; original recipes remain immutable."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

sys.dont_write_bytecode = True
ARMS = ('base_z', 'local_z', 'base_constant', 'local_constant')
SOURCE_MANIFEST = 'f144aaebb3ee8980aab1aa4e24cf4e3c395ef87fdd6f1f22ae5743a0385f18c1'
TERMINAL_HELPER = 'a3430beacb0ffb3eeff8bb5ce8988428ac418b4fc8c3fab5345e3cb49d5689db'
SCHEMA = 'gs_local_condition_checkpoint_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def protocol_contract(p):
    require(p['schema'] == 'gs_local_condition_v8' and p['seed'] == 20261031, 'Protocol identity changed')
    require(p['source_manifest_sha256'] == SOURCE_MANIFEST, 'Unexpected source manifest')
    require(set(p['arms']) == set(ARMS), 'Four fixed arms required')
    for arm in ARMS:
        expected = 'constant_delta' if arm.endswith('constant') else 'z_delta'
        require(p['arms'][arm]['source_arm'] == expected and
                p['arms'][arm]['local_skip_enabled'] is arm.startswith('local_'), 'Arm design changed')
    for section, expected in {
        'train': dict(source_step=400, total_steps=600, additional_steps=200, save_steps=[500, 600], max_seconds=1800),
        'evaluation': dict(max_seconds=2400, windows=168, future_exposures=1512, physical_future_frames=176),
        'preflight': dict(max_seconds=600, starts=[0, 42, 84, 126], parity_starts=[35, 123], updates_per_arm=1),
        'resources': dict(cpu_threads=6, peak_limit_gib=48),
        'baseline_tolerances': dict(mae=1e-4, psnr=.02),
    }.items():
        require(all(p[section].get(k) == v for k, v in expected.items()), section + ' contract changed')
    require(p['delta_weight'] == 1 and p['automatic_retry'] is False and p['goal_complete'] is False,
            'Loss/retry/scope changed')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest_path(root, name):
    candidate = Path(name)
    require(not candidate.is_absolute(), 'Absolute manifest entry rejected')
    resolved = (root / candidate).resolve()
    require(resolved.is_relative_to(root.resolve().parent), 'Manifest path escaped outputs parent')
    return resolved


def load_verified(root):
    p = read(root / 'protocol.json'); protocol_contract(p)
    manifest = read(root / 'CODE_MANIFEST.json')
    require(all(name in manifest for name in ('worker.py', 'local_skip.py', 'protocol.json')), 'Incomplete code manifest')
    for name, value in manifest.items(): require(sha(manifest_path(root, name)) == value, 'Bound file changed: ' + name)
    source = Path(p['source_root']).resolve()
    require(sha(source / 'CODE_MANIFEST.json') == SOURCE_MANIFEST, 'Source manifest changed')
    helper_path = source / 'postfit_code/render_six_panel.py'
    require(sha(helper_path) == TERMINAL_HELPER, 'Terminal helper changed')
    helper = module_at('_v8_terminal', helper_path)
    context = helper.load_terminal(source)
    require(context['step'] == 400, 'Source selection must be actual400')
    plan = context['runner'].make_plan(context['evidence']['source_cfg']['seed'], 600)
    require(plan[:400] == context['evidence']['plan'], 'Continuation is not the same seeded schedule')
    return dict(root=root, protocol=p, context=context, helper=helper, plan=plan,
                manifest_hash=sha(root / 'CODE_MANIFEST.json'), protocol_hash=sha(root / 'protocol.json'))


def check_time(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Explicit v8 wall-time budget exceeded')


def verify_unchanged(e):
    require(sha(e['root'] / 'CODE_MANIFEST.json') == e['manifest_hash'] and
            sha(e['root'] / 'protocol.json') == e['protocol_hash'], 'v8 manifest/protocol changed')
    for name, value in read(e['root'] / 'CODE_MANIFEST.json').items():
        require(sha(manifest_path(e['root'], name)) == value, 'v8 bound file changed: ' + name)
    for path, value in e['context']['hashes'].items(): require(sha(path) == value, 'Source input changed: ' + path)


def runtime(e):
    torch, core, staged, temporal, cfg, data = e['context']['runner'].setup_runtime(e['context']['evidence'])
    require(len(data.states) == 177 and len(data.sources) == 354, 'Incomplete original data')
    require(cfg['train']['batch_size'] == 4 and cfg['data']['frames_per_window'] == 10 and
            cfg['resources']['peak_limit_gib'] == 48, 'Original B4/full-window/resource contract changed')
    return dict(torch=torch, core=core, temporal=temporal, cfg=cfg, data=data,
                data_hash=digest(data.sources), runner=e['context']['runner'])


def rng_state(r):
    t = r['torch']
    return dict(torch_rng=t.get_rng_state(), cuda_rng=t.cuda.get_rng_state(),
                numpy_rng=r['core'].np.random.get_state(), python_rng=random.getstate())


def restore_rng(r, state):
    r['torch'].set_rng_state(state['torch_rng']); r['torch'].cuda.set_rng_state(state['cuda_rng'])
    r['core'].np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])


def tensor_digest(items):
    h = hashlib.sha256()
    for name, p in items:
        x = p.detach().cpu().contiguous()
        h.update((name + str(tuple(x.shape)) + str(x.dtype)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def source_metadata(e, arm):
    c = e['context']; parent = e['protocol']['arms'][arm]['source_arm']; item = c['arms'][parent]
    ev = c['evidence']; ancestor = ev['protocol']['arms'][parent]['source_arm']
    return dict(schema='gs_native_v7_checkpoint_v1', arm=parent, source_arm=ancestor,
                source_steps=100, intended_total_steps=400, source_recipe_config_preserved=True,
                source_checkpoint_sha256=ev['source_records'][ancestor]['sha256'],
                code_manifest_sha256=SOURCE_MANIFEST, protocol_sha256=ev['protocol_sha256'],
                fixed_null_key_bias=item['source']['fixed_null_key_bias'],
                null_key_bias_source_sha256=ev['benchmark']['null_key_bias_sha256'], delta_weight=1.)


def restore400(e, r, arm):
    t, core, cfg = r['torch'], r['core'], r['cfg']
    parent = e['protocol']['arms'][arm]['source_arm']; item = e['context']['arms'][parent]
    require(sha(item['path']) == item['evaluation']['checkpoint_sha256'], 'Source400 bytes changed')
    saved = t.load(item['path'], map_location='cpu', weights_only=False)
    require(saved['completed_steps'] == 400 and saved['config'] == cfg and
            saved['plan'] == e['plan'][:400], 'Source400 step/config/plan mismatch')
    for key, value in source_metadata(e, arm).items():
        require(saved.get('continuation', {}).get(key) == value, 'Source400 metadata mismatch: ' + key)
    kind = 'constant' if parent == 'constant_delta' else 'partial'
    model, bridge, optimizer = core.build(cfg, r['data'], kind)
    model.load_state_dict(saved['model'], strict=True); bridge.codec.load_state_dict(saved['codec'], strict=True)
    optimizer.load_state_dict(saved['optimizer']); restore_rng(r, saved)
    loaded_model_hash, loaded_codec_hash = core.tensor_hash(model), core.tensor_hash(bridge.codec)
    require(t.equal(model.output_std.cpu(), saved['model']['output_std']) and bool(model.output_calibrated),
            'TRAIN output units changed')
    from null_key_bias import freeze_null_key_bias
    guard = freeze_null_key_bias(model, optimizer)
    original_fixed = r['runner'].null_key_bias_state(model, optimizer, guard)
    original_groups = [[id(p) for p in g['params']] for g in optimizer.param_groups]
    original_ids = {id(p) for p in model.parameters()}
    before_rng = rng_state(r)
    from local_skip import attach_local_skip, local_skip_grad_norm
    enabled = e['protocol']['arms'][arm]['local_skip_enabled']
    added = attach_local_skip(model, enabled)
    require(t.equal(before_rng['torch_rng'], t.get_rng_state()) and
            t.equal(before_rng['cuda_rng'], t.cuda.get_rng_state()), 'Attachment consumed RNG')
    extra = [(n, p) for n, p in model.named_parameters() if id(p) not in original_ids]
    require(len(extra) == 1 and extra[0][1].shape == (64, 64) and
            bool((extra[0][1] == 0).all()) and extra[0][1].requires_grad == enabled,
            'Added projection must be zero64x64 and enabled exactly by arm')
    require([id(p) for p in added] == ([id(extra[0][1])] if enabled else []), 'Unexpected new parameters')
    if added: optimizer.add_param_group(dict(params=added, lr=cfg['train']['motion_lr']))
    require([[id(p) for p in g['params']] for g in optimizer.param_groups[:len(original_groups)]] == original_groups,
            'Original optimizer parameter membership/order changed')
    # The guard binds ALL model names/identities and optimizer groups. Rebind
    # after the intentional wrapper/group addition, retaining every protected
    # value/moment and original membership; never mutate the old guard.
    guard = freeze_null_key_bias(model, optimizer)
    require(r['runner'].null_key_bias_state(model, optimizer, guard) == original_fixed,
            'Attachment changed protected null-key values/moments/membership')
    model.train(); bridge.train(); r['runner'].ensure_eval_codec(bridge)
    encoder_hash = core.tensor_hash(bridge.codec.ae.encoder)
    require(encoder_hash == item['source']['frozen_encoder_sha256'] and
            bridge.binding == item['evaluation']['source_binding'] and
            r['data'].reference_hash == item['evaluation']['frozen_reference_sha256'], 'Runtime source mismatch')
    require(not any(p.requires_grad for p in bridge.codec.ae.encoder.parameters()), 'Native encoder is not frozen')
    allowed = {id(p) for module in (bridge.codec.ae.decoder.out,bridge.codec.ae.feature_heads,
                                    bridge.codec.ae.gaussian_multiplier) for p in module.parameters()}
    require({id(p) for p in bridge.codec.parameters() if p.requires_grad} == allowed,
            'Native partial unfreezing changed')
    require(bridge.codec.protocol['max_decoder_candidates'] == cfg['resources']['max_decoder_candidates'] and
            bridge.codec.protocol['max_decoded_gaussians'] == cfg['resources']['max_decoded_gaussians'] and
            len(bridge.codec._guards) == len(bridge.codec.ae.decoder.ups)+1, 'Native guards changed')
    state = dict(model=model, bridge=bridge, optimizer=optimizer, guard=guard, extra=extra,
                 encoder_hash=encoder_hash, output_std_hash=tensor_digest([('output_std', model.output_std)]),
                 loaded_model_hash=loaded_model_hash, loaded_codec_hash=loaded_codec_hash,
                 initial_projection_hash=tensor_digest(extra),
                 projection_grad=lambda: local_skip_grad_norm(model))
    del saved
    return state


def invariants(r, state):
    r['runner'].ensure_eval_codec(state['bridge'])
    require(r['core'].tensor_hash(state['bridge'].codec.ae.encoder) == state['encoder_hash'], 'Frozen encoder changed')
    require(tensor_digest([('output_std', state['model'].output_std)]) == state['output_std_hash'], 'TRAIN std changed')
    return r['runner'].null_key_bias_state(state['model'], state['optimizer'], state['guard'])


def provenance(e, r, state, arm, verify=True):
    parent = e['protocol']['arms'][arm]['source_arm']; item = e['context']['arms'][parent]
    return dict(arm=arm, source_arm=parent, source_steps=400, source_checkpoint_path=str(item['path']),
                source_checkpoint_sha256=item['evaluation']['checkpoint_sha256'],
                source_manifest_sha256=SOURCE_MANIFEST, source_selection_sha256=sha(Path(e['protocol']['source_root'])/'SELECTION.json'),
                code_manifest_sha256=e['manifest_hash'], protocol_sha256=e['protocol_hash'],
                data_hash=r['data_hash'], source_binding=state['bridge'].binding,
                frozen_reference_sha256=r['data'].reference_hash, frozen_encoder_sha256=state['encoder_hash'],
                frozen_reference_path=str(r['core'].V4_ROOT/'vw1_h9_lfq/evaluation/targets.jsonl'),
                loaded_model_sha256=state['loaded_model_hash'], loaded_codec_sha256=state['loaded_codec_hash'],
                zero_initial_projection_sha256=state['initial_projection_hash'],
                output_std_sha256=state['output_std_hash'], local_parameter_names=[n for n, _ in state['extra']],
                local_skip_enabled=e['protocol']['arms'][arm]['local_skip_enabled'],
                original_config=r['cfg'], full_plan_sha256=digest(e['plan']), continuation_plan_sha256=digest(e['plan'][400:]),
                fixed_null_key_bias=invariants(r, state) if verify else None,
                source_recipe_config_preserved=True, delta_weight=1., goal_complete=False)


def parity(e, r, state, arm, deadline):
    model, bridge, data = state['model'], state['bridge'], r['data']
    item = e['context']['arms'][e['protocol']['arms'][arm]['source_arm']]
    model.eval(); bridge.eval(); rows = []
    with r['torch'].no_grad():
        for start in e['protocol']['preflight']['parity_starts']:
            check_time(deadline)
            # Match the original evaluation's B4 encoding batches exactly;
            # B1 padding/reduction changes could otherwise flip marginal LFQ bits.
            first = start//4*4
            encoded = model.encode_windows([data.window(s) for s in range(first,first+4)])
            ids = encoded['indices'][start-first:start-first+1]
            require(ids[0].cpu().tolist() == item['rows'][(start,1)]['motion_ids'], 'Source400 action IDs changed')
            frames = model.decode_ids(ids, [r['core'].make_condition(data.states[start])])[0]
            require(len(frames) == 10, 'Parity missing h0..9')
            for h, frame in enumerate(frames):
                require(r['torch'].equal(frame['coordinates'], data.states[start]['coordinates']), 'Parity lost support')
                actual = data.metrics(bridge.render(frame, data.records[start+h]), start, h)
                expected = (item['initial'][start] if h == 0 else item['rows'][(start, h)])['correct']
                for key, value in expected.items():
                    tolerance = e['protocol']['baseline_tolerances']['psnr' if 'psnr' in key else 'mae']
                    require(math.isfinite(actual[key]) and abs(actual[key]-value) <= tolerance, 'Source400 render parity: '+key)
                rows.append(dict(start=start, horizon=h, actual=actual, expected=expected))
    model.train(); bridge.train()
    return rows


def check_gradients(measured, posterior):
    z = [v for row in measured['rgb_quantized_grad_per_sample_h'] for v in row]
    f = [v for row in measured['feature_grad_norm_per_sample_h'] for v in row]
    require(len(z) == 36 and len(f) == 40 and all(math.isfinite(v) and v > 0 for v in z+f), 'Missing36Z/40F gradients')
    require(len(measured['initial_rgb_grad_norm_per_sample']) == 4 and
            all(math.isfinite(v) and v > 0 for v in measured['initial_rgb_grad_norm_per_sample']), 'Missing h0 gradients')
    if posterior:
        q = measured['rgb_query_grad_per_h'] or []
        require(len(q) == 9 and all(math.isfinite(v) and v > 0 for v in q), 'Missing9 query gradients')
    else: require(measured['auxiliary'] == 0, 'Constant posterior auxiliary is not zero')


def preflight(e, out, deadline):
    r = runtime(e); results = {}
    for arm in ARMS:
        check_time(deadline); state = restore400(e, r, arm)
        r['torch'].cuda.reset_peak_memory_stats()
        baseline = parity(e, r, state, arm, deadline)
        bridge = state['bridge']; groups = dict(native_out=bridge.codec.ae.decoder.out,
            native_heads=bridge.codec.ae.feature_heads, native_multiplier=bridge.codec.ae.gaussian_multiplier)
        before = {k:r['core'].tensor_hash(v) for k, v in groups.items()}; p_before = tensor_digest(state['extra'])
        measured = r['temporal'].step_native(state['model'], bridge, state['optimizer'], r['data'],
            e['protocol']['preflight']['starts'], r['cfg'], delta_weight=1.)
        r['torch'].cuda.synchronize(); check_time(deadline)
        check_gradients(measured, not arm.endswith('constant'))
        updated = {k:r['core'].tensor_hash(v) != before[k] for k, v in groups.items()}
        require(all(updated.values()) and all(measured['gradient_norms'][k] > 0 for k in groups), 'Native groups did not update')
        pg = state['projection_grad'](); changed = tensor_digest(state['extra']) != p_before
        enabled = e['protocol']['arms'][arm]['local_skip_enabled']
        require((pg > 0 and changed) if enabled else (pg == 0 and not changed), 'Projection update contract failed')
        require(r['torch'].cuda.max_memory_allocated()/2**30 <= 48, 'Peak48GiB exceeded')
        results[arm] = dict(**provenance(e, r, state, arm), parity=baseline, optimizer_updates=1,
            starts=e['protocol']['preflight']['starts'], frames_per_window=10, all_initial_points_retained=True,
            native_groups_updated=updated, frozen_encoder_unchanged=True, projection_gradient_norm=pg,
            projection_changed=changed, measurement=measured)
        del state, bridge, groups
        r['torch'].cuda.empty_cache()
    verify_unchanged(e); check_time(deadline)
    receipt = dict(schema='gs_local_condition_preflight_v1', status='passed', arms=results,
        code_manifest_sha256=e['manifest_hash'], protocol_sha256=e['protocol_hash'], gpu_witness=True,
        trial_optimizer_updates=4, production_optimizer_updates=0, parameters_discarded=True, goal_complete=False)
    write(out/'RESULT.json', receipt); write(e['root']/'PREFLIGHT.json', receipt)


def witness(e):
    p = e['root']/'PREFLIGHT.json'; value = read(p)
    require(value.get('schema') == 'gs_local_condition_preflight_v1' and value.get('status') == 'passed' and
            value.get('gpu_witness') is True and value.get('parameters_discarded') is True and
            value.get('code_manifest_sha256') == e['manifest_hash'] and value.get('protocol_sha256') == e['protocol_hash'] and
            set(value.get('arms', {})) == set(ARMS), 'Matching complete GPU preflight is required')
    for arm, result in value['arms'].items():
        parent = e['protocol']['arms'][arm]['source_arm']
        require(result['source_checkpoint_sha256'] == e['context']['arms'][parent]['evaluation']['checkpoint_sha256'] and
                result['optimizer_updates'] == 1 and result['frozen_encoder_unchanged'] is True and
                set(result['native_groups_updated']) == {'native_out','native_heads','native_multiplier'} and
                all(result['native_groups_updated'].values()) and len(result['parity']) == 20 and
                result['all_initial_points_retained'] is True and result['frames_per_window'] == 10 and
                result['starts'] == [0,42,84,126], 'Preflight source/native witness mismatch')
        enabled = e['protocol']['arms'][arm]['local_skip_enabled']
        require(result['projection_changed'] is enabled and
                (result['projection_gradient_norm'] > 0 if enabled else result['projection_gradient_norm'] == 0),
                'Preflight projection witness mismatch')
        check_gradients(result['measurement'], not arm.endswith('constant'))
    return sha(p)


def save_checkpoint(path, e, r, state, arm, step, partial=False):
    metadata = dict(schema=SCHEMA, **provenance(e, r, state, arm, verify=not partial), intended_total_steps=600,
                    actual_step=None if partial else step, last_verified_step=step,
                    in_flight_update_may_have_applied=partial, partial=partial, resumable=False)
    saved = dict(model=state['model'].state_dict(), codec=state['bridge'].codec.state_dict(),
        optimizer=state['optimizer'].state_dict(), completed_steps=step, config=r['cfg'], plan=e['plan'],
        continuation=metadata, **rng_state(r))
    with path.open('xb') as stream: r['torch'].save(saved, stream)


def train(e, out, arm, deadline):
    preflight_hash = witness(e); r = runtime(e); state = restore400(e, r, arm)
    source = provenance(e, r, state, arm); write(out/'SOURCE.json', dict(schema='gs_local_condition_source_v1', **source))
    write(out/'PLAN.json', dict(schema='gs_local_condition_plan_v1', starts=e['plan'], source_steps=400,
        intended_total_steps=600, continued_batch_indices=list(range(400,600)),
        code_manifest_sha256=e['manifest_hash'],protocol_sha256=e['protocol_hash'],
        full_plan_sha256=digest(e['plan']), continuation_plan_sha256=digest(e['plan'][400:]), goal_complete=False))
    completed = 400; hashes = {}; started = time.monotonic(); r['torch'].cuda.reset_peak_memory_stats()
    try:
        with (out/'train.jsonl').open('x', encoding='utf-8') as log:
            for index in range(400,600):
                check_time(deadline); r['runner'].ensure_eval_codec(state['bridge'])
                measured = r['temporal'].step_native(state['model'], state['bridge'], state['optimizer'],
                    r['data'], e['plan'][index], r['cfg'], delta_weight=1.)
                completed = index+1; fixed = invariants(r, state)
                measured.update(step=completed, added_steps=completed-400, plan_index=index, starts=e['plan'][index],
                    local_projection_grad_norm_after_clip=state['projection_grad'](),
                    frozen_encoder_sha256=state['encoder_hash'], output_std_sha256=state['output_std_hash'],
                    fixed_null_key_bias=fixed, elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(measured, allow_nan=False)+'\n'); log.flush()
                if completed%10 == 0: print(arm, 'STEP', completed, 'L1', measured['full_l1'], flush=True)
                check_time(deadline)
                if completed in e['protocol']['train']['save_steps']:
                    path=out/f'step_{completed:04d}.pt'; save_checkpoint(path,e,r,state,arm,completed); hashes[str(completed)]=sha(path)
        require(completed == 600, 'Incomplete200-step continuation'); verify_unchanged(e); check_time(deadline)
        write(out/'TRAIN_DONE.json', dict(schema='gs_local_condition_train_v1', status='complete', **source,
            completed_steps=completed, steps=completed, additional_steps=200, checkpoint_sha256=hashes,
            source_sha256=sha(out/'SOURCE.json'), plan_sha256=sha(out/'PLAN.json'), preflight_sha256=preflight_hash,
            train_jsonl_sha256=sha(out/'train.jsonl'), training_seconds=time.monotonic()-started,
            actual_wall_budget_seconds=e['protocol']['train']['max_seconds'], original_config_preserved=True,
            encoder_unchanged=True, last=measured))
    except BaseException:
        try: save_checkpoint(out/'partial.pt',e,r,state,arm,completed,partial=True)
        except BaseException: pass
        raise


def restore600(e, r, arm):
    directory=e['root']/'arms'/arm; done=read(directory/'TRAIN_DONE.json'); checkpoint=directory/'step_0600.pt'
    require(not (directory/'FAILED.json').exists() and done.get('status')=='complete' and
            done.get('completed_steps')==600 and done.get('steps')==600, 'Complete600 training required')
    for key, path in [('source_sha256',directory/'SOURCE.json'),('plan_sha256',directory/'PLAN.json'),
                      ('preflight_sha256',e['root']/'PREFLIGHT.json'),('train_jsonl_sha256',directory/'train.jsonl')]:
        require(done.get(key)==sha(path), 'Train completion binding mismatch: '+key)
    require(done.get('checkpoint_sha256',{}).get('600')==sha(checkpoint), 'Fixed600 checkpoint changed')
    state=restore400(e,r,arm); saved=r['torch'].load(checkpoint,map_location='cpu',weights_only=False)
    require(saved['completed_steps']==600 and saved['plan']==e['plan'] and saved['config']==r['cfg'], 'New checkpoint internal mismatch')
    expected=dict(schema=SCHEMA,**provenance(e,r,state,arm),intended_total_steps=600,actual_step=600,
                  last_verified_step=600,in_flight_update_may_have_applied=False,partial=False,resumable=False)
    require(saved.get('continuation')==expected, 'New checkpoint metadata mismatch')
    state['model'].load_state_dict(saved['model'],strict=True);state['bridge'].codec.load_state_dict(saved['codec'],strict=True)
    state['optimizer'].load_state_dict(saved['optimizer']);restore_rng(r,saved);invariants(r,state)
    # no_grad below suppresses graphs; preserve requires_grad flags bound by
    # the original null-key guard instead of silently invalidating its scope.
    state['model'].eval();state['bridge'].eval()
    return state,done,checkpoint


def snapshot_mapping(context):
    mapping={row['source_frame']: (row['start'],row['horizon']) for row in context['timing'].frame_mapping()}
    selected={frame:mapping[frame] for frame in (0,88,176)}
    selected.update({44:(35,9),132:(123,9)})
    return selected


def save_image(path,image):
    from PIL import Image
    require(tuple(image.shape)==(252,448,3), 'Snapshot native resolution changed')
    Image.fromarray((image.detach().clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(path)


def evaluate(e,out,arm,deadline):
    witness(e);r=runtime(e);state,done,checkpoint=restore600(e,r,arm)
    t,core,data,v7=r['torch'],r['core'],r['data'],r['runner'];model,bridge=state['model'],state['bridge']
    checkpoint_hash=sha(checkpoint);rows=[];initial_rows=[];mapping=snapshot_mapping(e['context']);snapshot_records=[]
    (out/'snapshots').mkdir();t.cuda.reset_peak_memory_stats()
    def capture(start,h,images):
        for frame,pair in mapping.items():
            if pair!=(start,h):continue
            paths={}
            for control in ('real','correct','shuffle84','blinded','adapted_held'):
                path=out/'snapshots'/f'frame_{frame:03d}_{control}.png';save_image(path,images[control]);paths[control]=dict(path=str(path),sha256=sha(path))
            snapshot_records.append(dict(source_frame=frame,start=start,horizon=h,source_pts=float(data.states[frame]['pts']),files=paths))
    with t.no_grad():
        ids=[];blinded=[];constant=model.code_mode=='constant'
        for start in range(0,168,4):
            check_time(deadline);windows=[data.window(s) for s in range(start,start+4)]
            ids.append(model.encode_windows(windows)['indices'])
            if not constant:blinded.append(model.encode_windows([e['context']['probe'].blind_window(w) for w in windows])['indices'])
        ids=t.cat(ids,0);blinded_ids=ids if constant else t.cat(blinded,0)
        require(ids.shape==(168,9) and blinded_ids.shape==(168,9) and bool(((ids>=0)&(ids<65536)).all()), 'Invalid168x9 action IDs')
        require(not constant or bool((ids==model.constant_id).all()), 'Constant IDs not constant')
        with (out/'targets.jsonl').open('x',encoding='utf-8') as log:
            for start in range(168):
                check_time(deadline);condition=[core.make_condition(data.states[start])]
                predictions={'correct':model.decode_ids(ids[start:start+1],condition)[0]}
                if not constant:
                    for offset in (42,84,126):
                        i=(start+offset)%168;predictions[f'shuffle{offset}']=model.decode_ids(ids[i:i+1],condition)[0]
                    predictions['blinded']=model.decode_ids(blinded_ids[start:start+1],condition)[0]
                require(all(len(frames)==10 and all(t.equal(f['coordinates'],data.states[start]['coordinates']) for f in frames)
                    for frames in predictions.values()), 'All10/full initial support violated')
                initial_images={name:bridge.render(frames[0],data.records[start]) for name,frames in predictions.items()}
                if constant:
                    for name in ('shuffle42','shuffle84','shuffle126','blinded'):initial_images[name]=initial_images['correct']
                initial_images['adapted_held']=bridge.render(data.states[start],data.records[start])
                initial_row=dict(start=start,end=start,horizon=0,held_initial=None,frozen_target=None)
                initial_row.update({name:data.metrics(image,start,0) for name,image in initial_images.items()});initial_rows.append(initial_row)
                capture(start,0,dict(initial_images,real=data.records[start]['real']))
                for h in range(1,10):
                    check_time(deadline)
                    row=dict(start=start,end=start+h,horizon=h,motion_ids=ids[start].cpu().tolist(),
                        blinded_motion_ids=blinded_ids[start].cpu().tolist(),blinded_horizon_id_changed=bool(ids[start,h-1]!=blinded_ids[start,h-1]),
                        **data.reference[(start,h)],delta={'held_initial':None,'frozen_target':None})
                    snapshots={};wanted=(start,h) in mapping.values()
                    for name,frames in predictions.items():
                        image=bridge.render(frames[h],data.records[start+h]);row[name]=data.metrics(image,start,h)
                        row['delta'][name]=v7.delta_metrics(image,initial_images[name],data,start,h)
                        if wanted and name in ('correct','shuffle84','blinded'):snapshots[name]=image
                    if constant:
                        for name in ('shuffle42','shuffle84','shuffle126','blinded'):
                            row[name]=dict(row['correct']);row['delta'][name]=dict(row['delta']['correct'])
                            if wanted and name in ('shuffle84','blinded'):snapshots[name]=snapshots['correct']
                    image=bridge.render(data.states[start],data.records[start+h]);row['adapted_held']=data.metrics(image,start,h)
                    row['delta']['adapted_held']=v7.delta_metrics(image,initial_images['adapted_held'],data,start,h)
                    if wanted:capture(start,h,dict(snapshots,adapted_held=image,real=data.records[start+h]['real']))
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                    require(t.cuda.max_memory_allocated()/2**30<=48, 'Peak48GiB exceeded')
                log.flush();del predictions,initial_images
                if start%21==0:print(arm,'EVAL',start+1,'/168',flush=True)
        v7.validate_targets(rows);require(len(initial_rows)==168 and len(snapshot_records)==5,'Incomplete h0/snapshots')
        actions=dict(schema='gs_local_condition_actions_v1',arm=arm,steps=600,ids=ids.cpu(),blinded_ids=blinded_ids.cpu(),
            starts=list(range(168)),horizons=list(range(1,10)),source_frames=177,frame_gap=1,
            checkpoint_sha256=checkpoint_hash,code_manifest_sha256=e['manifest_hash'],protocol_sha256=e['protocol_hash'],
            code_mode=model.code_mode,constant_id=model.constant_id if constant else None,
            semantics='Whole nine-code posterior bundles, not independent adjacent AR actions')
        with (out/'actions.pt').open('xb') as stream:t.save(actions,stream)
        write(out/'initial_reconstruction.json',dict(schema='gs_local_condition_initial_reconstruction_v1',rows=initial_rows,unique_initial_frames=168,
            metrics={c:core.mean_rows(initial_rows,c,['psnr','hand_activity_psnr','mae','hand_activity_mae']) for c in v7.LIVE_CONTROLS},
            uncomputed_controls=['held_initial','frozen_target']))
        write(out/'snapshots.json',dict(rows=sorted(snapshot_records,key=lambda x:x['source_frame']),rerendered=False))
        entropy=[];distinct=[]
        for h in range(9):
            _,counts=ids[:,h].unique(return_counts=True);prob=counts.float()/168
            entropy.append(float(-(prob*prob.log()).sum()));distinct.append(len(counts))
        invariants(r,state);verify_unchanged(e);check_time(deadline);require(sha(checkpoint)==checkpoint_hash,'Checkpoint changed')
        write(out/'EVALUATION.json',dict(schema='gs_local_condition_evaluation_v1',status='complete',
            **provenance(e,r,state,arm),completed_steps=600,steps=600,additional_steps=200,selected_step=600,checkpoint_step=600,
            checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_hash,optimizer_updates=0,
            scope='single_clip_TRAIN_fixed600_posterior_reconstruction_not_AR_or_crosssource',
            source_record_sha256=sha(e['root']/'arms'/arm/'SOURCE.json'),train_done_sha256=sha(e['root']/'arms'/arm/'TRAIN_DONE.json'),
            plan_sha256=sha(e['root']/'arms'/arm/'PLAN.json'),train=done,
            windows=168,real_future_exposures=1512,unique_future_frames=176,physical_frames_with_some_reconstruction=177,
            all_initial_latent_points_retained=True,all_ten_frames=True,full_native_resolution=[252,448],
            metrics={c:core.aggregate(rows,c) for c in v7.ALL_CONTROLS},delta_metrics={c:v7.aggregate_delta(rows,c) for c in v7.ALL_CONTROLS},
            delta_definition='(pred_h-own_pred_0)-(real_h-real_0); each control has its own h0',
            adapted_held_definition='Current step600 arm codec renders unchanged initial state; frozen old reference is separate',
            uncomputed_delta_controls=['held_initial','frozen_target'],constant_shuffle_render_reused=constant,constant_blinded_render_reused=constant,
            blinded_id_statistics=e['context']['probe'].id_statistics(ids.cpu().tolist(),blinded_ids.cpu().tolist()),
            conditional_entropy_nats=sum(entropy)/9,per_h_entropy_nats=entropy,per_h_distinct_ids=distinct,distinct_bundles=len(t.unique(ids,dim=0)),
            targets_sha256=sha(out/'targets.jsonl'),initial_reconstruction_sha256=sha(out/'initial_reconstruction.json'),
            actions_sha256=sha(out/'actions.pt'),snapshots_sha256=sha(out/'snapshots.json'),gpu_peak_gib=t.cuda.max_memory_allocated()/2**30))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('preflight','train','eval'),required=True);p.add_argument('--arm',choices=ARMS)
    args=p.parse_args();root=args.root.resolve()
    if (args.mode=='preflight')!=(args.arm is None):p.error('--arm is required only for train/eval')
    section={'preflight':'preflight','train':'train','eval':'evaluation'}[args.mode]
    raw=read(root/'protocol.json');protocol_contract(raw);deadline=time.monotonic()+raw[section]['max_seconds']
    out=root/'preflight' if args.mode=='preflight' else root/('arms' if args.mode=='train' else 'evaluations')/args.arm
    require(not out.exists(),'Output exists; no overwrite/retry/resume');out.mkdir(parents=True,exist_ok=False)
    try:
        e=load_verified(root);check_time(deadline)
        if args.mode=='preflight':preflight(e,out,deadline)
        elif args.mode=='train':train(e,out,args.arm,deadline)
        else:evaluate(e,out,args.arm,deadline)
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='failed',mode=args.mode,arm=args.arm,error=repr(exc),traceback=traceback.format_exc(),goal_complete=False))
        raise


if __name__=='__main__':main()
