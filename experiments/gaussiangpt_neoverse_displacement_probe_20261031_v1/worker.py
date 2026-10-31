"""Frozen v9 GS-position reachability probe. CPU import does not access CUDA.

Run and replay are separate CLI processes. Replay loads saved native GS/SH and
camera packets, never a model, Z or in-memory optimized displacement. All modes
require an externally frozen manifest and passed fresh CPU witness; no queue or
GPU launch is performed when importing this module or running its CPU tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import signal
import statistics
import sys
import time
import traceback

import torch

sys.dont_write_bytecode = True
STARTS = (35, 123)
BRANCHES = ('xyz_only', 'bounded_features')
CHECKPOINTS = (0, 25, 50, 75, 100)
SOURCE = '/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_spatial_action_20261031_v9'
PACKET_SCHEMA = 'gsworld_native_gs_oracle_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_tensor(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def protocol_contract(p):
    require(p['schema'] == 'gs_displacement_probe_v1' and p['source_root'] == SOURCE
            and p['source_arm'] == 'spatial_z' and p['selected_step'] == 600, 'Source identity changed')
    require(p['starts'] == list(STARTS) and p['branches'] == list(BRANCHES), 'Probe matrix changed')
    require(p['optimization'] == dict(steps=100, optimizer='Adam', betas=[.9, .999], eps=1e-8,
            weight_decay=0., xyz_lr=.003, feature_lr=.03, checkpoints=list(CHECKPOINTS)), 'Optimization changed')
    require(p['scope'] == dict(source_frames=177, bound_source_files=354, frames_per_window=10,
            frame_gap=1, resolution_hw=[252, 448], all_initial_rows=True, h0_frozen=True,
            native_pruning_unchanged=True, model_updates=0), 'Data/full-point contract changed')
    require(p['resources'] == dict(cpu_threads=6, peak_limit_gib=16, minimum_free_gib=20, preflight_seconds=180,
            window_seconds=1800, replay_seconds=300), 'Resource budget changed')
    require(p['baseline_tolerances'] == dict(mae=1e-4, psnr=.02)
            and p['replay_tolerances'] == dict(max_abs=1e-6, mae=1e-7), 'Tolerance changed')
    require(p['delta_weight'] == 1. and p['automatic_retry'] is False
            and p['goal_complete'] is False, 'Scope/loss changed')
    require(p['numeric_gate'] == dict(future_motion_reduction_fraction=.1,
            h9_motion_reduction_fraction=.1, future_full_roi_nonworse=True,
            visual_acceptance_separate=True), 'Prespecified diagnostic gate changed')


def owned_path(root, name):
    candidate = (Path(root) / name).resolve()
    require(not Path(name).is_absolute() and candidate.is_relative_to(Path(root).resolve()),
            'Artifact path escapes dedicated root')
    return candidate


def verify(root, cpu_witness=True):
    root = Path(root).resolve(); p = read(root/'protocol.json'); protocol_contract(p)
    manifest = read(root/'CODE_MANIFEST.json')
    required = {'worker.py', 'test_worker.py', 'protocol.json', 'DESIGN.md', 'EXECUTION.md'}
    require(required <= set(manifest), 'Incomplete own-source manifest')
    require({f.name for f in root.glob('*.py')} <= set(manifest), 'Unbound root Python source')
    for name, expected in manifest.items():
        require(isinstance(expected, str) and sha(owned_path(root, name)) == expected,
                'Own source changed: '+name)
    binding = dict(code_manifest_sha256=sha(root/'CODE_MANIFEST.json'), protocol_sha256=sha(root/'protocol.json'))
    if cpu_witness:
        witness = read(root/'CPU_VALIDATION.json')
        require(witness.get('status') == 'passed' and witness.get('gpu_executed') is False
                and all(witness.get(k) == v for k, v in binding.items()), 'Fresh bound CPU witness required')
    return p, binding


def verify_sources(p):
    root = Path(p['source_root'])
    for name, expected in p['source_hashes'].items():
        require(sha(owned_path(root, name)) == expected, 'Immutable v9 source changed: '+name)
    require(sha(p['render_source']['path']) == p['render_source']['sha256'], 'Native renderer source changed')
    done = read(root/'queue/DONE.json')
    require(not (root/'queue/FAILED.json').exists() and done['status'] == 'complete'
            and done['summary_sha256'] == p['source_hashes']['SUMMARY.json'], 'Unique terminal v9 DONE required')


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_frozen(*modules):
    for module in modules:
        require(not any(m.training for m in module.modules()), 'Frozen module entered train mode')
        require(not any(p.requires_grad or p.grad is not None for p in module.parameters()),
                'Model acquired a trainable parameter or gradient')


def move_means(gs, displacement):
    require(gs.ndim == 2 and gs.shape[1] == 14 and displacement.shape == (len(gs), 3),
            'Expected complete native GS [M,14] and displacement [M,3]')
    require(gs.device == displacement.device and gs.dtype == displacement.dtype, 'XYZ dtype/device mismatch')
    return torch.cat((gs[:, :3] + displacement, gs[:, 3:]), dim=1)


def normalized_residual(features, initial, units):
    require(features.ndim == 3 and features.shape[1:] == initial.shape
            and units.shape == (initial.shape[-1],), 'Feature parameter shapes differ')
    require(all(bool(torch.isfinite(x).all()) for x in (features, initial, units))
            and bool((units >= 0).all()), 'Nonfinite or negative feature units')
    positive = units > 0; difference = features - initial[None]
    require(bool((difference[..., ~positive] == 0).all()), 'Zero-unit channel changed')
    result = torch.zeros_like(features); result[..., positive] = difference[..., positive]/units[positive]
    require(float(result.abs().max()) <= 1.00001, 'Posterior lies outside original feature range')
    require(torch.allclose(initial[None]+units*result, features, rtol=1e-6, atol=1e-6),
            'Feature initialization is not the posterior')
    return result


def frame_loss(image, initial_image, target, initial_target, roi, motion_weight, horizon):
    require(0 <= horizon <= 9 and image.shape == target.shape == initial_image.shape == initial_target.shape,
            'Complete-window loss shape/horizon mismatch')
    require(roi.shape == motion_weight.shape == image.shape[:2] and roi.dtype == torch.bool
            and bool(roi.any()), 'Invalid ROI/motion mask')
    error = (image-target).abs().mean(-1)
    loss = (error.mean()+error[roi].mean())/10
    if horizon:
        loss = loss + (error*motion_weight).sum()/motion_weight.sum().clamp_min(1e-6)/9
        delta = ((image-initial_image)-(target-initial_target)).abs().mean(-1)
        loss = loss + (delta.mean()+delta[roi].mean())/9
    return loss


def render_pair(gs, sh, record, render_fn, return_alpha=False):
    require(record['cameras'].shape == (2, 4, 4), 'Original two-camera contract changed')
    images, alphas = [], []
    for camera in record['cameras']:
        image, alpha = render_fn(gs, camera, record['intrinsics'], record['near'], record['far'], sh)
        require(image.shape == (252, 448, 3) and bool(torch.isfinite(image).all()), 'Invalid native RGB')
        require(alpha.shape[:2] == (252, 448) and bool(torch.isfinite(alpha).all()), 'Invalid native alpha')
        images.append(image); alphas.append(alpha)
    images = torch.stack(images)
    return (images, torch.stack(alphas)) if return_alpha else images


def validate_packet(packet):
    require(packet.get('schema') == PACKET_SCHEMA and packet.get('resolution_hw') == [252, 448], 'Packet schema/resolution')
    gs, sh = packet['gs'], packet['sh']
    require(gs.dtype == sh.dtype == torch.float32 and gs.device.type == sh.device.type == 'cpu', 'Packet must be CPU float32')
    require(gs.ndim == 2 and gs.shape[1] == 14 and len(gs) > 0 and sh.shape == (len(gs), 4, 3), 'Packet lost GS/SH rows')
    require(all(bool(torch.isfinite(x).all()) for x in (gs, sh, packet['cameras'], packet['intrinsics'])), 'Nonfinite packet')
    require(bool((gs[:, 6:9] > 0).all()) and bool(((gs[:, 13] >= 0) & (gs[:, 13] <= 1)).all()), 'Invalid GS attributes')
    require(bool((gs[:, 9:13].norm(dim=-1) > 0).all()), 'Zero quaternion')
    require(packet['cameras'].shape == (2, 4, 4) and packet['intrinsics'].shape == (4,), 'Camera shape changed')
    require(packet['start'] in STARTS and 0 <= packet['horizon'] <= 9
            and packet['frame'] == packet['start']+packet['horizon'], 'Packet frame mapping changed')
    require(math.isfinite(packet['pts']) and packet['near'] > 0 and packet['far'] > packet['near'], 'Packet time/clipping invalid')
    require(packet['target_fitting'] is True and packet['goal_complete'] is False, 'Oracle scope missing')


def make_packet(gs, sh, record, start, h, pts, normalization, provenance):
    require(gs.dtype == sh.dtype == torch.float32, 'Do not silently convert native GS precision')
    packet = dict(schema=PACKET_SCHEMA, gs=gs.detach().cpu().clone(), sh=sh.detach().cpu().clone(),
        cameras=record['cameras'].detach().cpu().float().clone(), intrinsics=record['intrinsics'].detach().cpu().float().clone(),
        near=float(record['near']), far=float(record['far']), resolution_hw=[252, 448], start=start,
        horizon=h, frame=start+h, pts=float(pts), normalization=normalization, provenance=provenance,
        target_fitting=True, goal_complete=False)
    validate_packet(packet)
    return packet


def compare_render(actual, reference, tolerance):
    require(actual.shape == reference.shape == (2, 252, 448, 3), 'Replay omitted a view or changed resolution')
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all()), 'Nonfinite replay')
    error = (actual-reference).abs()
    result = dict(max_abs=float(error.max()), mae=float(error.mean()))
    require(result['max_abs'] <= tolerance['max_abs'] and result['mae'] <= tolerance['mae'], 'Independent replay mismatch')
    return result


def parity(rows, expected, tolerances):
    require(len(rows) == len(expected) == 10, 'Parity must cover all ten frames')
    for row, ref in zip(rows, expected):
        for k, value in ref.items():
            require(k in row and math.isfinite(row[k]) and abs(row[k]-value) <= tolerances['psnr' if 'psnr' in k else 'mae'],
                    'Posterior parity failed: '+k)


def summarize(rows):
    require(len(rows) == 10 and [r['horizon'] for r in rows] == list(range(10)), 'Incomplete output window')
    keys = rows[0]['metrics'].keys()
    return dict(per_horizon=rows, h0=rows[0]['metrics'],
                mean_future9={k:statistics.fmean(r['metrics'][k] for r in rows[1:]) for k in keys})


def numeric_gate(final, baseline):
    f, b = final['mean_future9'], baseline['mean_future9']
    h9f, h9b = final['per_horizon'][9]['metrics'], baseline['per_horizon'][9]['metrics']
    return dict(future_motion_reduction_ge_10pct=f['motion_mae'] <= .9*b['motion_mae'],
        h9_motion_reduction_ge_10pct=h9f['motion_mae'] <= .9*h9b['motion_mae'],
        future_full_nonworse=f['mae'] <= b['mae'], future_roi_nonworse=f['hand_activity_mae'] <= b['hand_activity_mae'],
        horizon_degradations=[dict(horizon=h, metric=k, increase=final['per_horizon'][h]['metrics'][k]-baseline['per_horizon'][h]['metrics'][k])
            for h in range(1,10) for k in ('mae','hand_activity_mae','motion_mae')
            if final['per_horizon'][h]['metrics'][k] > baseline['per_horizon'][h]['metrics'][k]],
        visual_acceptance='pending', learned_action=False, goal_complete=False)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic(); self.seconds = seconds
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, self.expired); signal.alarm(seconds)
    @staticmethod
    def expired(*unused):
        raise TimeoutError('Fixed displacement probe wall budget expired')
    def check(self):
        require(time.monotonic()-self.started < self.seconds, 'Fixed wall budget expired')
        require(torch.cuda.max_memory_allocated()/2**30 <= 16
                and torch.cuda.max_memory_reserved()/2**30 <= 16, '16 GiB allocation/reservation guard exceeded')


def admit_memory(p):
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    require(free >= p['resources']['minimum_free_gib']*2**30, 'Less than20GiB free; leave shared workloads untouched')
    fraction=p['resources']['peak_limit_gib']*2**30/total
    require(0 < fraction <= 1, 'Device cannot honor16GiB allocator contract')
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.cuda.reset_peak_memory_stats()
    return dict(free_gib=free/2**30,total_gib=total/2**30,allocator_fraction=fraction,
                allocator_cap_gib=16,source_checkpoint_config_unchanged=True,
                measurement_scope='PyTorch allocated/reserved; not NVML whole-process peak')


def frozen_source_runtime(e,p):
    """New resource admission; do not monkeypatch or mutate immutable core.setup/config."""
    evidence=e['context']['evidence']
    for path in (evidence['source']/'code',evidence['acceleration'],evidence['root']):sys.path.insert(0,str(path))
    import native_experiment as core
    import temporal_step
    require(Path(core.__file__).resolve()==(evidence['source']/'code/native_experiment.py').resolve()
            and Path(temporal_step.__file__).resolve()==(evidence['root']/'temporal_step.py').resolve(),
            'Imported runtime modules differ from bound source paths')
    cfg=read(evidence['source']/'protocol.json')
    for name,digest in read(evidence['source']/'CODE_MANIFEST.json').items():
        require(sha(owned_path(evidence['source'],name))==digest,'Original source recipe/code changed')
    require(cfg==evidence['source_cfg'],'Original scientific checkpoint config changed')
    require(cfg['train']['batch_size']==4 and cfg['data']['frames_per_window']==10
            and cfg['resources']['peak_limit_gib']==48,'Historical native recipe changed')
    torch.set_num_threads(6);torch.manual_seed(cfg['seed']);core.np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    data=core.Data()
    return dict(torch=torch,core=core,temporal=temporal_step,cfg=cfg,data=data,
                data_hash=hashlib.sha256(json.dumps(data.sources,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                runner=e['context']['runner'])


def load_runtime(root, p, budget):
    verify_sources(p); source = Path(p['source_root']); sys.path.insert(0, str(source))
    worker = import_path('_displacement_bound_v9_worker', source/'worker.py')
    e = worker.load_verified(source); r = frozen_source_runtime(e,p)
    require(r['data_hash'] == p['data_hash'] and len(r['data'].states) == 177
            and len(r['data'].sources) == 354, 'All 177 source frames must remain bound')
    state, done, checkpoint = worker.restore600(e, r, 'spatial_z')
    require(sha(checkpoint) == p['source_hashes']['arms/spatial_z/step_0600.pt'], 'Final checkpoint changed')
    model, bridge = state['model'], state['bridge']; del state
    model.eval().requires_grad_(False); bridge.eval().requires_grad_(False)
    for module in (model, bridge):
        for parameter in module.parameters(): parameter.grad = None
    assert_frozen(model, bridge)
    core, data = r['core'], r['data']
    initial_hashes = dict(model=core.tensor_hash(model), codec=core.tensor_hash(bridge.codec))
    actions = torch.load(source/'evaluations/spatial_z/actions.pt', map_location='cpu', weights_only=True)
    require(actions['ids'].shape == (168,9) and bool(((actions['ids'] >= 0) & (actions['ids'] < 65536)).all()), 'Invalid saved action IDs')
    targets = {(row['start'],row['horizon']):row for row in (json.loads(line) for line in
               (source/'evaluations/spatial_z/targets.jsonl').read_text().splitlines())}
    require(set(targets) == {(s,h) for s in range(168) for h in range(1,10)}, 'Incomplete v9 target evidence')
    initial = read(source/'evaluations/spatial_z/initial_reconstruction.json')
    initial_rows = {row['start']:row for row in initial['rows']}
    require(set(initial_rows) == set(range(168)), 'Incomplete v9 h0 evidence')
    raw = next(x for x in data.sources if '/cache/' in x['path'])
    normalization = torch.load(raw['path'], map_location='cpu', weights_only=True)['normalization']
    runtime = dict(worker=worker, evidence=e, runtime=r, model=model, bridge=bridge, core=core,
        data=data, hashes=initial_hashes, actions=actions, targets=targets, initial=initial_rows,
        normalization=normalization)
    torch.set_num_threads(6); budget.check()
    return runtime


def frozen_check(r):
    assert_frozen(r['model'], r['bridge'])
    require(dict(model=r['core'].tensor_hash(r['model']), codec=r['core'].tensor_hash(r['bridge'].codec)) == r['hashes'],
            'Frozen model/codec tensor or buffer changed')


def verify_all_data(r):
    for item in r['data'].sources:
        path = Path(item['path'])
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Bound source data changed')
    r['worker'].verify_unchanged(r['evidence'])


def native_gs(r, coordinates, features):
    require(features.shape == (len(coordinates),64) and len(coordinates) > 0, 'All native latent rows required')
    points = r['bridge'].codec.decode_state(coordinates, features)
    return r['bridge'].prediction_gs(points)


@torch.no_grad()
def prepare_window(r, start, budget):
    data = r['data']; initial = data.states[start]; coords = initial['coordinates']
    ids = r['actions']['ids'][start:start+1].to(initial['features'].device)
    require(ids[0].tolist() == r['targets'][(start,1)]['motion_ids'], 'Saved posterior IDs disagree with evaluation')
    predictions = r['model'].decode_ids(ids, [r['core'].make_condition(initial)])[0]
    require(len(predictions) == 10 and all(torch.equal(p['coordinates'], coords) for p in predictions), 'Initial support changed')
    features = torch.stack([p['features'].detach() for p in predictions])
    baseline = [tuple(v.detach().clone() for v in native_gs(r, coords, f)) for f in features]
    unit = (r['model'].output_std*r['model'].residual_scale).detach()
    residual = normalized_residual(features[1:], initial['features'], unit)
    window = dict(start=start, coords=coords, initial_features=initial['features'], baseline_features=features,
        baseline=baseline, units=unit, residual=residual, ids=ids.cpu(), initial_image=None)
    image = render_pair(*baseline[0], data.records[start], r['bridge'].render_fn)[0].clamp(0,1)
    window['initial_image'] = image.detach(); budget.check()
    return window


def observation(r, w, outputs, budget, save_dir=None, provenance=None):
    data=r['data']; start=w['start']; rows=[];own_initial=None
    if save_dir is not None: save_dir.mkdir()
    with torch.no_grad():
        for h in range(10):
            budget.check(); gs, sh = outputs(h)
            record = data.records[start+h]
            images, alphas = render_pair(gs, sh, record, r['bridge'].render_fn, return_alpha=True)
            image=images[0].clamp(0,1); metrics=data.metrics(image,start,h)
            if h == 0:own_initial=image.detach()
            delta=r['runtime']['runner'].delta_metrics(image,own_initial,data,start,h) if h else None
            homogeneous=torch.cat((gs[:,:3],torch.ones_like(gs[:,:1])),dim=1)
            camera_points=homogeneous @ torch.linalg.inv(record['cameras'][0]).T
            outside=(camera_points[:,2] < float(record['near'])) | (camera_points[:,2] > float(record['far']))
            rows.append(dict(horizon=h,frame=start+h,metrics=metrics,delta=delta,gs_count=len(gs),
                outside_depth_fraction=float(outside.float().mean()),
                opacity_weighted_outside_depth=float((outside*gs[:,13]).sum()/gs[:,13].sum().clamp_min(1e-12)),
                view0_alpha_coverage_fraction=float((alphas[0]>1e-6).float().mean())))
            if save_dir is not None:
                packet=make_packet(gs,sh,data.records[start+h],start,h,data.states[start+h]['pts'],r['normalization'],provenance)
                save_tensor(save_dir/f'h{h:02d}.pt',packet)
                save_tensor(save_dir/f'h{h:02d}_reference_rgb.pt',images.detach().cpu())
                from PIL import Image
                for v in range(2):
                    with (save_dir/f'h{h:02d}_view{v}.png').open('xb') as stream:
                        Image.fromarray((images[v].clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(stream,format='PNG')
    return summarize(rows)


def baseline_parity(r,w,actual,p):
    expected=[r['initial'][w['start']]['correct']]+[r['targets'][(w['start'],h)]['correct'] for h in range(1,10)]
    parity([x['metrics'] for x in actual['per_horizon']],expected,p['baseline_tolerances'])


def outputs_for(r,w,branch,parameter):
    def output(h):
        if h == 0: return w['baseline'][0]
        if branch == 'xyz_only':
            gs,sh=w['baseline'][h]
            return move_means(gs,parameter[h-1]),sh
        return native_gs(r,w['coords'],w['initial_features']+w['units']*parameter[h-1])
    return output


def optimize(root,p,binding,r,w,out,budget):
    provenance=dict(**binding,source_checkpoint_sha256=p['source_hashes']['arms/spatial_z/step_0600.pt'],
        posterior_ids=w['ids'].tolist(),source_frames_bound=177,model_updates=0)
    base=observation(r,w,lambda h:w['baseline'][h],budget,out/'posterior_baseline',provenance)
    baseline_parity(r,w,base,p)
    teacher=observation(r,w,lambda h:native_gs(r,r['data'].states[w['start']+h]['coordinates'],
        r['data'].states[w['start']+h]['features']),budget,out/'teacher_state_reference',provenance)
    save_tensor(out/'baseline_state.pt',dict(coordinates=w['coords'].cpu(),features=w['baseline_features'].cpu(),
        initial_features=w['initial_features'].cpu(),units=w['units'].cpu(),ids=w['ids'],stride=4,grid=.008))
    branches={}
    for branch in BRANCHES:
        frozen_check(r)
        parameter=(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(gs[:,:3])) for gs,sh in w['baseline'][1:]])
                   if branch=='xyz_only' else torch.nn.Parameter(w['residual'].clone()))
        params=list(parameter.parameters()) if branch=='xyz_only' else [parameter]
        optimizer=torch.optim.Adam(params,lr=p['optimization']['xyz_lr' if branch=='xyz_only' else 'feature_lr'],
            betas=(.9,.999),eps=1e-8,weight_decay=0.)
        output=outputs_for(r,w,branch,parameter)
        observations={'0':observation(r,w,output,budget)}
        parity([x['metrics'] for x in observations['0']['per_horizon']],
            [x['metrics'] for x in base['per_horizon']],p['baseline_tolerances'])
        with (out/f'{branch}_trace.jsonl').open('x',encoding='utf-8') as trace:
            for step in range(1,101):
                budget.check(); optimizer.zero_grad(set_to_none=True); loss_value=0.; counts=[]
                for h in range(1,10):
                    gs,sh=output(h); record=r['data'].records[w['start']+h]
                    image,_=r['bridge'].render_fn(gs,record['cameras'][0],record['intrinsics'],record['near'],record['far'],sh)
                    require(image.shape==(252,448,3) and bool(torch.isfinite(image).all()),'Invalid optimization render')
                    loss=frame_loss(image.clamp(0,1),w['initial_image'],record['real'],r['data'].records[w['start']]['real'],
                        r['data'].roi,r['data'].motion_weight(w['start'],h),h)
                    require(bool(torch.isfinite(loss)),'Nonfinite loss'); loss.backward(); loss_value+=float(loss.detach());counts.append(len(gs))
                    del gs,sh,image,loss
                gradients=[]
                for variable in params:
                    require(variable.grad is not None and bool(torch.isfinite(variable.grad).all()),'Disconnected/nonfinite oracle gradient')
                    gradients.append(dict(norm=float(variable.grad.norm()),nonzero_fraction=float((variable.grad!=0).float().mean())))
                assert_frozen(r['model'],r['bridge']);optimizer.step()
                if branch=='bounded_features':
                    with torch.no_grad():parameter.clamp_(-1,1)
                require(all(bool(torch.isfinite(x).all()) for x in params),'Nonfinite oracle parameter')
                trace.write(json.dumps(dict(step=step,loss_future_before_update=loss_value,gradients=gradients,
                    gs_counts=counts,elapsed_seconds=time.monotonic()-budget.started),allow_nan=False)+'\n')
                if step in CHECKPOINTS:
                    observations[str(step)]=observation(r,w,output,budget,out/branch if step==100 else None,
                        dict(provenance,branch=branch,steps=100,selected_by='fixed_step100'))
                    trace.flush()
        with torch.no_grad():
            if branch=='xyz_only':
                audit=[dict(horizon=h,norm_quantiles=torch.quantile(d.detach().norm(dim=-1),d.new_tensor([.5,.9,.99,1.])).cpu().tolist())
                       for h,d in enumerate(params,1)]
                values=[x.detach().cpu() for x in params]
                for h in range(1,10):
                    gs,sh=output(h); old,oldsh=w['baseline'][h]
                    require(gs.shape==old.shape and torch.equal(gs[:,3:],old[:,3:]) and torch.equal(sh,oldsh),'XYZ changed nonposition attributes/count')
            else:
                audit=dict(max_abs_residual=float(parameter.abs().max()),near_boundary_fraction=float((parameter.abs()>=.99).float().mean()))
                values=parameter.detach().cpu()
                final_features=torch.cat((w['baseline_features'][:1],w['initial_features'][None]+w['units']*parameter),dim=0)
                save_tensor(out/'bounded_features_state.pt',dict(schema='gsworld_continuous_state_sequence_v1',
                    coordinates=w['coords'].cpu(),features=final_features.cpu(),stride=4,grid=.008,
                    codec_checkpoint_sha256=p['source_hashes']['arms/spatial_z/step_0600.pt'],
                    frames=list(range(w['start'],w['start']+10)),target_fitting=True,goal_complete=False))
            save_tensor(out/f'{branch}_variables.pt',dict(branch=branch,values=values,
                semantic='target_fitting_oracle_not_action_codes',model_updates=0,goal_complete=False))
        frozen_check(r)
        branches[branch]=dict(steps=100,parameters=sum(x.numel() for x in params),observations=observations,
            audit=audit,numeric_gate=numeric_gate(observations['100'],base),legal_action_learning=False)
        del optimizer,parameter,params,output
    return dict(baseline=base,teacher_state_reference=teacher,branches=branches)


def preflight(root,p,binding,budget):
    require(not (root/'PREFLIGHT.json').exists(),'Preserve existing preflight')
    r=load_runtime(root,p,budget);windows=[]
    for start in STARTS:
        w=prepare_window(r,start,budget);observed=observation(r,w,lambda h:w['baseline'][h],budget)
        baseline_parity(r,w,observed,p)
        gradient_checks={}
        for branch in BRANCHES:
            parameter=(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(gs[:,:3])) for gs,sh in w['baseline'][1:]])
                       if branch=='xyz_only' else torch.nn.Parameter(w['residual'].clone()))
            outputs=outputs_for(r,w,branch,parameter);loss_value=0.;metrics=[]
            for h in range(10):
                budget.check();gs,sh=outputs(h);record=r['data'].records[start+h]
                image,_=r['bridge'].render_fn(gs,record['cameras'][0],record['intrinsics'],record['near'],record['far'],sh)
                require(image.shape==(252,448,3) and bool(torch.isfinite(image).all()),'Preflight native image invalid')
                image=image.clamp(0,1)
                metrics.append(r['data'].metrics(image,start,h))
                loss=frame_loss(image,w['initial_image'],record['real'],r['data'].records[start]['real'],
                    r['data'].roi,r['data'].motion_weight(start,h),h)
                loss_value+=float(loss.detach())
                if h:loss.backward()
                else:require(not image.requires_grad,'Preflight h0 acquired oracle gradient')
                del gs,sh,image,loss
            parity(metrics,[x['metrics'] for x in observed['per_horizon']],p['baseline_tolerances'])
            params=list(parameter.parameters()) if branch=='xyz_only' else [parameter]
            require(all(x.grad is not None and bool(torch.isfinite(x.grad).all()) and float(x.grad.norm())>0 for x in params),
                    'Preflight disconnected/nonfinite/zero actual full-window gradient')
            gradient_checks[branch]=dict(loss=loss_value,gradient_norms=[float(x.grad.norm()) for x in params],
                forward_frames=10,backward_future_frames=9,h0_frozen=True,optimizer_updates=0)
            del parameter,outputs,params
        windows.append(dict(start=start,latent_rows=len(w['coords']),posterior_ids=w['ids'].tolist(),baseline=observed,
                            actual_native_gradients=gradient_checks))
    frozen_check(r);verify_all_data(r);verify_sources(p);verify(root);budget.check()
    write_new(root/'PREFLIGHT.json',dict(schema='gs_displacement_preflight_v1',status='passed',**binding,
        source_checkpoint_sha256=p['source_hashes']['arms/spatial_z/step_0600.pt'],frozen_hashes=r['hashes'],
        source_files=r['data'].sources,source_frames=177,windows=windows,model_updates=0,oracle_updates=0,
        gpu_witness=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,resource_adapter=p['resource_adapter'],
        memory_admission=budget.admission,goal_complete=False))


def run_window(root,p,binding,start,out,budget):
    pre=read(root/'PREFLIGHT.json')
    require(pre.get('status')=='passed' and pre.get('gpu_witness') is True and all(pre.get(k)==v for k,v in binding.items()),
            'Bound actual GPU preflight required')
    r=load_runtime(root,p,budget);require(r['hashes']==pre['frozen_hashes'],'Preflight model differs')
    require(pre.get('source_files')==r['data'].sources and pre.get('source_frames')==177
            and {x['start'] for x in pre['windows']}==set(STARTS), 'Preflight source/window scope differs')
    w=prepare_window(r,start,budget);result=optimize(root,p,binding,r,w,out,budget)
    frozen_check(r);verify_all_data(r);verify_sources(p);verify(root);budget.check()
    artifacts={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write_new(out/'RUN_DONE.json',dict(schema='gs_displacement_window_v1',status='complete_pending_independent_replay',**binding,
        start=start,frames=list(range(start,start+10)),source_checkpoint_sha256=p['source_hashes']['arms/spatial_z/step_0600.pt'],
        source_files=r['data'].sources,frozen_hashes=r['hashes'],preflight_sha256=sha(root/'PREFLIGHT.json'),
        **result,artifact_sha256=artifacts,model_updates=0,oracle_updates=200,goal_complete=False,
        all_initial_rows=True,resolution_hw=[252,448],elapsed_seconds=time.monotonic()-budget.started,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,
        resource_adapter=p['resource_adapter'],memory_admission=budget.admission))


def replay(root,p,binding,start,out,budget):
    window=root/'windows'/f'start_{start:04d}'; done=read(window/'RUN_DONE.json')
    require(done['status']=='complete_pending_independent_replay' and done['start']==start
            and all(done.get(k)==v for k,v in binding.items()),'Unbound/incomplete window')
    for name,digest in done['artifact_sha256'].items():require(sha(owned_path(window,name))==digest,'Saved oracle artifact changed')
    renderer=p['render_source'];require(sha(renderer['path'])==renderer['sha256'],'Renderer source changed')
    common=import_path('_displacement_replay_common',renderer['path']); records=[]
    for label in ('posterior_baseline','teacher_state_reference')+BRANCHES:
        for h in range(10):
            budget.check();packet_path=window/label/f'h{h:02d}.pt'
            packet=torch.load(packet_path,map_location='cpu',weights_only=True);validate_packet(packet)
            require(packet['start']==start and packet['horizon']==h,'Wrong replay frame')
            record={k:packet[k].cuda() if torch.is_tensor(packet[k]) else packet[k] for k in ('cameras','intrinsics','near','far')}
            actual=render_pair(packet['gs'].cuda(),packet['sh'].cuda(),record,common.render).cpu()
            reference=torch.load(window/label/f'h{h:02d}_reference_rgb.pt',map_location='cpu',weights_only=True)
            comparison=compare_render(actual,reference,p['replay_tolerances'])
            records.append(dict(label=label,horizon=h,frame=start+h,gs_count=len(packet['gs']),packet_sha256=sha(packet_path),**comparison))
    require(len(records)==40,'Replay omitted native packets')
    verify(root);budget.check()
    write_new(out/'REPLAY_DONE.json',dict(schema='gs_displacement_replay_v1',status='passed',**binding,start=start,
        run_done_sha256=sha(window/'RUN_DONE.json'),records=records,model_loaded=False,optimizer_loaded=False,
        z_required=False,displacement_required=False,views_per_frame=2,memory_admission=budget.admission,
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu_reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--mode',choices=('preflight','run','replay'),required=True)
    parser.add_argument('--start',type=int,choices=STARTS);args=parser.parse_args();root=args.root.resolve();out=None;owned=False
    require((args.mode=='preflight')==(args.start is None),'--start required exactly for run/replay')
    try:
        p,binding=verify(root);seconds=p['resources'][{'preflight':'preflight_seconds','run':'window_seconds','replay':'replay_seconds'}[args.mode]]
        budget=Budget(seconds);budget.admission=admit_memory(p)
        if args.mode=='preflight':preflight(root,p,binding,budget)
        else:
            out=root/('windows' if args.mode=='run' else 'replays')/f'start_{args.start:04d}';out.mkdir(parents=True,exist_ok=False)
            owned=True
            (run_window if args.mode=='run' else replay)(root,p,binding,args.start,out,budget)
        print(json.dumps(dict(status='complete',mode=args.mode,start=args.start,model_updates=0,goal_complete=False)))
    except BaseException as exc:
        if owned and out is not None and not (out/'FAILED.json').exists():
            write_new(out/'FAILED.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),goal_complete=False))
        raise
    finally:
        if hasattr(signal,'SIGALRM'):signal.alarm(0)


if __name__=='__main__':main()
