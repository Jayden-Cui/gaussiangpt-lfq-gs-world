"""Bounded GS-native Z experiments; no runtime architecture/quality retuning.

Frozen state encoder permits the provenance-bound continuous latent cache.
Only decoder.out, attribute heads and multiplier may be unfrozen. All initial
latent coordinates and full 448x252 targets are retained; native AE pruning
is unchanged. RGB target-derived weights occur only inside the loss.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anchored_ldm import AnchoredLDM, make_condition
from gs_bridge import STATE_ROOT, MOTION_ROOT, V4_ROOT, sha, verify_native_sources
from sampling import Segment, Window
from sequence_data import _validate_state

CLIP = 'buildingblock_tower_17_176'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def setup(root):
    cfg = json.loads((root/'protocol.json').read_text())
    manifest = json.loads((root/'CODE_MANIFEST.json').read_text())
    for rel, digest in manifest.items():
        if sha(root/rel) != digest:
            raise RuntimeError('Code/protocol manifest mismatch: '+rel)
    torch.set_num_threads(cfg['resources']['cpu_threads'])
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(cfg['resources']['allocator_fraction'])
    free, _ = torch.cuda.mem_get_info()
    if free < cfg['resources']['minimum_free_gib']*2**30:
        raise RuntimeError('Insufficient free GPU memory; shared tasks must remain untouched')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    return cfg


class NativeBridge(nn.Module):
    def __init__(self, partial=False):
        super().__init__()
        self.binding = verify_native_sources()
        sys.path.insert(0, str(STATE_ROOT/'code'))
        from codec import load_codec
        from common import prediction_gs, render
        self.codec = load_codec(STATE_ROOT, STATE_ROOT/'codec_final.pt', device='cuda')
        self.codec.eval().requires_grad_(False)
        if self.codec.grid != .008 or self.codec.ae.stride != 4:
            raise RuntimeError('Bound native codec grid/stride differs from0.032 latent coordinates')
        if not isinstance(self.codec.ae.vq, nn.Identity):
            raise RuntimeError('State quantization must be absent')
        if partial:
            for module in (self.codec.ae.decoder.out, self.codec.ae.feature_heads,
                           self.codec.ae.gaussian_multiplier):
                module.requires_grad_(True)
        self.partial = partial
        self.prediction_gs = prediction_gs
        self.render_fn = render
        self.last_count = None

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def render(self, prediction, record):
        coordinates, features = prediction['coordinates'], prediction['features']
        if features.shape != (len(coordinates), 64) or len(coordinates) == 0:
            raise RuntimeError('Invalid complete initial latent support')
        points = self.codec.decode_state(coordinates, features)
        gs, sh = self.prediction_gs(points)
        image, _ = self.render_fn(gs, record['cameras'][0], record['intrinsics'],
                                 record['near'], record['far'], sh)
        if image.shape != (252, 448, 3) or not bool(torch.isfinite(image).all()):
            raise RuntimeError('Invalid native-resolution RGB')
        self.last_count = len(gs)
        return image.clamp(0, 1)


class Data:
    def __init__(self, device='cuda'):
        manifest = {item['path']: item for item in json.loads((V4_ROOT/'cache_manifest.json').read_text())}
        self.states, self.records, self.sources = [], [], []
        segment = Segment(CLIP, 0, 176)
        normalized = None
        for frame in range(177):
            state_path = MOTION_ROOT/'states'/CLIP/f'{frame:03d}.pt'
            raw_path = STATE_ROOT/'cache'/CLIP/f'{frame:03d}.pt'
            for path in (state_path, raw_path):
                entry = manifest.get(str(path))
                if entry is None or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                    raise RuntimeError('Cache provenance mismatch: '+str(path))
                self.sources.append(entry)
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            _validate_state(state, state_path, Window(segment, min(frame, 167)), frame)
            state['frame_index'] = frame
            state = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in state.items()}
            if int(state.get('stride', 4)) != 4 or float(state.get('grid', .008)) != .008:
                raise RuntimeError('Frozen state grid/stride changed')
            raw = torch.load(raw_path, map_location='cpu', weights_only=True)
            norm = json.dumps(raw['normalization'], sort_keys=True)
            if normalized is None: normalized = norm
            if norm != normalized: raise RuntimeError('Cross-frame normalization mismatch')
            record = {k:raw[k] for k in ['real','cameras','intrinsics','near','far']}
            record = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in record.items()}
            if record['real'].shape != (252,448,3): raise RuntimeError('Native target resolution changed')
            self.states.append(state); self.records.append(record)
        pts = [float(s['pts']) for s in self.states]
        if any(b <= a for a,b in zip(pts,pts[1:])): raise RuntimeError('Nonmonotone source PTS')
        self.boxes = json.loads((STATE_ROOT/'protocol.json').read_text())['evaluation']['region_boxes'][CLIP]
        self.roi = torch.zeros((252,448),dtype=torch.bool,device=device)
        for x0,y0,x1,y1 in self.boxes['hand_activity']: self.roi[y0:y1,x0:x1] = True
        ref_path = V4_ROOT/'vw1_h9_lfq/evaluation/targets.jsonl'
        self.reference_hash = sha(ref_path)
        self.reference = {}
        for line in ref_path.read_text().splitlines():
            row = json.loads(line)
            if row['clip'] == CLIP and row['start'] < 168:
                key = (row['start'], row['horizon'])
                if key in self.reference: raise RuntimeError('Duplicate frozen reference')
                self.reference[key] = {k:row[k] for k in ('held_initial','frozen_target')}
        if set(self.reference) != {(s,h) for s in range(168) for h in range(1,10)}:
            raise RuntimeError('Incomplete fixed-reference metrics')

    def window(self, start):
        if not 0 <= start <= 167: raise ValueError('Incomplete window requested')
        return self.states[start:start+10]

    def motion_weight(self, start, horizon):
        if horizon == 0: return torch.zeros_like(self.roi, dtype=torch.float32)
        return ((self.records[start+horizon]['real']-self.records[start]['real'])
                .abs().mean(-1)/.1).clamp(0,1).detach()

    def loss(self, image, start, horizon):
        error = (image-self.records[start+horizon]['real']).abs().mean(-1)
        full, roi = error.mean(), error[self.roi].mean()
        weight = self.motion_weight(start,horizon)
        motion = (weight*error).sum()/weight.sum().clamp_min(1e-6)
        # Full/ROI each average all 10; motion averages only 9 real futures.
        loss = full/10+roi/10+(motion/9 if horizon else 0)
        return loss, dict(full_l1=float(full.detach()), roi_l1=float(roi.detach()),
                         motion_l1=float(motion.detach()))

    def metrics(self, image, start, horizon):
        error = image-self.records[start+horizon]['real']
        mse = error.square(); mae = error.abs().mean(-1)
        weight = self.motion_weight(start,horizon)
        return dict(psnr=float(-10*torch.log10(mse.mean().clamp_min(1e-12))),
                    hand_activity_psnr=float(-10*torch.log10(mse[self.roi].mean().clamp_min(1e-12))),
                    mae=float(mae.mean()),hand_activity_mae=float(mae[self.roi].mean()),
                    motion_mae=float((mae*weight).sum()/weight.sum().clamp_min(1e-6)))


def build(cfg, data, arm):
    model = AnchoredLDM(blocks=cfg['model']['blocks'], checkpoint_blocks=True,
                       residual_scale=cfg['model']['residual_scale'],
                       inv_temperature=cfg['model']['inv_temperature'],
                       code_mode='constant' if arm == 'constant' else 'posterior').cuda()
    model.initialize_output_units(data.states[0]['features'])
    bridge = NativeBridge(partial=arm != 'frozen')
    groups = [{'params':[p for p in model.parameters() if p.requires_grad], 'lr':cfg['train']['motion_lr']}]
    native = [p for p in bridge.parameters() if p.requires_grad]
    if native: groups.append({'params':native,'lr':cfg['train']['native_lr']})
    optimizer = torch.optim.AdamW(groups, betas=(.9,.95), weight_decay=0.)
    return model, bridge, optimizer


def grad_norm(module):
    values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def step(model, bridge, optimizer, data, starts, cfg):
    optimizer.zero_grad(set_to_none=True)
    windows = [data.window(s) for s in starts]
    conditions = [make_condition(w[0]) for w in windows]
    z = model.encode_windows(windows)
    z['z_quantized'].retain_grad()
    predictions = model.decode_latents(z['z_quantized'],conditions,initial_contexts=z['initial_contexts'])
    terms=[]; counts=[]
    for b,start in enumerate(starts):
        if len(predictions[b]) != 10: raise RuntimeError('Missing initial/future output')
        for h,prediction in enumerate(predictions[b]):
            if not torch.equal(prediction['coordinates'], windows[b][0]['coordinates']):
                raise RuntimeError('Initial latent support was changed')
            image = bridge.render(prediction, data.records[start+h])
            loss, values = data.loss(image,start,h)
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite RGB loss')
            (loss/len(starts)).backward(retain_graph=True)
            terms.append(values); counts.append(bridge.last_count)
            del image,loss
    rgb_qgrad_all = z['z_quantized'].grad.detach().norm(dim=2).cpu().tolist()
    rgb_qgrad = z['z_quantized'].grad.detach().norm(dim=2).mean(0).cpu().tolist()
    rgb_query_grad = None if model.queries.weight.grad is None else model.queries.weight.grad.detach().norm(dim=1).cpu().tolist()
    aux = z['aux_loss']*cfg['train']['aux_multiplier']
    if aux.requires_grad: aux.backward()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise RuntimeError('Nonfinite gradient')
    norms = dict(residual_head=grad_norm(model.features),motion_encoder=grad_norm(model.encoder_backbone),
                 motion_decoder=grad_norm(model.decoder_backbone),native_trainable=grad_norm(bridge),
                 native_out=grad_norm(bridge.codec.ae.decoder.out),
                 native_heads=grad_norm(bridge.codec.ae.feature_heads),
                 native_multiplier=grad_norm(bridge.codec.ae.gaussian_multiplier))
    torch.nn.utils.clip_grad_norm_(parameters,cfg['train']['grad_clip'],error_if_nonfinite=True)
    optimizer.step()
    result = dict(**{k:float(np.mean([v[k] for v in terms])) for k in terms[0]},
                  auxiliary=float(aux.detach()),rgb_quantized_grad_per_h=rgb_qgrad,
                  rgb_quantized_grad_per_sample_h=rgb_qgrad_all,
                  rgb_query_grad_per_h=rgb_query_grad,gradient_norms=norms,
                  gs_count_min=min(counts),gs_count_max=max(counts),
                  batch_ids=z['indices'].detach().cpu().tolist(),
                  gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    if result['gpu_peak_gib'] > cfg['resources']['peak_limit_gib']:
        raise RuntimeError('Recorded GPU allocation exceeded admitted limit')
    return result


def save_checkpoint(path, model, bridge, optimizer, cfg, completed, plan):
    state = dict(model=model.state_dict(),codec=bridge.codec.state_dict(),optimizer=optimizer.state_dict(),
                 completed_steps=completed,plan=plan,config=cfg,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                 numpy_rng=np.random.get_state(),python_rng=random.getstate())
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def preflight(root):
    cfg=setup(root); begun=time.monotonic(); data=Data()
    # Zero residual must preserve the original state decoder rendering.
    model,bridge,optimizer=build(cfg,data,'partial')
    model.train();bridge.train()
    source_hash=tensor_hash(bridge.codec.ae.encoder)
    partial_modules=dict(native_out=bridge.codec.ae.decoder.out,native_heads=bridge.codec.ae.feature_heads,
                         native_multiplier=bridge.codec.ae.gaussian_multiplier)
    partial_before={name:tensor_hash(module) for name,module in partial_modules.items()}
    with torch.no_grad():
        for start,h in [(0,9),(84,1)]:
            window=data.window(start)
            codes=model.encode_windows([window])
            prediction=model.decode_latents(codes['z_quantized'],[make_condition(window[0])],
                       initial_contexts=codes['initial_contexts'])[0][h]
            if not torch.equal(prediction['features'],window[0]['features']):
                raise RuntimeError('Zero-head model changed the initial features')
            observed=data.metrics(bridge.render(prediction,data.records[start+h]),start,h)
            expected=data.reference[(start,h)]['held_initial']
            if any(abs(observed[k]-expected[k])>.02 for k in ['psnr','hand_activity_psnr']):
                raise RuntimeError('Frozen-reference renderer PSNR parity failed')
            if any(abs(observed[k]-expected[k])>1e-4 for k in ['mae','hand_activity_mae']):
                raise RuntimeError('Frozen-reference renderer MAE parity failed')
    results=[]
    for number in range(3):
        result=step(model,bridge,optimizer,data,[0,42,84,167],cfg)
        results.append(result)
        if number>0 and (min(v for row in result['rgb_quantized_grad_per_sample_h'] for v in row)<=0 or
                         min(result['rgb_query_grad_per_h'] or [0])<=0):
            raise RuntimeError('Missing post-zero-initialization RGB gradient to every Z/query')
        if any(result['gradient_norms'][name]<=0 for name in partial_modules):
            raise RuntimeError('One of the three partial decoder groups has no gradient')
    if tensor_hash(bridge.codec.ae.encoder)!=source_hash: raise RuntimeError('Frozen encoder changed')
    partial_changed={name:tensor_hash(module)!=partial_before[name] for name,module in partial_modules.items()}
    if not all(partial_changed.values()): raise RuntimeError('A partial decoder group did not update')
    with torch.no_grad():
        states=data.window(40);model.eval()
        encoded=model.encode_windows([states]);cond=[make_condition(states[0])]
        direct=model.decode_latents(encoded['z_quantized'],cond,initial_contexts=encoded['initial_contexts'])
        replay=model.decode_ids(encoded['indices'],cond)
        replay_error=max(float((a['features']-b['features']).abs().max()) for a,b in zip(direct[0],replay[0]))
        if replay_error>1e-3: raise RuntimeError('ID replay changed latent prediction')
    write(root/'PREFLIGHT.json',dict(status='passed',scope='B4_all10_RGB_three_optimizer_updates_partial_decoder',
          optimizer_updates=3,elapsed_seconds=time.monotonic()-begun,results=results,
          initial_encoder_unchanged=True,id_replay_max_abs=replay_error,
          partial_groups_updated=partial_changed,code_manifest_sha256=sha(root/'CODE_MANIFEST.json'),
          protocol_sha256=sha(root/'protocol.json'),
          fixed_reference_metrics_sha256=data.reference_hash,source_binding=bridge.binding,
          limit='Engineering check only; trained trial parameters discarded, no quality claim'))


def make_plan(cfg):
    rng=random.Random(cfg['seed']+17);starts=[]
    while len(starts)<cfg['train']['steps']*4:
        epoch=list(range(168));rng.shuffle(epoch);starts.extend(epoch)
    return [starts[i:i+4] for i in range(0,cfg['train']['steps']*4,4)]


def run_train(root,arm):
    cfg=setup(root)
    witness=json.loads((root/'PREFLIGHT.json').read_text())
    if (witness['status']!='passed' or witness['code_manifest_sha256']!=sha(root/'CODE_MANIFEST.json')
            or witness['protocol_sha256']!=sha(root/'protocol.json')):
        raise RuntimeError('Real B4 preflight has not passed for this exact code/protocol')
    out=root/arm;out.mkdir(exist_ok=False)
    begun=time.monotonic();data=Data();model,bridge,optimizer=build(cfg,data,arm)
    model.train();bridge.train();plan=make_plan(cfg);completed=0;last={}
    initial_hash=tensor_hash(model);encoder_hash=tensor_hash(bridge.codec.ae.encoder)
    write(out/'PLAN.json',dict(starts=plan,initial_model_hash=initial_hash,
          partial_native=bridge.partial,native_trainable_parameters=sum(p.numel() for p in bridge.parameters() if p.requires_grad),
          protocol_sha256=sha(root/'protocol.json'),code_manifest_sha256=sha(root/'CODE_MANIFEST.json')))
    started=time.monotonic()
    with (out/'train.jsonl').open('x') as log:
        for number,starts in enumerate(plan,1):
            if time.monotonic()-started>cfg['train']['max_seconds']:break
            last=step(model,bridge,optimizer,data,starts,cfg);completed=number
            last.update(step=number,starts=starts,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(last,allow_nan=False)+'\n');log.flush()
            if number==1 or number%10==0:
                write(out/'STATUS.json',dict(status='training',**last))
                print(arm,'STEP',number,'L1',round(last['full_l1'],6),'ROI',round(last['roi_l1'],6),flush=True)
            if number%50==0:
                save_checkpoint(out/f'step_{number:04d}.pt',model,bridge,optimizer,cfg,completed,plan)
    if tensor_hash(bridge.codec.ae.encoder)!=encoder_hash:raise RuntimeError('Frozen encoder changed during training')
    save_checkpoint(out/'final.pt',model,bridge,optimizer,cfg,completed,plan)
    visited=set(s for batch in plan[:completed] for s in batch)
    write(out/'TRAIN_DONE.json',dict(status='complete',steps=completed,planned_steps=cfg['train']['steps'],
          stop_reason='step_budget' if completed==cfg['train']['steps'] else 'wall_budget',
          elapsed_seconds=time.monotonic()-begun,windows_visited=len(visited),
          full_window_coverage=visited==set(range(168)),target_frames_seen=sorted({s+h for s in visited for h in range(10)}),
          encoder_unchanged=True,final_sha256=sha(out/'final.pt'),last=last))
    del model,bridge,optimizer,data
    torch.cuda.empty_cache()
    run_eval(root,arm)


def mean_rows(rows,control,keys):
    return {k:float(np.mean([r[control][k] for r in rows])) for k in keys}


def aggregate(rows,control):
    keys=['psnr','hand_activity_psnr','mae','hand_activity_mae']
    if 'motion_mae' in rows[0][control]:keys.append('motion_mae')
    per_frame={}
    for r in rows:per_frame.setdefault(r['end'],[]).append(r)
    return dict(exposure_mean=mean_rows(rows,control,keys),
                frame_uniform={k:float(np.mean([mean_rows(group,control,[k])[k] for group in per_frame.values()])) for k in keys},
                per_horizon={str(h):mean_rows([r for r in rows if r['horizon']==h],control,keys) for h in range(1,10)})


@torch.no_grad()
def run_eval(root,arm):
    cfg=setup(root);begun=time.monotonic();out=root/arm;data=Data()
    model,bridge,optimizer=build(cfg,data,arm);del optimizer
    saved=torch.load(out/'final.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['model']);bridge.codec.load_state_dict(saved['codec'])
    model.eval();bridge.eval();del saved
    ids=[]
    for start in range(0,168,4):
        ids.append(model.encode_windows([data.window(s) for s in range(start,start+4)])['indices'])
    ids=torch.cat(ids,0)
    torch.save(dict(schema='common_initial_h9_bundle_v1',ids=ids.cpu(),starts=list(range(168)),
                    horizons=list(range(1,10)),source_frames=177,frame_gap=1,
                    code_mode=model.code_mode,constant_id=model.constant_id if model.code_mode=='constant' else None,
                    all_windows_complete=True,cache_manifest_sha256=bridge.binding['cache_manifest_sha256'],
                    checkpoint_sha256=sha(out/'final.pt'),state_source_sha256=bridge.binding['state_checkpoint_sha256'],
                    semantics='Posterior labels or explicitly marked constant-control IDs; not a flattened physical adjacent-action stream or complete AR sample'),out/'actions.pt')
    rows=[];initial_rows=[];controls=['correct','shuffle42','shuffle84','shuffle126','adapted_held','held_initial','frozen_target']
    with (out/'targets.jsonl').open('x') as log:
        for start in range(168):
            if time.monotonic()-begun>cfg['evaluation']['max_seconds']:raise TimeoutError('Evaluation budget exhausted')
            condition=[make_condition(data.states[start])]
            predictions={'correct':model.decode_ids(ids[start:start+1],condition)[0]}
            for offset in [42,84,126]:
                source=(start+offset)%168
                predictions[f'shuffle{offset}']=model.decode_ids(ids[source:source+1],condition)[0]
            initial_image=bridge.render(predictions['correct'][0],data.records[start])
            initial_row=dict(start=start,end=start,horizon=0,correct=data.metrics(initial_image,start,0))
            held_image=bridge.render(data.states[start],data.records[start])
            initial_row['adapted_held']=data.metrics(held_image,start,0)
            initial_rows.append(initial_row)
            for h in range(1,10):
                row=dict(start=start,end=start+h,horizon=h,motion_ids=ids[start].cpu().tolist(),
                         **data.reference[(start,h)])
                for control,frames in predictions.items():
                    image=bridge.render(frames[h],data.records[start+h])
                    row[control]=data.metrics(image,start,h)
                image=bridge.render(data.states[start],data.records[start+h])
                row['adapted_held']=data.metrics(image,start,h)
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            if start%21==0:
                write(out/'STATUS.json',dict(status='evaluating',windows=start+1,targets=len(rows)))
    summary={control:aggregate(rows,control) for control in controls}
    write(out/'initial_reconstruction.json',dict(rows=initial_rows,unique_initial_frames=168,
          metrics={control:mean_rows(initial_rows,control,['psnr','hand_activity_psnr','mae','hand_activity_mae'])
                   for control in ['correct','adapted_held']}))
    entropy=[];distinct=[]
    for h in range(9):
        _,counts=ids[:,h].unique(return_counts=True);p=counts.float()/168
        entropy.append(float(-(p*p.log()).sum()));distinct.append(len(counts))
    write(out/'EVALUATION.json',dict(status='complete',scope='single_clip_TRAIN_posterior_reconstruction',
          checkpoint_sha256=sha(out/'final.pt'),windows=168,real_future_exposures=len(rows),unique_future_frames=176,
          metrics=summary,conditional_entropy_nats=float(np.mean(entropy)),per_h_entropy_nats=entropy,
          initial_reconstruction='initial_reconstruction.json',physical_frames_with_some_reconstruction=177,
          per_h_distinct_ids=distinct,distinct_bundles=len(torch.unique(ids,dim=0)),
          frozen_reference_sha256=data.reference_hash,elapsed_seconds=time.monotonic()-begun,
          gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30))
    write(out/'STATUS.json',dict(status='complete',training_steps=json.loads((out/'TRAIN_DONE.json').read_text())['steps'],
          evaluation='EVALUATION.json',elapsed_seconds=time.monotonic()-begun))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--mode',choices=['preflight','train','eval'],required=True)
    p.add_argument('--arm',choices=['frozen','partial','constant'])
    a=p.parse_args()
    try:
        if a.mode=='preflight':preflight(a.root)
        elif a.mode=='train':run_train(a.root,a.arm)
        else:run_eval(a.root,a.arm)
    except BaseException as error:
        target=a.root/(a.arm or 'preflight_failure')
        target.mkdir(parents=True,exist_ok=True)
        write(target/'FAILED.json',dict(status='failed',mode=a.mode,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
