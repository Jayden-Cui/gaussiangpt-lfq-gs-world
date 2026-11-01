"""Frozen native decoder, complete initial support, independently learned features."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch
from core import (future_features, image_losses, image_metrics, residual_statistics,
                  object_proxy, coordinate_overlap)
from io_utils import ROOT, read, write, sha, verify


class PruningMonitor:
    def __init__(self, decoder):
        self.rows, self.handles = [], []
        for index, module in enumerate(decoder.up_cls):
            def capture(module, inputs, output, i=index):
                if i == 0: self.rows = []
                logits = output.F.detach()
                self.rows.append(dict(stage=i, candidate_nodes=len(logits),
                    keep_predicted=int((logits > decoder.prune_thresh[i]).sum()),
                    prune_threshold=float(decoder.prune_thresh[i])))
            self.handles.append(module.register_forward_hook(capture))
        def after_prune(module, inputs, output):
            self.rows[-1]['retained_nodes'] = len(output.F)
        self.handles.append(decoder.pruning.register_forward_hook(after_prune))


def tensor_cpu(t):
    return t.detach().cpu().contiguous()


def save_tensor(path, obj):
    with Path(path).open('xb') as stream: torch.save(obj, stream)
    return dict(file=Path(path).name, sha256=sha(path), bytes=Path(path).stat().st_size)


def png(path, image):
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    Image.fromarray((array.clip(0,1)*255).round().astype('uint8')).save(path)


def panel(path, images):
    canvas = Image.new('RGB', (448*len(images), 276), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(images):
        array = (image.detach().cpu().numpy().clip(0,1)*255).round().astype('uint8')
        canvas.paste(Image.fromarray(array), (448*i,24))
        draw.text((448*i+8,6),label,fill='black')
    canvas.save(path)


def render(bridge, prediction, record, views=(0,)):
    points = bridge.decode_points(prediction['coordinates'],prediction['features'])
    gs, sh = bridge.prediction_gs(points)
    if not 0 < len(gs) <= 2400000 or tuple(sh.shape) != (len(gs),4,3):
        raise RuntimeError('Invalid full native GS output')
    images = torch.stack([bridge.render_fn(gs,record['cameras'][v],record['intrinsics'],
                          record['near'],record['far'],sh)[0].clamp(0,1) for v in views])
    if tuple(images.shape) != (len(views),252,448,3) or not bool(torch.isfinite(images).all()):
        raise RuntimeError('Invalid native render')
    return images, gs, sh


def main(start):
    p = verify()
    if start not in p['starts']: raise ValueError('Unregistered endpoint')
    out = ROOT/f's{start:03d}'
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    context = dict(phase='initializing', start=start, step=0)
    write(out/'STARTED.json',dict(pid=os.getpid(),start=start,protocol_sha256=sha(ROOT/'protocol.json'),
          code_sha256=sha(ROOT/'CODE.sha256')))
    try:
        sys.path.insert(0,p['formal_root'])
        from codec_binding import BaselineStates, require, CHECKPOINT_SHA
        from teacher_binding import BoundTeachers
        from runtime_bridge import BoundNativeBridge
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose one GPU')
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        torch.manual_seed(p['seed']); np.random.seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
        free,total = torch.cuda.mem_get_info()
        require(free >= p['minimum_free_gpu_gib']*2**30,'GPU free-memory requirement failed')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
        base = Path(p['formal_root']).parent
        states = BaselineStates(base/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline')
        teachers = BoundTeachers(states)
        target_frame = start+p['horizon']
        initial, actual = (states.load(frame,device='cuda') for frame in (start,target_frame))
        calibration = states.load(0,device='cuda')
        std = calibration['features'].std(0,unbiased=False).detach()
        initial_features = initial['features'].detach().clone()
        initial_coordinates = initial['coordinates'].detach().clone()
        samples = {frame:teachers.load_record(frame,device='cuda') for frame in (start,target_frame)}
        for frame in samples:
            asset = teachers.assets['records'][frame]
            require(sha(teachers.native_root/f'native_group_{frame:03d}.npz') == asset['source_sha256'],
                    'Native source changed')
        record = samples[target_frame]['record']
        target = record['real']
        motion_weight = ((target-samples[start]['record']['real']).abs().mean(-1)/.1).clamp(0,1).detach()
        roi = torch.zeros((252,448),dtype=torch.bool,device='cuda')
        for x0,y0,x1,y1 in p['hand_roi_xyxy']: roi[y0:y1,x0:x1]=True
        bridge = BoundNativeBridge(partial=False,device='cuda').eval()
        require(not any(x.requires_grad for x in bridge.parameters()),'Codec must be fully frozen')
        monitor = PruningMonitor(bridge.codec.ae.decoder)
        object_spec = p['object_proxies'][str(start)]
        target_array = tensor_cpu(target).numpy()
        provenance = dict(codec_sha256=CHECKPOINT_SHA,codec_tensor_digest=bridge.precision.mode_state_digest,
            initial_state_sha256=states.rows[start]['state_sha256'],target_state_sha256=states.rows[target_frame]['state_sha256'],
            calibration_state_sha256=states.rows[0]['state_sha256'],source_cache_sha256=states.rows[target_frame]['source_cache_sha256'],
            start=start,target_frame=target_frame,start_pts=initial['pts'],target_pts=actual['pts'],
            normalization=states.rows[start]['normalization'],output_std=std.tolist(),
            support_overlap=coordinate_overlap(initial_coordinates,actual['coordinates']),
            optimized_parameters=initial_features.numel(),protocol_sha256=sha(ROOT/'protocol.json'),
            code_sha256=sha(ROOT/'CODE.sha256'))
        write(out/'PROVENANCE.json',provenance)
        png(out/'real_target.png',target)
        png(out/'real_initial.png',samples[start]['record']['real'])
        png(out/'teacher_view1.png',samples[target_frame]['teacher_targets'][1])
        packet_rows = []
        references, reference_images = {}, {}

        def inspect(images, gs, sh):
            result = image_metrics(images[0],target,roi,motion_weight)
            teacher = samples[target_frame]['teacher_targets'][1]
            mask = samples[target_frame]['mask']
            result['teacher_view1_masked_mae'] = float((images[1]-teacher).abs().mean(-1)[mask].mean())
            return dict(metrics=result, object_proxy=object_proxy(tensor_cpu(images[0]).numpy(),target_array,object_spec),
                        sparse_nodes=copy.deepcopy(monitor.rows),gs_count=len(gs))

        def save_native(name, images, gs, sh):
            packet = dict(schema='free_feature_native_packet_v1',name=name,start=start,frame=target_frame,
                gs=tensor_cpu(gs),sh=tensor_cpu(sh),cameras=tensor_cpu(record['cameras']),
                intrinsics=tensor_cpu(record['intrinsics']),near=record['near'],far=record['far'],
                resolution_hw=[252,448],provenance=provenance,posterior_prediction=False)
            entry = save_tensor(out/(name+'_native.pt'),packet)
            ref = save_tensor(out/(name+'_render.pt'),tensor_cpu(images))
            packet_rows.append(dict(name=name,packet=entry,reference=ref,gs_count=len(gs)))

        with torch.no_grad():
            for name, prediction in (('held_initial',initial),('decoded_actual_target_state',actual)):
                images,gs,sh = render(bridge,prediction,record,views=(0,1))
                references[name] = inspect(images,gs,sh)
                reference_images[name] = tensor_cpu(images)
                for view in (0,1): png(out/f'{name}_v{view}.png',images[view])
                save_native(name,images,gs,sh)
                del images,gs,sh
        write(out/'REFERENCES.json',references)

        # Real decoder/rasterizer backward, discarded and reset before fitting.
        context.update(phase='gradient_preflight')
        preflight = {}
        for mode in p['modes']:
            residual = torch.zeros_like(initial_features,requires_grad=True)
            prediction = dict(coordinates=initial_coordinates,
                              features=future_features(initial_features,residual,std,mode))
            images,gs,sh = render(bridge,prediction,record)
            loss,_ = image_losses(images[0],target,roi,motion_weight)
            loss.backward()
            require(residual.grad is not None and bool(torch.isfinite(residual.grad).all())
                    and float(residual.grad.norm())>0,'No finite feature reconstruction gradient')
            parity = float((tensor_cpu(images[0])-reference_images['held_initial'][0]).abs().mean())
            require(parity <= p['replay_mae_max'],'Zero-residual initialization differs from held reference')
            optimizer = torch.optim.Adam([residual],lr=p['learning_rate'],betas=tuple(p['betas']),eps=p['eps'])
            preflight[mode] = dict(loss=float(loss.detach()),feature_gradient_norm=float(residual.grad.norm()),
                                  initial_held_mae=parity,discarded_updates=1)
            optimizer.step()
            require(float((future_features(initial_features,residual,std,mode)-initial_features).abs().max())>0,
                    'Optimizer made no feature change')
            bridge.assert_frozen()
            require(all(x.grad is None for x in bridge.parameters()),'Frozen codec acquired gradients')
            del residual,prediction,images,gs,sh,loss,optimizer
        write(out/'PREFLIGHT.json',dict(status='passed',modes=preflight,codec_frozen=True))

        results, final_images = {}, {}
        for mode in p['modes']:
            context.update(phase='optimizing',mode=mode,step=0)
            residual = torch.zeros_like(initial_features,requires_grad=True)
            optimizer = torch.optim.Adam([residual],lr=p['learning_rate'],betas=tuple(p['betas']),eps=p['eps'])
            with (out/f'{mode}_train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
                for step in range(p['steps_per_case']+1):
                    if time.monotonic()-began >= p['worker_max_seconds']:
                        raise RuntimeError('Worker time budget reached')
                    context['step'] = step
                    if step in p['snapshots']:
                        with torch.no_grad():
                            features = future_features(initial_features,residual,std,mode)
                            prediction = dict(coordinates=initial_coordinates,features=features)
                            images,gs,sh = render(bridge,prediction,record,views=(0,1))
                            snapshot = dict(step=step,mode=mode,**inspect(images,gs,sh),
                                residual=residual_statistics(initial_features,residual,std,mode))
                            write(out/f'{mode}_{step:04d}.json',snapshot)
                            for view in (0,1): png(out/f'{mode}_{step:04d}_v{view}.png',images[view])
                            if step == p['steps_per_case']:
                                save_native(mode+'_final',images,gs,sh)
                                save_tensor(out/f'{mode}_final_state.pt',dict(schema='free_feature_endpoint_state_v1',
                                    coordinates=tensor_cpu(initial_coordinates),features=tensor_cpu(features),
                                    residual=tensor_cpu(residual),output_std=tensor_cpu(std),mode=mode,step=step,
                                    provenance=provenance,optimizer=optimizer.state_dict()))
                                results[mode] = snapshot
                                final_images[mode] = tensor_cpu(images)
                            del images,gs,sh,features,prediction
                    if step == p['steps_per_case']: break
                    optimizer.zero_grad(set_to_none=True)
                    fraction = step/max(p['steps_per_case']-1,1)
                    lr = p['final_learning_rate'] + .5*(p['learning_rate']-p['final_learning_rate'])*(1+math.cos(math.pi*fraction))
                    optimizer.param_groups[0]['lr'] = lr
                    features = future_features(initial_features,residual,std,mode)
                    images,gs,sh = render(bridge,dict(coordinates=initial_coordinates,features=features),record)
                    loss,terms = image_losses(images[0],target,roi,motion_weight)
                    require(bool(torch.isfinite(loss)),'Nonfinite loss')
                    loss.backward()
                    require(residual.grad is not None and bool(torch.isfinite(residual.grad).all()),'Nonfinite gradient')
                    grad_norm = float(residual.grad.norm())
                    row = dict(update=step+1,metrics_at_completed_updates=step,mode=mode,loss=float(loss.detach()),
                               terms={k:float(v.detach()) for k,v in terms.items()},gradient_norm=grad_norm,lr=lr,
                               gs_count=len(gs),sparse_nodes=copy.deepcopy(monitor.rows))
                    optimizer.step()
                    require(bool(torch.isfinite(residual).all()),'Nonfinite optimized feature residual')
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    if step==0 or (step+1)%p['progress_every']==0:
                        write(out/'STATUS.json',dict(status='running',**context,completed_updates=step+1,
                            loss_before_update=row['loss'],elapsed_seconds=time.monotonic()-began),replace=True)
                    del features,images,gs,sh,loss,terms
            bridge.assert_frozen()
            require(torch.equal(initial_features,initial['features']) and torch.equal(initial_coordinates,initial['coordinates']),
                    'Original state mutated')
            del residual,optimizer

        for view in (0,1):
            true = tensor_cpu(target) if view==0 else tensor_cpu(samples[target_frame]['teacher_targets'][1])
            panel(out/f'comparison_v{view}.png',[
                ('REAL TARGET' if view==0 else 'NATIVE TEACHER (eval only)',true),
                ('HELD INITIAL',reference_images['held_initial'][view]),
                ('ACTUAL FUTURE STATE',reference_images['decoded_actual_target_state'][view]),
                ('BOUNDED FREE FEATURES',final_images['bounded'][view]),
                ('UNBOUNDED FREE FEATURES',final_images['unbounded'][view])])
        bridge.assert_frozen()
        write(out/'PACKETS.json',dict(start=start,rows=packet_rows,provenance=provenance))
        write(out/'RESULT.json',dict(status='complete',start=start,target_frame=target_frame,
            steps_per_mode=p['steps_per_case'],provenance=provenance,references=references,results=results,
            codec_frozen=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            elapsed_seconds=time.monotonic()-began,action_success='pending visual review',
            independent_native_replay='pending',packets_sha256=sha(out/'PACKETS.json')))
        write(out/'STATUS.json',dict(status='complete',start=start,completed_updates_per_mode=p['steps_per_case'],
                                    elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        write(out/'FAILED.json',dict(status='failed',context=context,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,required=True)
    main(parser.parse_args().start)
