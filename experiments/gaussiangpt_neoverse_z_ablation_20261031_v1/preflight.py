"""Matched real B4/T10 preflight: two discarded updates per action arm.

No formal training is launched. All sources, native replay and teacher assets
are checked by BoundData; the fixed action scale is calibrated on all 168
windows from one common initialization before the arms are cloned.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from codec_binding import CHECKPOINT_SHA, require, sha
from model_adapter import load_pinned, attach_adapters, clone_pair
from paired_step import step_native, grad_norm
from runner_core import candidate_plan
from runtime_bridge import BoundNativeBridge, _bound_module
from runtime_data import BoundData

BASE = Path('/data/CJH/AGS-V6Direct/outputs')
ARM = BASE/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline'
NATIVE = BASE/'gaussiangpt_neoverse_native_codec_replay_20261031_v1/results'
TEMPORAL = BASE/'gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py'
TEACHER = BASE/'gaussiangpt_neoverse_gs_supervision_20261031_v10b/loss_step.py'
NULL_BIAS = BASE/'gaussiangpt_neoverse_step_acceleration_20261031_v2/null_key_bias.py'


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def model_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        t=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parent
    out=root/'preflight_v1'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    context={'phase':'initializing'}
    write(out/'STARTED.json',dict(pid=os.getpid(),formal_training=False,discarded_updates_per_arm=2))
    try:
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one assigned GPU required')
        torch.set_num_threads(6); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info()
        require(free>=32*2**30,'Preflight requires 32 GiB actually free on the shared GPU')
        torch.cuda.set_per_process_memory_fraction(24*2**30/total)
        torch.cuda.reset_peak_memory_stats()
        bridge_smoke=json.loads((root/'bridge_smoke/RESULT.json').read_text())
        require(bridge_smoke['status']=='passed' and bridge_smoke['checkpoint_sha256']==CHECKPOINT_SHA
                and bridge_smoke['optimizer_updates']==0 and bridge_smoke['formal_training'] is False
                and bridge_smoke['bridge_source_sha256']==sha(root/'runtime_bridge.py'), 'Bound native bridge smoke required')
        api=load_pinned(BASE)
        temporal=_bound_module('_z_preflight_temporal',TEMPORAL,'6a1de737b3e87d88c6678352236258c031f9e8d66f17d79a04e2b944560562ef')
        teacher=_bound_module('_z_preflight_teacher',TEACHER,'ce304e2d861e303f1780c1267cf05146e7d5d4a13f8b30f54b28368608ea16ad')
        null_bias=_bound_module('_z_preflight_null_bias',NULL_BIAS,'7ef41a420de732a11f6938303f18bfc69cc7655c85b2ad870a1105a1585f83fb')
        context['phase']='verify_and_load_data'
        data=BoundData(ARM,NATIVE,device='cuda')
        torch.manual_seed(20261031); torch.cuda.manual_seed_all(20261031)
        np.random.seed(20261031); random.seed(20261031)
        common=api.AnchoredLDM(blocks=4,checkpoint_blocks=True,residual_scale=3.,
                               inv_temperature=1.,code_mode='posterior').cuda().eval()
        common.initialize_output_units(data.states[0]['features'])
        api.attach_spatial_action(common,True)
        adapter_receipt=attach_adapters(common,mode='lfq16')
        context['phase']='common_scale_calibration'
        def batches():
            for start in range(0,168,4):
                if time.monotonic()-start_time>=3500:
                    raise TimeoutError('Preflight time budget')
                if start % 28 == 0:
                    print('CALIBRATION_WINDOWS',start,'/168',flush=True)
                yield [data.window(s) for s in range(start,start+4)]
        start_time=start
        calibration=common.calibrate_actions(batches())
        require(calibration['entries']==168*9*16 and calibration['s_ref']>1e-6, 'Incomplete common action scale')
        common_sha=model_digest(common)
        save_path=out/'common_init_preflight.pt'
        with save_path.open('xb') as stream:
            torch.save(dict(schema='preflight_common_init_v1', model=common.state_dict(),
                            calibration=calibration, checkpoint_sha256=CHECKPOINT_SHA,
                            native_replay_sha256=data.native_evidence['native_replay_sha256'],
                            state_manifest_sha256=data.state_evidence['state_manifest_sha256'],
                            teacher_assets_sha256=data.teacher_evidence['teacher_assets_sha256'],
                            common_model_sha256=common_sha, formal_training=False),stream)
        pair=clone_pair(common)
        require(model_digest(pair['continuous16'])==model_digest(pair['lfq16'])==common_sha,
                'Common initial parameters differ')
        del common
        plan=candidate_plan()
        starts=plan['batches'][:2]
        rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state().clone()
        rng_python=random.getstate();rng_numpy=np.random.get_state()
        results={}
        initial_codec_digest=None
        for mode in ('continuous16','lfq16'):
            context.update(phase='discarded_updates',arm=mode,step=0)
            model=pair[mode].train()
            bridge=BoundNativeBridge(partial=True,device='cuda').train(True)
            runtime_digest=bridge.precision.report['mode_state_digest']
            if initial_codec_digest is None: initial_codec_digest=runtime_digest
            require(runtime_digest==initial_codec_digest,'Arms do not start from identical baseline codec')
            optimizer=torch.optim.AdamW([
                {'params':[p for p in model.parameters() if p.requires_grad],'lr':3e-4},
                {'params':[p for p in bridge.parameters() if p.requires_grad],'lr':1e-5}],
                betas=(.9,.95),weight_decay=0.)
            require(not optimizer.state,'Optimizer must be fresh')
            bias_guard=null_bias.freeze_null_key_bias(model,optimizer)
            torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
            random.setstate(rng_python);np.random.set_state(rng_numpy)
            updates=[]
            for index,batch in enumerate(starts,1):
                context['step']=index
                require(time.monotonic()-start<3500,'Preflight time budget exceeded')
                stats,codes,ids=step_native(model,bridge,optimizer,data,batch,make_condition=api.make_condition,
                    temporal=temporal,teacher_loss=teacher.teacher_loss,aux_multiplier=.1,grad_clip=1.)
                require(stats['batch_size']==4 and stats['action_mode']==mode and list(codes.shape)==[4,9,16]
                        and ((ids is None) if mode=='continuous16' else list(ids.shape)==[4,9]),'Action result mismatch')
                null_bias.assert_null_key_bias_unchanged(model,optimizer,bias_guard)
                if index==2:
                    norms=stats['reconstruction_latent_grad_per_sample_h']
                    q=stats['reconstruction_query_grad_per_h']
                    require(len(norms)==4 and all(len(row)==9 and all(v>0 for v in row) for row in norms)
                            and q is not None and len(q)==9 and all(v>0 for v in q)
                            and stats['reconstruction_project_in_grad']>0
                            and stats['reconstruction_project_out_grad']>0, 'Missing second-step reconstruction gradient')
                    groups=stats['gradient_norms']
                    require(all(groups[key]>0 for key in ('residual_head','motion_encoder','motion_decoder',
                            'native_out','native_heads','native_multiplier')), 'Missing second-step model/native gradient')
                updates.append(stats)
                print('PREFLIGHT',mode,'DISCARDED',index,'/2',flush=True)
            bridge.assert_frozen()
            results[mode]=dict(discarded_updates=2,formal_updates=0,initial_model_sha256=common_sha,
                               initial_codec_digest=runtime_digest,updates=updates,
                               null_key_bias=null_bias.assert_null_key_bias_unchanged(model,optimizer,bias_guard),
                               frozen_codec_parameters_buffers_unchanged=True)
            del optimizer,bridge,model,codes,ids
            pair.pop(mode)
            gc.collect();torch.cuda.empty_cache()
        require(torch.cuda.max_memory_reserved()<=24*2**30,'Preflight allocator cap exceeded')
        write(out/'RESULT.json',dict(status='passed',formal_training=False,formal_updates_per_arm=0,
            discarded_updates_per_arm=2, batch_size=4,frames_per_window=10,source_frames=177,frame_gap=1,
            checkpoint_sha256=CHECKPOINT_SHA,native_replay_sha256=data.native_evidence['native_replay_sha256'],
            state_manifest_sha256=data.state_evidence['state_manifest_sha256'],
            teacher_assets_sha256=data.teacher_evidence['teacher_assets_sha256'],
            initial_model_sha256=common_sha,common_preflight_checkpoint_sha256=sha(save_path),
            initial_codec_digest=initial_codec_digest,calibration=calibration,adapter=adapter_receipt,
            training_plan_candidate_sha256=plan['plan_sha256'],preflight_batches=starts,
            optimizer=dict(motion_lr=3e-4,native_lr=1e-5,betas=[.9,.95],weight_decay=0.,grad_clip=1.),
            arms=results,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,elapsed_seconds=time.monotonic()-start))
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='failed',context=context,error=repr(exc),traceback=traceback.format_exc(),
                                    formal_training=False,elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__': main()
