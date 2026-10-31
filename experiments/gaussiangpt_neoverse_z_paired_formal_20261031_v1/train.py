"""Train one arm from the identical approved preflight initialization."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from codec_binding import require, sha
from formal_common import verify_sources, gpu_setup, guard, build, seed_common, atomic_checkpoint, write_new, model_digest
from paired_step import step_native
from runner_core import candidate_plan


def main(mode):
    root=Path(__file__).resolve().parent
    p,source=verify_sources(root)
    require(mode in p['arms'],'Unknown arm')
    out=root/'arms'/mode
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();step=0;completed=0;event='initializing'
    write_new(out/'STARTED.json',dict(pid=os.getpid(),mode=mode,source=source,status='initializing'))
    try:
        memory=gpu_setup(p)
        rt=build(p,source,mode,training=True)
        model,bridge=rt['model'],rt['bridge']
        optimizer=torch.optim.AdamW([
            {'params':[x for x in model.parameters() if x.requires_grad],'lr':p['motion_lr']},
            {'params':[x for x in bridge.parameters() if x.requires_grad],'lr':p['native_lr']}],
            betas=tuple(p['betas']),weight_decay=p['weight_decay'])
        require(not optimizer.state,'AdamW must start with empty state')
        bias_guard=rt['null_bias'].freeze_null_key_bias(model,optimizer)
        rng=seed_common(p['seed'])
        plan=candidate_plan()
        require(plan['plan_sha256']==p['plan_sha256'] and len(plan['batches'])==800,'Sampling plan mismatch')
        write_new(out/'TRAIN_SOURCE.json',dict(mode=mode,source=source,seed=p['seed'],rng=rng,resource=memory,
            initial_model_sha256=model_digest(model),initial_codec_digest=bridge.precision.report['mode_state_digest'],
            training_plan_sha256=plan['plan_sha256'],formal_updates=0))
        checkpoints={}
        with (out/'train_metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for step,batch in enumerate(plan['batches'],1):
                guard(p,started,'training')
                event='inside_update'
                stats,codes,ids=step_native(model,bridge,optimizer,rt['data'],batch,
                    make_condition=rt['api'].make_condition,temporal=rt['temporal'],
                    teacher_loss=rt['teacher'].teacher_loss,aux_multiplier=p['aux_multiplier'],grad_clip=p['grad_clip'])
                completed=step;event='post_update'
                require(stats['action_mode']==mode and stats['starts']==batch and tuple(codes.shape)==(4,9,16)
                        and ((ids is None) if mode=='continuous16' else tuple(ids.shape)==(4,9)),'Actual action mismatch')
                stats.update(step=step,elapsed_seconds=time.monotonic()-started,
                    gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                log.write(json.dumps(stats,allow_nan=False)+'\n')
                if step in p['checkpoints']:
                    bridge.assert_frozen()
                    bias=rt['null_bias'].assert_null_key_bias_unchanged(model,optimizer,bias_guard)
                    npstate=np.random.get_state()
                    saved=dict(schema='paired_action_checkpoint_v1',mode=mode,completed_steps=step,source=source,
                        plan_sha256=plan['plan_sha256'],next_plan_index=step,model=model.state_dict(),
                        codec=bridge.codec.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),
                        numpy_rng=dict(kind=npstate[0],keys=npstate[1].tolist(),pos=int(npstate[2]),
                                       has_gauss=int(npstate[3]),cached_gaussian=float(npstate[4])),
                        initial_rng=rng,null_key_bias=bias,decoder_precision='float64',
                        recovery_state_complete=True,exact_resume_verified=False)
                    path=out/f'step_{step:04d}.pt'
                    checkpoints[str(step)]=atomic_checkpoint(path,saved)
                    print('TRAIN',mode,step,'/800',flush=True)
                del codes,ids
        require(step==800 and set(checkpoints)=={'200','400','600','800'},'Incomplete training')
        bridge.assert_frozen()
        require(verify_sources(root)[1]==source,'Sources changed during training')
        write_new(out/'TRAIN_DONE.json',dict(status='complete',mode=mode,source=source,steps=step,
            actual_window_exposures=3200,frame_exposures=32000,future_exposures=28800,
            plan_sha256=plan['plan_sha256'],checkpoints=checkpoints,checkpoint_sha256=checkpoints['800'],
            metrics_sha256=sha(out/'train_metrics.jsonl'),initial_rng=rng,frozen_codec_unchanged=True,
            final_model_digest=model_digest(model),final_codec_digest=model_digest(bridge.codec),
            elapsed_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,goal_complete=False))
    except BaseException as error:
        write_new(out/'TRAIN_FAILED.json',dict(status='failed',mode=mode,last_confirmed_completed_step=completed,
            attempted_step=step,event=event,update_state_uncertain=event=='inside_update',
            error=repr(error),traceback=traceback.format_exc(),source=source,goal_complete=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['continuous16','lfq16'])
    main(parser.parse_args().mode)
