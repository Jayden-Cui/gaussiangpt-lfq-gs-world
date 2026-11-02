import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from object_masks import MaskBank
from trial_utils import ROOT,PARENT,B_PREP,CHECKPOINT,read,write,sha,verify,require,module,tree_digest,schedule


def run(arm,phase,out):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    protocol=verify(need_masks=True,need_tests=True);items,sampler=schedule(protocol)
    seed=protocol['restart_rng_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    out=Path(out).resolve();require(out.is_relative_to(ROOT) and not out.exists(),'Use new isolated output')
    out.mkdir(parents=True);began=time.monotonic()
    write(out/'STARTED.json',dict(pid=os.getpid(),arm=arm,phase=phase,code_sha256=sha(ROOT/'CODE.sha256')))
    try:
        sys.path.insert(0,str(B_PREP))
        from b_core import FreeCodes,load_bound_sources,optimizer_for,train_step as original_step
        from gpu_runtime import setup_gpu_once
        from loss_step import train_step
        bundle=load_bound_sources('cpu');p=bundle['p'];runtime=bundle['runtime'];fm=bundle['fm'];store=bundle['store']
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        require(payload['source']==read(PARENT/'run/training/TRAIN_DONE.json')['source']
            and payload['sweep']==5 and payload['starts']==list(range(168)),'Parent checkpoint schema changed')
        decoder=bundle['decoder'];decoder.load_state_dict(payload['decoder'],strict=True)
        require(runtime.digest(decoder)==payload['decoder_digest'] and fm.frozen_digest(decoder)==bundle['frozen_digest'],
            'Parent decoder or frozen initial-state encoder differs')
        setup_gpu_once(p);torch.cuda.manual_seed_all(seed);decoder=decoder.cuda().train()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        bank=FreeCodes(list(range(168)),payload['codes'],'cuda');std=store.frames['std'].cuda()
        optimizer=optimizer_for(decoder,bank,p,protocol['decoder_lr'],protocol['code_lr'])
        optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
        initial_optimizer_digest=tree_digest(payload['optimizer'])
        require(tree_digest(optimizer.state_dict())==initial_optimizer_digest,'Adam state restoration differs')
        initial_code_digest=tree_digest(payload['codes']);require(tree_digest(bank.stack())==initial_code_digest,'Code restoration differs')
        mask_bank=MaskBank(ROOT/'masks/masks.npz');weight=protocol['arms'][arm]['object_weight']
        source=dict(schema='object_supervision_matched_continuation_v1',arm=arm,object_weight=weight,
            parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],parent_decoder_digest=payload['decoder_digest'],
            parent_optimizer_digest=initial_optimizer_digest,parent_code_digest=initial_code_digest,
            protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),sampler=sampler,
            mask_receipt_sha256=sha(ROOT/'masks/MASKS.json'),mask_review_sha256=sha(ROOT/'MASK_REVIEW.json'),
            codec_sha256=p['codec_checkpoint_sha256'],cache_manifest_sha256=payload['source']['cache_manifest_sha256'],
            initial_context_digest=bundle['frozen_digest'],restart_rng_seed=seed,quantized=False,
            training_parameter_signature=tree_digest([(n,list(v.shape),str(v.dtype)) for n,v in decoder.named_parameters() if v.requires_grad]))

        def frozen():
            bridge.assert_frozen()
            require(fm.frozen_digest(decoder)==bundle['frozen_digest'],'Frozen initial context changed')

        def restore():
            decoder.load_state_dict(payload['decoder'],strict=True)
            with torch.no_grad():
                for i,value in enumerate(bank.values):value.copy_(payload['codes'][i].cuda())
            optimizer.load_state_dict(copy.deepcopy(payload['optimizer']));optimizer.zero_grad(set_to_none=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            require(runtime.digest(decoder)==payload['decoder_digest'] and tree_digest(bank.stack())==initial_code_digest
                and tree_digest(optimizer.state_dict())==initial_optimizer_digest,'Preflight restore differs')

        sys.path.insert(0,str(PARENT))
        parent_worker=module('_object_trial_preview',ROOT/'preview.py')
        old_mean=read(PARENT/'run/SUMMARY.json')['original_two_window_anchor_motion_mae']
        if phase=='preflight':
            preview=parent_worker.preview(5,decoder,bank,bridge,store,p,runtime,runtime.digest(decoder),old_mean,out)
            expected=read(PARENT/'run/SUMMARY.json')['pilot_anchor_motion_mae']
            require(abs(preview['anchor_motion_mae']-expected)<1e-5,'Parent anchor renders not reproduced')
            batch=[dict(start=s,horizon=h) for s,h in ((0,9),(35,3),(123,3),(123,6))]
            control_parity=None
            if arm=='control':
                original=original_step(decoder,bank,bridge,store,std,optimizer,batch,p,probe_rgb=True)
                original_model={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
                original_codes=bank.stack();restore()
            probe=train_step(decoder,bank,bridge,store,std,optimizer,batch,p,masks=mask_bank,object_weight=weight,probe_rgb=True)
            if arm=='control':
                model_error=max(float((v.detach().cpu()-original_model[k]).abs().max()) for k,v in decoder.state_dict().items() if v.is_floating_point())
                code_error=float((bank.stack()-original_codes).abs().max())
                loss_error=max(abs(a['loss']-b['loss']) for a,b in zip(original['windows'],probe['windows']))
                require(max(model_error,code_error,loss_error)<=2e-5,'Zero-weight control diverged from original training step')
                control_parity=dict(model_max_abs=model_error,code_max_abs=code_error,loss_max_abs=loss_error,tolerance=2e-5)
            frozen();restore();frozen()
            write(out/'PREFLIGHT.json',dict(status='passed',arm=arm,source=source,probe=probe,control_parity=control_parity,
                parent_anchor_motion_mae=preview['anchor_motion_mae'],restored_model_codes_adam=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-began))
            print(json.dumps(dict(status='preflight_passed',arm=arm,control_parity=control_parity,source=source)),flush=True)
            return

        preflight=read(ROOT/'preflight'/arm/'PREFLIGHT.json')
        require(preflight['status']=='passed' and preflight['source']==source,'Preflight does not bind this training run')
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for item in items:
                require(time.monotonic()-began<protocol['max_training_seconds_per_arm'],'Fixed training budget exceeded')
                result=train_step(decoder,bank,bridge,store,std,optimizer,item['batch'],p,masks=mask_bank,object_weight=weight)
                log.write(json.dumps(dict(**item,arm=arm,elapsed_seconds=time.monotonic()-began,**result),allow_nan=False)+'\n')
                if item['local_step']==1 or item['local_step']%protocol['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',arm=arm,additional_updates=item['local_step'],
                        planned_updates=846,cumulative_step=item['step'],sweep=item['sweep'],elapsed_seconds=time.monotonic()-began),replace=True)
                if item['step']%423==0:
                    frozen();sweep=item['sweep'];digest=runtime.digest(decoder);path=out/f'sweep_{sweep:02d}.pt'
                    with path.open('xb') as f:
                        torch.save(dict(schema='stage_b_shared_decoder_free_codes_v1',sweep=sweep,starts=list(range(168)),
                            decoder=decoder.state_dict(),decoder_digest=digest,codes=bank.stack(),optimizer=optimizer.state_dict(),
                            source=source,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),f)
                    disk=torch.load(path,map_location='cpu',weights_only=True)
                    require(torch.equal(disk['codes'],bank.stack()) and tree_digest(disk['optimizer'])==tree_digest(optimizer.state_dict()),'Saved training state differs')
                    write(out/f'sweep_{sweep:02d}.json',dict(status='complete',sweep=sweep,sha256=sha(path),decoder_digest=digest))
                    parent_worker.preview(sweep,decoder,bank,bridge,store,p,runtime,digest,old_mean,out)
                    del disk
        frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',arm=arm,source=source,additional_updates=846,
            cumulative_updates=2961,sweeps=7,code_shape=[168,9,16],final_checkpoint_sha256=sha(out/'sweep_07.pt'),
            codec_and_initial_encoder_frozen=True,elapsed_seconds=time.monotonic()-began))
        write(out/'STATUS.json',dict(status='complete',arm=arm,additional_updates=846,cumulative_updates=2961,evaluation='pending'),replace=True)
    except BaseException as error:
        write(out/'FAILED.json',dict(status='failed',phase=phase,arm=arm,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','train']);parser.add_argument('--arm',choices=['control','object'],required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();run(args.arm,args.phase,args.out)
