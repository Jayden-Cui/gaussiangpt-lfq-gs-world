"""Train the full-168 replay pilot from the original two-window warm model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from sampling import schedule, audit


ROOT=Path(__file__).resolve().parent
OUTPUTS=ROOT.parent
B_PREP=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
DIAG=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_diagnosis_20261102_v1'
REFINE=OUTPUTS/'gaussiangpt_neoverse_stage_b_code_refine_20261102_v1'
PARENT_EVAL_SHA='74692f5757622aa150e6617304fbb8786060ec0ceb7f0dd7b105d898cc4d53d7'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(path)
    else:
        with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def validate_protocol(p):
    if (p.get('schema')!='anchor_replay_full168_pilot_v1' or p.get('sweeps')!=5 or p.get('batch_size')!=4
            or p.get('base_sampler_seed')!=20261102 or p.get('replay_sampler_seed')!=20261103
            or p.get('decoder_lr')!=.0001 or p.get('code_lr')!=.003 or p.get('max_training_seconds')!=9000
            or p.get('evaluation_timeout_seconds')!=10890 or p.get('replay_timeout_seconds')!=1230
            or p.get('progress_every')!=25 or p.get('checkpoint_sweeps')!=[1,3,5]
            or p.get('anchors')!=[35,123] or p.get('quantized') is not False
            or p.get('native_codec_frozen') is not True or p.get('initial_state_context_frozen') is not True
            or p.get('predefined_full_evaluation_sweep')!=5 or p.get('manual_visual_review_required') is not True):
        raise RuntimeError('Anchor replay pilot protocol differs')
    steps=schedule(p['sweeps'],p['base_sampler_seed'],p['replay_sampler_seed'])
    return steps,audit(steps,p['sweeps'])


def verify_manifest(folder,witness_name='CPU_TESTS.json'):
    witness=read(folder/witness_name)
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(folder/'CODE.sha256'):
        raise RuntimeError('Required source witness changed: '+str(folder))
    for line in (folder/'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(folder/name)!=digest:raise RuntimeError('Source changed: '+name)


def verify_sources():
    for folder in (ROOT,B_PREP,DIAG,REFINE):verify_manifest(folder)
    p=read(ROOT/'protocol.json');steps,sampler=validate_protocol(p)
    paths={'base_config_sha256':B_RUN/'CONFIG.json',
           'baseline_sweep05_evaluation_sha256':B_RUN/'training/sweep_05/free_z/evaluation/EVALUATION.json',
           'stage_b_summary_sha256':B_RUN/'SUMMARY.json',
           'anchor_diagnosis_sha256':DIAG/'result/DIAGNOSIS.json',
           'fixed_decoder_refine_done_sha256':REFINE/'result/DONE.json',
           'fixed_decoder_refine_final_eval_sha256':REFINE/'result/step_00360/EVALUATION.json'}
    for key,path in paths.items():
        if sha(path)!=p[key]:raise RuntimeError('Decision evidence changed: '+key)
    b=read(B_RUN/'SUMMARY.json');diag=read(DIAG/'result/DIAGNOSIS.json');refine=read(REFINE/'result/DONE.json')
    final=read(REFINE/'result/step_00360/EVALUATION.json')
    if (b.get('status')!='complete' or b.get('stage')!='B' or b.get('action_codec_complete') is not False
            or diag.get('status')!='complete' or diag.get('oracle_is_post_inference_only') is not True
            or refine.get('status')!='complete' or refine.get('updates')!=360
            or refine.get('decoder_and_codec_frozen') is not True or refine.get('full_action_codec_complete') is not False
            or final.get('status')!='complete' or final.get('step')!=360 or len(final.get('rows',[]))!=18):
        raise RuntimeError('Completed B or fixed-decoder diagnostic evidence differs')
    return p,steps,sampler


def image_array(x):
    a=x.detach().cpu().numpy()
    if a.shape!=(252,448,3) or not np.isfinite(a).all():raise RuntimeError('Unexpected preview image')
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)


@torch.no_grad()
def preview(sweep,decoder,bank,bridge,store,p,runtime,digest,original_mean,out):
    from data import evaluate_metrics
    from runtime import render
    from support_decoder import select_state,support_metrics
    from b_evaluate_v2 import anchor_metrics
    folder=out/('preview_%02d'%sweep);folder.mkdir();rows=[]
    decoder.eval()
    for start in (35,123):
        w=store.get(start);prediction=decoder(w['condition'],bank.code(start))
        for h in range(1,10):
            sample=store.sample(start,h)
            state=select_state(w['condition'].coordinates,prediction['features'][h-1],prediction['logits'][h-1],
                               p['max_predicted_latent_nodes'])
            images,_,_=render(bridge,state,sample['record'],views=(0,1))
            metrics=evaluate_metrics(images,w,sample,p)
            rows.append(dict(start=start,horizon=h,frame=start+h,metrics={'correct':metrics},
                             support={'correct':support_metrics(prediction['logits'][h-1],w['targets']['occupancy'][h-1])}))
            if h in (1,3,6,9):
                image=np.concatenate((image_array(sample['record']['real']),image_array(images[0])),axis=1)
                Image.fromarray(image).save(folder/('s%03d_h%02d.png'%(start,h)))
    current=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in rows)/18
    if not math.isfinite(current) or runtime.digest(decoder)!=digest:raise RuntimeError('Preview changed decoder')
    bridge.assert_frozen()
    if sweep==0 and abs(current-original_mean)>1e-4:raise RuntimeError('Warm initial anchor regression failed')
    result=dict(status='complete',sweep=sweep,windows=[35,123],future_predictions=18,
        anchor_motion_mae=current,original_anchor_motion_mae=original_mean,
        dynamic_anchors=anchor_metrics(rows,store,p),rows=rows,decoder_digest=digest,
        codec_frozen=True,independent_native_replay='not performed for preview',manual_visual_review_required=True,
        scope='Two training windows only; full 168-window evaluation follows the five-sweep pilot')
    write(folder/'PREVIEW.json',result)
    decoder.train()
    return result


def fit(out):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    protocol,items,sampler=verify_sources()
    out=Path(out)
    if not out.is_absolute() or out.exists() or out.parent.parent.resolve()!=ROOT.resolve():
        raise RuntimeError('Expected a new training directory under the isolated run')
    sys.path.insert(0,str(B_PREP))
    from b_core import FreeCodes,load_bound_sources,optimizer_for,train_step
    from b_worker_v3 import load_initial_codes,validate_review
    from gpu_runtime import setup_gpu_once
    bundle=load_bound_sources('cpu');binding=bundle['binding'];p=bundle['p'];fm=bundle['fm'];runtime=bundle['runtime']
    old_config=read(B_RUN/'CONFIG.json')
    review=validate_review(old_config,binding)
    initial,initial_source=load_initial_codes(old_config,bundle)
    original_preflight=read(B_RUN/'training/PREFLIGHT.json')
    if (original_preflight.get('status')!='passed' or original_preflight['source']['initial_codes']!=initial_source
            or original_preflight['source']['warm_decoder_digest']!=runtime.digest(bundle['decoder'])):
        raise RuntimeError('Pilot is not initialized from the same B warm model and codes')
    parent=Path(p['parent_root'])/'free16/evaluation/EVALUATION.json'
    if sha(parent)!=PARENT_EVAL_SHA:raise RuntimeError('Original two-window evaluation changed')
    old=[r for r in read(parent)['rows'] if r['horizon']>0]
    if len(old)!=18 or {(r['start'],r['horizon']) for r in old}!={(s,h) for s in (35,123) for h in range(1,10)}:
        raise RuntimeError('Original warm reference is incomplete')
    original_mean=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in old)/18
    out.mkdir();began=time.monotonic();phase='initializing'
    write(out/'STARTED.json',dict(pid=os.getpid(),planned_updates=len(items),quantized=False,
                                  scope='full-168 five-sweep replay pilot; not a teacher acceptance'))
    try:
        setup_gpu_once(p)
        decoder=bundle['decoder'].cuda();store=bundle['store'];decoder.train()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        std=store.frames['std'].cuda();bank=FreeCodes(list(range(168)),initial,'cuda')
        initial_decoder={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
        digest=runtime.digest(decoder);frozen=bundle['frozen_digest']
        def assert_frozen():
            bridge.assert_frozen()
            if fm.frozen_digest(decoder)!=frozen:raise RuntimeError('Initial-state encoder/context changed')
        warm=preview(0,decoder,bank,bridge,store,p,runtime,digest,original_mean,out)
        def optimizer():return optimizer_for(decoder,bank,p,protocol['decoder_lr'],protocol['code_lr'])
        opt=optimizer();phase='gpu_preflight'
        probe=train_step(decoder,bank,bridge,store,std,opt,
            [dict(start=s,horizon=9) for s in (0,35,123,167)],p,probe_rgb=True)
        assert_frozen();decoder.load_state_dict(initial_decoder,strict=True)
        if runtime.digest(decoder)!=digest:raise RuntimeError('Preflight decoder reset failed')
        del bank,opt,initial_decoder
        bank=FreeCodes(list(range(168)),initial,'cuda');opt=optimizer()
        if opt.state or not torch.equal(bank.stack(),initial):raise RuntimeError('Preflight code/optimizer reset failed')
        source=dict(schema='stage_b_anchor_replay_pilot_source_v1',protocol_sha256=sha(ROOT/'protocol.json'),
            code_sha256=sha(ROOT/'CODE.sha256'),base_config_sha256=protocol['base_config_sha256'],
            stage_b_summary_sha256=protocol['stage_b_summary_sha256'],
            fixed_decoder_refine_done_sha256=protocol['fixed_decoder_refine_done_sha256'],
            anchor_diagnosis_sha256=protocol['anchor_diagnosis_sha256'],
            stage_a_review=review,initial_codes=initial_source,sampler=sampler,warm_decoder_digest=digest,
            frozen_initial_context_digest=frozen,codec_sha256=p['codec_checkpoint_sha256'],
            cache_manifest_sha256=binding.sha(bundle['paths']['cache_manifest']),quantized=False)
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=1,restored_decoder_codes_optimizer=True,
            rgb_to_free_code_and_decoder_verified=True,codec_and_initial_encoder_frozen=True,probe=probe,source=source))
        phase='training'
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for item in items:
                if time.monotonic()-began>protocol['max_training_seconds']:raise RuntimeError('Fixed training budget exceeded')
                result=train_step(decoder,bank,bridge,store,std,opt,item['batch'],p)
                log.write(json.dumps(dict(**item,elapsed_seconds=time.monotonic()-began,**result),allow_nan=False)+'\n')
                if item['step']==1 or item['step']%protocol['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase=phase,completed_updates=item['step'],
                          total_updates=len(items),sweep=item['sweep'],elapsed_seconds=time.monotonic()-began),replace=True)
                if item['sweep'] in protocol['checkpoint_sweeps'] and item['step']%423==0:
                    assert_frozen();current=runtime.digest(decoder);path=out/('sweep_%02d.pt'%item['sweep'])
                    with path.open('xb') as stream:
                        torch.save(dict(schema='stage_b_shared_decoder_free_codes_v1',sweep=item['sweep'],
                            starts=list(range(168)),decoder=decoder.state_dict(),decoder_digest=current,
                            codes=bank.stack(),optimizer=opt.state_dict(),source=source),stream)
                    restored=torch.load(path,map_location='cpu',weights_only=True)
                    if restored['decoder_digest']!=current or not torch.equal(restored['codes'],bank.stack()):
                        raise RuntimeError('Pilot checkpoint replay differs')
                    write(out/('sweep_%02d.json'%item['sweep']),dict(status='complete',sweep=item['sweep'],
                          sha256=sha(path),decoder_digest=current,shape=[168,9,16]))
                    preview(item['sweep'],decoder,bank,bridge,store,p,runtime,current,original_mean,out)
        assert_frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',updates=len(items),sweeps=5,source=source,
             final_checkpoint_sha256=sha(out/'sweep_05.pt'),decoder_digest=runtime.digest(decoder),
             code_shape=[168,9,16],codec_and_initial_encoder_frozen=True,evaluation='pending',
             elapsed_seconds=time.monotonic()-began))
        write(out/'STATUS.json',dict(status='complete',phase='training',completed_updates=len(items),
              evaluation='pending'),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();fit(args.out)
