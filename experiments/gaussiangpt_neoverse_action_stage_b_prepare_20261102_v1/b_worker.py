"""Conditional shared-decoder/free-code worker; not launched during stage A."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback
import torch

from b_core import FreeCodes,initialize_codes,load_bound_sources,optimizer_for,train_step
from b_sampling import schedule,audit

ROOT=Path(__file__).resolve().parent


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def validate_config(config):
    if (config.get('schema')!='stage_b_shared_decoder_free_codes_v1' or config.get('batch_size')!=4
            or config.get('starts')!=list(range(168)) or config.get('quantized') is not False
            or config.get('anchors')!=[35,123]):
        raise ValueError('Stage-B scope, batch, quantization or anchors differ')
    if (type(config.get('sweeps'))!=int or config['sweeps']<=0
            or any(not math.isfinite(config[k]) or config[k]<=0 for k in ('decoder_lr','code_lr','max_seconds'))
            or type(config.get('progress_every'))!=int or config['progress_every']<=0):
        raise ValueError('Invalid fixed training budget')
    checkpoints=config['checkpoint_sweeps']
    if sorted(set(checkpoints))!=checkpoints or not checkpoints or checkpoints[-1]!=config['sweeps']:
        raise ValueError('Sorted checkpoint sweeps must include the final sweep')
    if any(not 1<=v<=config['sweeps'] for v in checkpoints):raise ValueError('Invalid checkpoint sweep')
    items=schedule(config['sweeps'],config['sampler_seed'])
    return items,audit(items,config['sweeps'])


def validate_review(config,binding):
    entry=config['stage_a_review'];path=Path(entry['path'])
    if not path.is_absolute() or not path.is_file() or binding.sha(path)!=entry['sha256']:
        raise RuntimeError('Stage-A review missing or changed')
    review=binding.read(path)
    if (review.get('status')!='reviewed' or review.get('decision')!='stage_B'
            or not {35,123}.issubset(set(review.get('reviewed_starts',[])))
            or len(set(review.get('reviewed_starts',[])))<8):
        raise RuntimeError('Stage B requires completed stage-A dynamic review')
    result=Path(review['evaluation']['path'])
    if not result.is_absolute() or binding.sha(result)!=review['evaluation']['sha256']:
        raise RuntimeError('Reviewed stage-A evaluation changed')
    value=binding.read(result)
    if (value.get('status')!='complete' or value.get('window_predictions')!=1512
            or value.get('correction',{}).get('status')!='applied'
            or value.get('source',{}).get('sweep')!=20):
        raise RuntimeError('Stage-A decision is not bound to the corrected final evaluation')
    return dict(review_sha256=entry['sha256'],evaluation_sha256=review['evaluation']['sha256'])


def fit(config_path,out):
    config_path=Path(config_path);config=json.loads(config_path.read_text(encoding='utf-8'))
    items,sampler=validate_config(config)
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    # Loading the bound two-window models is read-only; no GPU training starts
    # until the final stage-A review and exact source hashes are verified.
    bundle=load_bound_sources('cpu');binding=bundle['binding']
    witness=binding.read(ROOT/'CPU_TESTS.json')
    if witness['status']!='passed' or witness['code_sha256']!=binding.sha(ROOT/'CODE.sha256'):
        raise RuntimeError('Stage-B preparation witness changed')
    worker_test=binding.read(ROOT/'WORKER_TEST.json')
    if (worker_test['status']!='passed' or worker_test['worker_sha256']!=binding.sha(ROOT/'b_worker.py')
            or worker_test['worker_test_sha256']!=binding.sha(ROOT/'worker_tests.py')
            or worker_test['core_manifest_sha256']!=binding.sha(ROOT/'CODE.sha256')):
        raise RuntimeError('Stage-B worker launch tests changed')
    review=validate_review(config,binding)
    out=Path(out);out.mkdir();began=time.monotonic();phase='initializing'
    try:
        p=bundle['p'];binding.setup_gpu(p)
        decoder=bundle['decoder'].cuda();posterior=bundle['posterior'].cuda();store=bundle['store']
        fm,runtime=bundle['fm'],bundle['runtime']
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        std=store.frames['std'].cuda();decoder.train()
        phase='initializing_all_window_codes'
        initial=initialize_codes(posterior,decoder,store,config['starts'],'cuda')
        if tuple(initial.shape)!=(168,9,16):raise RuntimeError('Full code initialization incomplete')
        bank=FreeCodes(config['starts'],initial,'cuda')
        initial_decoder={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
        initial_digest=runtime.digest(decoder);frozen=bundle['frozen_digest']
        def assert_frozen():
            bridge.assert_frozen()
            if fm.frozen_digest(decoder)!=frozen:raise RuntimeError('Frozen initial encoder/context changed')
            if any(v.grad is not None for name,v in decoder.named_parameters() if not v.requires_grad):
                raise RuntimeError('Frozen decoder component acquired gradient')
        def optimizer():return optimizer_for(decoder,bank,p,config['decoder_lr'],config['code_lr'])
        opt=optimizer();phase='gpu_preflight'
        probe=train_step(decoder,bank,bridge,store,std,opt,
            [dict(start=s,horizon=9) for s in (0,35,123,167)],p,probe_rgb=True)
        assert_frozen();decoder.load_state_dict(initial_decoder,strict=True)
        if runtime.digest(decoder)!=initial_digest:raise RuntimeError('Preflight decoder reset failed')
        del bank,opt,initial_decoder
        bank=FreeCodes(config['starts'],initial,'cuda');opt=optimizer()
        if opt.state or not torch.equal(bank.stack(),initial):raise RuntimeError('Preflight code/optimizer reset failed')
        source=dict(schema='stage_b_shared_decoder_source_v1',stage_a_review=review,
          config_sha256=binding.sha(config_path),training_core_sha256=binding.sha(ROOT/'b_core.py'),
          worker_sha256=binding.sha(ROOT/'b_worker.py'),sampler=sampler,
          warm_decoder_checkpoint_sha256=p['teacher_checkpoint_sha256'],
          warm_posterior_checkpoint_sha256=p['posterior_checkpoint_sha256'],
          warm_decoder_digest=initial_digest,frozen_initial_context_digest=frozen,
          cache_manifest_sha256=binding.sha(bundle['paths']['cache_manifest']),
          codec_sha256=p['codec_checkpoint_sha256'],anchors=[35,123],quantized=False)
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=1,logical_batch_size=4,
              restored_decoder_codes_optimizer=True,codec_and_initial_encoder_frozen=True,
              rgb_to_free_code_and_decoder_verified=True,probe=probe,source=source))
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for item in items:
                phase='training'
                if time.monotonic()-began>config['max_seconds']:raise RuntimeError('Training time budget exceeded')
                row=train_step(decoder,bank,bridge,store,std,opt,item['batch'],p)
                row.update(step=item['step'],sweep=item['sweep'],elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if item['step']==1 or item['step']%config['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase=phase,completed_updates=item['step'],
                        total_updates=len(items),sweep=item['sweep'],elapsed_seconds=time.monotonic()-began),replace=True)
                if item['sweep'] in config['checkpoint_sweeps'] and item['step']%378==0:
                    assert_frozen();decoder_digest=runtime.digest(decoder)
                    path=out/f"sweep_{item['sweep']:02d}.pt"
                    with path.open('xb') as stream:
                        torch.save(dict(schema='stage_b_shared_decoder_free_codes_v1',sweep=item['sweep'],
                            starts=config['starts'],decoder=decoder.state_dict(),decoder_digest=decoder_digest,
                            codes=bank.stack(),optimizer=opt.state_dict(),source=source),stream)
                    replay=torch.load(path,map_location='cpu',weights_only=True)
                    if (replay['decoder_digest']!=decoder_digest or not torch.equal(replay['codes'],bank.stack())
                            or replay['starts']!=config['starts']):raise RuntimeError('Checkpoint replay differs')
                    write(out/f"sweep_{item['sweep']:02d}.json",dict(status='complete',sweep=item['sweep'],
                        sha256=binding.sha(path),decoder_digest=decoder_digest,shape=[168,9,16]))
        assert_frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',updates=len(items),sweeps=config['sweeps'],
             source=source,final_checkpoint_sha256=binding.sha(out/f"sweep_{config['sweeps']:02d}.pt"),
             decoder_digest=runtime.digest(decoder),code_shape=[168,9,16],
             codec_and_initial_encoder_frozen=True,evaluation='pending',elapsed_seconds=time.monotonic()-began))
        write(out/'STATUS.json',dict(status='complete',phase='training',completed_updates=len(items),
              evaluation='pending'),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();fit(args.config,args.out)
