"""A fixed-teacher posterior training worker; no teacher is selected here."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import torch

from c_schedule import schedule,audit
from teacher_receipt_v2 import read,sha,validate


VARIANTS=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_posterior_variants_20261102_v2')


def load_training_core():
    witness=read(VARIANTS/'TRAINING_CORE_WITNESS.json')
    if (witness['status']!='passed' or sha(VARIANTS/'training_core.py')!=witness['training_core_sha256']
            or sha(VARIANTS/'CODE.sha256')!=witness['source_model_code_sha256']):
        raise RuntimeError('Previously witnessed posterior training core changed')
    spec=importlib.util.spec_from_file_location('_bound_continuous_training_core',VARIANTS/'training_core.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)


def fit(model,decoder,bridge,store,p,runtime,teacher_receipt,config,out):
    """Train C0/C1/C2 against a reviewed shared teacher; caller constructs models.

    Formal execution requires a separately frozen config and a real GPU
    preflight. This helper never chooses a teacher or starts a process itself.
    """
    out=Path(out);out.mkdir();began=time.monotonic();phase='initializing'
    try:
        if model.variant not in ('C0','C1','C2') or config['batch_size']!=4:
            raise RuntimeError('Unexpected posterior variant or batch size')
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Formal preflight requires exactly one exposed GPU')
        if any(x.requires_grad for x in decoder.parameters()):raise RuntimeError('Teacher decoder is not frozen')
        bridge.eval();decoder.eval()
        teacher=validate(teacher_receipt,decoder,runtime,config['codec_sha256'],config['cache_manifest_sha256'])
        core=load_training_core()
        items=schedule(config['distill_sweeps'],config['joint_sweeps'],config['sampler_seed'])
        sampler=audit(items,config['distill_sweeps'],config['joint_sweeps'])
        checkpoints=config['checkpoint_steps']
        if (not checkpoints or sorted(set(checkpoints))!=checkpoints or checkpoints[-1]!=len(items)
                or any(not 1<=x<=len(items) for x in checkpoints)):
            raise RuntimeError('Checkpoints must be sorted, unique, and include the fixed final step')
        device=next(model.parameters()).device
        if device.type!='cuda':raise RuntimeError('Posterior must be on the exposed GPU')
        std=store.frames['std'].to(device);scale=core.teacher_scale(teacher['codes'])
        decoder_digest=runtime.digest(decoder)
        def assert_frozen():
            bridge.assert_frozen()
            if runtime.digest(decoder)!=decoder_digest or any(x.grad is not None for x in decoder.parameters()):
                raise RuntimeError('Frozen teacher decoder changed or acquired gradients')
        def optimizer():return torch.optim.Adam(model.parameters(),lr=config['learning_rate'],
                                                betas=tuple(config['betas']),eps=config['eps'])
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        initial_digest=runtime.digest(model);model.train();opt=optimizer()
        batch=[dict(start=s,horizon=9) for s in (0,35,123,167)]
        phase='preflight'
        probes=[core.step(model,decoder,bridge,store,std,opt,batch,teacher['codes'],scale,p,'distill'),
                core.step(model,decoder,bridge,store,std,opt,batch,teacher['codes'],scale,p,'joint',probe_rgb=True)]
        assert_frozen();model.load_state_dict(initial,strict=True)
        if runtime.digest(model)!=initial_digest:raise RuntimeError('GPU preflight model reset failed')
        del opt,initial
        opt=optimizer()
        if opt.state:raise RuntimeError('GPU preflight optimizer state was not reset')
        source=dict(teacher_receipt_sha256=teacher['receipt_sha256'],decoder_digest=decoder_digest,
                    model_initial_digest=initial_digest,sampler=sampler,variant=model.variant,
                    original_codec_sha256=config['codec_sha256'],
                    original_cache_sha256=config['cache_manifest_sha256'],quantized=False)
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=2,restored_model_and_optimizer=True,
              rgb_to_posterior_gradients_verified=True,decoder_and_codec_frozen=True,rows=probes,source=source))
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for item in items:
                phase=item['phase']
                if time.monotonic()-began>config['max_seconds']:raise RuntimeError('Fixed training time budget exceeded')
                result=core.step(model,decoder,bridge,store,std,opt,item['batch'],teacher['codes'],scale,p,phase)
                row=dict(step=item['step'],phase=phase,sweep=item['sweep'],elapsed_seconds=time.monotonic()-began,**result)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if item['step']==1 or item['step']%config['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',phase=phase,completed_updates=item['step'],
                          total_updates=len(items),sweep=item['sweep'],elapsed_seconds=time.monotonic()-began),replace=True)
                if item['step'] in checkpoints:
                    assert_frozen();path=out/f"step_{item['step']:05d}.pt"
                    with path.open('xb') as stream:
                        torch.save(dict(schema='fixed_teacher_posterior_v1',step=item['step'],variant=model.variant,
                            model=model.state_dict(),optimizer=opt.state_dict(),source=source),stream)
                    disk=torch.load(path,map_location='cpu',weights_only=True)
                    if disk['step']!=item['step'] or disk['source']!=source:
                        raise RuntimeError('Posterior checkpoint provenance mismatch')
        assert_frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',updates=len(items),source=source,
              model_digest=runtime.digest(model),final_checkpoint_sha256=sha(out/f"step_{len(items):05d}.pt"),
              decoder_and_codec_frozen=True,elapsed_seconds=time.monotonic()-began,
              evaluation='not yet run; dynamic acceptance remains required'))
        write(out/'STATUS.json',dict(status='complete',phase='training',completed_updates=len(items),
              evaluation='pending'),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),
                     no_automatic_retry=True)
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise
