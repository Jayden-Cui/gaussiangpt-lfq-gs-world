"""CPU audit of schedule-prefix and optimizer-state restoration, without training."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


ROOT=Path(__file__).resolve().parent
OUTPUTS=ROOT.parent
B_PREP=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PILOT=OUTPUTS/'gaussiangpt_neoverse_anchor_replay_pilot_20261102_v1'
CHECKPOINT_SHA='e1613f6688dd2dda1ac6b6e8870da99ac96ecc586fe23ccb930a13303b320a69'
PILOT_CODE_SHA='b6a85d22f8a27a6c79869a0c8bff381cabb139464cf4b277dda8e49c7c8f1c5c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def equal(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def verify_manifest(folder):
    witness=read(folder/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(folder/'CODE.sha256'):
        raise RuntimeError('Required source witness changed: '+str(folder))
    for line in (folder/'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(folder/name)!=digest:raise RuntimeError('Source changed: '+name)


def main(output):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    for folder in (ROOT,B_PREP,PILOT):verify_manifest(folder)
    if sha(PILOT/'CODE.sha256')!=PILOT_CODE_SHA:raise RuntimeError('Pilot source version changed')
    checkpoint=B_RUN/'training/sweep_20.pt';metadata=read(B_RUN/'training/sweep_20.json')
    if sha(checkpoint)!=CHECKPOINT_SHA or metadata.get('sha256')!=CHECKPOINT_SHA:
        raise RuntimeError('Historical B checkpoint changed')
    spec=importlib.util.spec_from_file_location('_pilot_sampler_audit',PILOT/'sampling.py')
    sampling=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampling)
    config=read(PILOT/'protocol.json')
    first=sampling.schedule(5,config['base_sampler_seed'],config['replay_sampler_seed'])
    complete=sampling.schedule(20,config['base_sampler_seed'],config['replay_sampler_seed'])
    first_report=sampling.audit(first,5);full_report=sampling.audit(complete,20)
    if first!=complete[:len(first)] or len(first)!=2115 or len(complete)-len(first)!=6345:
        raise RuntimeError('Pilot schedule is not the exact prefix of the 20-sweep schedule')
    sys.path.insert(0,str(B_PREP))
    from b_core import FreeCodes,load_bound_sources,optimizer_for
    began=time.monotonic();bundle=load_bound_sources('cpu');p=bundle['p'];runtime=bundle['runtime'];fm=bundle['fm']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    done=read(B_RUN/'training/TRAIN_DONE.json')
    if (payload.get('schema')!='stage_b_shared_decoder_free_codes_v1' or payload.get('sweep')!=20
            or payload.get('starts')!=list(range(168)) or tuple(payload['codes'].shape)!=(168,9,16)
            or payload.get('source')!=done.get('source') or payload.get('decoder_digest')!=metadata['decoder_digest']):
        raise RuntimeError('Historical checkpoint ordering or provenance differs')
    decoder=bundle['decoder'];decoder.load_state_dict(payload['decoder'],strict=True)
    if runtime.digest(decoder)!=payload['decoder_digest'] or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
        raise RuntimeError('Historical decoder or frozen initial context differs')
    decoder2=copy.deepcopy(decoder)
    bank1=FreeCodes(list(range(168)),payload['codes'],'cpu')
    bank2=FreeCodes(list(range(168)),payload['codes'],'cpu')
    opt1=optimizer_for(decoder,bank1,p,.0001,.003);opt2=optimizer_for(decoder2,bank2,p,.0001,.003)
    opt1.load_state_dict(copy.deepcopy(payload['optimizer']))
    opt2.load_state_dict(copy.deepcopy(payload['optimizer']))
    if not equal(opt1.state_dict(),payload['optimizer']) or not equal(opt2.state_dict(),payload['optimizer']):
        raise RuntimeError('Adam moments, steps or parameter-group settings did not restore exactly')
    trainable1=[(n,v) for n,v in decoder.named_parameters() if v.requires_grad]
    trainable2=[(n,v) for n,v in decoder2.named_parameters() if v.requires_grad]
    signature=[(n,list(v.shape),str(v.dtype)) for n,v in trainable1]
    if signature!=[(n,list(v.shape),str(v.dtype)) for n,v in trainable2]:
        raise RuntimeError('Trainable decoder parameter order changed')
    before=bank1.code(0).detach().clone();selected_before=bank1.code(35).detach().clone()
    for optimizer,group,bank in ((opt1,trainable1,bank1),(opt2,trainable2,bank2)):
        optimizer.zero_grad(set_to_none=True)
        group[0][1].grad=torch.full_like(group[0][1],1e-4)
        bank.code(35).grad=torch.full_like(bank.code(35),2e-4)
        bank.code(123).grad=torch.full_like(bank.code(123),-3e-4)
        optimizer.step()
    if (not torch.equal(bank1.code(0),before) or torch.equal(bank1.code(35),selected_before)
            or not torch.equal(bank1.stack(),bank2.stack()) or runtime.digest(decoder)!=runtime.digest(decoder2)
            or not equal(opt1.state_dict(),opt2.state_dict())):
        raise RuntimeError('Synthetic resumed Adam step or untouched-window behavior differs')
    output=Path(output)
    if not output.is_absolute() or output.exists() or output.parent.resolve()!=ROOT.resolve():
        raise RuntimeError('Expected a new audit output directory')
    output.mkdir()
    result=dict(status='passed',scope='CPU audit using completed original B20 checkpoint; not a pilot resume or quality result',
        checkpoint_sha256=CHECKPOINT_SHA,pilot_code_sha256=PILOT_CODE_SHA,first_five_schedule=first_report,
        full_twenty_schedule=full_report,first_five_are_exact_schedule_prefix=True,remaining_updates=6345,
        restored_adam_state_exact=True,synthetic_step_reproduced=True,unselected_code_not_updated=True,
        trainable_decoder_tensors=len(signature),optimizer_parameter_groups=len(opt1.param_groups),
        decoder_parameter_signature_sha256=hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest(),
        historical_checkpoint_only=True,pilot_checkpoint_requires_separate_validation=True,
        rng_state_saved_in_historical_checkpoint='not saved; do not claim bitwise-identical GPU continuation',
        sparse_backend_determinism='MinkowskiEngine configured nondeterministic; GPU bitwise identity is not guaranteed',
        gpu_used=False,training_performed=False,elapsed_seconds=time.monotonic()-began,
        code_sha256=sha(ROOT/'CODE.sha256'))
    with (output/'AUDIT.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.output)
