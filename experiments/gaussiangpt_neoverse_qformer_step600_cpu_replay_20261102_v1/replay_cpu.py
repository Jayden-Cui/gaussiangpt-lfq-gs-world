"""Independently reload the intermediate Q checkpoint on CPU and recompute codes."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
BROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
QROOT=OUTPUTS/'gaussiangpt_neoverse_two_future_qformer_20261102_v1'
PAIRED=OUTPUTS/'gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def verify_self():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('Independent Q replay test witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        expected,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=expected:raise ValueError('Q replay source changed: '+name)


def code_stats(q,teacher,scale):
    if (tuple(q.shape)!=(9,16) or tuple(teacher.shape)!=(9,16) or not bool(torch.isfinite(q).all())
            or not bool(torch.isfinite(teacher).all()) or not math.isfinite(float(scale)) or float(scale)<=0):
        raise ValueError('Expected finite [9,16] code sequences and positive scale')
    diff=q-teacher
    return {'mse':float(diff.square().mean()),'rms':float(diff.square().mean().sqrt()),
            'max_abs_error':float(diff.abs().max()),'normalized_mse':float(diff.square().mean()/scale)}


def run():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    verify_self();cfg=read(ROOT/'protocol.json')
    if (cfg['schema']!='two_future_qformer_step600_independent_cpu_replay_v1' or cfg['step']!=600
            or cfg['windows']!=[35,123] or cfg['horizons']!=list(range(1,10)) or cfg['gpu_used'] is not False):
        raise ValueError('Independent Q replay protocol changed')
    files=[(QROOT/'result/checkpoint_00600.pt','q_checkpoint_sha256'),
           (QROOT/'result/step_00600/EVALUATION.json','q_evaluation_sha256'),
           (QROOT/'protocol.json','q_protocol_sha256'),(QROOT/'CODE.sha256','q_code_sha256'),
           (PAIRED/'result/checkpoint_00360.pt','teacher_checkpoint_sha256')]
    for path,key in files:
        if sha(path)!=cfg[key]:raise RuntimeError('Bound Q source changed: '+str(path))
    evaluation=read(QROOT/'result/step_00600/EVALUATION.json')
    qcheckpoint=torch.load(QROOT/'result/checkpoint_00600.pt',map_location='cpu',weights_only=True)
    teacher=torch.load(PAIRED/'result/checkpoint_00360.pt',map_location='cpu',weights_only=True)
    if (evaluation['status']!='complete' or evaluation['step']!=600 or qcheckpoint['step']!=600
            or qcheckpoint['schema']!='two_future_fixed_decoder_qformer_pilot_v1'
            or qcheckpoint['posterior_digest']!=cfg['posterior_digest']
            or evaluation['source']['posterior_digest']!=cfg['posterior_digest']
            or evaluation['source']['posterior_checkpoint_sha256']!=cfg['q_checkpoint_sha256']
            or teacher['decoder_digest']!=cfg['decoder_digest']):
        raise RuntimeError('Q or teacher checkpoint metadata differs')
    for key,value in qcheckpoint['source'].items():
        if evaluation['source'].get(key)!=value:raise RuntimeError('Checkpoint and evaluation provenance differs')
    sys.path.insert(0,str(BROOT))
    from b_core import load_bound_sources
    bundle=load_bound_sources('cpu');store,posterior,runtime=(bundle[k] for k in ('store','posterior','runtime'))
    if sha(bundle['paths']['cache_manifest'])!=cfg['cache_manifest_sha256']:
        raise RuntimeError('Original Q input cache differs')
    posterior.load_state_dict(qcheckpoint['posterior'],strict=True)
    posterior.requires_grad_(False).eval()
    if runtime.digest(posterior)!=cfg['posterior_digest']:
        raise RuntimeError('Reloaded CPU posterior digest differs')
    codes={branch:teacher[branch+'_codes'] for branch in ('motion','static')}
    scale=torch.stack([v.square().mean() for v in codes.values()]).mean().clamp_min(1e-8)
    expected={(r['start'],r['horizon']):r for r in evaluation['rows']}
    rows=[];causal=[];discrepancies=[];output_codes={};began=time.monotonic()
    with torch.no_grad():
        for i,start in enumerate(cfg['windows']):
            window=store.get(start,device='cpu')
            inferred={branch:posterior([window['memories'] if branch=='motion' else window['initial_memories']])['codes'][0]
                      for branch in ('motion','static')}
            output_codes[str(start)]={k:v.detach().cpu() for k,v in inferred.items()}
            row={'start':start,'teacher_code_scale':float(scale),
                 'motion':code_stats(inferred['motion'],codes['motion'][i],scale),
                 'static':code_stats(inferred['static'],codes['static'][i],scale),
                 'q_motion_static_code_rms':float((inferred['motion']-inferred['static']).square().mean().sqrt()),
                 'teacher_motion_static_code_rms':float((codes['motion'][i]-codes['static'][i]).square().mean().sqrt())}
            rows.append(row)
            for h in cfg['horizons']:
                d=float((inferred['motion'][h-1]-inferred['static'][h-1]).square().mean().sqrt())
                discrepancies.append(abs(d-expected[(start,h)]['q_motion_static_code_rms']))
            memory=window['memories']
            for h in (3,7):
                changed=list(memory)
                for j in range(h,len(changed)):
                    item=changed[j]
                    changed[j]=type(item)(keys=item.keys+3.,coordinates=item.coordinates+100)
                altered=posterior([changed])['codes'][0,:h]
                err=float((inferred['motion'][:h]-altered).abs().max())
                if not math.isfinite(err) or err>cfg['cached_prefix_causality_tolerance']:
                    raise RuntimeError('Cached prefix causality failed')
                causal.append({'start':start,'horizon':h,'max_abs_error':err})
    if (len(discrepancies)!=18 or max(discrepancies)>cfg['code_rms_reproduction_tolerance']
            or runtime.digest(posterior)!=cfg['posterior_digest']):
        raise RuntimeError('CPU checkpoint outputs do not match the recorded GPU evaluation')
    output=ROOT/'result';output.mkdir()
    codes_path=output/'Q_CODES.pt'
    with codes_path.open('xb') as f:
        torch.save({'schema':cfg['schema'],'source_checkpoint_sha256':cfg['q_checkpoint_sha256'],
                    'codes':output_codes,'quantized':False},f)
    write(output/'REPLAY.json',{'status':'passed','rows':rows,'cached_prefix_causality':causal,
        'max_code_difference_from_gpu_record':max(discrepancies),'q_codes_sha256':sha(codes_path),
        'q_checkpoint_sha256':cfg['q_checkpoint_sha256'],'teacher_checkpoint_sha256':cfg['teacher_checkpoint_sha256'],
        'producer_pid':os.getpid(),'gpu_used':False,'training_updates':0,'training_signalled':False,
        'elapsed_seconds':time.monotonic()-began,'scope':cfg['scope']})
    print('PASS: independently reloaded step-600 Q on CPU; four code sequences and cached-prefix causality')


if __name__=='__main__':run()
