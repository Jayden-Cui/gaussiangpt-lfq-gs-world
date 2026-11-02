"""Cross-decode warm and replay-sweep-1 models/codes, without updates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch


ROOT=Path(__file__).resolve().parent
OUTPUTS=ROOT.parent
B_PREP=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PILOT=OUTPUTS/'gaussiangpt_neoverse_anchor_replay_pilot_20261102_v1'
OLD=OUTPUTS/'gaussiangpt_neoverse_stage_b_cross_decode_20261102_v1'
PLAN=(('warm_decoder_warm_codes','warm','warm'),
      ('warm_decoder_replay_codes','warm','replay'),
      ('replay_decoder_warm_codes','replay','warm'),
      ('replay_decoder_replay_codes','replay','replay'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def validate_plan(p):
    if (p.get('schema')!='anchor_replay_sweep01_cross_decode_v1' or p.get('windows')!=[35,123]
            or p.get('horizons')!=list(range(1,10)) or p.get('conditions')!=[x[0] for x in PLAN]
            or p.get('optimizer_updates')!=0 or p.get('gpu')!=1 or p.get('allocator_limit_gib')!=12
            or p.get('minimum_free_gpu_gib')!=32 or p.get('max_seconds')!=600
            or p.get('manual_visual_review_required') is not True):
        raise RuntimeError('Cross-decoding protocol differs')
    if {(d,c) for _,d,c in PLAN}!={(d,c) for d in ('warm','replay') for c in ('warm','replay')}:
        raise RuntimeError('Cross-decoding factors are incomplete')


def verify_manifest(folder):
    witness=read(folder/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(folder/'CODE.sha256'):
        raise RuntimeError('Source witness changed: '+str(folder))
    for line in (folder/'CODE.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(folder/name)!=digest:raise RuntimeError('Source changed: '+name)


def verify_sources():
    for folder in (ROOT,B_PREP,PILOT,OLD):verify_manifest(folder)
    p=read(ROOT/'protocol.json');validate_plan(p)
    paths={'pilot_code_sha256':PILOT/'CODE.sha256',
           'old_cross_code_sha256':OLD/'CODE.sha256',
           'pilot_preflight_sha256':PILOT/'run/training/PREFLIGHT.json',
           'warm_preview_sha256':PILOT/'run/training/preview_00/PREVIEW.json',
           'replay_preview_sha256':PILOT/'run/training/preview_01/PREVIEW.json',
           'checkpoint_metadata_sha256':PILOT/'run/training/sweep_01.json',
           'checkpoint_sha256':PILOT/'run/training/sweep_01.pt'}
    for key,path in paths.items():
        if sha(path)!=p[key]:raise RuntimeError('Pilot input changed: '+key)
    pre=read(paths['pilot_preflight_sha256']);warm=read(paths['warm_preview_sha256']);now=read(paths['replay_preview_sha256'])
    meta=read(paths['checkpoint_metadata_sha256'])
    if (pre.get('status')!='passed' or pre.get('restored_decoder_codes_optimizer') is not True
            or pre.get('codec_and_initial_encoder_frozen') is not True
            or warm.get('status')!='complete' or warm.get('sweep')!=0
            or now.get('status')!='complete' or now.get('sweep')!=1
            or meta.get('status')!='complete' or meta.get('sweep')!=1
            or meta.get('sha256')!=p['checkpoint_sha256'] or meta.get('decoder_digest')!=now.get('decoder_digest')
            or warm.get('decoder_digest')!=pre['source']['warm_decoder_digest']
            or pre['source']['code_sha256']!=p['pilot_code_sha256']
            or pre['source']['codec_sha256']!=p['codec_sha256']
            or pre['source']['cache_manifest_sha256']!=p['cache_manifest_sha256']
            or pre['source']['quantized'] is not False):
        raise RuntimeError('Pilot source, preflight or previews differ')
    return p,pre,warm,now,meta,paths['checkpoint_sha256']


def rgb(x):
    a=x.detach().cpu().numpy()
    if a.shape!=(252,448,3) or not np.isfinite(a).all():raise RuntimeError('Unexpected rendered image')
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p,pre,warm_preview,replay_preview,meta,checkpoint=verify_sources()
    spec=importlib.util.spec_from_file_location('_bound_old_cross',OLD/'cross_decode.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.verify_self();old.PLAN=PLAN
    sys.path.insert(0,str(B_PREP))
    from b_core import load_bound_sources
    from b_worker_v3 import load_initial_codes
    from gpu_runtime import setup_gpu_once
    from b_evaluate_v2 import anchor_metrics
    bundle=load_bound_sources('cpu');runtime=bundle['runtime'];fm=bundle['fm'];store=bundle['store']
    base=read(B_RUN/'CONFIG.json');warm_codes,initial_source=load_initial_codes(base,bundle)
    source=pre['source'];item=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (initial_source!=source['initial_codes'] or item.get('schema')!='stage_b_shared_decoder_free_codes_v1'
            or item.get('sweep')!=1 or item.get('starts')!=list(range(168)) or item.get('source')!=source
            or item.get('decoder_digest')!=meta['decoder_digest'] or tuple(item['codes'].shape)!=(168,9,16)
            or item['codes'].dtype!=torch.float32 or not bool(torch.isfinite(item['codes']).all())):
        raise RuntimeError('Pilot checkpoint or initial codes differ')
    decoder=bundle['decoder']
    if runtime.digest(decoder)!=source['warm_decoder_digest'] or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
        raise RuntimeError('Warm decoder or initial context differs')
    gpu_p=dict(bundle['p'],allocator_limit_gib=p['allocator_limit_gib'],minimum_free_gpu_gib=p['minimum_free_gpu_gib'])
    out=ROOT/'result';out.mkdir();began=time.monotonic();phase='initializing'
    write(out/'STARTED.json',dict(pid=os.getpid(),checkpoint_sha256=p['checkpoint_sha256'],
          protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256')))
    try:
        setup_gpu_once(gpu_p);decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from runtime import render
        from support_decoder import select_state
        from data import evaluate_metrics
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        rows=[];pixels={};targets={};phase='cross_decoding'
        with torch.no_grad():
            for decoder_name in ('warm','replay'):
                if decoder_name=='replay':decoder.load_state_dict(item['decoder'],strict=True)
                expected=source['warm_decoder_digest'] if decoder_name=='warm' else meta['decoder_digest']
                if runtime.digest(decoder)!=expected or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder or frozen initial context differs')
                for start in (35,123):
                    w=store.get(start)
                    for name,d_name,c_name in PLAN:
                        if d_name!=decoder_name:continue
                        codes=(warm_codes if c_name=='warm' else item['codes'])[start].cuda()
                        prediction=decoder(w['condition'],codes)
                        for h in range(1,10):
                            if time.monotonic()-began>p['max_seconds']:raise RuntimeError('Diagnostic deadline')
                            sample=store.sample(start,h)
                            state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                               prediction['logits'][h-1],gpu_p['max_predicted_latent_nodes'])
                            images,_,_=render(bridge,state,sample['record'],views=(0,1))
                            metric=evaluate_metrics(images,w,sample,gpu_p)
                            rows.append(dict(condition=name,decoder=d_name,codes=c_name,start=start,horizon=h,
                                             frame=start+h,metrics=metric))
                            key=(start,h);target=rgb(sample['record']['real'])
                            if key in targets and not np.array_equal(targets[key],target):
                                raise RuntimeError('Source target changed across conditions')
                            targets[key]=target;pixels[(name,start,h)]=rgb(images[0])
                        del prediction,codes
                    del w
                if runtime.digest(decoder)!=expected or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder or initial context changed during inference')
            bridge.assert_frozen()
        phase='verification_and_packaging'
        summary=old.aggregate(rows)
        reproductions={'warm':old.verify_preview(rows,'warm_decoder_warm_codes',warm_preview),
                       'replay':old.verify_preview(rows,'replay_decoder_replay_codes',replay_preview)}
        anchors={}
        for name,_,_ in PLAN:
            wrapped=[dict(start=r['start'],horizon=r['horizon'],frame=r['frame'],
                          metrics={'correct':r['metrics']}) for r in rows if r['condition']==name]
            anchors[name]=anchor_metrics(wrapped,store,gpu_p)
        for start in (35,123):
            folder=out/('s%03d'%start);folder.mkdir()
            for h in range(1,10):
                image=np.concatenate([targets[(start,h)]]+[pixels[(name,start,h)] for name,_,_ in PLAN],axis=1)
                Image.fromarray(image).save(folder/('frame_%02d.png'%h))
            subprocess.run(['ffmpeg','-nostdin','-n','-loglevel','error','-framerate','5','-start_number','1','-i',
                str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                str(out/('s%03d_five_columns_5fps.mp4'%start))],check=True,timeout=90)
        drift={str(s):dict(warm_rms=float(warm_codes[s].square().mean().sqrt()),
            replay_rms=float(item['codes'][s].square().mean().sqrt()),
            difference_rms=float((item['codes'][s]-warm_codes[s]).square().mean().sqrt())) for s in (35,123)}
        payload=dict(status='complete',protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
          checkpoint_sha256=p['checkpoint_sha256'],warm_decoder_digest=source['warm_decoder_digest'],
          replay_decoder_digest=meta['decoder_digest'],warm_preview_sha256=p['warm_preview_sha256'],
          replay_preview_sha256=p['replay_preview_sha256'],codec_sha256=source['codec_sha256'],
          cache_manifest_sha256=source['cache_manifest_sha256'],summary=summary,preview_reproduction=reproductions,
          dynamic_anchors=anchors,code_drift=drift,rows=rows,optimizer_updates=0,training_signalled=False,
          independent_native_replay='not performed',manual_visual_review_required=True,
          max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-began,
          scope='Two known training windows only; crossed decoder/code combinations can be out of distribution')
        write(out/'CROSS_DECODE.json',payload)
        print(json.dumps(dict(status='complete',summary=summary,preview_reproduction=reproductions,
                              max_allocated_gib=payload['max_allocated_gib']),ensure_ascii=False))
    except BaseException as error:
        write(out/'FAILED.json',dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),
                                    optimizer_updates=0,training_signalled=False,no_automatic_retry=True))
        raise


if __name__=='__main__':main()
