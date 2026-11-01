"""Replay old Q controls against moving/static codec references at identical cameras."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
BROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PREVIEW=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
ORIGINAL=OUTPUTS/'gaussiangpt_neoverse_qformer_support_20261101_v1/qformer_continuous/evaluation'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def verify_self():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('CPU witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise ValueError('Source changed: '+name)


def weighted_controls(motion_prediction,static_prediction,motion_reference,static_reference):
    import torch
    tensors=(motion_prediction,static_prediction,motion_reference,static_reference)
    if any(x.ndim!=3 or x.shape[-1]!=3 or x.shape!=tensors[0].shape or not bool(torch.isfinite(x).all()) for x in tensors):
        raise ValueError('Expected matching finite RGB tensors')
    weight=((motion_reference-static_reference).abs().mean(-1)/.1).clamp(0,1)
    total=weight.sum()
    if not bool(torch.isfinite(total)) or float(total)<=1e-6:raise ValueError('No measurable reference change')
    def error(pred,target):return float((weight*(pred-target).abs().mean(-1)).sum()/total)
    return {'motion_code_to_motion_reference':error(motion_prediction,motion_reference),
            'motion_code_to_static_reference':error(motion_prediction,static_reference),
            'static_code_to_static_reference':error(static_prediction,static_reference),
            'static_code_to_motion_reference':error(static_prediction,motion_reference),
            'reference_motion_vs_static':error(motion_reference,static_reference),
            'prediction_motion_vs_static':error(motion_prediction,static_prediction),
            'dynamic_weight_sum':float(total)}


def aggregate(rows):
    expected={(s,h) for s in (35,123) for h in range(1,10)}
    if len(rows)!=18 or {(r['start'],r['horizon']) for r in rows}!=expected:
        raise ValueError('Incomplete, duplicated or mismatched diagnostic frames')
    keys=('motion_code_to_motion_reference','motion_code_to_static_reference',
          'static_code_to_static_reference','static_code_to_motion_reference',
          'reference_motion_vs_static','prediction_motion_vs_static')
    def mean(selected):
        d={k:sum(r['metrics'][k] for r in selected)/len(selected) for k in keys}
        if not all(math.isfinite(x) for x in d.values()):raise ValueError('Nonfinite control metric')
        d['static_prediction_closer_to_static_than_motion']=(d['static_code_to_static_reference']<d['static_code_to_motion_reference'])
        return d
    return {'overall':mean(rows),'per_window':{str(s):mean([r for r in rows if r['start']==s]) for s in (35,123)},
            'h9':mean([r for r in rows if r['horizon']==9])}


def run():
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    verify_self();p0=read(ROOT/'protocol.json')
    if (p0['schema']!='two_window_same_camera_static_control_v1' or p0['windows']!=[35,123]
            or p0['future_horizons']!=list(range(1,10)) or p0['training_updates']!=0):
        raise ValueError('Diagnostic protocol changed')
    result=ROOT/'result';result.mkdir();began=time.monotonic();phase='source_binding'
    write(result/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
                                 'code_sha256':sha(ROOT/'CODE.sha256')})
    try:
        evaluation_path=ORIGINAL/'EVALUATION.json';codes_path=ORIGINAL/'action_codes.pt'
        if (sha(evaluation_path)!=p0['historical_evaluation_sha256']
                or sha(codes_path)!=p0['historical_codes_sha256']):raise ValueError('Historical source changed')
        evaluation=read(evaluation_path);payload=torch.load(codes_path,map_location='cpu',weights_only=True)
        if (evaluation['status']!='complete' or evaluation['windows']!=2 or evaluation['future_states']!=18
                or payload['schema']!='qformer_continuous_codes_v1' or payload['shape']!=[2,9,16]
                or payload['quantized'] is not False or payload['source']!=evaluation['source']
                or evaluation['source']['codec_sha256']!=p0['codec_sha256']
                or evaluation['source']['teacher_tensor_digest']!=p0['warm_decoder_digest']):
            raise ValueError('Historical Q controls, decoder or codec do not match')
        correct=payload['controls']['correct'];static=payload['controls']['initial_only']
        if any(x.dtype!=torch.float32 or tuple(x.shape)!=(2,9,16) or not bool(torch.isfinite(x).all()) for x in (correct,static)):
            raise ValueError('Invalid historical continuous codes')
        historical={(r['start'],r['horizon']):r for r in evaluation['rows'] if r['horizon']>0}
        if set(historical)!={(s,h) for s in p0['windows'] for h in p0['future_horizons']}:
            raise ValueError('Historical prediction coverage differs')
        sys.path.insert(0,str(PREVIEW));import preview
        preview.verify_sources()
        sys.path.insert(0,str(BROOT))
        from b_core import load_bound_sources
        from gpu_runtime import setup_gpu_once
        bundle=load_bound_sources('cpu');runtime=bundle['runtime'];store=bundle['store'];decoder=bundle['decoder']
        if (sha(bundle['paths']['cache_manifest'])!=p0['cache_manifest_sha256']
                or runtime.digest(decoder)!=p0['warm_decoder_digest']):
            raise ValueError('Frame cache or frozen decoder differs')
        for s in p0['windows']:
            if store.manifest['windows'][str(s)]['state_sources']!=evaluation['source']['states'][str(s)]:
                raise ValueError('GS state bytes differ from historical Q experiment')
        p=dict(bundle['p'],allocator_limit_gib=p0['allocator_limit_gib'],minimum_free_gpu_gib=p0['minimum_free_gpu_gib'])
        phase='gpu_preflight';setup_gpu_once(p);decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import select_state
        from runtime import render
        from data import evaluate_metrics
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        rows=[];reproduction=[];phase='render_same_camera_controls'
        with torch.no_grad():
            for i,start in enumerate(p0['windows']):
                w=store.get(start)
                a=decoder(w['condition'],correct[i].cuda())
                b=decoder(w['condition'],static[i].cuda())
                folder=result/f's{start:03d}';folder.mkdir()
                for h in p0['future_horizons']:
                    sample=store.sample(start,h);record=sample['record'];images=[]
                    for prediction in (a,b):
                        state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                           prediction['logits'][h-1],p['max_predicted_latent_nodes'])
                        rendered,_,_=render(bridge,state,record,views=(0,1));images.append(rendered)
                    moving,_,_=render(bridge,w['states'][h],record,views=(0,1))
                    still,_,_=render(bridge,w['states'][0],record,views=(0,1))
                    values=weighted_controls(images[0][0],images[1][0],moving[0],still[0])
                    previous=historical[(start,h)]
                    for prediction,name in zip(images,('correct','initial_only')):
                        metrics=evaluate_metrics(prediction,w,sample,p)['metrics']
                        for key in ('mae','motion_mae'):
                            reproduction.append(abs(metrics[key]-previous['metrics'][name]['metrics'][key]))
                    rows.append({'start':start,'horizon':h,'frame':start+h,'metrics':values,
                        'code_difference_rms':float((correct[i,h-1]-static[i,h-1]).square().mean().sqrt()),
                        'same_initial_state_and_decoder_condition':True,'same_target_camera':True})
                    panels=[record['real'],moving[0],images[0][0],still[0],images[1][0]]
                    combined=np.concatenate([preview.rgb_array(x) for x in panels],axis=1)
                    Image.fromarray(combined,mode='RGB').save(folder/f'frame_{h:02d}.png')
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1','-i',
                    str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                    str(result/f's{start:03d}_same_camera_5fps.mp4')],check=True,timeout=90)
        phase='verification'
        bridge.assert_frozen()
        if runtime.digest(decoder)!=p0['warm_decoder_digest'] or any(x.requires_grad for x in decoder.parameters()):
            raise RuntimeError('Motion decoder changed')
        if len(reproduction)!=72 or max(reproduction)>5e-5:
            raise RuntimeError('Historical Q moving/static predictions did not reproduce')
        summary=aggregate(rows)
        write(result/'STATIC_CONTROL.json',{'status':'complete','summary':summary,'rows':rows,
          'source_evaluation_sha256':sha(evaluation_path),'source_codes_sha256':sha(codes_path),
          'source_cache_sha256':sha(bundle['paths']['cache_manifest']),'codec_sha256':p0['codec_sha256'],
          'decoder_digest':p0['warm_decoder_digest'],
          'historical_prediction_reproduction':{'comparisons':len(reproduction),'max_abs_error':max(reproduction),'tolerance':5e-5},
          'same_initial_state_and_target_camera':True,'training_updates':0,'training_signalled':False,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.monotonic()-began,
          'independent_native_replay':'not performed','visual_dynamic_review':'pending','scope':p0['limits']})
        print('PASS: two-window same-camera historical Q static control, no updates')
    except BaseException as error:
        write(result/'FAILED.json',{'status':'failed','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
             'training_updates':0,'training_signalled':False,'no_automatic_retry':True})
        raise


if __name__=='__main__':run()
