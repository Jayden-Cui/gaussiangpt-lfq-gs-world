"""Decode a bound 2x2 stage-B checkpoint/code matrix without updating training."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path('/data/CJH/AGS-V6Direct/outputs')
RUN = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PREVIEW = OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
B_ROOT = OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
PLAN = (('warm_decoder_warm_codes','warm','warm'),
        ('warm_decoder_sweep05_codes','warm','sweep05'),
        ('sweep05_decoder_warm_codes','sweep05','warm'),
        ('sweep05_decoder_sweep05_codes','sweep05','sweep05'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def verify_plan(protocol):
    if (protocol['schema'] != 'stage_b_cross_decode_diagnostic_v1'
            or protocol['conditions'] != [name for name,_,_ in PLAN]
            or protocol['windows'] != [35,123] or protocol['future_horizons'] != list(range(1,10))
            or {(d,c) for _,d,c in PLAN} != {(d,c) for d in ('warm','sweep05') for c in ('warm','sweep05')}
            or protocol['updates'] != 0 or protocol['cuda_visible_device'] != 1):
        raise ValueError('Diagnostic plan differs from fixed 2x2 contract')


def aggregate(rows):
    expected={(name,start,h) for name,_,_ in PLAN for start in (35,123) for h in range(1,10)}
    actual=[(r['condition'],r['start'],r['horizon']) for r in rows]
    if len(actual)!=72 or set(actual)!=expected:
        raise ValueError('Cross-decode rows contain missing, extra or duplicate pairs')
    factors={name:(d,c) for name,d,c in PLAN}
    if any((r['decoder'],r['codes'])!=factors[r['condition']]
           or r['frame']!=r['start']+r['horizon'] for r in rows):
        raise ValueError('Cross-decode factor or target-frame provenance differs')
    result={}
    for name,_,_ in PLAN:
        selected=[r for r in rows if r['condition']==name]
        metrics={}
        for key in ('mae','motion_mae'):
            values=[r['metrics']['metrics'][key] for r in selected]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
                raise ValueError('Nonfinite image metric')
            metrics[key]=sum(values)/len(values)
        metrics['per_window_motion_mae']={str(s):sum(r['metrics']['metrics']['motion_mae'] for r in selected if r['start']==s)/9
                                          for s in (35,123)}
        result[name]=metrics
    return result


def verify_preview(rows, condition, preview_data, tolerance=5e-5):
    future=[r for r in preview_data['rows'] if r['horizon']>0]
    expected={(r['start'],r['horizon']):r for r in future}
    actual={(r['start'],r['horizon']):r for r in rows if r['condition']==condition}
    if len(future)!=18 or set(expected)!={(s,h) for s in (35,123) for h in range(1,10)} or set(actual)!=set(expected):
        raise ValueError('Preview frame coverage differs')
    diffs=[]
    for key in expected:
        for metric in ('mae','motion_mae'):
            a=actual[key]['metrics']['metrics'][metric]
            b=expected[key]['metrics']['correct']['metrics'][metric]
            diffs.append(abs(a-b))
    if not all(math.isfinite(d) for d in diffs) or max(diffs)>tolerance:
        raise ValueError('Cross-decoder result did not reproduce original preview')
    return {'pairs':len(expected),'metrics_per_pair':2,'max_abs_metric_difference':max(diffs),'tolerance':tolerance}


def verify_self():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('CPU witness differs')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:
            raise ValueError('Bound source changed: '+name)


def run():
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    verify_self()
    protocol=read(ROOT/'protocol.json')
    verify_plan(protocol)
    sys.path.insert(0,str(PREVIEW))
    import preview
    preview.verify_sources()
    sys.path.insert(0,str(B_ROOT))
    from b_core import load_bound_sources
    from b_worker_v3 import load_initial_codes
    from gpu_runtime import setup_gpu_once
    from b_evaluate_v2 import anchor_metrics
    bundle=load_bound_sources('cpu')
    binding=bundle['binding']; runtime=bundle['runtime']; fm=bundle['fm']; store=bundle['store']
    p=dict(bundle['p'],allocator_limit_gib=protocol['allocator_limit_gib'],
           minimum_free_gpu_gib=protocol['minimum_free_gpu_gib'])
    config=read(RUN/'CONFIG.json')
    preflight=read(RUN/'training/PREFLIGHT.json')
    source=preflight['source']
    checkpoint=RUN/'training/sweep_05.pt'
    metadata=read(RUN/'training/sweep_05.json')
    if (preflight['status']!='passed' or source['codec_sha256']!=protocol['codec_sha256']
            or source['cache_manifest_sha256']!=protocol['cache_manifest_sha256']
            or sha(checkpoint)!=protocol['source_checkpoint_sha256']):
        raise ValueError('Wrong bound codec, frame cache or stage-B checkpoint')
    item=torch.load(checkpoint,map_location='cpu',weights_only=True)
    codes05=preview.validate_checkpoint(item,metadata,source,5,sha(checkpoint))
    codes0,initial_source=load_initial_codes(config,bundle)
    decoder=bundle['decoder']
    if initial_source!=source['initial_codes'] or runtime.digest(decoder)!=source['warm_decoder_digest']:
        raise ValueError('Warm decoder/code initialization differs')
    if fm.frozen_digest(decoder)!=bundle['frozen_digest']:
        raise ValueError('Warm frozen context differs')
    ref0=read(PREVIEW/'warm_start/PREVIEW.json')
    ref5=read(PREVIEW/'sweep_05/PREVIEW.json')
    if (ref0['status']!='complete' or ref5['status']!='complete'
            or ref5['checkpoint_sha256']!=sha(checkpoint)
            or ref0['decoder_digest']!=source['warm_decoder_digest']
            or ref5['decoder_digest']!=metadata['decoder_digest']
            or any(ref.get('config_sha256')!=sha(RUN/'CONFIG.json')
                   or ref.get('codec_and_initial_context_frozen') is not True for ref in (ref0,ref5))):
        raise ValueError('Anchor previews are not bound to the same pair of models')
    out=ROOT/'result'
    out.mkdir()
    began=time.monotonic();phase='gpu_initializing'
    write_new(out/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
                                'code_sha256':sha(ROOT/'CODE.sha256'),'checkpoint_sha256':sha(checkpoint)})
    try:
        setup_gpu_once(p)
        decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import select_state
        from runtime import render
        from data import evaluate_metrics
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        rows=[];pixels={};targets={}
        phase='cross_decoding'
        with torch.no_grad():
            for decoder_name in ('warm','sweep05'):
                if decoder_name=='sweep05':
                    decoder.load_state_dict(item['decoder'],strict=True)
                expected_digest=source['warm_decoder_digest'] if decoder_name=='warm' else metadata['decoder_digest']
                if runtime.digest(decoder)!=expected_digest or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder or frozen initial context differs')
                for start in (35,123):
                    w=store.get(start)
                    for name,d_name,c_name in PLAN:
                        if d_name!=decoder_name:continue
                        code=(codes0 if c_name=='warm' else codes05)[start].cuda()
                        prediction=decoder(w['condition'],code)
                        for h in range(1,10):
                            sample=store.sample(start,h)
                            state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                               prediction['logits'][h-1],p['max_predicted_latent_nodes'])
                            images,_,_=render(bridge,state,sample['record'],views=(0,1))
                            metric=evaluate_metrics(images,w,sample,p)
                            row={'condition':name,'decoder':d_name,'codes':c_name,'start':start,'horizon':h,
                                 'frame':start+h,'metrics':metric}
                            rows.append(row)
                            key=(start,h)
                            target=preview.rgb_array(sample['record']['real'])
                            if key in targets and not np.array_equal(targets[key],target):
                                raise RuntimeError('Source target changed across conditions')
                            targets[key]=target
                            pixels[(name,start,h)]=preview.rgb_array(images[0])
                        del prediction,code
                    del w
                if runtime.digest(decoder)!=expected_digest or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder or frozen context changed during inference')
            bridge.assert_frozen()
        phase='verifying_and_packaging'
        summary=aggregate(rows)
        reproductions={
          'warm':verify_preview(rows,'warm_decoder_warm_codes',ref0),
          'sweep05':verify_preview(rows,'sweep05_decoder_sweep05_codes',ref5)}
        anchors={}
        for name,_,_ in PLAN:
            wrapped=[{'start':r['start'],'horizon':r['horizon'],'frame':r['frame'],
                      'metrics':{'correct':r['metrics']}} for r in rows if r['condition']==name]
            anchors[name]=anchor_metrics(wrapped,store,p)
        for start in (35,123):
            folder=out/f's{start:03d}'
            folder.mkdir()
            for h in range(1,10):
                tile=np.concatenate([targets[(start,h)]]+[pixels[(name,start,h)] for name,_,_ in PLAN],axis=1)
                Image.fromarray(tile,mode='RGB').save(folder/f'frame_{h:02d}.png')
            subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1','-i',
                str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                str(out/f's{start:03d}_five_column_5fps.mp4')],check=True,timeout=90)
        drift={str(s):{'code_rms_warm':float(codes0[s].square().mean().sqrt()),
            'code_rms_sweep05':float(codes05[s].square().mean().sqrt()),
            'code_difference_rms':float((codes05[s]-codes0[s]).square().mean().sqrt())} for s in (35,123)}
        payload={'status':'complete','protocol_sha256':sha(ROOT/'protocol.json'),
          'checkpoint_sha256':sha(checkpoint),'warm_decoder_digest':source['warm_decoder_digest'],
          'sweep05_decoder_digest':metadata['decoder_digest'],'codec_sha256':source['codec_sha256'],
          'cache_manifest_sha256':source['cache_manifest_sha256'],
          'warm_preview_sha256':sha(PREVIEW/'warm_start/PREVIEW.json'),
          'sweep05_preview_sha256':sha(PREVIEW/'sweep_05/PREVIEW.json'),
          'preview_reproduction':reproductions,'summary':summary,'dynamic_anchors':anchors,'code_drift':drift,
          'rows':rows,'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'elapsed_seconds':time.monotonic()-began,'optimizer_updates':0,'training_signalled':False,
          'independent_native_replay':'not performed','manual_visual_review':'pending','scope':protocol['limits']}
        write_new(out/'CROSS_DECODE.json',payload)
        print('PASS: 72 predictions, 2 prior previews reproduced, no updates')
    except BaseException as error:
        write_new(out/'FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
            'traceback':traceback.format_exc(),'optimizer_updates':0,'training_signalled':False,'no_automatic_retry':True})
        raise


if __name__=='__main__':
    run()
