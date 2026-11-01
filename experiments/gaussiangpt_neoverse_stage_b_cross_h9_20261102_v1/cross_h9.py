"""Bounded inference-only 2x2 comparison across all 168 windows at h9."""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
CROSS=OUTPUTS/'gaussiangpt_neoverse_stage_b_cross_decode_20261102_v1'
PREVIEW=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
GROUPS=OUTPUTS/'gaussiangpt_neoverse_motion_groups_20261102_v1'
BROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
ASTAGE=OUTPUTS/'gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v3/sweep_20/EVALUATION.json'
PLAN=(('warm_decoder_warm_codes','warm','warm'),('warm_decoder_sweep05_codes','warm','sweep05'),
      ('sweep05_decoder_warm_codes','sweep05','warm'),('sweep05_decoder_sweep05_codes','sweep05','sweep05'))
GROUP_NAMES=('lowest','lower_middle','upper_middle','highest')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value,replace=False):
    path=Path(path);content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content,encoding='utf8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf8') as f:f.write(content)


def verify_self():
    w=read(ROOT/'CPU_TESTS.json')
    if w.get('status')!='passed' or w.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('CPU witness differs')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise ValueError('Source changed: '+name)


def validate_groups(data):
    groups=data['groups']
    if set(groups)!=set(GROUP_NAMES) or any(len(groups[n])!=42 for n in GROUP_NAMES):
        raise ValueError('Expected four disjoint 42-window source-only groups')
    starts=[s for name in GROUP_NAMES for s in groups[name]]
    if len(starts)!=168 or set(starts)!=set(range(168)):
        raise ValueError('Group windows duplicated or missing')
    return groups


def summarize(rows,groups):
    expected={(name,s) for name,_,_ in PLAN for s in range(168)}
    if len(rows)!=672 or {(r['condition'],r['start']) for r in rows}!=expected:
        raise ValueError('Expected exactly 672 unique h9 predictions')
    factor={n:(d,c) for n,d,c in PLAN}
    for row in rows:
        if ((row['decoder'],row['codes'])!=factor[row['condition']] or row['horizon']!=9
                or row['frame']!=row['start']+9):raise ValueError('Prediction provenance differs')
        for k in ('mae','motion_mae'):
            if not math.isfinite(row['metrics']['metrics'][k]):raise ValueError('Nonfinite image metric')
    def reduce(chosen):
        return {k:sum(r['metrics']['metrics'][k] for r in chosen)/len(chosen) for k in ('mae','motion_mae')}
    out={}
    for name,_,_ in PLAN:
        current=[r for r in rows if r['condition']==name]
        out[name]={'overall':reduce(current),'by_source_change_quartile':{}}
        for group in GROUP_NAMES:
            subset=[r for r in current if r['start'] in set(groups[group])]
            if len(subset)!=42:raise ValueError('Missing group predictions')
            out[name]['by_source_change_quartile'][group]=reduce(subset)
    return out


def validate_anchors(rows,earlier):
    ref={(r['condition'],r['start'],r['horizon']):r for r in earlier['rows'] if r['horizon']==9}
    sample=[r for r in rows if r['start'] in (35,123)]
    if len(ref)!=8 or len(sample)!=8:raise ValueError('Missing prior h9 anchor comparisons')
    diff=[]
    for r in sample:
        q=ref[(r['condition'],r['start'],9)]
        for k in ('mae','motion_mae'):
            diff.append(abs(r['metrics']['metrics'][k]-q['metrics']['metrics'][k]))
    if max(diff)>5e-5:raise ValueError('h9 anchors differ from earlier cross-decoding')
    return {'predictions':8,'metrics_per_prediction':2,'max_abs_metric_difference':max(diff),'tolerance':5e-5}


def run():
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    verify_self();protocol=read(ROOT/'protocol.json')
    if (protocol['schema']!='stage_b_cross_h9_all168_v1' or protocol['conditions']!=[n for n,_,_ in PLAN]
            or protocol['windows']!=168 or protocol['horizon']!=9 or protocol['updates']!=0):
        raise ValueError('Unexpected h9 comparison protocol')
    out=ROOT/'result';out.mkdir()
    phase='loading_bound_sources';began=time.monotonic()
    write(out/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),'code_sha256':sha(ROOT/'CODE.sha256')})
    try:
        sys.path.insert(0,str(CROSS));import cross_decode
        cross_decode.verify_self()
        if cross_decode.PLAN!=PLAN or sha(CROSS/'result/CROSS_DECODE.json')!=protocol['matched_cross_decode_sha256']:
            raise ValueError('Earlier cross-decode source changed')
        earlier=read(CROSS/'result/CROSS_DECODE.json')
        if earlier['status']!='complete' or earlier['optimizer_updates']!=0:raise ValueError('Earlier cross result incomplete')
        if sha(GROUPS/'MOTION_GROUPS.json')!=protocol['source_groups_sha256']:
            raise ValueError('Source change groups differ')
        grouping=read(GROUPS/'MOTION_GROUPS.json');groups=validate_groups(grouping)
        if grouping['cache_manifest_sha256']!=protocol['cache_manifest_sha256']:
            raise ValueError('Grouping uses another source')
        if sha(ASTAGE)!=protocol['stage_a_corrected_evaluation_sha256']:
            raise ValueError('Actual-target reference evaluation changed')
        a=read(ASTAGE)
        if (a['status']!='complete' or a['window_predictions']!=1512
                or a['source']['cache_manifest_sha256']!=protocol['cache_manifest_sha256']
                or a['source']['codec_sha256']!=protocol['codec_sha256']):
            raise ValueError('Actual-target reference provenance differs')
        targets={r['start']:r['metrics']['actual_target']['metrics'] for r in a['rows'] if r['horizon']==9}
        if set(targets)!=set(range(168)):raise ValueError('Reference h9 coverage differs')
        sys.path.insert(0,str(PREVIEW));import preview
        preview.verify_sources()
        sys.path.insert(0,str(BROOT))
        from b_core import load_bound_sources
        from b_worker_v3 import load_initial_codes
        from gpu_runtime import setup_gpu_once
        bundle=load_bound_sources('cpu')
        runtime=bundle['runtime'];fm=bundle['fm'];store=bundle['store']
        p=dict(bundle['p'],allocator_limit_gib=protocol['allocator_limit_gib'],
               minimum_free_gpu_gib=protocol['minimum_free_gpu_gib'])
        if p['diagnostic_starts']!=protocol['diagnostic_starts']:raise ValueError('Representative windows changed')
        config=read(RUN/'CONFIG.json');preflight=read(RUN/'training/PREFLIGHT.json');source=preflight['source']
        checkpoint=RUN/'training/sweep_05.pt';metadata=read(RUN/'training/sweep_05.json')
        if (preflight['status']!='passed' or source['cache_manifest_sha256']!=protocol['cache_manifest_sha256']
                or source['codec_sha256']!=protocol['codec_sha256'] or sha(checkpoint)!=protocol['stage_b_sweep05_checkpoint_sha256']):
            raise ValueError('Sweep5 model, data or native codec differs')
        item=torch.load(checkpoint,map_location='cpu',weights_only=True)
        codes5=preview.validate_checkpoint(item,metadata,source,5,sha(checkpoint))
        codes0,initial_source=load_initial_codes(config,bundle)
        decoder=bundle['decoder']
        if (initial_source!=source['initial_codes'] or runtime.digest(decoder)!=source['warm_decoder_digest']
                or fm.frozen_digest(decoder)!=bundle['frozen_digest']):raise ValueError('Warm model or context differs')
        phase='gpu_preflight';setup_gpu_once(p);decoder.requires_grad_(False).eval().cuda()
        from runtime_bridge import BoundNativeBridge
        from support_decoder import select_state,support_metrics
        from runtime import render
        from data import evaluate_metrics
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        rows=[];pixels={};target_pixels={}
        phase='rendering_h9'
        with torch.no_grad():
            for d in ('warm','sweep05'):
                if d=='sweep05':decoder.load_state_dict(item['decoder'],strict=True)
                expected=source['warm_decoder_digest'] if d=='warm' else metadata['decoder_digest']
                if runtime.digest(decoder)!=expected or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder or frozen initial context changed')
                for start in range(168):
                    w=store.get(start);sample=store.sample(start,9)
                    for name,d_name,c_name in PLAN:
                        if d_name!=d:continue
                        code=(codes0 if c_name=='warm' else codes5)[start].cuda()
                        prediction=decoder(w['condition'],code)
                        state=select_state(w['condition'].coordinates,prediction['features'][8],
                                           prediction['logits'][8],p['max_predicted_latent_nodes'])
                        images,_,_=render(bridge,state,sample['record'],views=(0,1))
                        metrics=evaluate_metrics(images,w,sample,p)
                        support=support_metrics(prediction['logits'][8],w['targets']['occupancy'][8])
                        rows.append({'condition':name,'decoder':d,'codes':c_name,'start':start,'horizon':9,
                                     'frame':start+9,'metrics':metrics,'support':support})
                        if start in p['diagnostic_starts']:
                            target=preview.rgb_array(sample['record']['real'])
                            if start in target_pixels and not np.array_equal(target_pixels[start],target):
                                raise RuntimeError('Targets differ across conditions')
                            target_pixels[start]=target
                            pixels[(name,start)]=preview.rgb_array(images[0])
                        del code,prediction,state,images
                    del w,sample
                    if start%28==27 or start==167:
                        write(out/'STATUS.json',{'status':'running','decoder':d,'completed_windows':start+1,
                              'total_windows':168,'elapsed_seconds':time.monotonic()-began},replace=True)
                if runtime.digest(decoder)!=expected or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                    raise RuntimeError('Decoder changed during inference')
            bridge.assert_frozen()
        phase='validating_and_writing'
        summary=summarize(rows,groups);anchor_match=validate_anchors(rows,earlier)
        ref_summary={'overall':{k:sum(targets[s][k] for s in range(168))/168 for k in ('mae','motion_mae')},
                     'by_source_change_quartile':{g:{k:sum(targets[s][k] for s in groups[g])/42
                         for k in ('mae','motion_mae')} for g in GROUP_NAMES},
                     'source':'corrected A evaluation of true future state through native codec; offline reference only'}
        panels=[]
        for start in p['diagnostic_starts']:
            data=np.concatenate([target_pixels[start]]+[pixels[(name,start)] for name,_,_ in PLAN],axis=1)
            path=out/f's{start:03d}_h09.png';Image.fromarray(data,mode='RGB').save(path)
            panels.append({'start':start,'file':path.name,'sha256':sha(path)})
        result={'status':'complete','protocol_sha256':sha(ROOT/'protocol.json'),'source':source,
          'checkpoint_sha256':sha(checkpoint),'earlier_cross_decode_sha256':sha(CROSS/'result/CROSS_DECODE.json'),
          'source_groups_sha256':sha(GROUPS/'MOTION_GROUPS.json'),'reference_evaluation_sha256':sha(ASTAGE),
          'window_predictions':672,'unique_h9_target_frames':168,'aggregation':'h9 has one unique target physical frame per start',
          'summary':summary,'actual_target_reference':ref_summary,'anchor_reproduction':anchor_match,
          'representative_panels':panels,'rows':rows,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'elapsed_seconds':time.monotonic()-began,'optimizer_updates':0,'training_signalled':False,
          'independent_native_replay':'not performed','visual_review':'pending','scope':protocol['scope']}
        write(out/'CROSS_H9.json',result)
        write(out/'STATUS.json',{'status':'complete','phase':'awaiting_visual_review'},replace=True)
        print('PASS: all 168 h9 windows x four, earlier anchors reproduced')
    except BaseException as error:
        failure={'status':'failed','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
                 'optimizer_updates':0,'training_signalled':False,'no_automatic_retry':True}
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':run()
