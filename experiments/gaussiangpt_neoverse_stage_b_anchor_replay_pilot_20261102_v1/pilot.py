"""Run two bounded, matched stage-B continuations without touching the live run."""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

import sampling
import train_core

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
BROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
PREVIEW=OUTPUTS/'gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1'
CROSS=OUTPUTS/'gaussiangpt_neoverse_stage_b_cross_h9_20261102_v1'
ANCHOR_CROSS=OUTPUTS/'gaussiangpt_neoverse_stage_b_cross_decode_20261102_v1'
GROUPS=OUTPUTS/'gaussiangpt_neoverse_motion_groups_20261102_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value,replace=False):
    path=Path(path);text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf8');tmp.replace(path)
    else:
        with path.open('x',encoding='utf8') as f:f.write(text)


def verify_self():
    w=read(ROOT/'CPU_TESTS.json')
    if w.get('status')!='passed' or w.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('Pilot CPU witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise ValueError('Pilot source changed: '+name)


def equal_tree(a,b):
    import torch
    if torch.is_tensor(a) and torch.is_tensor(b):return a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def max_state_difference(a,b):
    import torch
    if set(a)!=set(b):raise ValueError('State keys differ')
    maximum=0.0
    for key in a:
        x,y=a[key].detach().cpu(),b[key].detach().cpu()
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('State tensor metadata differs')
        if x.is_floating_point():maximum=max(maximum,float((x-y).abs().max()))
        elif not torch.equal(x,y):raise ValueError('Integer state differs')
    return maximum


def evaluate(decoder,bank,bridge,store,p,protocol,groups,out,preview,anchor_metrics):
    import numpy as np
    from PIL import Image
    import torch
    from support_decoder import select_state,support_metrics
    from runtime import render
    from data import evaluate_metrics
    decoder.eval();rows=[];image_dir=out/'paired_images';image_dir.mkdir()
    selected=set(protocol['diagnostic_starts'])|set(protocol['anchors'])
    with torch.no_grad():
        for start in range(168):
            w=store.get(start);prediction=decoder(w['condition'],bank.code(start))
            horizons=range(1,10) if start in protocol['anchors'] else (9,)
            for h in horizons:
                sample=store.sample(start,h)
                state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                   prediction['logits'][h-1],p['max_predicted_latent_nodes'])
                images,_,_=render(bridge,state,sample['record'],views=(0,1))
                metrics=evaluate_metrics(images,w,sample,p)
                support=support_metrics(prediction['logits'][h-1],w['targets']['occupancy'][h-1])
                rows.append({'start':start,'horizon':h,'frame':start+h,
                             'metrics':{'correct':metrics},'support':support})
                if start in selected and (h==9 or start in protocol['anchors']):
                    target=preview.rgb_array(sample['record']['real']);pred=preview.rgb_array(images[0])
                    Image.fromarray(np.concatenate((target,pred),axis=1),mode='RGB').save(image_dir/f's{start:03d}_h{h:02d}.png')
                del sample,state,images
            del w,prediction
    h9=[r for r in rows if r['horizon']==9]
    anchors=[r for r in rows if r['start'] in protocol['anchors']]
    if (len(h9)!=168 or {r['start'] for r in h9}!=set(range(168)) or len(anchors)!=18
            or {(r['start'],r['horizon']) for r in anchors}!={(s,h) for s in (35,123) for h in range(1,10)}):
        raise RuntimeError('Pilot evaluation coverage differs')
    mean=lambda xs:sum(r['metrics']['correct']['metrics']['motion_mae'] for r in xs)/len(xs)
    result={'status':'complete','anchor_motion_mae':mean(anchors),'h9_motion_mae':mean(h9),
      'h9_high_change_motion_mae':mean([r for r in h9 if r['start'] in set(groups['highest'])]),
      'by_source_change_quartile':{g:mean([r for r in h9 if r['start'] in set(groups[g])]) for g in groups},
      'dynamic_anchors':anchor_metrics(anchors,store,p),'rows':rows,
      'independent_native_replay':'not performed','visual_review':'pending',
      'scope':'One TRAIN video, all 168 windows at h9, and two 9-frame anchor windows; free-code diagnostic'}
    write(out/'EVALUATION.json',result)
    return result


def render_comparisons(root,control,replay,protocol,cross,anchor_manifest):
    import numpy as np
    from PIL import Image,ImageChops
    visuals=root/'visuals';visuals.mkdir()
    def compose(start,h,reference):
        a=Image.open(control/'paired_images'/f's{start:03d}_h{h:02d}.png').convert('RGB')
        b=Image.open(replay/'paired_images'/f's{start:03d}_h{h:02d}.png').convert('RGB')
        old=Image.open(reference).convert('RGB')
        if a.size!=(896,252) or b.size!=a.size or old.size!=(2240,252):raise ValueError('Unexpected panel dimensions')
        target=a.crop((0,0,448,252))
        if (ImageChops.difference(target,b.crop((0,0,448,252))).getbbox() is not None
                or ImageChops.difference(target,old.crop((0,0,448,252))).getbbox() is not None):
            raise ValueError('Target image pixels differ between matched runs')
        parts=[np.asarray(x) for x in (target,old.crop((1792,0,2240,252)),a.crop((448,0,896,252)),b.crop((448,0,896,252)))]
        return Image.fromarray(np.concatenate(parts,axis=1),mode='RGB')
    records=[]
    h9_hashes={x['start']:x['sha256'] for x in cross['representative_panels']}
    if set(h9_hashes)!=set(protocol['diagnostic_starts']):raise ValueError('H9 panel index differs')
    anchor_hashes={(x['start'],v['horizon']):v['sha256'] for x in anchor_manifest['rows'] for v in x['inputs']}
    if set(anchor_hashes)!={(s,h) for s in protocol['anchors'] for h in range(1,10)}:
        raise ValueError('Anchor frame index differs')
    for start in protocol['diagnostic_starts']:
        old=CROSS/'result'/f's{start:03d}_h09.png'
        if sha(old)!=h9_hashes[start]:raise ValueError('Prior h9 panel changed')
        img=compose(start,9,old);path=visuals/f's{start:03d}_h09.png';img.save(path)
        records.append({'start':start,'file':path.name,'sha256':sha(path),'source_equals_across_arms':True})
    for start in protocol['anchors']:
        folder=visuals/f's{start:03d}';folder.mkdir()
        for h in range(1,10):
            old=ANCHOR_CROSS/'result'/f's{start:03d}'/f'frame_{h:02d}.png'
            if sha(old)!=anchor_hashes[(start,h)]:raise ValueError('Prior anchor frame changed')
            compose(start,h,old).save(folder/f'frame_{h:02d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-start_number','1','-i',
            str(folder/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
            str(visuals/f's{start:03d}_four_column_5fps.mp4')],check=True,timeout=90)
    return records


def run():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    verify_self();protocol=read(ROOT/'protocol.json')
    if (protocol['schema']!='stage_b_anchor_replay_pilot_v1' or protocol['arms']!=['control','anchor_replay']
            or protocol['updates_per_arm']!=189 or protocol['anchor_replay_weight']!=0.1
            or protocol['main_logical_batch_size']!=4 or protocol['anchors']!=[35,123]):
        raise ValueError('Pilot protocol differs from fixed experiment')
    schedule=sampling.make_schedule(protocol['main_sampler_seed'],protocol['anchor_sampler_seed'])
    sampler=sampling.audit(schedule)
    out=ROOT/'result';out.mkdir();began=time.monotonic();phase='loading'
    write(out/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
                              'code_sha256':sha(ROOT/'CODE.sha256'),'sampler':sampler})
    try:
        checkpoint=RUN/'training/sweep_05.pt'
        for path,key in ((checkpoint,'source_checkpoint_sha256'),(CROSS/'result/CROSS_H9.json','cross_h9_result_sha256'),
                         (GROUPS/'MOTION_GROUPS.json','source_groups_sha256'),
                         (ANCHOR_CROSS/'result/CROSS_DECODE.json','anchor_cross_result_sha256'),
                         (ANCHOR_CROSS/'result/MONTAGES.json','anchor_frame_manifest_sha256')):
            if sha(path)!=protocol[key]:raise ValueError('Required input hash differs: '+key)
        cross=read(CROSS/'result/CROSS_H9.json');grouping=read(GROUPS/'MOTION_GROUPS.json')
        anchor_cross=read(ANCHOR_CROSS/'result/CROSS_DECODE.json')
        anchor_manifest=read(ANCHOR_CROSS/'result/MONTAGES.json')
        if cross['status']!='complete' or cross['checkpoint_sha256']!=sha(checkpoint):
            raise ValueError('Cross-h9 evidence is not bound to source checkpoint')
        if anchor_cross['status']!='complete' or anchor_cross['checkpoint_sha256']!=sha(checkpoint):
            raise ValueError('Anchor cross evidence is not bound to source checkpoint')
        groups=grouping['groups'];starts=[s for values in groups.values() for s in values]
        if len(starts)!=168 or set(starts)!=set(range(168)) or any(len(v)!=42 for v in groups.values()):
            raise ValueError('Frozen source groups incomplete')
        sys.path.insert(0,str(PREVIEW));import preview
        preview.verify_sources()
        sys.path.insert(0,str(BROOT))
        from b_core import FreeCodes,load_bound_sources,optimizer_for,train_step as native_step
        from b_evaluate_v2 import anchor_metrics
        from gpu_runtime import setup_gpu_once
        bundle=load_bound_sources('cpu');runtime=bundle['runtime'];fm=bundle['fm'];store=bundle['store']
        config=read(RUN/'CONFIG.json');preflight=read(RUN/'training/PREFLIGHT.json');source=preflight['source']
        metadata=read(RUN/'training/sweep_05.json')
        if (source['codec_sha256']!=protocol['codec_sha256'] or source['cache_manifest_sha256']!=protocol['cache_manifest_sha256']
                or bundle['p']['diagnostic_starts']!=protocol['diagnostic_starts']):
            raise ValueError('Native codec, cache or diagnostics differ')
        item=torch.load(checkpoint,map_location='cpu',weights_only=True)
        codes=preview.validate_checkpoint(item,metadata,source,5,sha(checkpoint))
        if not isinstance(item.get('optimizer'),dict):raise ValueError('Checkpoint missing Adam state')
        p=dict(bundle['p'],allocator_limit_gib=protocol['allocator_limit_gib'],
               minimum_free_gpu_gib=protocol['minimum_free_gpu_gib'])
        phase='gpu_preflight';setup_gpu_once(p);decoder=bundle['decoder'].cuda()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        std=store.frames['std'].cuda()
        def assert_frozen():
            bridge.assert_frozen()
            if fm.frozen_digest(decoder)!=bundle['frozen_digest']:
                raise RuntimeError('Initial encoder or decoder context changed')
            if any(v.grad is not None for v in decoder.parameters() if not v.requires_grad):
                raise RuntimeError('Frozen decoder component acquired a gradient')
        def reset():
            torch.manual_seed(protocol['seed_per_arm']);torch.cuda.manual_seed_all(protocol['seed_per_arm'])
            decoder.load_state_dict(item['decoder'],strict=True);decoder.train()
            if runtime.digest(decoder)!=item['decoder_digest']:raise RuntimeError('Stage5 decoder failed to restore')
            bank=FreeCodes(list(range(168)),codes,'cuda')
            opt=optimizer_for(decoder,bank,p,config['decoder_lr'],config['code_lr'])
            # Adam keeps some step tensors on CPU; isolate every arm from the
            # loaded checkpoint so optimizer updates cannot mutate the source.
            opt.load_state_dict(copy.deepcopy(item['optimizer']))
            if not torch.equal(bank.stack(),codes) or not equal_tree(opt.state_dict(),item['optimizer']):
                raise RuntimeError('Stage5 codes or Adam state failed to restore')
            return bank,opt
        probe_batch=[{'start':s,'horizon':9} for s in (0,35,123,167)]
        bank,opt=reset();native=native_step(decoder,bank,bridge,store,std,opt,probe_batch,p,probe_rgb=True)
        snapshot={k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()};code_snapshot=bank.stack()
        del bank,opt
        bank,opt=reset();control=train_core.step(decoder,bank,bridge,store,std,opt,probe_batch,
            {'start':35,'horizon':9},p,0.0)
        model_diff=max_state_difference(snapshot,decoder.state_dict())
        code_diff=float((code_snapshot-bank.stack()).abs().max())
        if model_diff>1e-5 or code_diff>1e-5:raise RuntimeError('Zero-replay step does not reproduce native B step')
        del bank,opt
        bank,opt=reset();replay_probe=train_core.step(decoder,bank,bridge,store,std,opt,probe_batch,
            {'start':35,'horizon':9},p,0.1,probe=True)
        effect=max_state_difference(snapshot,decoder.state_dict())
        if effect<=1e-8:raise RuntimeError('Replay preflight did not affect the decoder update')
        assert_frozen();del bank,opt,snapshot,code_snapshot
        write(out/'PREFLIGHT.json',{'status':'passed','native_step_matched_zero_replay':True,
             'max_decoder_parameter_difference':model_diff,'max_code_difference':code_diff,
             'replay_changed_decoder_update_max':effect,'replay_rgb_gradient':replay_probe['replay']['rgb_to_code_and_decoder_gradients'],
             'discarded_updates':3,'arms_restore_checkpoint_and_adam_after_preflight':True,
             'codec_and_initial_context_frozen':True,'sampler':sampler})
        results={}
        for arm in protocol['arms']:
            phase='training_'+arm
            arm_dir=out/arm;arm_dir.mkdir()
            bank,opt=reset()
            assert_frozen()
            write(arm_dir/'STARTED.json',{'status':'running','arm':arm,'initial_decoder_digest':runtime.digest(decoder),
                 'initial_codes_match_checkpoint':True,'initial_adam_matches_checkpoint':True,'sampler':sampler})
            weight=0.0 if arm=='control' else protocol['anchor_replay_weight']
            with (arm_dir/'train.jsonl').open('x',encoding='utf8',buffering=1) as log:
                for sample in schedule:
                    if time.monotonic()-began>protocol['deadline_seconds']-60:raise TimeoutError('Pilot deadline near')
                    result=train_core.step(decoder,bank,bridge,store,std,opt,sample['batch'],sample['anchor'],p,weight)
                    log.write(json.dumps({'step':sample['step'],'elapsed_seconds':time.monotonic()-began,**result},allow_nan=False)+'\n')
                    if sample['step']==1 or sample['step']%15==0:
                        write(out/'STATUS.json',{'status':'running','phase':phase,'completed_updates':sample['step'],
                              'total_updates':189,'elapsed_seconds':time.monotonic()-began},replace=True)
            assert_frozen();digest=runtime.digest(decoder);path=arm_dir/'pilot_checkpoint.pt'
            with path.open('xb') as f:
                torch.save({'schema':'stage_b_anchor_replay_pilot_checkpoint_v1','arm':arm,'updates':189,
                    'starts':list(range(168)),'decoder':decoder.state_dict(),'decoder_digest':digest,
                    'codes':bank.stack(),'optimizer':opt.state_dict(),'source_checkpoint_sha256':sha(checkpoint),
                    'protocol_sha256':sha(ROOT/'protocol.json'),'sampler':sampler},f)
            disk=torch.load(path,map_location='cpu',weights_only=True)
            if (disk['decoder_digest']!=digest or not torch.equal(disk['codes'],bank.stack())
                    or disk['source_checkpoint_sha256']!=sha(checkpoint)):
                raise RuntimeError('Pilot checkpoint did not round-trip')
            phase='evaluating_'+arm
            result=evaluate(decoder,bank,bridge,store,p,protocol,groups,arm_dir,preview,anchor_metrics)
            assert_frozen()
            if runtime.digest(decoder)!=digest:raise RuntimeError('Decoder changed during evaluation')
            results[arm]=result
            write(arm_dir/'DONE.json',{'status':'complete','updates':189,'checkpoint_sha256':sha(path),
                 'evaluation_sha256':sha(arm_dir/'EVALUATION.json'),'decoder_digest':digest})
            del bank,opt,disk
        phase='comparison_and_visuals'
        panels=render_comparisons(out,out/'control',out/'anchor_replay',protocol,cross,anchor_manifest)
        control=results['control'];replay=results['anchor_replay']
        numeric=(replay['anchor_motion_mae']<=0.9*control['anchor_motion_mae']
                 and replay['h9_high_change_motion_mae']<=1.05*control['h9_high_change_motion_mae'])
        proxies=all(v['retention_fraction']>=.95 and v['maximum_consecutive_missing_frames']==0
                    for v in replay['dynamic_anchors'].values())
        summary={'status':'complete','source_checkpoint_sha256':sha(checkpoint),'sampler':sampler,
          'arms':{name:{k:results[name][k] for k in ('anchor_motion_mae','h9_motion_mae','h9_high_change_motion_mae',
                                                    'by_source_change_quartile','dynamic_anchors')} for name in protocol['arms']},
          'preliminary_numeric_gate':numeric,'preliminary_anchor_proxy_gate':proxies,'manual_visual_review_required':True,
          'teacher_accepted':False,'independent_native_replay':'not performed','panels':panels,
          'source_h9_result_sha256':sha(CROSS/'result/CROSS_H9.json'),
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.monotonic()-began,
          'training_signalled':False,'scope':protocol['limits']}
        write(out/'SUMMARY.json',summary)
        write(out/'STATUS.json',{'status':'complete','phase':'awaiting_visual_review','teacher_accepted':False},replace=True)
        print('PASS: matched pilot arms, no automatic teacher selection')
    except BaseException as error:
        failure={'status':'failed','phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
                 'training_signalled':False,'no_automatic_retry':True}
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':run()
