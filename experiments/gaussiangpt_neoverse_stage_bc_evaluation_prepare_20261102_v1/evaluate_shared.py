"""Full fixed-teacher posterior evaluation with independently decoded controls."""
import subprocess,time
import torch
from io_utils import ROOT,write,sha
from runtime import render,cpu,save,panel,digest
from data import evaluate_metrics
from support_decoder import select_state,support_metrics
from full_model import aggregate


def build_controls(student_codes,initial_codes,teacher_codes,starts):
    correct=student_codes.detach().cpu().contiguous()
    initial=initial_codes.detach().cpu().contiguous()
    teacher=teacher_codes.detach().cpu().contiguous()
    if starts!=list(range(168)) or any(x.shape!=(168,9,16) or x.dtype!=torch.float32 or
                                      not bool(torch.isfinite(x).all()) for x in (correct,initial,teacher)):
        raise RuntimeError('Invalid continuous codes or starts')
    neighbors=torch.tensor([s+1 if s<167 else 166 for s in starts],dtype=torch.long)
    return dict(correct=correct,initial_only=initial,teacher_free=teacher,swapped_window_codes=correct[neighbors])


def build_predictions(decoder,condition,controls,start,device):
    """Decode each distinct control explicitly; experiment mode must not alias them."""
    return {name:decoder(condition,controls[name][start].to(device))
            for name in ('correct','initial_only','swapped_window_codes')}


def validate_teacher_reference(source,teacher_evaluation,teacher_code_path,decoder):
    """Verify that copied teacher metrics refer to the identical decoder and codes."""
    reference=teacher_evaluation['source'];code_sha=sha(teacher_code_path)
    if (teacher_evaluation.get('status')!='complete' or teacher_evaluation.get('window_predictions')!=1512
            or source['decoder_digest']!=digest(decoder) or any(x.requires_grad for x in decoder.parameters())
            or reference['decoder_digest']!=source['decoder_digest']
            or reference['codec_sha256']!=source['codec_sha256']
            or reference['cache_manifest_sha256']!=source['cache_manifest_sha256']
            or reference.get('merged_codes_sha256',reference.get('codes_sha256'))!=code_sha
            or source['teacher_codes_sha256']!=code_sha):
        raise RuntimeError('Reference metrics are not bound to the identical frozen teacher')
    item=torch.load(teacher_code_path,map_location='cpu',weights_only=True)
    if item['starts']!=list(range(168)) or item['decoder_digest']!=source['decoder_digest']:
        raise RuntimeError('Teacher code ordering or decoder binding differs')
    future=[r for r in teacher_evaluation['rows'] if r['horizon']>0]
    rows={(r['start'],r['horizon']):r for r in future}
    if len(future)!=1512 or set(rows)!={(s,h) for s in range(168) for h in range(1,10)}:
        raise RuntimeError('Teacher evaluation coverage differs')
    return item['codes'],rows


def evaluate(decoder,bridge,store,p,out,source,mode,student_codes,initial_codes,teacher_evaluation_path,teacher_code_path):
    if mode not in ('C0','C1','C2'):raise ValueError('Expected a posterior variant')
    if sha(teacher_evaluation_path)!=source['teacher_evaluation_sha256']:
        raise RuntimeError('Teacher evaluation bytes changed')
    import json
    with open(teacher_evaluation_path,encoding='utf-8') as stream:teacher_evaluation=json.load(stream)
    teacher_codes,teacher_rows=validate_teacher_reference(source,teacher_evaluation,teacher_code_path,decoder)
    p=dict(p,eval_controls=['correct','initial_only','swapped_window_codes','actual_target','teacher_free'])
    began=time.monotonic();folder=out/'evaluation';folder.mkdir()
    for name in ['packets','references','snapshots','videos']:(folder/name).mkdir()
    decoder.eval();bridge.eval();rows=[];packets=[];actual_images={}
    def deadline():
        if time.monotonic()-began>p['evaluation_max_seconds']:raise RuntimeError('Evaluation deadline')
    with torch.no_grad():
        controls=build_controls(student_codes,initial_codes,teacher_codes,p['starts'])
        entry=save(folder/'action_codes.pt',dict(schema='fixed_teacher_posterior_codes_v1',source=source,starts=p['starts'],
            controls=controls,shape=[168,9,16],quantized=False,
            control_meaning=dict(correct='student posterior on real prefix',teacher_free='reviewed free-code teacher',
                initial_only='same student posterior on repeated initial state',
                swapped_window_codes='student code from adjacent window')))
        loaded=torch.load(folder/'action_codes.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(v,loaded['controls'][k]) for k,v in controls.items()):raise RuntimeError('Continuous code replay differs')
        controls=loaded['controls']
        code_stats={name:dict(rms=float(value.square().mean().sqrt()),across_window_std=float(value.std(0,unbiased=False).mean())) for name,value in controls.items()}

        def packet(name,images,gs,sh,record,start,h):
            value=dict(schema='learned_support_native_v1',name=name,mode=mode,start=start,horizon=h,frame=start+h,
                gs=cpu(gs),sh=cpu(sh),cameras=cpu(record['cameras']),intrinsics=cpu(record['intrinsics']),
                near=record['near'],far=record['far'],resolution_hw=[252,448],source=source)
            packets.append(dict(name=name,start=start,horizon=h,gs_count=len(gs),
                packet=save(folder/'packets'/(name+'.pt'),value),reference=save(folder/'references'/(name+'.pt'),cpu(images))))

        for start in p['starts']:
            deadline();w=store.get(start);diagnostic=start in p['diagnostic_starts']
            prediction=build_predictions(decoder,w['condition'],controls,start,'cuda')
            if diagnostic:
                frame_folder=folder/'snapshots'/f's{start:03d}';frame_folder.mkdir()
                sample=store.sample(start,0)
                images,gs,sh=render(bridge,w['states'][0],sample['record'],views=(0,1))
                packet(f's{start:03d}_h00_correct',images,gs,sh,sample['record'],start,0)
                panel(frame_folder/'frame_00.png',[(name,cpu(value)) for name,value in [
                    ('REAL TARGET',sample['record']['real']),('POSTERIOR Z',images[0]),('REPEATED INITIAL Z',images[0]),('ACTUAL STATE REFERENCE',images[0])]])
                rows.append(dict(start=start,horizon=0,frame=start,metrics={'correct':evaluate_metrics(images,w,sample,p)}))
                del sample,images,gs,sh
            for h in range(1,10):
                deadline();sample=store.sample(start,h);record=sample['record'];frame=start+h
                row=dict(start=start,horizon=h,frame=frame,metrics={},support={});rendered={};states={}
                for name in ['correct','initial_only','swapped_window_codes']:
                    pred=prediction[name]
                    state=select_state(w['condition'].coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_predicted_latent_nodes'])
                    states[name]=state
                    rendered[name]=render(bridge,state,record,views=(0,1))
                    images,gs,sh=rendered[name]
                    row['metrics'][name]=evaluate_metrics(images,w,sample,p)
                    row['support'][name]=support_metrics(pred['logits'][h-1],w['targets']['occupancy'][h-1])
                    if diagnostic and (name=='correct' or h==9):
                        stem=f's{start:03d}_h{h:02d}_{name}';packet(stem,images,gs,sh,record,start,h)
                        if name in ['correct','swapped_window_codes']:
                            path=folder/(stem+'_state.pt');save(path,dict(coordinates=cpu(state['coordinates']),features=cpu(state['features']),source=source))
                            disk=torch.load(path,map_location='cpu',weights_only=True)
                            replay,_,_=render(bridge,{k:disk[k].cuda() for k in ['coordinates','features']},record,views=(0,1))
                            difference=replay-images;mae=float(difference.abs().mean());psnr=float(-10*torch.log10(difference.square().mean().clamp_min(1e-12)))
                            if mae>p['replay_mae_max'] or psnr<p['replay_psnr_min']:raise RuntimeError('Latent disk replay failed')
                            row.setdefault('latent_replay',{})[name]=dict(mae=mae,psnr=psnr,passed=True)
                            del disk,replay,difference
                if frame not in actual_images or (diagnostic and h==9):
                    images,gs,sh=render(bridge,w['states'][h],record,views=(0,1));actual_images[frame]=cpu(images)
                else:images=actual_images[frame].cuda();gs=sh=None
                row['metrics']['actual_target']=evaluate_metrics(images,w,sample,p)
                reference=teacher_rows[(start,h)]
                row['metrics']['teacher_free']=reference['metrics']['correct']
                row['support']['teacher_free']=reference['support']['correct']
                if diagnostic and h==9:packet(f's{start:03d}_h09_actual_target',images,gs,sh,record,start,h)
                if diagnostic:
                    panel(frame_folder/f'frame_{h:02d}.png',[
                        ('REAL TARGET',cpu(record['real'])),('POSTERIOR Z',cpu(rendered['correct'][0][0])),
                        ('REPEATED INITIAL Z',cpu(rendered['initial_only'][0][0])),('ACTUAL STATE REFERENCE',cpu(images[0]))])
                    if h==9:
                        for view in (0,1):
                            target=cpu(record['real']) if view==0 else cpu(sample['teacher_targets'][1])
                            panel(folder/f's{start:03d}_comparison_v{view}.png',[
                                ('REAL / NATIVE TEACHER',target),('POSTERIOR Z',cpu(rendered['correct'][0][view])),
                                ('REPEATED INITIAL Z',cpu(rendered['initial_only'][0][view])),
                                ('NEIGHBOR WINDOW Z',cpu(rendered['swapped_window_codes'][0][view]))])
                rows.append(row)
                del sample,record,rendered,states,images,gs,sh,state,pred
            if diagnostic:
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-i',str(frame_folder/'frame_%02d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'videos'/f's{start:03d}_slow5fps.mp4')],check=True,timeout=120)
                subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'videos'/f's{start:03d}_slow5fps.mp4'),
                    '-vf','crop=896:252:0:24','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
                    str(folder/'videos'/f's{start:03d}_source_vs_posterior_clean_5fps.mp4')],check=True,timeout=120)
            if start%8==0 or start==167:
                write(out/'EVAL_STATUS.json',dict(status='running',completed_windows=start+1,total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
            del prediction,w
        future=[row for row in rows if row['horizon']>0]
        if len(future)!=1512 or {r['frame'] for r in future}!=set(range(1,177)):raise RuntimeError('Evaluation coverage mismatch')
        summary=aggregate(future,p['eval_controls'])
        support_rows=[dict(frame=r['frame'],metrics={name:dict(metrics=value) for name,value in r['support'].items()}) for r in future]
        support_summary=aggregate(support_rows,['correct','initial_only','swapped_window_codes','teacher_free'])
        segments={f'{lo:03d}_{hi:03d}':aggregate([r for r in future if lo<=r['frame']<=hi],p['eval_controls'])
                  for lo,hi in [(1,44),(45,88),(89,132),(133,176)]}
    bridge.assert_frozen()
    if digest(decoder)!=source['decoder_digest']:raise RuntimeError('Fixed teacher decoder changed during evaluation')
    write(folder/'PACKETS.json',dict(mode=mode,source=source,rows=packets))
    write(folder/'EVALUATION.json',dict(status='complete',mode=mode,source=source,rows=rows,metrics=summary,
        support=support_summary,temporal_segments=segments,code_stats=code_stats,action_codes=entry,
        training_windows=168,window_predictions=1512,unique_future_frames=176,
        all_source_frames_preserved=True,decoder_future_geometry_input=False,
        initial_only_reference='Same posterior on repeated initial state, freshly decoded by the fixed teacher',
        teacher_free_reference='Metrics from the hash-bound complete free-code teacher evaluation on identical decoder and state cache',
        serialized_code_replay='passed',packets_sha256=sha(folder/'PACKETS.json'),
        action_success='pending visual review',scope='posterior action reconstruction on one TRAIN video; no AR or cross-video claim',elapsed_seconds=time.monotonic()-began))
    write(out/'EVAL_STATUS.json',dict(status='complete',completed_windows=168,total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
