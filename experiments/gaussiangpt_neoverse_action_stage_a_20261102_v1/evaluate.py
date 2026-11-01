"""Derived from the hash-bound 9560 evaluator; controls are explicitly relabeled."""
import subprocess,time
import torch
from io_utils import ROOT,write,sha
from runtime import render,cpu,save,panel
from data import evaluate_metrics
from support_decoder import select_state,support_metrics
from full_model import aggregate


def build_controls(optimized,baseline_codes,starts):
    correct=optimized.detach().cpu().contiguous()
    if starts!=list(range(168)) or correct.shape!=(168,9,16) or not bool(torch.isfinite(correct).all()):
        raise RuntimeError('Invalid optimized continuous codes or starts')
    neighbors=torch.tensor([s+1 if s<167 else 166 for s in starts],dtype=torch.long)
    return dict(correct=correct,initial_only=baseline_codes['initial_only'],
                baseline_qformer=baseline_codes['correct'],swapped_window_codes=correct[neighbors])


def evaluate(decoder,bridge,store,p,out,source,mode,optimized,baseline_codes,baseline_rows):
    began=time.monotonic();folder=out/'evaluation';folder.mkdir()
    for name in ['packets','references','snapshots','videos']:(folder/name).mkdir()
    decoder.eval();bridge.eval();rows=[];packets=[];actual_images={}
    def deadline():
        if time.monotonic()-began>p['evaluation_max_seconds']:raise RuntimeError('Evaluation deadline')
    with torch.no_grad():
        controls=build_controls(optimized,baseline_codes,p['starts'])
        entry=save(folder/'action_codes.pt',dict(schema='stage_a_free_codes_v1',source=source,starts=p['starts'],
            controls=controls,shape=[168,9,16],quantized=False,
            control_meaning=dict(correct='optimized free code',baseline_qformer='9560-step Q-Former code',
                initial_only='9560-step Q-Former with repeated initial state',
                swapped_window_codes='optimized code from adjacent window')))
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
            prediction={'correct':decoder(w['condition'],controls['correct'][start].cuda())}
            prediction['initial_only']=decoder(w['condition'],controls['initial_only'][start].cuda()) if mode=='posterior' else prediction['correct']
            prediction['swapped_window_codes']=decoder(w['condition'],controls['swapped_window_codes'][start].cuda())
            if diagnostic:
                frame_folder=folder/'snapshots'/f's{start:03d}';frame_folder.mkdir()
                sample=store.sample(start,0)
                images,gs,sh=render(bridge,w['states'][0],sample['record'],views=(0,1))
                packet(f's{start:03d}_h00_correct',images,gs,sh,sample['record'],start,0)
                panel(frame_folder/'frame_00.png',[(name,cpu(value)) for name,value in [
                    ('REAL TARGET',sample['record']['real']),('OPTIMIZED FREE Z',images[0]),('BASELINE INITIAL ONLY',images[0]),('ACTUAL STATE REFERENCE',images[0])]])
                rows.append(dict(start=start,horizon=0,frame=start,metrics={'correct':evaluate_metrics(images,w,sample,p)}))
                del sample,images,gs,sh
            for h in range(1,10):
                deadline();sample=store.sample(start,h);record=sample['record'];frame=start+h
                row=dict(start=start,horizon=h,frame=frame,metrics={},support={});rendered={};states={}
                for name in ['correct','initial_only','swapped_window_codes']:
                    pred=prediction[name]
                    state=select_state(w['condition'].coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_predicted_latent_nodes'])
                    states[name]=state
                    rendered[name]=rendered['correct'] if name=='initial_only' and mode=='initial_only' else render(bridge,state,record,views=(0,1))
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
                baseline=baseline_rows[(start,h)]
                row['metrics']['baseline_qformer']=baseline['metrics']['correct']
                row['support']['baseline_qformer']=baseline['support']['correct']
                if diagnostic and h==9:packet(f's{start:03d}_h09_actual_target',images,gs,sh,record,start,h)
                if diagnostic:
                    panel(frame_folder/f'frame_{h:02d}.png',[
                        ('REAL TARGET',cpu(record['real'])),('OPTIMIZED FREE Z',cpu(rendered['correct'][0][0])),
                        ('9560 Q-FORMER: INITIAL ONLY',cpu(rendered['initial_only'][0][0])),('ACTUAL STATE REFERENCE',cpu(images[0]))])
                    if h==9:
                        for view in (0,1):
                            target=cpu(record['real']) if view==0 else cpu(sample['teacher_targets'][1])
                            panel(folder/f's{start:03d}_comparison_v{view}.png',[
                                ('REAL / NATIVE TEACHER',target),('OPTIMIZED FREE Z',cpu(rendered['correct'][0][view])),
                                ('9560 Q-FORMER: INITIAL ONLY',cpu(rendered['initial_only'][0][view])),
                                ('NEIGHBOR WINDOW FREE Z',cpu(rendered['swapped_window_codes'][0][view]))])
                rows.append(row)
                del sample,record,rendered,states,images,gs,sh,state,pred
            if diagnostic:
                subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-i',str(frame_folder/'frame_%02d.png'),
                    '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'videos'/f's{start:03d}_slow5fps.mp4')],check=True,timeout=120)
                subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'videos'/f's{start:03d}_slow5fps.mp4'),
                    '-vf','crop=896:252:0:24','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
                    str(folder/'videos'/f's{start:03d}_source_vs_free_clean_5fps.mp4')],check=True,timeout=120)
            if start%8==0 or start==167:
                write(out/'EVAL_STATUS.json',dict(status='running',completed_windows=start+1,total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
            del prediction,w
        future=[row for row in rows if row['horizon']>0]
        if len(future)!=1512 or {r['frame'] for r in future}!=set(range(1,177)):raise RuntimeError('Evaluation coverage mismatch')
        summary=aggregate(future,p['eval_controls'])
        support_rows=[dict(frame=r['frame'],metrics={name:dict(metrics=value) for name,value in r['support'].items()}) for r in future]
        support_summary=aggregate(support_rows,['correct','initial_only','swapped_window_codes','baseline_qformer'])
        segments={f'{lo:03d}_{hi:03d}':aggregate([r for r in future if lo<=r['frame']<=hi],p['eval_controls'])
                  for lo,hi in [(1,44),(45,88),(89,132),(133,176)]}
    bridge.assert_frozen()
    write(folder/'PACKETS.json',dict(mode=mode,source=source,rows=packets))
    write(folder/'EVALUATION.json',dict(status='complete',mode=mode,source=source,rows=rows,metrics=summary,
        support=support_summary,temporal_segments=segments,code_stats=code_stats,action_codes=entry,
        training_windows=168,window_predictions=1512,unique_future_frames=176,
        all_source_frames_preserved=True,decoder_future_geometry_input=False,
        serialized_code_replay='passed',packets_sha256=sha(folder/'PACKETS.json'),
        action_success='pending visual review',scope='full TRAIN clip per-window free-code fit; no deployable encoder or AR evaluation',elapsed_seconds=time.monotonic()-began))
    write(out/'EVAL_STATUS.json',dict(status='complete',completed_windows=168,total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
