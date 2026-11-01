"""Full-window controls using one immutable learned-support decoder."""
import subprocess,time
from pathlib import Path
import torch
from io_utils import ROOT,read,write,sha
from runtime import render,cpu,save,panel,metrics
from support_decoder import select_state,support_metrics


def evaluate(teacher,posterior,bridge,windows,p,out,source):
    began=time.monotonic();teacher.eval();posterior.eval();bridge.eval()
    folder=out/'evaluation';folder.mkdir()
    for name in ['packets','references','snapshots','videos']:(folder/name).mkdir()
    rows=[];packets=[]
    previous=read(Path(p['parent_root'])/'free16/evaluation/EVALUATION.json')
    previous={(r['start'],r['horizon']):r for r in previous['rows']}
    controls=[name for name in p['eval_controls'] if name!='actual_target']
    with torch.no_grad():
        correct=posterior([w['memories'] for w in windows])['codes']
        initial_only=posterior([w['initial_memories'] for w in windows])['codes']
        learned=teacher.codes.detach()
        code_dict=dict(correct=correct,free_teacher=learned,initial_only=initial_only,
                       zero_codes=torch.zeros_like(correct),swapped_window_codes=correct.flip(0))
        save(folder/'action_codes.pt',dict(schema='qformer_continuous_codes_v1',source=source,
            controls={k:cpu(v) for k,v in code_dict.items()},quantized=False,shape=[2,9,16]))
        disk=torch.load(folder/'action_codes.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(cpu(v),disk['controls'][k]) for k,v in code_dict.items()):
            raise RuntimeError('Serialized continuous codes changed')
        code_dict={k:v.cuda() for k,v in disk['controls'].items()}
        rms=learned.square().mean().sqrt()
        code_metrics={name:dict(rmse=float((codes-learned).square().mean().sqrt()),
            normalized_rmse=float((codes-learned).square().mean().sqrt()/rms)) for name,codes in code_dict.items()}

        def packet(name,images,gs,sh,record,start,h):
            value=dict(schema='learned_support_native_v1',name=name,mode=p['modes'][0],start=start,
                horizon=h,frame=start+h,gs=cpu(gs),sh=cpu(sh),cameras=cpu(record['cameras']),
                intrinsics=cpu(record['intrinsics']),near=record['near'],far=record['far'],
                resolution_hw=[252,448],source=source)
            packets.append(dict(name=name,start=start,horizon=h,gs_count=len(gs),
                packet=save(folder/'packets'/(name+'.pt'),value),
                reference=save(folder/'references'/(name+'.pt'),cpu(images))))

        for k,w in enumerate(windows):
            prediction={name:teacher(w['condition'],code_dict[name][k]) for name in controls}
            frame_folder=folder/'snapshots'/f"s{w['start']:03d}";frame_folder.mkdir()
            for h in range(10):
                if time.monotonic()-began>p['evaluation_max_seconds']:raise RuntimeError('Evaluation deadline')
                record=w['samples'][h]['record'];images_by_control={}
                row=dict(start=w['start'],horizon=h,frame=w['start']+h,metrics={},support={})
                if h==0:
                    images,gs,sh=render(bridge,w['states'][0],record,views=(0,1))
                    images_by_control={name:cpu(images) for name in p['eval_controls']}
                    row['metrics']['correct']=metrics(images,w,h,p)
                    packet(f"s{w['start']:03d}_h00_correct",images,gs,sh,record,w['start'],h)
                    del images,gs,sh
                else:
                    for name in p['eval_controls']:
                        if name=='actual_target':state=w['states'][h]
                        else:
                            pred=prediction[name]
                            state=select_state(w['condition'].coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_predicted_latent_nodes'])
                            row['support'][name]=support_metrics(pred['logits'][h-1],w['targets']['occupancy'][h-1])
                        images,gs,sh=render(bridge,state,record,views=(0,1))
                        images_by_control[name]=cpu(images);row['metrics'][name]=metrics(images,w,h,p)
                        if name=='free_teacher':
                            old=previous[(w['start'],h)]['metrics']['correct']['metrics']
                            if any(abs(value-old[key])>p['teacher_metric_tolerance'] for key,value in row['metrics'][name]['metrics'].items()):
                                raise RuntimeError('Frozen free-teacher reference changed')
                        if name=='correct' or h==9:
                            stem=f"s{w['start']:03d}_h{h:02d}_{name}"
                            packet(stem,images,gs,sh,record,w['start'],h)
                            if name=='correct' or name=='swapped_window_codes':
                                state_file=folder/(stem+'_state.pt')
                                save(state_file,dict(coordinates=cpu(state['coordinates']),features=cpu(state['features']),source=source))
                                saved=torch.load(state_file,map_location='cpu',weights_only=True)
                                replay,_,_=render(bridge,{key:saved[key].cuda() for key in ('coordinates','features')},record,views=(0,1))
                                error=replay-images;mae=float(error.abs().mean())
                                psnr=float(-10*torch.log10(error.square().mean().clamp_min(1e-12)))
                                if mae>p['replay_mae_max'] or psnr<p['replay_psnr_min']:raise RuntimeError('Latent replay failed')
                                row.setdefault('latent_replay',{})[name]=dict(mae=mae,psnr=psnr,passed=True)
                                del replay,error,saved
                        del images,gs,sh,state
                panel(frame_folder/f'frame_{h:02d}.png',[
                    ('REAL TARGET',cpu(record['real'])),('Q-FORMER CONTINUOUS Z',images_by_control['correct'][0]),
                    ('FROZEN DECODER + FREE Z REFERENCE',images_by_control['free_teacher'][0]),
                    ('ACTUAL FUTURE STATE REFERENCE',images_by_control['actual_target'][0])])
                if h==9:
                    for view in (0,1):
                        target=cpu(record['real']) if view==0 else cpu(w['samples'][h]['teacher_targets'][1])
                        panel(folder/f"s{w['start']:03d}_comparison_v{view}.png",[
                            ('REAL / NATIVE TEACHER',target),('FREE Z REFERENCE',images_by_control['free_teacher'][view]),
                            ('Q-FORMER CONTINUOUS Z',images_by_control['correct'][view]),
                            ('SAME INITIAL + SWAPPED Z',images_by_control['swapped_window_codes'][view])])
                        panel(folder/f"s{w['start']:03d}_controls_v{view}.png",[
                            ('ACTUAL FUTURE STATE',images_by_control['actual_target'][view]),
                            ('Q-FORMER CONTINUOUS Z',images_by_control['correct'][view]),
                            ('POSTERIOR SEES REPEATED S0 ONLY',images_by_control['initial_only'][view]),
                            ('SAME DECODER + ZERO Z',images_by_control['zero_codes'][view])])
                rows.append(row)
            subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-i',str(frame_folder/'frame_%02d.png'),
                '-c:v','libx264','-pix_fmt','yuv420p',str(folder/'videos'/f"s{w['start']:03d}_slow5fps.mp4")],check=True,timeout=120)
            del prediction
    bridge.assert_frozen()
    write(folder/'PACKETS.json',dict(mode=p['modes'][0],source=source,rows=packets))
    write(folder/'EVALUATION.json',dict(status='complete',mode=p['modes'][0],source=source,rows=rows,
        code_metrics=code_metrics,windows=2,future_states=18,initial_states=2,free_reference_matches_parent=True,
        posterior_uses_prefix_states=True,decoder_uses_only_initial_and_codes=True,quantized=False,
        all_source_frames_preserved=True,serialized_code_replay='passed',packets_sha256=sha(folder/'PACKETS.json'),
        action_success='pending visual review',elapsed_seconds=time.monotonic()-began))
