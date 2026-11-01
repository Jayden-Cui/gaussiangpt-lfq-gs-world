"""Two-window capacity training: predicted support, free/fixed continuous codes."""
import argparse,copy,json,math,os,shutil,subprocess,time,traceback
from pathlib import Path
import torch
from io_utils import ROOT,read,write,sha,verify
from runtime import build_model,load_data,digest,render,cpu,save,png,panel,metrics
from support_decoder import state_loss,select_state,support_metrics
from core import image_losses


def gradients(model):
    groups={}
    for name,module in [('occupancy',model.occupancy),('features',model.base.features),
                        ('decoder',model.base.decoder_backbone),('geometry',model.geometry)]:
        values=[p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        groups[name]=float(torch.stack(values).sum().sqrt()) if values else 0.
    groups['codes']=float(model.codes.grad.norm()) if model.codes.grad is not None else 0.
    return groups


def update(model,bridge,windows,std,optimizer,phase,index,p):
    optimizer.zero_grad(set_to_none=True);rows=[]
    horizon=1+(index%9)
    for k,window in enumerate(windows):
        prediction=model(window['condition'],model.action_codes(k))
        latent,terms=state_loss(prediction,window['targets'],std)
        loss=latent
        row=dict(start=window['start'],latent_loss=float(latent.detach()),
                 latent_terms={a:float(v.detach()) for a,v in terms.items()})
        if phase=='render':
            state=select_state(window['condition'].coordinates,prediction['features'][horizon-1],
                               prediction['logits'][horizon-1],p['max_predicted_latent_nodes'])
            sample=window['samples'][horizon]
            images,gs,sh=render(bridge,state,sample['record'])
            reconstruction,_=image_losses(images[0],sample['record']['real'],window['roi'],window['motion'][horizon])
            loss=reconstruction+.1*latent
            row.update(horizon=horizon,rgb_loss=float(reconstruction.detach()),predicted_nodes=len(state['coordinates']),gs_count=len(gs))
            del images,gs,sh,state,reconstruction
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss')
        row['loss']=float(loss.detach());(loss/len(windows)).backward();rows.append(row)
        del prediction,latent,terms,loss
    parameters=[v for v in model.parameters() if v.requires_grad]
    if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in parameters):
        raise RuntimeError('Nonfinite learned-decoder gradient')
    norms=gradients(model)
    clipped=float(torch.nn.utils.clip_grad_norm_(parameters,p['grad_clip']))
    optimizer.step()
    return dict(windows=rows,gradients=norms,preclip_gradient_norm=clipped,clip_triggered=clipped>p['grad_clip'])


def export_evaluation(model,bridge,windows,p,out,source):
    began=time.monotonic();model.eval();bridge.eval();records=[];packets=[]
    evaluation=out/'evaluation';evaluation.mkdir()
    for name in ['packets','references','snapshots','videos']:(evaluation/name).mkdir()
    codes=torch.stack([model.action_codes(i) for i in range(2)]).detach()
    save(evaluation/'action_codes.pt',dict(schema='free_window_codes_v1',mode=model.mode,codes=cpu(codes),source=source,
        shape=[2,9,16],quantized=False,posterior_encoder=False))
    def save_packet(name,images,gs,sh,record,start,h):
        packet=dict(schema='learned_support_native_v1',name=name,mode=model.mode,start=start,horizon=h,frame=start+h,
                    gs=cpu(gs),sh=cpu(sh),cameras=cpu(record['cameras']),intrinsics=cpu(record['intrinsics']),
                    near=record['near'],far=record['far'],resolution_hw=[252,448],source=source)
        packets.append(dict(name=name,start=start,horizon=h,gs_count=len(gs),
            packet=save(evaluation/'packets'/(name+'.pt'),packet),
            reference=save(evaluation/'references'/(name+'.pt'),cpu(images))))
    with torch.no_grad():
        for k,w in enumerate(windows):
            initial=w['states'][0];c=w['condition']
            correct=model(c,codes[k]);swapped=model(c,codes[1-k])
            frame_folder=evaluation/'snapshots'/f"s{w['start']:03d}";frame_folder.mkdir()
            for h in range(10):
                if time.monotonic()-began>p['evaluation_max_seconds']:raise RuntimeError('Evaluation deadline')
                sample=w['samples'][h];record=sample['record'];images_by_control={};row=dict(start=w['start'],horizon=h,frame=w['start']+h)
                if h==0:
                    images,gs,sh=render(bridge,initial,record,views=(0,1))
                    images_by_control['correct']=cpu(images);images_by_control['actual_target']=cpu(images)
                    row['metrics']={'correct':metrics(images,w,h,p)}
                    save_packet(f"s{w['start']:03d}_h00_correct",images,gs,sh,record,w['start'],h)
                    del images,gs,sh
                else:
                    row['metrics']={};row['support']={}
                    for name,pred in [('correct',correct),('swapped_window_codes',swapped)]:
                        state=select_state(c.coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_predicted_latent_nodes'])
                        images,gs,sh=render(bridge,state,record,views=(0,1))
                        images_by_control[name]=cpu(images)
                        row['metrics'][name]=metrics(images,w,h,p)
                        row['support'][name]=support_metrics(pred['logits'][h-1],w['targets']['occupancy'][h-1])
                        if name=='correct' or h==9:
                            stem=f"s{w['start']:03d}_h{h:02d}_{name}"
                            save_packet(stem,images,gs,sh,record,w['start'],h)
                            state_path=evaluation/(stem+'_state.pt')
                            save(state_path,dict(coordinates=cpu(state['coordinates']),features=cpu(state['features']),source=source))
                            # Rebuild sparse state from saved CPU values, never feed targets.
                            disk=torch.load(state_path,map_location='cpu',weights_only=True)
                            replay,_,_=render(bridge,{key:disk[key].cuda() for key in ('coordinates','features')},record,views=(0,1))
                            error=(replay-images).abs();mae=float(error.mean())
                            psnr=float(-10*torch.log10((replay-images).square().mean().clamp_min(1e-12)))
                            if mae>p['replay_mae_max'] or psnr<p['replay_psnr_min']:raise RuntimeError('Latent disk replay failed')
                            row.setdefault('latent_replay',{})[name]=dict(mae=mae,psnr=psnr,passed=True)
                            del replay,error,disk
                        del state,images,gs,sh
                    images,gs,sh=render(bridge,w['states'][h],record,views=(0,1))
                    images_by_control['actual_target']=cpu(images)
                    row['metrics']['actual_target']=metrics(images,w,h,p)
                    if h==9:save_packet(f"s{w['start']:03d}_h09_actual_target",images,gs,sh,record,w['start'],h)
                    del images,gs,sh
                panel(frame_folder/f'frame_{h:02d}.png',[
                    ('REAL TARGET',cpu(record['real'])),('PREDICTED SUPPORT + FREE/FIXED Z',images_by_control['correct'][0]),
                    ('ACTUAL FUTURE STATE REFERENCE',images_by_control['actual_target'][0])])
                if h==9:
                    for view in (0,1):
                        target=cpu(record['real']) if view==0 else cpu(sample['teacher_targets'][1])
                        panel(evaluation/f"s{w['start']:03d}_comparison_v{view}.png",[
                            ('REAL / NATIVE TEACHER',target),('ACTUAL FUTURE STATE',images_by_control['actual_target'][view]),
                            ('PREDICTED SUPPORT + CORRECT Z',images_by_control['correct'][view]),
                            ('SAME INITIAL + SWAPPED Z',images_by_control['swapped_window_codes'][view])])
                records.append(row)
            subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5','-i',str(frame_folder/'frame_%02d.png'),
                '-c:v','libx264','-pix_fmt','yuv420p',str(evaluation/'videos'/f"s{w['start']:03d}_slow5fps.mp4")],
                check=True,timeout=120)
            del correct,swapped
    bridge.assert_frozen()
    write(evaluation/'PACKETS.json',dict(mode=model.mode,source=source,rows=packets))
    write(evaluation/'EVALUATION.json',dict(status='complete',mode=model.mode,source=source,rows=records,
        windows=2,future_states=18,initial_states=2,all_source_frames_preserved=True,
        inference_uses_target_coordinates=False,inference_uses_target_counts=False,
        candidate_set_initial_only=True,packets_sha256=sha(evaluation/'PACKETS.json'),
        action_success='pending visual review',elapsed_seconds=time.monotonic()-began))


def main(mode):
    p=verify();out=ROOT/mode;out.mkdir(exist_ok=False);began=time.monotonic();context={'phase':'initializing'}
    write(out/'STARTED.json',dict(pid=os.getpid(),mode=mode))
    try:
        if mode not in p['modes'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Invalid mode/GPU')
        if shutil.which('ffmpeg') is None:raise RuntimeError('Existing ffmpeg is required for full-window videos')
        torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info()
        if free<p['minimum_free_gpu_gib']*2**30:raise RuntimeError('Insufficient free GPU memory')
        torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
        torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
        model=build_model(p,mode,'cuda')
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        windows,std,binding=load_data(p,model)
        initial_digest=digest(model)
        if initial_digest!=read(ROOT/'CPU_TESTS.json')['common_initial_digest']:
            raise RuntimeError('Actual model differs from witnessed common initialization')
        source=dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
                    codec_sha256=p['codec_checkpoint_sha256'],codec_tensor_digest=bridge.precision.mode_state_digest,
                    common_model_initial_digest=initial_digest,initial_checkpoint_sha256=p['common_checkpoint_sha256'],
                    candidate_radius=p['candidate_radius'],initial_only_candidates=True,future_inputs_to_decoder=False,
                    code_shape=p['code_shape'],quantized=False,posterior_encoder_used=False,
                    state_sources={str(w['start']):[binding.rows[f]['state_sha256'] for f in range(w['start'],w['start']+10)] for w in windows})
        geometry={str(w['start']):dict(initial_nodes=len(w['states'][0]['coordinates']),candidate_nodes=len(w['condition'].coordinates),
                  coverage=[dict(horizon=h,target_nodes=len(w['states'][h]['coordinates']),missing_nodes=0) for h in range(1,10)],
                  transfer=w['condition'].report) for w in windows}
        write(out/'PROVENANCE.json',dict(source=source,geometry=geometry))
        frozen_encoder=digest(model.base.encoder_backbone)
        initial_state={k:cpu(v).clone() for k,v in model.state_dict().items()}
        parameters=[v for v in model.parameters() if v.requires_grad]
        def optimizer():return torch.optim.Adam(parameters,lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'])
        opt=optimizer();model.train()
        preflight=[]
        for index in range(2):
            row=update(model,bridge,windows,std,opt,'render',index,p);preflight.append(row)
        if mode=='free16' and preflight[1]['gradients']['codes']<=0:raise RuntimeError('No free-code gradient after second update')
        for name in ('occupancy','features','decoder'):
            if preflight[1]['gradients'][name]<=0:raise RuntimeError('No gradient in '+name)
        bridge.assert_frozen();model.load_state_dict(initial_state,strict=True)
        if digest(model)!=initial_digest:raise RuntimeError('Preflight reset failed')
        write(out/'PREFLIGHT.json',dict(status='passed',discarded_updates=2,rows=preflight,source=source,
                                      restored_exact_initialization=True,codec_frozen=True))
        del initial_state,opt
        opt=optimizer();model.train()
        with (out/'train.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for step in range(p['total_steps']):
                if time.monotonic()-began>p['worker_max_seconds']:raise RuntimeError('Training time budget exceeded')
                phase='latent' if step<p['latent_steps'] else 'render'
                index=step if phase=='latent' else step-p['latent_steps']
                context=dict(phase=phase,completed_updates=step)
                progress=step/max(p['total_steps']-1,1)
                lr=p['final_lr']+.5*(p['lr']-p['final_lr'])*(1+math.cos(math.pi*progress))
                opt.param_groups[0]['lr']=lr
                row=update(model,bridge,windows,std,opt,phase,index,p)
                row.update(step=step+1,phase=phase,lr=lr,elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if step==0 or (step+1)%p['progress_every']==0:
                    write(out/'STATUS.json',dict(status='running',mode=mode,phase=phase,completed_updates=step+1,
                        total_updates=p['total_steps'],mean_loss=sum(r['loss'] for r in row['windows'])/2,
                        elapsed_seconds=time.monotonic()-began),replace=True)
                if step+1 in p['checkpoints']:
                    bridge.assert_frozen()
                    save(out/f'step_{step+1:04d}.pt',dict(schema='learned_support_capacity_checkpoint_v1',mode=mode,
                        step=step+1,model=model.state_dict(),optimizer=opt.state_dict(),source=source))
        if digest(model.base.encoder_backbone)!=frozen_encoder:raise RuntimeError('Frozen initial encoder changed')
        bridge.assert_frozen()
        write(out/'TRAIN_DONE.json',dict(status='complete',mode=mode,steps=p['total_steps'],source=source,
            final_model_digest=digest(model),checkpoint_sha256=sha(out/f"step_{p['total_steps']:04d}.pt"),
            codec_frozen=True,initial_encoder_frozen=True,elapsed_seconds=time.monotonic()-began,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        context={'phase':'evaluating'}
        write(out/'STATUS.json',dict(status='running',mode=mode,phase='evaluation',completed_updates=p['total_steps']),replace=True)
        export_evaluation(model,bridge,windows,p,out,source)
        write(out/'STATUS.json',dict(status='complete',mode=mode,completed_updates=p['total_steps'],
              native_replay='pending',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(out/'FAILED.json',dict(status='failed',mode=mode,context=context,error=repr(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True);main(parser.parse_args().mode)
