"""One bounded B1/H9 real-GS RGB-gradient witness; no optimizer updates."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch
from gs_bridge import NativeGSBridge,STATE_ROOT,MOTION_ROOT,V4_ROOT,verify_window_sources,sha
from sampling import Segment,Window
from online_state import load_online_window
from aligned_ldm import make_condition
from native_aligned_ldm import NativeAlignedLDM

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def norm(module):
    values=[p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.
def finite_nonzero(value,label):
    if not torch.isfinite(torch.as_tensor(value)).all() or not bool((torch.as_tensor(value)>0).all()):
        raise AssertionError('Missing/nonfinite real-RGB gradient: '+label)

def run(root):
    started=time.monotonic();stage='initialization';details=[]
    def progress(**extra):
        write(root/'GPU_RENDER_PROGRESS.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started,details=details,**extra))
    try:
        torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
        free,total=torch.cuda.mem_get_info()
        if free<55*2**30:raise RuntimeError('Less than55GiB free before witness; do not disturb shared jobs')
        torch.manual_seed(20261030)
        bridge=NativeGSBridge().train()
        spec=Window(Segment('buildingblock_tower_17_176',0,176),40)
        sources=verify_window_sources(spec)
        calibration=MOTION_ROOT/'states'/spec.clip/'000.pt'
        row=next(v for v in json.loads((V4_ROOT/'cache_manifest.json').read_text()) if v['path']==str(calibration))
        if sha(calibration)!=row['sha256']:raise ValueError('TRAIN calibration source mismatch')
        reference=torch.load(calibration,map_location='cpu',weights_only=True)['features']
        model=NativeAlignedLDM(coordinate_scale=bridge.codec.grid*bridge.codec.ae.stride,checkpoint_blocks=True).cuda().train()
        model.initialize_output_units(reference.cuda())
        stage='online_state_encoder';progress()
        record=load_online_window(STATE_ROOT,MOTION_ROOT,spec,bridge.codec,'cuda')
        states=record['window']
        for state in states:
            if not state['features'].requires_grad:raise AssertionError('Online encoder was detached')
            state['features'].retain_grad()
        stage='native64_ldm';progress()
        condition=make_condition(states[0],radius=6)
        candidates={tuple(c) for c in condition.coordinates.detach().cpu().tolist()}
        for state in states:
            if any(tuple(c) not in candidates for c in state['coordinates'].detach().cpu().tolist()):
                raise RuntimeError('Real future outside initial candidate domain; no expansion from targets')
        z=model.encode_windows([states]);z['z_quantized'].retain_grad()
        predictions=model.decode_latents(z['z_quantized'],[condition],initial_contexts=z['initial_contexts'])[0]
        if len(predictions)!=10:raise AssertionError('Must reconstruct initial plus nine futures')
        stage='native_rgb_backward'
        progress(latent_candidates=len(condition.coordinates),state_counts=[len(t['coordinates']) for t in states])
        losses=[]
        for h,prediction in enumerate(predictions):
            image=bridge.render_prediction(prediction['coordinates'],prediction['features'],prediction['occupancy_logits'],record['records'][h])
            target=record['records'][h]['real']
            loss=torch.nn.functional.l1_loss(image*2-1,target*2-1)/10
            if not bool(torch.isfinite(loss)):raise AssertionError('Nonfinite native RGB L1')
            loss.backward(retain_graph=h<9)
            losses.append(float(loss.detach())*10)
            details.append(dict(horizon=h,source_frame=spec.frames[h],rgb_l1=losses[-1],
                gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,**bridge.last_stats))
            progress()
            print('REAL_RGB',h,'frame',spec.frames[h],'loss',round(losses[-1],6),'GS',bridge.last_stats['decoded_gaussians'],flush=True)
            del image,loss
        gradients=dict(query_per_horizon=model.queries.weight.grad.detach().norm(dim=1).cpu().tolist(),
            quantized_per_horizon=z['z_quantized'].grad.detach().norm(dim=2).cpu().tolist()[0],
            online_state_per_frame=[float(s['features'].grad.norm()) if s['features'].grad is not None else 0. for s in states],
            motion_encoder=norm(model.encoder_backbone),motion_decoder=norm(model.decoder_backbone),
            motion_occupancy=norm(model.occupancy),native_encoder=norm(bridge.codec.ae.encoder),
            native_decoder=norm(bridge.codec.ae.decoder))
        for key,value in gradients.items():finite_nonzero(value,key)
        peak=torch.cuda.max_memory_allocated()/2**30
        if peak>48:raise RuntimeError('Witness peak exceeded48GiB')
        stage='passed'
        result=dict(status='passed',scope='B1_all10_native_RGB_L1_backward_only',optimizer_updates=0,
            native_state_quantizer='Identity',native_codec_weights_require_grad=True,
            native_codec_mode='eval to disable target forcing and freeze BN statistics, autograd enabled',
            source_frames=list(spec.frames),image_shape=[252,448,3],latent_shape=list(z['z_quantized'].shape),
            ids_shape=list(z['indices'].shape),gradients=gradients,mean_rgb_l1=sum(losses)/10,
            details=details,gpu_peak_gib=peak,elapsed_seconds=time.monotonic()-started,
            calibration_source=row,source_files=sources,native_binding=bridge.binding,
            code_sha256={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py'))},
            limits=['B1, not final B4 joint objective','No LPIPS/GAN or optimizer update in this witness',
                'Random motion model; no quality claim','Single TRAIN window; no generalization or AR claim'])
        write(root/'GPU_RENDER_WITNESS.json',result);progress()
        print('GPU_RENDER_WITNESS_PASSED',round(result['elapsed_seconds'],3),'seconds',flush=True)
    except BaseException as error:
        write(root/'GPU_RENDER_FAILURE.json',dict(status='failed',stage=stage,error=repr(error),
            traceback=traceback.format_exc(),details=details,elapsed_seconds=time.monotonic()-started))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

