"""All177-frame continuous-state reconstruction and saved-feature replay."""
import argparse,gc,subprocess,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import CLIPS,H,W,read,write,sha,metrics,region_metrics,render
from codec import load_codec,save_state
from train import load_frame,render_prediction
from perceptual import ssim


def api_check(model,p):
    model.eval();assert isinstance(model.ae.vq,torch.nn.Identity)
    with torch.no_grad():
        e=model.encode_state(p['representatives'],p['set_statistics'])
        coords=e['coordinates'].clone();features=e['features'].detach().clone()
        direct,_=model.ae.decode(e['sparse'],target_key=None);direct=model._world(direct)
        a=render_prediction(direct,p,0)
        decoded=model.decode_state(coords,features,4);b=render_prediction(decoded,p,0)
        check=metrics(a,b);assert check['psnr']>=60 and check['mae']<=1e-5,check
        assert torch.equal(coords,e['coordinates'])
        repeat=model.encode_state(p['representatives'],p['set_statistics'])
        def order(x):
            c=x['coordinates'].cpu().numpy();i=np.lexsort((c[:,2],c[:,1],c[:,0]));return c[i],x['features'].cpu().numpy()[i]
        c,f=order(e);rc,rf=order(repeat);assert np.array_equal(c,rc)
        feature_error=float(np.max(np.abs(f-rf)));assert feature_error<1e-4,feature_error
        maximum=float((a-b).abs().max())
    del e,repeat,direct,decoded,a,b;gc.collect();torch.cuda.empty_cache();model.requires_grad_(False)
    with torch.enable_grad():
        features.requires_grad_(True);out=model.decode_state(coords,features,4)
        (render_prediction(out,p,0)-p['real']).abs().mean().backward()
        assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum()>0
        assert all(x.grad is None for x in model.parameters())
        norm=float(features.grad.norm())
    return {'passed':True,'state_lfq_removed':True,'representation':'Nx64 float32 continuous features plus integer sparse coordinates',
        'state_render_roundtrip':check,'render_max_abs':maximum,'repeat_feature_max_abs':feature_error,
        'frozen_decoder_input_gradient_norm':norm,'decoder_target_key':None}


def image_metrics(image,target,boxes):return {**metrics(image,target),**region_metrics(image,target,boxes),'ssim':ssim(image,target)}
def means(rows,key):return {k:float(np.mean([r[key][k] for r in rows])) for k in rows[0][key]}

@torch.no_grad()
def main(root):
    started=time.monotonic();torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    cfg=read(root/'protocol.json');model=load_codec(root,root/'codec_final.pt').eval()
    api=api_check(model,load_frame(root,CLIPS[0],0));write(root/'api_acceptance.json',api)
    rows=[]
    for clip in CLIPS:
        folder=root/'evaluation'/clip;folder.mkdir(parents=True,exist_ok=True)
        states=root/'states'/clip;states.mkdir(parents=True,exist_ok=True)
        err=(folder/'ffmpeg.log').open('wb')
        proc=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{W*4}x{H}','-framerate','29.97','-i','pipe:0','-an','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'real_teacher_representative_codec.mp4')],stdin=subprocess.PIPE,stderr=err)
        boxes=cfg['evaluation']['region_boxes'][clip]
        try:
            for t in range(177):
                p=load_frame(root,clip,t);e=model.encode_state(p['representatives'],p['set_statistics'])
                out=model.decode_state(e['coordinates'],e['features'],4);image=render_prediction(out,p,0);near=render_prediction(out,p,1)
                teacher=p['targets'][0].clamp(0,1);rep=render(p['representatives'],p['cameras'][0],p['intrinsics'],p['near'],p['far'])[0].clamp(0,1)
                path=states/f'{t:03d}.pt';save_state(path,e,{'clip':clip,'frame':t,'normalization':p['normalization']})
                row={'clip':clip,'frame':t,'codec_real':image_metrics(image,p['real'],boxes),'teacher_real':image_metrics(teacher,p['real'],boxes),
                    'representative_real':image_metrics(rep,p['real'],boxes),'codec_teacher':metrics(image,teacher),'pseudo_view_teacher':metrics(near,p['targets'][1].clamp(0,1)),
                    'latent_positions':len(e['coordinates']),'input_gaussians':p['n_input'],'input_voxels':len(p['representatives']),
                    'decoded_gaussians':len(next(iter(out.values()))),'serialized_state_bytes':path.stat().st_size,
                    'unpacked_feature_and_coordinate_bytes':len(e['coordinates'])*(64*4+3*4),'raw_gs_float32_bytes':p['n_input']*14*4,'state_sha256':sha(path)}
                rows.append(row)
                panel=(torch.cat([p['real'],teacher,rep,image],1).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
                proc.stdin.write(panel.tobytes())
                if t in cfg['calibration_frames']:Image.fromarray(panel).save(folder/f'comparison_{t:03d}.png')
                if t%20==0 or t==176:write(root/'eval_progress.json',{'frames':t+1,'total':177,'elapsed_seconds':time.monotonic()-started})
                del p,e,out,image,near,teacher,rep
        finally:
            proc.stdin.close();code=proc.wait(timeout=90);err.close()
        assert code==0
        write(folder/'metrics.json',{'frames':rows,'summary':means(rows,'codec_real')})
    summary={k:means(rows,k) for k in ['codec_real','teacher_real','representative_real','codec_teacher','pseudo_view_teacher']}
    summary['mean_latent_positions']=float(np.mean([r['latent_positions'] for r in rows]))
    summary['mean_serialized_state_bytes']=float(np.mean([r['serialized_state_bytes'] for r in rows]))
    training=read(root/'train_done.json');gates={'all177_frames':len(rows)==177,'training_coverage':training['all_frames_seen'],
        'training_without_forcing':training['no_target_forcing_steps']>0,'api':api['passed'],'teacher_reconstruction_at_least25db':summary['codec_teacher']['psnr']>=25}
    for k in ['psnr','workspace_roi_psnr','object_psnr','hand_activity_psnr']:gates[k+'_drop_at_most1db']=summary['teacher_real'][k]-summary['codec_real'][k]<=1
    write(root/'evaluation.json',{'status':'evaluated','quality_passed':all(gates.values()),'gates':gates,'summary':summary,'api':api,
        'training':training,'scope':'Single-source TRAIN reconstruction; offline whole-video teacher; continuous state AE, LFQ absent',
        'checkpoint_sha256':sha(root/'codec_final.pt'),'protocol_sha256':sha(root/'protocol.json')})
    print('STATE_EVAL_COMPLETE',summary,gates,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
