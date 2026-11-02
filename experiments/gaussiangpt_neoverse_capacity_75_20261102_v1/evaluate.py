from runtime_env import configure
configure()
import argparse,json,time,traceback,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image,ImageDraw
from bindings import ROOT,ASSETS,Store,NativeBridge,native_modules,reference_model,read,write,sha,require,tree,select_state,support_metrics,image_metrics,object_proxy
from capacity_model import CapacityDecoder
from worker import verify

def array(t):return (t.detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
def panel(path,images):
    path.parent.mkdir(parents=True,exist_ok=True);canvas=Image.new('RGB',(448*len(images),276),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,im) in enumerate(images):canvas.paste(Image.fromarray(array(im)),(448*i,24));draw.text((448*i+8,6),label,fill='black')
    canvas.save(path)
def mean_metrics(rows):
    names=list(rows[0]);return {k:sum(x[k] for x in rows)/len(rows) for k in names}

@torch.no_grad()
def evaluate(arm,training,out):
    p=verify();training=Path(training);out=Path(out);require(not out.exists() and out.resolve().is_relative_to(ROOT),'New isolated evaluation required')
    out.mkdir(parents=True);torch.set_num_threads(6);torch.set_num_interop_threads(1);began=time.monotonic()
    require(torch.cuda.device_count()==1,'Expose exactly one GPU')
    free,total=torch.cuda.mem_get_info();require(free>=p['minimum_free_gpu_gib']*2**30,'Insufficient evaluation memory')
    torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
    try:
        checkpoint=training/'step_0846.pt';done=read(training/'TRAIN_DONE.json')
        require(done['status']=='complete' and done['updates']==846 and sha(checkpoint)==done['checkpoint_sha256'],'Training incomplete')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);a=p['arms'][arm]
        require(saved['source']==done['source'] and saved['source']['code_sha256']==sha(ROOT/'CODE.sha256'),'Training code/source differs')
        reference,_=reference_model();model=CapacityDecoder(reference,a['code_dim'],a['width'],p['seed'])
        model.load_state_dict(saved['model'],strict=True);model=model.cuda().eval();codes=saved['codes'].cuda();del reference,saved
        bridge=NativeBridge().eval();store=Store();rows=[];packets=[];diags=set(p['diagnostic_starts'])
        (out/'packets').mkdir();(out/'timeline').mkdir()
        with (out/'rows.jsonl').open('x',buffering=1) as log:
            for start in range(168):
                require(time.monotonic()-began<p['evaluation_timeout_seconds'],'Evaluation time budget exceeded')
                w=store.get(start);pred=model(w['condition'],codes[start]);swapped=model(w['condition'],codes[(start+84)%168]) if start in diags else None
                if start in diags:
                    sample=store.sample(start,0);initial,_,_=bridge.render(w['states'][0],sample['record'])
                    panel(out/'snapshots'/f's{start:03d}'/'frame_00.png',[('source',sample['record']['real']),('initial codec',initial[0]),('initial codec',initial[0]),('initial codec',initial[0])])
                    if start==0:Image.fromarray(np.concatenate((array(sample['record']['real']),array(initial[0])),1)).save(out/'timeline/000.png')
                for h in range(1,10):
                    sample=store.sample(start,h);state=select_state(w['condition'].coordinates,pred['features'][h-1],pred['logits'][h-1],p['max_nodes'])
                    views=(0,1) if start in diags else (0,)
                    images,gs,sh=bridge.render(state,sample['record'],views)
                    m=image_metrics(images[0],sample['record']['real'],w['roi'],sample['motion'])
                    row=dict(start=start,horizon=h,frame=start+h,correct=m,support=support_metrics(pred['logits'][h-1],w['targets']['occupancy'][h-1]))
                    if start in diags:
                        alt=select_state(w['condition'].coordinates,swapped['features'][h-1],swapped['logits'][h-1],p['max_nodes'])
                        alt_im,_,_=bridge.render(alt,sample['record']);actual_im,_,_=bridge.render(w['states'][h],sample['record'])
                        row['swapped']=image_metrics(alt_im[0],sample['record']['real'],w['roi'],sample['motion'])
                        row['actual_target']=image_metrics(actual_im[0],sample['record']['real'],w['roi'],sample['motion'])
                        spec=store.p['object_proxies'].get(str(start))
                        if spec:row['object_proxy']=object_proxy(array(images[0])/255.,array(sample['record']['real'])/255.,spec)
                        panel(out/'snapshots'/f's{start:03d}'/f'frame_{h:02d}.png',[('source',sample['record']['real']),('correct',images[0]),('swapped',alt_im[0]),('actual target codec',actual_im[0])])
                        packet=out/'packets'/f's{start:03d}_h{h:02d}.npz';r=sample['record']
                        np.savez_compressed(packet,gs=gs.cpu().numpy(),sh=sh.cpu().numpy(),cameras=r['cameras'].cpu().numpy(),
                            intrinsics=r['intrinsics'].cpu().numpy(),near=np.array(r['near']),far=np.array(r['far']),images=images.cpu().numpy())
                        packets.append(dict(start=start,horizon=h,file=str(packet.relative_to(out)),sha256=sha(packet)))
                    if start==max(0,start+h-9):
                        Image.fromarray(np.concatenate((array(sample['record']['real']),array(images[0])),1)).save(out/'timeline'/f'{start+h:03d}.png')
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n')
                if start in (0,41,83,125,167):write(out/'STATUS.json',dict(status='running',completed_windows=start+1,total_windows=168,elapsed_seconds=time.monotonic()-began),True)
        require(len(rows)==1512 and len({r['frame'] for r in rows})==176 and len(packets)==72,'Incomplete evaluation')
        by_frame=defaultdict(list);by_h=defaultdict(list);by_window=defaultdict(list)
        for row in rows:by_frame[row['frame']].append(row['correct']);by_h[row['horizon']].append(row['correct']);by_window[row['start']].append(row['correct'])
        physical=mean_metrics([mean_metrics(x) for x in by_frame.values()]);by_h={str(k):mean_metrics(v) for k,v in by_h.items()}
        by_window={str(k):mean_metrics(v) for k,v in by_window.items()};anchors=mean_metrics([r['correct'] for r in rows if r['start'] in (35,123)])
        bridge.assert_frozen();write(out/'PACKETS.json',dict(packets=packets,encoder_or_codec_needed_for_replay=False))
        result=dict(status='complete',arm=arm,checkpoint_sha256=sha(checkpoint),source=done['source'],window_predictions=1512,unique_future_frames=176,
            metrics=dict(unique_frame_equal=physical,window_equal=mean_metrics([r['correct'] for r in rows])),by_horizon=by_h,by_window=by_window,
            anchors=anchors,rows=rows,packets_sha256=sha(out/'PACKETS.json'),elapsed_seconds=time.monotonic()-began,
            scope='Free-code reconstruction on one TRAIN source, no Q extraction or AR rollout',teacher_accepted=False)
        write(out/'EVALUATION.json',result);write(out/'STATUS.json',dict(status='complete',replay='pending'),True)
    except BaseException as e:write(out/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

@torch.no_grad()
def replay(out):
    p=verify();out=Path(out);torch.set_num_threads(6);torch.set_num_interop_threads(1);began=time.monotonic()
    require(torch.cuda.device_count()==1,'Expose exactly one replay GPU')
    _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(p['allocator_limit_gib']*2**30/total)
    try:
        data=read(out/'EVALUATION.json');require(sha(out/'PACKETS.json')==data['packets_sha256'],'Packet list changed')
        common,_=native_modules();require('codec' not in sys.modules and 'model.gaussian_vqvae' not in sys.modules,'Replay imported codec')
        rows=[]
        for row in read(out/'PACKETS.json')['packets']:
            require(time.monotonic()-began<p['replay_timeout_seconds'],'Replay budget exceeded')
            path=out/row['file'];require(sha(path)==row['sha256'],'Native packet changed')
            with np.load(path,allow_pickle=False) as z:
                gs=torch.from_numpy(z['gs'].copy()).cuda();sh=torch.from_numpy(z['sh'].copy()).cuda()
                camera=torch.from_numpy(z['cameras'].copy()).cuda();intr=torch.from_numpy(z['intrinsics'].copy()).cuda();expected=torch.from_numpy(z['images'].copy()).cuda()
                for v in range(2):
                    im=common.render(gs,camera[v],intr,float(z['near']),float(z['far']),sh)[0].clamp(0,1)
                    mae=float((im-expected[v]).abs().mean());psnr=float(-10*torch.log10((im-expected[v]).square().mean().clamp_min(1e-12)))
                    require(mae<=1e-5 and psnr>=60,'Independent native replay mismatch')
                    rows.append(dict(start=row['start'],horizon=row['horizon'],view=v,mae=mae,psnr=psnr))
        require(len(rows)==144,'Replay coverage incomplete')
        write(out/'REPLAY.json',dict(status='passed',packets=72,views=144,packets_sha256=data['packets_sha256'],max_mae=max(r['mae'] for r in rows),
            encoder_or_codec_loaded=False,rows=rows,elapsed_seconds=time.monotonic()-began))
    except BaseException as e:write(out/'REPLAY_FAILED.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

def videos(out):
    import cv2
    out=Path(out);manifest=[]
    for name,paths,fps,crop in [('full177_source_prediction.mp4',sorted((out/'timeline').glob('*.png')),30000/1001,False)]+[
        (f's{start:03d}_source_prediction_5fps.mp4',sorted((out/'snapshots'/f's{start:03d}').glob('*.png')),5,True) for start in (35,123)]:
        dest=out/name;writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),fps,(896,252));require(writer.isOpened(),'Video encoder unavailable')
        try:
            for path in paths:
                im=cv2.imread(str(path));im=im[24:276,:896] if crop else im
                require(im.shape==(252,896,3),'Unexpected video frame');writer.write(im)
        finally:writer.release()
        manifest.append(dict(file=name,frames=len(paths),fps=fps,sha256=sha(dest)))
    require(manifest[0]['frames']==177 and all(r['frames']==10 for r in manifest[1:]),'Video sequence incomplete')
    write(out/'VIDEOS.json',dict(rows=manifest,labels=False,interpolation=False,timeline_rule='frame0 codec; frames1..9 from start0, thereafter h9 from start=frame-9; teacher-forced free codes, not AR'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['evaluate','replay','videos']);parser.add_argument('--arm');parser.add_argument('--training');parser.add_argument('--out',required=True)
    a=parser.parse_args()
    if a.phase=='evaluate':evaluate(a.arm,a.training,a.out)
    elif a.phase=='replay':replay(a.out)
    else:videos(a.out)
