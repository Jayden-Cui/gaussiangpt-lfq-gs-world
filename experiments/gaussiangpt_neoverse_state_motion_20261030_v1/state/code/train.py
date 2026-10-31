"""Fixed3000-step full-state adaptation with recoverable failure checkpoints."""
import argparse,gc,json,signal,time,random
from pathlib import Path
import numpy as np
import torch
from common import CLIPS,read,write,sha,render,prediction_gs,metrics
from codec import load_codec
from perceptual import Vgg19

def load_frame(root,clip,t):
    p=torch.load(root/'cache'/clip/f'{t:03d}.pt',map_location='cpu',weights_only=True)
    for k in ['representatives','set_statistics','cameras','intrinsics','targets','real']:p[k]=p[k].cuda()
    return p

def render_prediction(points,p,view):
    gs,sh=prediction_gs(points)
    return render(gs,p['cameras'][view],p['intrinsics'],p['near'],p['far'],sh)[0].clamp(0,1)

def terms(model,p,vgg,force):
    prediction,info=model(p['representatives'],p['set_statistics'],force_target=force)
    image=render_prediction(prediction,p,0);nearby=render_prediction(prediction,p,1)
    return {'real_l1':(image-p['real']).abs().mean(),'real_vgg19':vgg(image,p['real']),
        'teacher_l1':(image-p['targets'][0]).abs().mean(),'pseudo_view_l1':(nearby-p['targets'][1]).abs().mean(),
        'workspace_roi_l1':(image[38:227,67:380]-p['real'][38:227,67:380]).abs().mean(),
        'occupancy_bce':info['occupancy']},info

def main(root):
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    protocol=read(root/'protocol.json');cfg=protocol['train'];torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    for clip in CLIPS:
        prepared=read(root/'prepare'/f'{clip}.json');assert len(prepared['rows'])==177
        for r in prepared['rows']:assert sha(root/'cache'/clip/f'{r["frame"]:03d}.pt')==r['cache_sha256']
    model=load_codec(root);vgg=Vgg19(root/'torch_cache').cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
    with torch.no_grad():
        model.eval();rows=[]
        for t in protocol['calibration_frames']:
            p=load_frame(root,CLIPS[0],t);e=model.encode_state(p['representatives'],p['set_statistics'])
            points=model.decode_state(e['coordinates'],e['features'],4)
            rows.append({'frame':t,'real':metrics(render_prediction(points,p,0),p['real']),'tokens':len(e['coordinates'])})
        write(root/'frozen_baseline.json',{'scope':'Official initialized model,9 fixed TRAIN frames','records':rows})
        del p,e,points
    model.train();rng=random.Random(cfg['seed']);frames=[(c,t) for c in CLIPS for t in range(177)]
    order=list(range(len(frames)));rng.shuffle(order);seen=set();completed=0;no_force=0;context={};event='start'
    started=time.monotonic()
    def save(name,resumable=True):
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':completed,
            'frames_seen':sorted(seen),'selection':read(root/'selection.json'),'protocol_sha256':sha(root/'protocol.json'),
            'resumable':resumable,'sampler':{'order':order,'rng':rng.getstate()},'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
            'elapsed_seconds':time.monotonic()-started,'no_target_forcing_steps':no_force}
        path=root/name;tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    def interrupt(signum,frame):raise RuntimeError(f'Interrupted by signal{signum}')
    old=signal.signal(signal.SIGTERM,interrupt)
    try:
        with (root/'train_metrics.jsonl').open('x',buffering=1) as log:
            for step in range(1,cfg['max_steps']+1):
                elapsed=time.monotonic()-started
                if elapsed>=cfg['max_seconds']:raise TimeoutError('State train wall-time budget')
                offset=(step-1)%len(frames)
                if step>1 and offset==0:rng.shuffle(order)
                clip,t=frames[order[offset]];context={'attempted_step':step,'clip':clip,'frame':t};event='forward'
                p=load_frame(root,clip,t);optimizer.zero_grad(set_to_none=True)
                force=max((step-1)/cfg['max_steps'],elapsed/cfg['max_seconds'])<.8
                losses,info=terms(model,p,vgg,force);loss=sum(cfg['loss'][k]*v for k,v in losses.items())
                assert torch.isfinite(loss);event='backward';loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True);assert norm>0
                event='optimizer_step';optimizer.step();completed=step;event='step_complete';seen.add(f'{clip}/{t:03d}')
                no_force+=int(not force)
                row={'step':step,'max_steps':cfg['max_steps'],**context,'force_target':force,'loss':float(loss.detach()),
                    'terms':{k:float(v.detach()) for k,v in losses.items()},'gradient_norm':float(norm),
                    **info,'occupancy':float(info['occupancy'].detach()),
                    'seen_frames':len(seen),'elapsed_seconds':time.monotonic()-started,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if step==1 or step%10==0:
                    write(root/'train_progress.json',row);print('STATE_TRAIN',step,'loss',row['loss'],flush=True)
                if step==1 or step%cfg['checkpoint_every']==0:save('last.pt')
                del p,losses,info,loss
        assert len(seen)==177 and no_force>0;save('codec_final.pt')
        write(root/'train_done.json',{'status':'complete','steps':completed,'frames_seen':len(seen),'all_frames_seen':len(seen)==177,
            'no_target_forcing_steps':no_force,'elapsed_seconds':time.monotonic()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'checkpoint_sha256':sha(root/'codec_final.pt')})
    except BaseException as e:
        try:save('exception.pt',event!='optimizer_step');saved=True
        except BaseException:saved=False
        write(root/'training_failure.json',{'error':repr(e),'event':event,'last_completed_step':completed,'context':context,'exception_checkpoint_saved':saved})
        raise
    finally:signal.signal(signal.SIGTERM,old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
