import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from io_utils import ROOT,read,sha,sources
from support_decoder import SupportDecoder,make_condition,target_labels
from core import image_metrics,object_proxy


def build_model(p,mode,device):
    sources(p)
    sys.path.insert(0,p['formal_root'])
    from model_adapter import load_pinned,attach_adapters
    api=load_pinned(Path(p['formal_root']).parent)
    base=api.AnchoredLDM(blocks=4,checkpoint_blocks=True,residual_scale=3.,inv_temperature=1.,code_mode='posterior')
    api.attach_spatial_action(base,True)
    attach_adapters(base,mode='continuous16',s_ref=p['s_ref'])
    initial=torch.load(p['common_checkpoint'],map_location='cpu',weights_only=True)
    base.load_state_dict(initial['model'],strict=True)
    return SupportDecoder(base,mode,p['code_initial_std']).to(device)


def digest(module):
    import hashlib
    h=hashlib.sha256()
    for name,v in sorted(module.state_dict().items()):
        t=v.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_data(p,model):
    from codec_binding import BaselineStates
    from teacher_binding import BoundTeachers
    base=Path(p['formal_root']).parent
    states=BaselineStates(base/'gaussiangpt_neoverse_codec_complete_eval_20261031_v1/evaluations/baseline')
    teachers=BoundTeachers(states);std=states.load(0,device='cuda')['features'].std(0,unbiased=False)
    if not torch.allclose(std,model.base.output_std,atol=1e-6,rtol=1e-6):raise RuntimeError('Output scale differs')
    windows=[]
    model.eval()
    for start in p['starts']:
        seq=[states.load(f,device='cuda') for f in range(start,start+10)]
        cond=make_condition(model.base,seq[0],p['candidate_radius'],p['max_candidates'])
        targets=target_labels(cond,seq[1:])
        samples=[teachers.load_record(f,device='cuda') for f in range(start,start+10)]
        for frame in range(start,start+10):
            entry=teachers.assets['records'][frame]
            if sha(teachers.native_root/f'native_group_{frame:03d}.npz')!=entry['source_sha256']:
                raise RuntimeError('Native source changed')
        roi=torch.zeros((252,448),device='cuda',dtype=torch.bool)
        for x0,y0,x1,y1 in p['hand_roi_xyxy']:roi[y0:y1,x0:x1]=True
        motion=[((sample['record']['real']-samples[0]['record']['real']).abs().mean(-1)/.1).clamp(0,1) for sample in samples]
        windows.append(dict(start=start,states=seq,condition=cond,targets=targets,samples=samples,roi=roi,motion=motion))
    return windows,std,states


def render(bridge,state,record,views=(0,)):
    points=bridge.decode_points(state['coordinates'],state['features'])
    gs,sh=bridge.prediction_gs(points)
    if not 0<len(gs)<=2400000 or tuple(sh.shape)!=(len(gs),4,3):raise RuntimeError('Invalid full GS')
    images=torch.stack([bridge.render_fn(gs,record['cameras'][v],record['intrinsics'],record['near'],record['far'],sh)[0].clamp(0,1) for v in views])
    if not bool(torch.isfinite(images).all()):raise RuntimeError('Nonfinite renderer output')
    return images,gs,sh


def cpu(t):return t.detach().cpu().contiguous()

def save(path,value):
    with Path(path).open('xb') as f:torch.save(value,f)
    return dict(file=Path(path).name,sha256=sha(path),bytes=Path(path).stat().st_size)

def png(path,image):
    a=(cpu(image).numpy().clip(0,1)*255).round().astype(np.uint8);Image.fromarray(a).save(path)

def panel(path,images):
    canvas=Image.new('RGB',(448*len(images),276),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,image) in enumerate(images):
        canvas.paste(Image.fromarray((cpu(image).numpy().clip(0,1)*255).round().astype(np.uint8)),(448*i,24))
        draw.text((448*i+8,6),label,fill='black')
    canvas.save(path)

def metrics(images,window,h,p):
    sample=window['samples'][h];target=sample['record']['real']
    m=image_metrics(images[0],target,window['roi'],window['motion'][h])
    m['teacher_view1_masked_mae']=float((images[1]-sample['teacher_targets'][1]).abs().mean(-1)[sample['mask']].mean())
    proxy=object_proxy(cpu(images[0]).numpy(),cpu(target).numpy(),p['object_proxies'][str(window['start'])])
    return dict(metrics=m,object_proxy=proxy)
