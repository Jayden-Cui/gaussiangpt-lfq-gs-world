"""Bounded floating-point diagnosis; no training or scientific tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from common import H,W,C0,render,metrics,canonical_quat,similarity,write
from prepare_neoverse import SOURCE
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
cams=np.load(SOURCE/'cameras.npz');normal=json.loads((root/'normalization.json').read_text());factor=normal['scale'];translation=-np.array(normal['center'])*factor
reports=[]
with torch.no_grad():
    for t in [0,88,176]:
        a=np.load(SOURCE/'gaussians'/f'native_group_{t:03d}.npz')
        gs=torch.from_numpy(np.concatenate([a['means'],a['harmonics'][:,0]*C0+.5,a['scales'],a['rotations'],a['opacities'][:,None]],-1)).float().cuda()
        gs[:,9:13]=canonical_quat(gs[:,9:13]);camera=torch.from_numpy(cams['camera_to_world'][t].copy()).cuda().float();k=cams['intrinsics'][t]
        intr=torch.tensor([k[0,0]/W,k[1,1]/H,k[0,2]/W,k[1,2]/H],device='cuda');ref=render(gs,camera,intr,.01,1e10)[0].clamp(0,1)
        variants={}
        variants['current_similarity']=similarity(gs,camera,np.eye(3),factor,translation)
        for dtype,name,center in [(torch.float32,'no_quat_repeat',True),(torch.float64,'double_transform',True),(torch.float64,'scale_only',False),(torch.float64,'translation_only',True)]:
            g=gs.clone();c=camera.clone();f=1. if name=='translation_only' else factor
            tr=translation if center else np.zeros(3)
            g[:,:3]=(gs[:,:3].to(dtype)*f+torch.as_tensor(tr,device='cuda',dtype=dtype)).float()
            g[:,6:9]=(gs[:,6:9].to(dtype)*f).float();c[:3,3]=(camera[:3,3].to(dtype)*f+torch.as_tensor(tr,device='cuda',dtype=dtype)).float()
            variants[name]=(g,c)
        g=gs.clone();c=camera.clone();dyadic=2.**np.floor(np.log2(factor));g[:,:3]*=dyadic;g[:,6:9]*=dyadic;c[:3,3]*=dyadic
        variants['dyadic_scale_only']=(g,c)
        variants['identity']=(gs,camera)
        result={}
        for name,(g,c) in variants.items():
            f=dyadic if name=='dyadic_scale_only' else (1. if name in ['identity','translation_only'] else factor)
            image=render(g,c,intr,.01*f,1e10*f)[0].clamp(0,1);d=(image-ref).abs()
            result[name]={**metrics(image,ref),'max_abs':float(d.max()),'fraction_channels_over_1_255':float((d>1/255).float().mean())}
        reports.append({'frame':t,'comparisons':result})
write(root/'similarity_diagnosis.json',{'normalization':normal,'reports':reports})
print(json.dumps({'dyadic':dyadic,'reports':[{'frame':r['frame'],'dyadic':r['comparisons']['dyadic_scale_only']} for r in reports]},indent=2))
