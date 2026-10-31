"""Two fixed teacher frames only: alpha coverage calibration, no model or optimizer."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
B=Path('/data/CJH/AGS-V6Direct/outputs');S=B/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
sys.path.insert(0,str(S/'code'));from common import render,canonical_quat,C0
out=B/'gaussiangpt_neoverse_gs_supervision_20261031_review/alpha_audit'
assert not out.exists();out.mkdir(parents=True)
torch.set_num_threads(4);torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();assert free>6*2**30
torch.cuda.set_per_process_memory_fraction(4*2**30/total)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prepared=S/'prepare/buildingblock_tower_17_176.json';assert sha(prepared)=='7b63c8c50eb9a3e7f3e2f4b55576894f272c6c0f72de3502b6b7636c56dd702a'
rows=json.loads(prepared.read_text())['rows'];results=[];begun=time.monotonic()
with torch.no_grad():
 for frame,box,color in [(44,(200,25,250,130),'yellow'),(132,(135,25,245,85),'green')]:
  rawpath=S/'cache/buildingblock_tower_17_176'/f'{frame:03d}.pt';assert sha(rawpath)==rows[frame]['cache_sha256']
  raw=torch.load(rawpath,map_location='cpu',weights_only=True)
  path=B/'neoverse_tower177_fps30_20261029_v1/native448/gaussians'/f'native_group_{frame:03d}.npz'
  assert sha(path)==rows[frame]['native_gs_sha256']
  with np.load(path) as a:gs=torch.tensor(np.concatenate([a['means'],a['harmonics'][:,0]*C0+.5,a['scales'],a['rotations'],a['opacities'][:,None]],-1),device='cuda')
  assert gs.shape==(112896,14);gs[:,9:13]=canonical_quat(gs[:,9:13]);s=raw['normalization']['scale'];gs[:,:3]*=s;gs[:,6:9]*=s
  rgb,alpha=render(gs,raw['cameras'][1].cuda(),raw['intrinsics'].cuda(),raw['near'],raw['far']);alpha=alpha[...,0]
  assert float((rgb.cpu()-raw['targets'][1]).abs().mean())<1e-5
  rgb=rgb.clamp(0,1);r,g,b=rgb.unbind(-1)
  colored=((r>.25)&(g>.2)&(b<.15)&(r>g*.8)&(r<g*1.6)) if color=='yellow' else ((g>.15)&(g>r*1.3)&(g>b*1.3))
  x0,y0,x1,y1=box;roi=torch.zeros_like(colored);roi[y0:y1,x0:x1]=True;colored &= roi
  candidates=[]
  for threshold in (.5,.75,.9,.95,.99):
   invalid=(alpha<threshold).float()[None,None]
   mask=F.max_pool2d(F.pad(invalid,(1,1,1,1),value=1),3,1)[0,0]==0
   candidates.append(dict(threshold=threshold,coverage=float(mask.float().mean()),colored_pixels=int(colored.sum()),colored_retained_fraction=float(mask[colored].float().mean())))
   visible=rgb.clone();visible[~mask]*=.15
   panel=torch.cat([rgb,mask[...,None].expand(-1,-1,3).float(),visible],1)
   Image.fromarray((panel.cpu().numpy()*255).round().astype('uint8')).save(out/f'frame{frame:03d}_alpha{threshold:.2f}.png')
  results.append(dict(frame=frame,box=box,color=color,alpha_quantiles=torch.quantile(alpha[colored],torch.tensor([0.,.1,.25,.5,.75,.9,1.],device='cuda')).cpu().tolist(),candidates=candidates))
data=dict(results=results,model_updates=0,frames=2,peak_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.monotonic()-begun,scope='teacher colored-pixel diagnostics, not segmentation or model selection',goal_complete=False)
(out/'RESULT.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
