from runtime_env import configure
configure()
import torch,json
from dataclasses import replace
from bindings import ROOT,ASSETS,reference_model,Store,verify_assets,write,sha,require
from capacity_model import CapacityDecoder,FreeCodes
from worker import sampling,verify

p=verify();torch.set_num_threads(2);torch.manual_seed(p['seed']);verify_assets(full=True)
reference,parent=reference_model();store=Store();w=store.get(35,'cpu');c=w['condition']
c=replace(c,coordinates=c.coordinates[:40],features=c.features[:40],initial_present=c.initial_present[:40],geometry=c.geometry[:40],context=c.context[:,:40])
with torch.no_grad():expected=reference(c,parent['codes'][35])
rows=[]
for arm,a in p['arms'].items():
    model=CapacityDecoder(reference,a['code_dim'],a['width'],p['seed']).eval();bank=FreeCodes(parent['codes'],a['code_dim'])
    prediction=model(c,bank.code(35));diff={k:float((prediction[k]-expected[k]).detach().abs().max()) for k in ('features','logits')}
    require(diff['features']<=5e-4 and diff['logits']<=5e-5,'Initial expansion changed behavior')
    changed=bank.code(35).detach().clone();changed[3:]+=.25
    other=model(c,changed);causal=max(float((other[k][:3]-prediction[k][:3]).detach().abs().max()) for k in ('features','logits'))
    require(causal<=1e-5,'Future-code leakage')
    (prediction['features'].square().mean()+prediction['logits'].square().mean()).backward()
    grad=bank.code(35).grad;require(bool(torch.isfinite(grad).all()) and float(grad.norm())>0,'Missing code gradient')
    extra=float(grad[:,16:].norm()) if a['code_dim']>16 else None
    if extra is not None:require(extra>0,'Extra dimensions are dead')
    rank=int(torch.linalg.matrix_rank(model.action.weight.detach()).item());require(rank==a['code_dim'],'Projection lacks declared code rank')
    rows.append(dict(arm=arm,code_shape=list(bank.stack().shape),hidden_width=model.width,output_width=prediction['features'].shape[-1],
        max_errors=diff,causality_max=causal,extra_code_gradient=extra,action_projection_rank=rank))
items=sampling(p);require(len(items)==846,'Wrong matched update budget')
# Verify a usable local video encoder before the long pipeline starts.
import cv2,numpy as np
probe=ROOT/'video_encoder_probe.mp4';require(not probe.exists(),'New encoder probe path required')
writer=cv2.VideoWriter(str(probe),cv2.VideoWriter_fourcc(*'mp4v'),5,(32,32));require(writer.isOpened(),'Video encoder unavailable')
for i in range(3):writer.write(np.full((32,32,3),i*40,dtype=np.uint8))
writer.release();reader=cv2.VideoCapture(str(probe));count=0
while True:
    ok,im=reader.read()
    if not ok:break
    count+=1
reader.release();require(count==3,'Video encoder round-trip failed')
write(ROOT/'CPU_TESTS.json',dict(status='passed',code_sha256=sha(ROOT/'CODE.sha256'),rows=rows,sampling_updates=len(items),video_roundtrip_frames=count,
    assets_fully_verified=True,scope='CPU parity/gradient/causality uses 40 real candidate rows; GPU preflight must use full windows'))
print('CPU_CAPACITY_TESTS passed')
