import math
import numpy as np
import torch
from PIL import Image
from trial_utils import write

def image_array(x):
    a=x.detach().cpu().numpy()
    if a.shape!=(252,448,3) or not np.isfinite(a).all():raise RuntimeError('Unexpected preview image')
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)

@torch.no_grad()
def preview(sweep,decoder,bank,bridge,store,p,runtime,digest,original_mean,out):
    from data import evaluate_metrics
    from runtime import render
    from support_decoder import select_state,support_metrics
    from b_evaluate_v2 import anchor_metrics
    folder=out/('preview_%02d'%sweep);folder.mkdir();rows=[]
    decoder.eval()
    for start in (35,123):
        w=store.get(start);prediction=decoder(w['condition'],bank.code(start))
        for h in range(1,10):
            sample=store.sample(start,h)
            state=select_state(w['condition'].coordinates,prediction['features'][h-1],prediction['logits'][h-1],
                               p['max_predicted_latent_nodes'])
            images,_,_=render(bridge,state,sample['record'],views=(0,1))
            metrics=evaluate_metrics(images,w,sample,p)
            rows.append(dict(start=start,horizon=h,frame=start+h,metrics={'correct':metrics},
                             support={'correct':support_metrics(prediction['logits'][h-1],w['targets']['occupancy'][h-1])}))
            if h in (1,3,6,9):
                image=np.concatenate((image_array(sample['record']['real']),image_array(images[0])),axis=1)
                Image.fromarray(image).save(folder/('s%03d_h%02d.png'%(start,h)))
    current=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in rows)/18
    if not math.isfinite(current) or runtime.digest(decoder)!=digest:raise RuntimeError('Preview changed decoder')
    bridge.assert_frozen()
    if sweep==0 and abs(current-original_mean)>1e-4:raise RuntimeError('Warm initial anchor regression failed')
    result=dict(status='complete',sweep=sweep,windows=[35,123],future_predictions=18,
        anchor_motion_mae=current,original_anchor_motion_mae=original_mean,
        dynamic_anchors=anchor_metrics(rows,store,p),rows=rows,decoder_digest=digest,
        codec_frozen=True,independent_native_replay='not performed for preview',manual_visual_review_required=True,
        scope='Two training windows only; full 168-window evaluation follows this two-sweep continuation')
    write(folder/'PREVIEW.json',result)
    decoder.train()
    return result
