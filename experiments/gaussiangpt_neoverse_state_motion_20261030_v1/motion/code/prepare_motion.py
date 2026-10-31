"""Prepare all continuous frozen endpoint states and all adjacent pair labels."""
import argparse,time
from pathlib import Path
import torch
from motion_common import BASE,CLIPS,read,write,sha,frozen_codec,load_target,render_points,metrics,region_metrics,ssim
from motion_common import load_state,state_path,candidate_geometry,alignment_indices,pair_ids


def image_metrics(image,target,boxes):return {**metrics(image,target),**region_metrics(image,target,boxes),'ssim':ssim(image,target)}

@torch.no_grad()
def prepare(root):
    cfg=read(root/'protocol.json');assert sha(BASE/'codec_final.pt')==cfg['source_checkpoint_sha256']
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.60)
    model=frozen_codec(root);started=time.monotonic();regions=read(BASE/'protocol.json')['evaluation']['region_boxes'];states=[]
    for clip in CLIPS:
        mapping=read(BASE/'mappings'/f'{clip}.json');source={r['frame']:r['state_sha256'] for r in read(BASE/'evaluation'/clip/'metrics.json')['frames']}
        for t in range(177):
            path=BASE/'states'/clip/f'{t:03d}.pt';assert sha(path)==source[t]
            p=torch.load(path,map_location='cpu',weights_only=True);coords=p['coordinates'].cuda();features=p['features'].cuda()
            assert p['schema']=='gsworld_continuous_state_v1' and not any(k in p for k in ['indices','canonical'])
            points=model.decode_state(coords,features,4);target=load_target(clip,t)
            images=torch.stack([render_points(points,target,v) for v in range(2)])
            dest=state_path(root,clip,t);dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'coordinates':coords.cpu(),'features':features.cpu(),'reconstruction':images.cpu(),'pts':float(mapping['source_pts_seconds'][t]),
                'clip':clip,'frame':t,'normalization':p['metadata']['normalization'],
                'baseline_real':image_metrics(images[0],target['real'],regions[clip]),'source_state_sha256':source[t]},dest)
            states.append({'clip':clip,'frame':t,'sha256':sha(dest),'positions':len(coords)})
            if t%20==0:write(root/'prepare_progress.json',{'stage':'states','states':len(states),'total_states':177})
    pairs=[]
    for clip,start,end in pair_ids(cfg):
        s,t=load_state(root,clip,start),load_state(root,clip,end);assert s['normalization']==t['normalization']
        g=candidate_geometry(s['coordinates'],cfg['support']['radius'],cfg['support']['max_candidates']);idx=alignment_indices(g['coordinates'],t['coordinates'])
        gp=root/'geometry'/clip/f'{start:03d}.pt';gp.parent.mkdir(parents=True,exist_ok=True);torch.save(g,gp)
        target=load_target(clip,end);points=model.decode_state(s['coordinates'],s['features'],4);held=render_points(points,target)
        dt=t['pts']-s['pts'];assert abs(dt-1/29.97)<1e-9
        lp=root/'labels'/clip/f'{start:03d}.pt';lp.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'future_candidate_indices':idx,'held_initial_real':image_metrics(held,target['real'],regions[clip]),
            'held_initial_image':held.cpu(),'dt':dt},lp)
        pairs.append({'clip':clip,'start':start,'end':end,'candidates':len(g['coordinates']),'future_positions':len(t['coordinates']),
            'coverage':1.,'geometry_sha256':sha(gp),'label_sha256':sha(lp),'dt':dt})
        if start%20==0:write(root/'prepare_progress.json',{'stage':'pairs','pairs':len(pairs),'total_pairs':len(pair_ids(cfg))})
    assert len(states)==177 and len(pairs)==176
    write(root/'prepare_done.json',{'status':'complete','states':states,'pairs':pairs,'total_states':len(states),'total_pairs':len(pairs),
        'all_support_coverage':True,'max_candidates':max(p['candidates'] for p in pairs),'elapsed_seconds':time.monotonic()-started})
    print('MOTION_PREPARED',len(states),len(pairs),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);prepare(p.parse_args().root)
