"""Render original full GS alpha once; teacher validity never depends on predictions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
from contract import read, sha, require, verify, verify_sources, write_new

def valid_mask(alpha, threshold=.90):
    require(alpha.ndim == 2 and bool(torch.isfinite(alpha).all()), 'Invalid teacher alpha')
    invalid = (alpha < threshold).float()[None,None]
    # Invalid padding excludes the one-pixel image boundary as well as hole edges.
    return (F.max_pool2d(F.pad(invalid, (1,1,1,1), value=1), 3, 1)[0,0] == 0).detach()

def prepare(root):
    import numpy as np
    root = Path(root).resolve(); p, proof = verify(root); verify_sources(p)
    require(not (root/'teachers.pt').exists() and not (root/'TEACHER_ASSETS.json').exists(), 'Preserve existing teacher assets')
    torch.set_num_threads(6); torch.cuda.set_device(0)
    free,total = torch.cuda.mem_get_info(); require(free >= 20*2**30, 'Teacher preparation requires20GiB free')
    torch.cuda.set_per_process_memory_fraction(16*2**30/total); torch.cuda.reset_peak_memory_stats()
    started=time.monotonic(); source=p['external_sources']['renderer']['path']
    spec=importlib.util.spec_from_file_location('_v10_teacher_common', source)
    common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    preparation=read(p['external_sources']['prepared_cache']['path'])
    require(len(preparation['rows']) == 177, 'All177 original frames required')
    masks=[]; records=[]; original_records=[]
    with torch.no_grad():
        for frame,row in enumerate(preparation['rows']):
            require(time.monotonic()-started < p['teacher_prepare']['max_seconds'], 'Teacher preparation time budget')
            require(row['frame'] == frame and row['source_count'] == 112896, 'Frame order/GS count changed')
            raw_path=Path(p['teacher_prepare']['cache_root'])/f'{frame:03d}.pt'
            require(sha(raw_path)==row['cache_sha256'], 'Original cache changed')
            raw=torch.load(raw_path,map_location='cpu',weights_only=True)
            path=Path(p['teacher_prepare']['native_root'])/'gaussians'/f'native_group_{frame:03d}.npz'
            require(sha(path)==row['native_gs_sha256'], 'Original full GS changed')
            with np.load(path) as a:
                gs=torch.from_numpy(np.concatenate([a['means'], a['harmonics'][:,0]*common.C0+.5,
                    a['scales'],a['rotations'],a['opacities'][:,None]],axis=-1)).float().cuda()
            require(gs.shape==(112896,14) and bool(torch.isfinite(gs).all()), 'No GS deletion/downsampling allowed')
            gs[:,9:13]=common.canonical_quat(gs[:,9:13])
            norm=raw['normalization']; require(norm['center']==[0.,0.,0.], 'Unexpected translation')
            gs[:,:3]*=norm['scale']; gs[:,6:9]*=norm['scale']
            rgb,alpha=common.render(gs,raw['cameras'][1].cuda(),raw['intrinsics'].cuda(),raw['near'],raw['far'])
            error=rgb-raw['targets'][1].cuda(); mae=float(error.abs().mean()); max_abs=float(error.abs().max())
            psnr=float(-10*torch.log10(error.square().mean().clamp_min(1e-12)))
            require(mae<=1e-5 and psnr>=60, 'Teacher RGB parity failed')
            mask=valid_mask(alpha.squeeze(-1),p['loss']['alpha_threshold'])
            require(mask.shape==(252,448) and bool(mask.any()), 'Empty/wrong teacher mask')
            masks.append(mask.cpu())
            records.append(dict(frame=frame,coverage=float(mask.float().mean()),
                object_box_coverage=float(mask[25:235,140:320].float().mean()),mae=mae,max_abs=max_abs,psnr=psnr,
                cache_path=str(raw_path),cache_sha256=row['cache_sha256'],source_path=str(path),source_sha256=row['native_gs_sha256']))
            original_records.append((raw_path,row['cache_sha256'],path,row['native_gs_sha256']))
            if frame in (44,132):
                from PIL import Image
                Image.fromarray((rgb.clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(root/f'teacher_{frame:03d}_nearby.png')
                Image.fromarray(mask.cpu().numpy().astype('uint8')*255).save(root/f'teacher_{frame:03d}_valid.png')
            del gs,raw,rgb,alpha,error,mask
    for a,ah,b,bh in original_records:
        require(sha(a)==ah and sha(b)==bh,'Original teacher changed during preparation')
    with (root/'teachers.pt').open('xb') as f: torch.save({'masks':torch.stack(masks)},f)
    peak=torch.cuda.max_memory_allocated()/2**30; reserved=torch.cuda.max_memory_reserved()/2**30
    require(max(peak,reserved)<=16,'Teacher preparation exceeded allocation/reservation limit')
    verify(root); verify_sources(p)
    result=dict(status='complete',**proof,teachers_sha256=sha(root/'teachers.pt'),frames=177,
        source_points_per_frame=112896,records=records,alpha_threshold=.90,erosion_kernel=3,
        model_updates=0,prediction_dependent_mask=False,gpu_peak_gib=peak,gpu_reserved_peak_gib=reserved,
        elapsed_seconds=time.monotonic()-started,finished_at=datetime.now(timezone.utc).isoformat(),goal_complete=False)
    write_new(root/'TEACHER_ASSETS.json',result)
    print('TEACHER_ASSETS_COMPLETE',len(records),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    prepare(parser.parse_args().root)

