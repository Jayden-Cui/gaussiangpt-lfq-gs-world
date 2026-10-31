"""Full native NeoVerse states, fixed shared-world similarity, immutable hashes."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import CLIPS,H,W,N,C0,read,write,sha,similarity,canonical_quat,render,metrics,voxelize

SOURCE=Path('/data/CJH/AGS-V6Direct/outputs/neoverse_tower177_fps30_20261029_v1/native448')

@torch.no_grad()
def prepare(root):
    started=time.monotonic();torch.set_num_threads(6);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.60)
    protocol=read(root/'protocol.json');prov=read(SOURCE/'provenance.json')
    assert prov['status']=='complete' and not prov['arguments']['use_motion']
    assert len(prov['saved_gaussians'])==177
    times=prov['input']['source_timestamps_seconds']
    assert len(times)==177 and np.all(np.diff(times)>0)
    cams=np.load(SOURCE/'cameras.npz');rgb=np.load(SOURCE/'input_rgb.npy',mmap_mode='r')
    reference=np.load(SOURCE/'rendered_rgb.npy',mmap_mode='r')
    assert rgb.shape==reference.shape==(177,H,W,3) and rgb.dtype==np.uint8
    assert np.array_equal(cams['timestamps'],np.arange(177))
    clip=CLIPS[0];rows=[];normal=None
    files=[SOURCE/n for n in ['cameras.npz','input_rgb.npy','rendered_rgb.npy','provenance.json']]
    binding=read(root.parent/'source_binding.json')
    hashes={v['path']:v['sha256'] for v in binding['files']}
    for path in files:assert sha(path)==hashes[str(path)],str(path)
    write(root/'mappings'/f'{clip}.json',{'source_pts_seconds':times,'source_frame_indices':list(range(177)),
        'model_timestamps':cams['timestamps'].tolist(),'source_sha256':prov['input']['source_video_sha256'],
        'note':'Seconds come from original source FPS; model integer timestamp is only a frame index'})
    for t in range(177):
        path=SOURCE/'gaussians'/f'native_group_{t:03d}.npz'
        assert sha(path)==hashes[str(path)],str(path)
        with np.load(path) as a:
            meta=json.loads(str(a['metadata_json']))
            assert meta['timestamp']==t and meta['forward_vel'] is None and meta['backward_vel'] is None
            assert a['harmonics'].shape==(N,1,3)
            gs=torch.from_numpy(np.concatenate([a['means'],a['harmonics'][:,0]*C0+.5,a['scales'],
                a['rotations'],a['opacities'][:,None]],-1)).float().cuda()
        assert gs.shape==(N,14) and torch.isfinite(gs).all() and (gs[:,6:9]>0).all()
        assert ((gs[:,13]>=0)&(gs[:,13]<=1)).all()
        gs[:,9:13]=canonical_quat(gs[:,9:13])
        camera=torch.from_numpy(cams['camera_to_world'][t].copy()).cuda().float()
        k=cams['intrinsics'][t]
        intr=torch.tensor([k[0,0]/W,k[1,1]/H,k[0,2]/W,k[1,2]/H],device='cuda')
        # NeoVerse uses gsplat defaults near=.01, far=1e10, classic SH0 black background.
        native,_=render(gs,camera,intr,.01,1e10)
        old=torch.from_numpy(np.array(reference[t])).cuda().float()
        replay=metrics(native.clamp(0,1),old)
        assert replay['psnr']>=60 and replay['mae']<=1e-5,('native parity',t,replay)
        if normal is None:
            xyz=gs[gs[:,13]>.01,:3].cpu().numpy()
            lo,hi=np.quantile(xyz,[.01,.99],axis=0)
            robust_scale=.9/float(np.max(hi-lo))
            normal={'center':[0.,0.,0.],'scale':float(2.**np.floor(np.log2(robust_scale))),
                'robust_scale_reference':robust_scale,'robust_center_reference':((lo+hi)/2).tolist(),
                'source_frame':0,'fixed_for_all_frames':True,'calibration_only_alpha_threshold':.01,
                'rule':'Power-of-two scale only, no translation; preserve float32 mantissas and native projected covariance'}
            write(root/'normalization.json',normal)
        scale=normal['scale'];translation=-np.array(normal['center'])*scale
        # Scaling by2**k is exact for these normal-range float32 positions/scales.
        # Arbitrary recentering/rescaling caused measurable cancellation for thin GS.
        gs=gs.clone();camera=camera.clone()
        gs[:,:3]*=scale;gs[:,6:9]*=scale;camera[:3,3]*=scale
        near,far=.01*scale,1e10*scale
        normalized,_=render(gs,camera,intr,near,far)
        parity=metrics(normalized.clamp(0,1),native.clamp(0,1))
        assert parity['psnr']>=60 and parity['mae']<=1e-5,('similarity parity',t,parity)
        nearby=camera.clone();nearby[:3,3]+=camera[:3,0]*.02
        pseudo,_=render(gs,nearby,intr,near,far)
        real=torch.from_numpy(np.array(rgb[t])).cuda().float()/255
        vox=voxelize(gs,.008)
        assert vox['n_input']==N and int(vox['counts'].sum())==N
        assert len(vox['representatives'])<=protocol['max_input_voxels']
        rep,_=render(vox['representatives'].cuda(),camera,intr,near,far)
        row={'clip':clip,'frame':t,'pts_seconds':times[t],'source_count':N,
            'native_render_parity':replay,'similarity_parity':parity,'teacher_real':metrics(normalized.clamp(0,1),real),
            'representative_real':metrics(rep.clamp(0,1),real),'representative_teacher':metrics(rep.clamp(0,1),normalized.clamp(0,1)),
            'voxels':len(vox['representatives']),'native_gs_sha256':hashes[str(path)]}
        payload={**vox,'cameras':torch.stack([camera,nearby]).cpu(),'intrinsics':intr.cpu(),
            'targets':torch.stack([normalized,pseudo]).cpu(),'real':real.cpu(),'near':near,'far':far,
            'clip':clip,'frame':t,'normalization':normal,'metrics':row}
        out=root/'cache'/clip/f'{t:03d}.pt';out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,out);row['cache_sha256']=sha(out);row['cache_bytes']=out.stat().st_size;rows.append(row)
        if t in protocol['calibration_frames']:
            again=voxelize(gs,.008)
            for key in ['coordinates','set_statistics','source_indices']:assert torch.equal(vox[key],again[key])
            panel=torch.cat([real,normalized,rep],1).clamp(0,1).cpu().numpy()
            preview=root/'previews'/f'{t:03d}.png';preview.parent.mkdir(exist_ok=True)
            Image.fromarray((panel*255).round().astype(np.uint8)).save(preview)
        if t==0 or t%20==0 or t==176:
            write(root/'prepare_progress.json',{'frames':t+1,'total':177,'elapsed_seconds':time.monotonic()-started,'last':row})
            print('PREPARE',t+1,'voxels',len(vox['representatives']),'parity',replay['psnr'],flush=True)
    write(root/'prepare'/f'{clip}.json',{'clip':clip,'normalization':normal,'rows':rows,'rgb_sha256':hashes[str(SOURCE/'input_rgb.npy')]})
    write(root/'prepare_done.json',{'status':'complete','states':177,'all_source_points_per_state':N,'grid':.008,
        'min_native_parity_psnr':min(r['native_render_parity']['psnr'] for r in rows),
        'min_similarity_parity_psnr':min(r['similarity_parity']['psnr'] for r in rows),
        'max_input_voxels':max(r['voxels'] for r in rows),'elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);prepare(p.parse_args().root)
