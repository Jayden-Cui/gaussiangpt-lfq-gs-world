"""Run once on 134, CPU only. Export bound data and immutable source dependencies."""
from pathlib import Path
import hashlib,json,shutil,sys,time
import torch

B=Path('/data/CJH/AGS-V6Direct/outputs')
DEST=B/'gaussiangpt_neoverse_capacity_75_transfer_20261102_v1/assets'
FULL=B/'gaussiangpt_neoverse_qformer_full177_20261101_v1'
PARENT=B/'gaussiangpt_neoverse_anchor_replay_pilot_20261102_v1'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else digest_stream(f)
def digest_stream(f):
    h=hashlib.sha256()
    for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def copy(source,target):
    target=DEST/target;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    shutil.copyfile(source,target)
    return dict(source=str(source),source_sha256=sha(source),file=str(target.relative_to(DEST)),sha256=sha(target),bytes=target.stat().st_size)
def main():
    torch.set_num_threads(2)
    DEST.mkdir();began=time.monotonic();files=[]
    checkpoint=PARENT/'run/training/sweep_05.pt'
    assert sha(checkpoint)=='3893e488174f5c44fc8b696526af71d6ba718ba073bbda936d1960644fbc2699'
    files.append(copy(checkpoint,'initial.pt'))
    files.append(copy(PARENT/'run/SUMMARY.json','parent_summary.json'))
    manifest=json.loads((FULL/'cache/MANIFEST.json').read_text())
    assert sha(FULL/'cache/MANIFEST.json')=='cb5c2b30f1b4fdac4db642046a46773cc0b96a4dabdab4917e0d3c79ff6eaa4a'
    files.append(copy(FULL/'cache/MANIFEST.json','original_cache_manifest.json'))
    files.append(copy(FULL/'protocol.json','original_protocol.json'))
    src=FULL/'cache'/manifest['frames']['file'];assert sha(src)==manifest['frames']['sha256']
    files.append(copy(src,'data/frames.pt'))
    windows=[]
    for start in range(168):
        row=manifest['windows'][str(start)];source=FULL/'cache'/row['file']
        assert sha(source)==row['sha256'] and source.stat().st_size==row['bytes']
        w=torch.load(source,map_location='cpu',weights_only=True)
        assert w['start']==start and len(w['states'])==10 and w['condition']['features'].shape[1]==64
        packet={k:w[k] for k in ('start','states','targets','condition')}
        path=DEST/'data'/f'window_{start:03d}.pt'
        with path.open('xb') as f:torch.save(packet,f)
        windows.append(dict(start=start,file=str(path.relative_to(DEST)),sha256=sha(path),bytes=path.stat().st_size,
            original_window_sha256=row['sha256'],retained_keys=list(packet),omitted_keys=['memories','initial_memories']))
    sources={
        'gaussiangpt_neoverse_native_z_20261031_v6/code':None,
        'gaussiangpt_neoverse_spatial_action_20261031_v9':['spatial_action.py'],
        'gaussiangpt_neoverse_z_paired_formal_20261031_v1':['model_adapter.py','initial_norm.py','action_bottleneck.py'],
        'gaussiangpt_neoverse_qformer_full177_20261101_v1':['support_decoder.py','transfer.py','core.py'],
    }
    for folder,names in sources.items():
        base=B/folder
        paths=[base/n for n in names] if names else [p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json')]
        for p in paths:files.append(copy(p,Path('reference/outputs')/p.relative_to(B)))
    state=B/'gaussiangpt_neoverse_state_motion_20261030_v1/state'
    for n in ('code/codec.py','code/common.py','codec_final.pt','protocol.json','selection.json'):
        files.append(copy(state/n,Path('codec')/Path(n).name))
    assert sha(DEST/'codec/codec_final.pt')=='a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917'
    files.append(copy(B/'gaussiangpt_neoverse_replay_probe_20261031_v7/precision.py','codec/precision.py'))
    official=B/'gaussiangpt_state_vae_20261028_v1'
    files.append(copy(official/'checkpoints/vqvae_photoshape_official.ckpt','codec/official.ckpt'))
    vendor=official/'vendor/GaussianGPT-00658a4ab3f48daed3bdf4ba50805bb2b7dc5e14'
    for p in vendor.rglob('*'):
        if p.is_file() and not set(p.parts)&{'.git','__pycache__','wandb'} and p.suffix in ('.py','.yaml','.yml','.json','.so'):
            files.append(copy(p,Path('codec/vendor')/p.relative_to(vendor)))
    # The existing gsplat binary avoids rebuilding against the host's CUDA 11.8.
    import importlib.util
    import gsplat.cuda._backend as backend
    extension=Path(backend._C.__file__)
    files.append(copy(extension,Path('extensions')/extension.name))
    import importlib.metadata as metadata
    write(DEST/'environment_packages.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        packages={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata['Name']}))
    files.append(dict(file='environment_packages.json',sha256=sha(DEST/'environment_packages.json'),bytes=(DEST/'environment_packages.json').stat().st_size))
    write(DEST/'MANIFEST.json',dict(status='complete',frames=177,starts=list(range(168)),windows=windows,files=files,
        source_cache_manifest_sha256=sha(FULL/'cache/MANIFEST.json'),elapsed_seconds=time.monotonic()-began,
        exported_for='/data/CJH/1',quantized=False,raw_state_and_context_unchanged=True))
    print(json.dumps(dict(status='assets_exported',files=len(files),windows=len(windows),root=str(DEST),
        bytes=sum(x['bytes'] for x in files+windows),manifest_sha256=sha(DEST/'MANIFEST.json'))),flush=True)
if __name__=='__main__':main()
