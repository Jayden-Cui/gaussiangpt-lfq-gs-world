"""Freeze new code and complete existing NeoVerse assets; never modify source runs."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_movies_state_codec_20261029_v1')
def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def verify(root):
    binding=read(root/'source_binding.json')
    for r in binding['files']:assert sha(Path(r['path']))==r['sha256'],r['path']
    return binding
def bind(root):
    cfg=read(root/'protocol.json');src=Path(cfg['source'])
    paths=list(root.rglob('*.py'))+list(root.glob('*.md'))
    paths=[p for p in paths if '__pycache__' not in str(p)]
    paths += [root/p for p in ['protocol.json','state/protocol.json','state/selection.json',
        'motion/protocol.template.json','motion/code/vendor/provenance.json']]
    paths += [src/n for n in ['provenance.json','cameras.npz','input_rgb.npy','rendered_rgb.npy']]
    paths += sorted((src/'gaussians').glob('native_group_*.npz'))
    assert len(list((src/'gaussians').glob('native_group_*.npz')))==177
    # Pin exact original dependencies already used by the existing codec. No installs.
    old=read(OLD/'source_binding.json')
    for row in old['files']:
        p=Path(row['path'])
        if p.suffix=='.py' or p.suffix=='.ckpt' or p.name.endswith('.pth') or p.name=='env-spec.json':paths.append(p)
    files=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]
    write(root/'source_binding.json',{'schema':'neoverse-state-motion-binding-v1','files':files,
        'historical_split':'DEV source, subsequently training-used; current single-clip TRAIN reconstruction only',
        'environment':'Warm reuse unchanged spec5813f763; no package installations'})
    cache=root/'state/torch_cache'
    if not cache.exists():cache.symlink_to(OLD/'torch_cache',target_is_directory=True)
    print('BOUND',len(files),'files',sum(r['bytes'] for r in files),'bytes')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--verify',action='store_true');a=p.parse_args()
    verify(a.root) if a.verify else bind(a.root)
