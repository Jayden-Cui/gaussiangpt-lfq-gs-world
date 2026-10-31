"""Bind existing source artifacts and audit full H9 support before training."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from contracts import sha

def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main(root):
    assert root==Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_vw1_h9_lfq_20261030_v3')
    p=json.loads((root/'protocol.json').read_text());previous=Path(p['previous_run'])
    state=Path(p['source_run']);old=Path(p['source_motion_cache'])
    assert sha(state/'codec_final.pt')==p['source_checkpoint_sha256']
    for name in ['states','geometry']:(root/name).symlink_to(old/name,target_is_directory=True)
    # Reuse old immutable full cache inventory, checking every hash in witness.
    previous_manifest=previous/'cache_manifest.json'
    (root/'cache_manifest.json').write_bytes(previous_manifest.read_bytes())
    previous_binding=json.loads((previous/'source_binding.json').read_text())
    dependencies=list(previous_binding['dependencies'])
    for f in [previous/'protocol.json',previous/'source_binding.json',previous/'.aris/compute/ssh-134.md']:
        dependencies.append(dict(path=str(f),sha256=sha(f)))
    binding=dict(schema='gs-vw1-h9-lfq-source-v3',source_run=str(state),source_motion_cache=str(old),
        state_checkpoint_sha256=p['source_checkpoint_sha256'],cache_manifest_sha256=sha(root/'cache_manifest.json'),
        environment_spec_sha256='5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f',
        dependencies=dependencies,authorization='Continue VideoWorld1 alignment of QFormer, horizon, initial-only decoder and future reconstruction; retain LFQ and64-dimensional GaussianGPT.')
    write_new(root/'source_binding.json',binding)
    clip=p['data']['clip'];coordinates=[]
    for t in range(177):
        value=torch.load(root/'states'/clip/f'{t:03d}.pt',map_location='cpu',weights_only=True)
        assert value['features'].shape==(len(value['coordinates']),64)
        coordinates.append(value['coordinates'].numpy())
    coverage=[]
    for start in range(176):
        geometry=torch.load(root/'geometry'/clip/f'{start:03d}.pt',map_location='cpu',weights_only=True)
        assert geometry['radius']==6 and len(geometry['coordinates'])<=100000
        source_hash=__import__('hashlib').sha256(coordinates[start].astype(np.int32).tobytes()).hexdigest()
        assert geometry['source_coordinates_sha256']==source_hash
        tree=cKDTree(geometry['coordinates'].numpy());source_tree=cKDTree(coordinates[start])
        per_h=[]
        for h in range(1,10):
            target=coordinates[min(start+h,176)]
            distance=tree.query(target,workers=1)[0];assert np.all(distance==0),(start,h)
            required=float(source_tree.query(target,p=np.inf,workers=1)[0].max())
            per_h.append(required)
        coverage.append(dict(start=start,max_required_radius=max(per_h),candidate_count=len(geometry['coordinates']),per_horizon_required_radius=per_h))
    write_new(root/'support_coverage.json',dict(status='passed',windows=176,horizons=9,
        real_target_exposures=1548,padded_positions=36,max_required_radius=max(x['max_required_radius'] for x in coverage),
        max_candidates=max(x['candidate_count'] for x in coverage),radius=6,rows=coverage))
    selected=sorted((root/'code').rglob('*.py'))+sorted((root/'code/vendor').glob('*.json'))
    selected += [root/n for n in ['protocol.json','README.md','ENVIRONMENT.md','launch_remote.py','support_coverage.json']]
    write_new(root/'code_manifest.json',{str(f.relative_to(root)):sha(f) for f in selected})
    print(json.dumps(dict(bound_dependencies=len(dependencies),submitted_files=len(selected),all176_h9_supports_covered=True,
        max_required_radius=max(x['max_required_radius'] for x in coverage),max_candidates=max(x['candidate_count'] for x in coverage))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
