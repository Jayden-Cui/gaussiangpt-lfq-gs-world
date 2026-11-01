"""CPU-only input-information audit on world-aligned cached prefix memories."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
CACHE=OUTPUTS/'gaussiangpt_neoverse_qformer_full177_20261101_v1/cache'
QROOT=OUTPUTS/'gaussiangpt_neoverse_two_future_qformer_20261102_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def verify_self():
    witness=read(ROOT/'CPU_TESTS.json')
    if witness.get('status')!='passed' or witness.get('code_sha256')!=sha(ROOT/'CODE.sha256'):
        raise ValueError('Input audit CPU witness changed')
    for line in (ROOT/'CODE.sha256').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or sha(ROOT/name)!=digest:raise ValueError('Input audit source changed: '+name)


def validate(coords,features):
    c=np.asarray(coords);f=np.asarray(features)
    if (c.ndim!=2 or c.shape[1]!=3 or not np.issubdtype(c.dtype,np.integer)
            or f.shape!=(len(c),64) or not np.issubdtype(f.dtype,np.floating)
            or not np.isfinite(f).all() or len(c)==0):
        raise ValueError('Expected nonempty integer world coordinates and finite 64D features')
    return c,f.astype(np.float64,copy=False)


def summarize_world(first_coords,first_features,second_coords,second_features,*,unique_per_input):
    """Align by world coordinate; allow repeated coordinates in prefix tokens."""
    a,x=validate(first_coords,first_features);b,y=validate(second_coords,second_features)
    coords,inv=np.unique(np.concatenate((a,b),axis=0),axis=0,return_inverse=True)
    ia,ib=inv[:len(a)],inv[len(a):]
    ca=np.bincount(ia,minlength=len(coords));cb=np.bincount(ib,minlength=len(coords))
    if unique_per_input and (np.any(ca>1) or np.any(cb>1)):
        raise ValueError('A GS state contains duplicate world coordinates')
    sa=np.zeros((len(coords),64),dtype=np.float64);sb=np.zeros_like(sa)
    np.add.at(sa,ia,x);np.add.at(sb,ib,y)
    shared=(ca>0)&(cb>0)
    na,nb=int((ca>0).sum()),int((cb>0).sum())
    common=int(shared.sum());union=len(coords)
    rms=relative=None
    if common:
        ma=sa[shared]/ca[shared,None];mb=sb[shared]/cb[shared,None]
        rms=float(np.sqrt(np.mean((ma-mb)**2)))
        scale=float(np.sqrt(np.mean((ma**2+mb**2)/2)))
        relative=rms/scale if scale>1e-12 else (0.0 if rms<=1e-12 else None)
        if not math.isfinite(rms) or (relative is not None and not math.isfinite(relative)):
            raise ValueError('Invalid aligned feature metric')
    token_tv=float(np.abs(ca/len(a)-cb/len(b)).sum()/2)
    return {'first_tokens':len(a),'second_tokens':len(b),'first_world_cells':na,'second_world_cells':nb,
        'shared_world_cells':common,'first_only_world_cells':int(((ca>0)&(cb==0)).sum()),
        'second_only_world_cells':int(((cb>0)&(ca==0)).sum()),'world_cell_jaccard':common/union,
        'world_cell_token_distribution_tv':token_tv,'shared_cell_mean_feature_difference_rms':rms,
        'shared_cell_mean_feature_relative_rms':relative,
        'interpretation':'Cell-mean pooling is a diagnostic; it does not compare all token multisets or prove learned use'}


def run():
    import os
    import torch
    torch.set_num_threads(2)
    verify_self();p=read(ROOT/'protocol.json')
    if (p['schema']!='paired_frozen_prefix_memory_audit_v1' or p['windows']!=[35,123]
            or p['horizons']!=list(range(1,10)) or p['device']!='cpu'):
        raise ValueError('Fixed input audit protocol differs')
    manifest_path=CACHE/'MANIFEST.json'
    if sha(manifest_path)!=p['cache_manifest_sha256'] or sha(QROOT/'protocol.json')!=p['q_training_protocol_sha256']:
        raise RuntimeError('Cache or active Q experiment protocol changed')
    q=read(QROOT/'protocol.json');manifest=read(manifest_path)
    if (q['cache_manifest_sha256']!=p['cache_manifest_sha256'] or q['windows']!=p['windows']
            or manifest['status']!='complete' or manifest['starts']!=list(range(168))):
        raise RuntimeError('Input audit is not bound to current Q inputs')
    output=ROOT/'result';output.mkdir();began=time.monotonic()
    write(output/'STARTED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
          'code_sha256':sha(ROOT/'CODE.sha256'),'gpu_used':False,'training_signalled':False})
    rows=[];window_hashes={}
    for start in p['windows']:
        entry=manifest['windows'][str(start)];path=CACHE/entry['file']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise RuntimeError('Cached window bytes changed')
        window_hashes[str(start)]=entry['sha256']
        packet=torch.load(path,map_location='cpu',weights_only=True)
        if packet['start']!=start or len(packet['states'])!=10 or len(packet['memories'])!=9 or len(packet['initial_memories'])!=9:
            raise RuntimeError('Unexpected cached window structure')
        s0=packet['states'][0]
        for h in p['horizons']:
            sh=packet['states'][h];motion=packet['memories'][h-1];static=packet['initial_memories'][h-1]
            state=summarize_world(s0['coordinates'].numpy(),s0['features'].numpy(),
                                  sh['coordinates'].numpy(),sh['features'].numpy(),unique_per_input=True)
            memory=summarize_world(static['coordinates'].numpy(),static['keys'].numpy(),
                                   motion['coordinates'].numpy(),motion['keys'].numpy(),unique_per_input=False)
            rows.append({'start':start,'horizon':h,'time_delta_seconds':float(sh['pts'])-float(s0['pts']),
                'state_initial_vs_future':state,'memory_static_vs_motion':memory})
    if len(rows)!=18 or {(r['start'],r['horizon']) for r in rows}!={(s,h) for s in p['windows'] for h in p['horizons']}:
        raise RuntimeError('Input audit row coverage differs')
    write(output/'AUDIT.json',{'status':'complete','rows':rows,'window_cache_sha256':window_hashes,
        'cache_manifest_sha256':p['cache_manifest_sha256'],'q_protocol_sha256':p['q_training_protocol_sha256'],
        'gpu_used':False,'model_inference':False,'training_signalled':False,'elapsed_seconds':time.monotonic()-began,
        'scope':p['limits']})
    print('PASS: two-window world-aligned state and frozen-memory audit, CPU only')


if __name__=='__main__':run()
