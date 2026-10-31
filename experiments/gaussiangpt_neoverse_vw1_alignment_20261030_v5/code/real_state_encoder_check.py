"""Bounded real-cache CPU encoder check. No renderer, GPU, training or quality claim."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from sampling import Segment,Window
from sequence_data import load_window
from aligned_ldm import AlignedLDM

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(root,cache_root,manifest):
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261030)
    spec=Window(Segment('buildingblock_tower_17_176',0,176),167)
    bindings={r['path']:r for r in json.loads(manifest.read_text())}
    checked=[]
    for frame in spec.frames:
        path=cache_root/'states'/spec.clip/f'{frame:03d}.pt'
        binding=bindings[str(path)]
        if path.stat().st_size!=binding['bytes'] or sha(path)!=binding['sha256']:
            raise ValueError('State cache binding mismatch: '+str(path))
        checked.append(binding)
    states=load_window(cache_root,spec,device='cpu')
    model=AlignedLDM().eval()
    with torch.no_grad():
        z=model.encode_windows([states])
        shared=z['initial_contexts'][0].values
        standalone=model.encode_initial(states[0])
        difference=float((shared-standalone).abs().max())
        torch.testing.assert_close(shared,standalone,atol=1e-5,rtol=1e-5)
        changed=copy.deepcopy(states)
        for j in range(5,10):
            changed[j]['coordinates']+=10000
            changed[j]['features']+=10
        altered=model.encode_windows([changed])
        prefix_difference=float((z['z_prequant'][:,:4]-altered['z_prequant'][:,:4]).abs().max())
        torch.testing.assert_close(z['z_prequant'][:,:4],altered['z_prequant'][:,:4],atol=1e-5,rtol=1e-5)
        if not torch.equal(z['indices'][:,:4],altered['indices'][:,:4]):raise AssertionError('Future changed prefix IDs')
    result=dict(status='passed',scope='real_cached_state_encoder_cpu_only',frames=list(spec.frames),
        state_counts=[len(s['coordinates']) for s in states],pts=[float(s['pts']) for s in states],
        prequant_shape=list(z['z_prequant'].shape),ids_shape=list(z['indices'].shape),
        initial_shared_max_abs=difference,future_perturb_prefix_max_abs=prefix_difference,
        prefix_ids_equal=True,source_bindings=checked,cache_manifest_sha256=sha(manifest),
        source_sha256={str(p.relative_to(root)):sha(p) for p in sorted((root/'code').rglob('*.py'))},
        elapsed_seconds=time.monotonic()-started,model_initialization='random fixed seed; no pretrained motion weights',
        real_renderer_tested=False,training_started=False,quality_claim=False)
    (root/'REAL_STATE_ENCODER_CHECK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_bindings','source_sha256']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.cache_root,args.manifest)

