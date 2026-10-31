"""Run once in a new output directory; reference old caches without editing them."""
import argparse
import json
from pathlib import Path
from contracts import sha


def write(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Existing binding differs: '+str(path)
        return
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main(root):
    assert root.resolve() == Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_causal64_lfq_20261030_v2')
    p = json.loads((root/'protocol.json').read_text())
    state = Path(p['source_run']); old = Path(p['source_motion_cache'])
    assert sha(state/'codec_final.pt') == p['source_checkpoint_sha256']
    assert json.loads((old/'prepare_done.json').read_text())['total_pairs'] == 176
    clip = 'buildingblock_tower_17_176'
    for name in ['states','geometry','labels']:
        destination = root/name
        if destination.is_symlink():
            assert destination.resolve() == (old/name).resolve()
        else:
            destination.symlink_to(old/name, target_is_directory=True)
    files = []
    for folder, count in [(old/'states'/clip,177), (old/'geometry'/clip,176),
            (old/'labels'/clip,176), (state/'cache'/clip,177)]:
        entries = sorted(folder.glob('[0-9][0-9][0-9].pt'))
        assert len(entries) == count, (str(folder),len(entries),count)
        files += entries
    manifest = [{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
    write(root/'cache_manifest.json', manifest)
    ledger = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_native64_motion_20261030_v1/.aris/compute/ssh-134.md')
    lfq = Path('/data/CJH/envs/gaussiangpt/lib/python3.10/site-packages/vector_quantize_pytorch/lookup_free_quantization.py')
    vendor = Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_state_vae_20261028_v1/vendor/GaussianGPT-00658a4ab3f48daed3bdf4ba50805bb2b7dc5e14')
    dependencies = [state/'codec_final.pt',state/'protocol.json',state.parent/'source_binding.json',
        state/'mappings'/f'{clip}.json',old/'protocol.json',old/'source_binding.json',old/'prepare_done.json',ledger,lfq]
    dependencies += sorted((state/'code').rglob('*.py'))
    dependencies += sorted(vendor.rglob('*.py'))
    write(root/'source_binding.json', {'schema':'causal64-lfq-source-binding-v2',
        'source_run':str(state), 'source_motion_cache':str(old),
        'state_checkpoint_sha256':p['source_checkpoint_sha256'],
        'environment_spec_sha256':'5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f',
        'cache_manifest_sha256':sha(root/'cache_manifest.json'),
        'dependencies':[{'path':str(f),'sha256':sha(f)} for f in sorted(set(dependencies))],
        'state_fidelity_gate_already_failed':True,
        'authorization':'User requested replacing the prior run with coordinate-aligned causalCNN and prefixQFormer64 training; LFQ-only actions, frozen continuous GaussianGPT state.'})
    selected = sorted((root/'code').rglob('*.py')) + sorted((root/'code/vendor').glob('*.json'))
    selected += [root/'protocol.json',root/'README.md',root/'ENVIRONMENT.md',root/'launch_remote.py']
    write(root/'code_manifest.json', {str(f.relative_to(root)):sha(f) for f in selected})
    print(json.dumps({'bound_cache_files':len(manifest),'bound_dependencies':len(set(dependencies)),
        'submitted_code_files':len(selected),'root':str(root)}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
