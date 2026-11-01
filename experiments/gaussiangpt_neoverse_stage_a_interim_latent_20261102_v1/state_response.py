"""Measure how the fixed decoder responds to sweep-10 free codes."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
EXPECTED_CODE='1bdca9aad668120f028f35e4abb351a664f8ab493d0266d1c9a6781b2ecc5f54'


def main():
    began=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('_state_response_context',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    if binding.sha(STAGE_A/'CODE.sha256')!=EXPECTED_CODE:raise RuntimeError('Stage-A source changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    decoder,modules=binding.load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    store=binding.dataset(modules,p,c);runtime=modules[2];base=binding.load_codes(paths,evaluation)['correct']
    std=store.frames['std'].clamp_min(1e-6);rows=[];hashes={}
    with torch.no_grad():
        for shard,start in ((0,35),(1,123)):
            folder=STAGE_A/f'shard_{shard}';path=folder/'sweep_10.pt';meta=binding.read(folder/'sweep_10.json')
            if meta['status']!='complete' or binding.sha(path)!=meta['checkpoint_sha256']:
                raise RuntimeError('Sweep-10 checkpoint differs')
            item=torch.load(path,map_location='cpu',weights_only=True)
            if (item['schema']!='fixed_decoder_free_codes_shard_v1' or item['shard']!=shard
                    or item['sweep']!=10 or item['starts']!=c['shards'][str(shard)]
                    or item['source']['decoder_digest']!=c['decoder_tensor_digest']
                    or item['source']['baseline_codes_sha256']!=c['bindings']['codes']):
                raise RuntimeError('Sweep-10 source mismatch')
            code=item['codes'][item['starts'].index(start)];window=store.get(start,device='cpu')
            old=decoder(window['condition'],base[start]);new=decoder(window['condition'],code)
            for h in range(1,10):
                old_occ=old['logits'][h-1]>0;new_occ=new['logits'][h-1]>0
                index=window['targets']['indices'][h-1]
                delta=(new['features'][h-1]-old['features'][h-1]).abs()/std
                rows.append(dict(start=start,horizon=h,frame=start+h,
                  candidate_count=int(old_occ.numel()),baseline_occupied=int(old_occ.sum()),
                  updated_occupied=int(new_occ.sum()),newly_occupied=int((~old_occ&new_occ).sum()),
                  no_longer_occupied=int((old_occ&~new_occ).sum()),
                  occupancy_flip_fraction=float((old_occ!=new_occ).float().mean()),
                  candidate_feature_delta_normalized_mae=float(delta.mean()),
                  target_cells_feature_delta_normalized_mae=float(delta[index].mean())))
            hashes[str(shard)]=meta['checkpoint_sha256']
    if len(rows)!=18 or runtime.digest(decoder)!=c['decoder_tensor_digest']:
        raise RuntimeError('Unexpected response coverage or decoder changed')
    output=dict(status='complete',rows=rows,starts=[35,123],sweep=10,
      source=dict(stage_a_code_sha256=EXPECTED_CODE,decoder_digest=c['decoder_tensor_digest'],
                  baseline_codes_sha256=c['bindings']['codes'],checkpoint_sha256=hashes),
      script_sha256=binding.sha(ROOT/'state_response.py'),elapsed_seconds=time.monotonic()-began,
      scope='CPU state response on two TRAIN windows; normalized differences are not motion success or RGB quality')
    with (ROOT/'STATE_RESPONSE.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: fixed decoder response to sweep-10 codes on both anchor windows')


if __name__=='__main__':main()
