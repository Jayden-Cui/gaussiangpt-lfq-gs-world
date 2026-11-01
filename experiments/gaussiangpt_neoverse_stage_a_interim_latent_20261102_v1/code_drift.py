"""Measure within-run free-code changes using saved stage-A checkpoints."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
EXPECTED_CODE='1bdca9aad668120f028f35e4abb351a664f8ab493d0266d1c9a6781b2ecc5f54'


def stats(base,value):
    difference=value-base
    base_rms=float(base.square().mean().sqrt());delta_rms=float(difference.square().mean().sqrt())
    per_window=difference.flatten(1).square().mean(1).sqrt()
    return dict(base_rms=base_rms,updated_rms=float(value.square().mean().sqrt()),delta_rms=delta_rms,
      delta_over_base_rms=delta_rms/max(base_rms,1e-12),
      per_window_delta_rms_median=float(per_window.median()),
      per_window_delta_rms_p95=float(torch.quantile(per_window,.95)),
      changed_windows=int((per_window>0).sum()),
      by_horizon=[dict(horizon=h+1,base_rms=float(base[:,h].square().mean().sqrt()),
                       delta_rms=float(difference[:,h].square().mean().sqrt())) for h in range(9)])


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('_drift_stage_a_context',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    if binding.sha(STAGE_A/'CODE.sha256')!=EXPECTED_CODE:raise RuntimeError('Stage-A code manifest changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    base=binding.load_codes(paths,evaluation)['correct'];saved={};hashes={}
    for sweep in (5,10):
        assembled=torch.empty_like(base);seen=set()
        for shard in (0,1):
            folder=STAGE_A/f'shard_{shard}';path=folder/f'sweep_{sweep:02d}.pt'
            meta=binding.read(folder/f'sweep_{sweep:02d}.json')
            if meta['status']!='complete' or meta['sweep']!=sweep or binding.sha(path)!=meta['checkpoint_sha256']:
                raise RuntimeError('Intermediate code checkpoint changed')
            item=torch.load(path,map_location='cpu',weights_only=True)
            if (item['schema']!='fixed_decoder_free_codes_shard_v1' or item['shard']!=shard
                or item['sweep']!=sweep or item['starts']!=c['shards'][str(shard)]
                or tuple(item['codes'].shape)!=(84,9,16) or item['codes'].dtype!=torch.float32
                or item['source']['decoder_digest']!=c['decoder_tensor_digest']
                or item['source']['baseline_codes_sha256']!=c['bindings']['codes']):
                raise RuntimeError('Code checkpoint binding differs')
            for i,start in enumerate(item['starts']):
                if start in seen:raise RuntimeError('Duplicate code window')
                assembled[start]=item['codes'][i];seen.add(start)
            hashes[f'{sweep:02d}_{shard}']=meta['checkpoint_sha256']
        if seen!=set(range(168)) or not bool(torch.isfinite(assembled).all()):
            raise RuntimeError('Incomplete or nonfinite code coverage')
        saved[sweep]=assembled
    output=dict(status='complete',windows=168,code_shape=[168,9,16],
      baseline_to_sweep05=stats(base,saved[5]),baseline_to_sweep10=stats(base,saved[10]),
      sweep05_to_sweep10=stats(saved[5],saved[10]),
      anchors={str(s):dict(baseline_to_sweep05=stats(base[s:s+1],saved[5][s:s+1]),
                            baseline_to_sweep10=stats(base[s:s+1],saved[10][s:s+1])) for s in (35,123)},
      source=dict(stage_a_code_sha256=EXPECTED_CODE,baseline_codes_sha256=c['bindings']['codes'],checkpoint_sha256=hashes),
      script_sha256=binding.sha(ROOT/'code_drift.py'),
      scope='Same decoder and numerical code coordinates, TRAIN video only; code distance alone does not measure action quality or semantics')
    with (ROOT/'CODE_DRIFT.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: all 168 windows of fixed-decoder free codes compared at sweeps 5 and 10')


if __name__=='__main__':main()
