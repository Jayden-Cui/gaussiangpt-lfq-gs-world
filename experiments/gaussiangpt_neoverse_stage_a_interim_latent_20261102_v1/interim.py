"""Read completed stage-A code checkpoints for two-window CPU diagnostics."""
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
    spec=importlib.util.spec_from_file_location('_interim_stage_a_context',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or binding.sha(ROOT/name)!=digest:raise RuntimeError('Interim source changed: '+name)
    if binding.sha(STAGE_A/'CODE.sha256')!=EXPECTED_CODE:raise RuntimeError('Stage-A code manifest changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    decoder,modules=binding.load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    fm,data,runtime=modules;store=binding.dataset(modules,p,c);baseline=binding.load_codes(paths,evaluation)['correct']
    from support_decoder import support_metrics
    codes={'baseline_qformer':{35:baseline[35],123:baseline[123]}};checkpoint_sha={}
    for sweep in (5,10):
        codes[f'free_sweep_{sweep:02d}']={}
        for shard,start in ((0,35),(1,123)):
            folder=STAGE_A/f'shard_{shard}';path=folder/f'sweep_{sweep:02d}.pt'
            meta=binding.read(folder/f'sweep_{sweep:02d}.json')
            if meta['status']!='complete' or meta['sweep']!=sweep or binding.sha(path)!=meta['checkpoint_sha256']:
                raise RuntimeError('Intermediate checkpoint incomplete or changed')
            packet=torch.load(path,map_location='cpu',weights_only=True)
            if (packet['schema']!='fixed_decoder_free_codes_shard_v1' or packet['sweep']!=sweep
                    or packet['shard']!=shard or packet['source']['decoder_digest']!=c['decoder_tensor_digest']
                    or packet['source']['baseline_codes_sha256']!=c['bindings']['codes']
                    or packet['starts']!=c['shards'][str(shard)]):
                raise RuntimeError('Intermediate code source or window order changed')
            codes[f'free_sweep_{sweep:02d}'][start]=packet['codes'][packet['starts'].index(start)]
            checkpoint_sha[f'{sweep:02d}_{shard}']=meta['checkpoint_sha256']
    rows=[];std=store.frames['std'].clamp_min(1e-6)
    with torch.no_grad():
        for start in (35,123):
            window=store.get(start,device='cpu')
            for condition,values in codes.items():
                code=values[start]
                if code.shape!=(9,16) or not bool(torch.isfinite(code).all()):raise RuntimeError('Invalid free code')
                prediction=decoder(window['condition'],code)
                for h in range(1,10):
                    indices=window['targets']['indices'][h-1];target=window['targets']['features'][h-1]
                    features=prediction['features'][h-1,indices]
                    normalized=float(((features-target).abs()/std).mean())
                    support=support_metrics(prediction['logits'][h-1],window['targets']['occupancy'][h-1])
                    saturation=float((prediction['raw_residual'][h-1,indices].tanh().abs()>.95).float().mean())
                    rows.append(dict(start=start,horizon=h,frame=start+h,condition=condition,
                         target_feature_normalized_mae=normalized,support=support,
                         target_grid_tanh_abs_over_095_fraction=saturation))
    if len(rows)!=54 or runtime.digest(decoder)!=c['decoder_tensor_digest']:
        raise RuntimeError('Intermediate diagnostic coverage or frozen decoder differs')
    summary={}
    for start in (35,123):
        summary[str(start)]={}
        for name in codes:
            selected=[row for row in rows if row['start']==start and row['condition']==name]
            summary[str(start)][name]=dict(mean_target_feature_normalized_mae=sum(r['target_feature_normalized_mae'] for r in selected)/9,
              mean_support_iou=sum(r['support']['iou'] for r in selected)/9,
              h9=next(r for r in selected if r['horizon']==9))
    output=dict(status='complete',training_performed=False,decoder_or_codec_modified=False,
      source=dict(stage_a_code_sha256=EXPECTED_CODE,decoder_digest=c['decoder_tensor_digest'],
        baseline_code_sha256=c['bindings']['codes'],cache_manifest_sha256=c['bindings']['cache_manifest'],
        checkpoint_sha256=checkpoint_sha),starts=[35,123],sweeps=[5,10],rows=rows,summary=summary,
      scope='CPU latent and support diagnostics for two trained windows only; no RGB rendering, no final checkpoint, no action-success claim',
      script_sha256=binding.sha(ROOT/'interim.py'),code_manifest_sha256=binding.sha(ROOT/'CODE.sha256'),
      elapsed_seconds=time.monotonic()-began)
    with (ROOT/'INTERIM.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: two-window h1..h9 latent/support comparison at baseline and fixed sweeps 5/10')


if __name__=='__main__':main()
