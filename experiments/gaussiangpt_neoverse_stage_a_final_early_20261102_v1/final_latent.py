"""CPU-only final-sweep code movement and two-window latent diagnostics."""
import importlib.util
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
EXPECTED_CODE='1bdca9aad668120f028f35e4abb351a664f8ab493d0266d1c9a6781b2ecc5f54'


def analyze(sweep):
    began=time.monotonic()
    if sweep not in (10,20):raise ValueError('Only verified test sweep 10 or final sweep 20')
    spec=importlib.util.spec_from_file_location('_final_latent_context',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    if binding.sha(STAGE_A/'CODE.sha256')!=EXPECTED_CODE:raise RuntimeError('Stage-A source changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    decoder,modules=binding.load_decoder(c,helper,orig,p,paths,receipt,'cpu')
    store=binding.dataset(modules,p,c);runtime=modules[2];base=binding.load_codes(paths,evaluation)['correct']
    from support_decoder import support_metrics
    updated=torch.empty_like(base);seen=set();hashes={}
    for shard in (0,1):
        folder=STAGE_A/f'shard_{shard}';path=folder/f'sweep_{sweep:02d}.pt'
        meta=binding.read(folder/f'sweep_{sweep:02d}.json')
        if meta['status']!='complete' or binding.sha(path)!=meta['checkpoint_sha256']:
            raise RuntimeError('Saved code checkpoint changed')
        item=torch.load(path,map_location='cpu',weights_only=True)
        if (item['schema']!='fixed_decoder_free_codes_shard_v1' or item['shard']!=shard
                or item['sweep']!=sweep or item['starts']!=c['shards'][str(shard)]
                or tuple(item['codes'].shape)!=(84,9,16) or item['codes'].dtype!=torch.float32
                or item['source']['decoder_digest']!=c['decoder_tensor_digest']
                or item['source']['baseline_codes_sha256']!=c['bindings']['codes']):
            raise RuntimeError('Saved code provenance differs')
        for i,start in enumerate(item['starts']):
            if start in seen:raise RuntimeError('Overlapping shard windows')
            updated[start]=item['codes'][i];seen.add(start)
        hashes[str(shard)]=meta['checkpoint_sha256']
    if seen!=set(range(168)) or not bool(torch.isfinite(updated).all()):
        raise RuntimeError('Incomplete/nonfinite full code set')
    rows=[];std=store.frames['std'].clamp_min(1e-6)
    with torch.no_grad():
        for start in (35,123):
            window=store.get(start,device='cpu')
            for label,code in (('baseline_qformer',base[start]),('free_code',updated[start])):
                prediction=decoder(window['condition'],code)
                for h in range(1,10):
                    indices=window['targets']['indices'][h-1];target=window['targets']['features'][h-1]
                    feature=float(((prediction['features'][h-1,indices]-target).abs()/std).mean())
                    support=support_metrics(prediction['logits'][h-1],window['targets']['occupancy'][h-1])
                    rows.append(dict(start=start,horizon=h,frame=start+h,condition=label,
                                     target_feature_normalized_mae=feature,support=support))
    if len(rows)!=36 or runtime.digest(decoder)!=c['decoder_tensor_digest']:
        raise RuntimeError('Decoder or anchor coverage differs')
    summary={}
    for start in (35,123):
        summary[str(start)]={}
        for label in ('baseline_qformer','free_code'):
            selected=[r for r in rows if r['start']==start and r['condition']==label]
            summary[str(start)][label]=dict(mean_target_feature_normalized_mae=sum(r['target_feature_normalized_mae'] for r in selected)/9,
                mean_support_iou=sum(r['support']['iou'] for r in selected)/9,
                h9=next(r for r in selected if r['horizon']==9))
    delta=updated-base;base_rms=float(base.square().mean().sqrt());delta_rms=float(delta.square().mean().sqrt())
    return dict(status='complete',sweep=sweep,code_shape=[168,9,16],
        code_drift=dict(initial_rms=base_rms,updated_rms=float(updated.square().mean().sqrt()),
          delta_rms=delta_rms,delta_over_initial_rms=delta_rms/max(base_rms,1e-12),
          changed_windows=int((delta.flatten(1).square().mean(1)>0).sum())),
        anchors=summary,rows=rows,source=dict(stage_a_code_sha256=EXPECTED_CODE,
          decoder_digest=c['decoder_tensor_digest'],baseline_codes_sha256=c['bindings']['codes'],
          cache_manifest_sha256=c['bindings']['cache_manifest'],checkpoint_sha256=hashes),
        elapsed_seconds=time.monotonic()-began,
        scope='CPU-only two-window latent diagnostics and code movement; no RGB render or action-success claim')
