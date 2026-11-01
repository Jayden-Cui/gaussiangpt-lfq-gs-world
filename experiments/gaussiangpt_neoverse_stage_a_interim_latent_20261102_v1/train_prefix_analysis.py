"""Analyze complete, fixed first-ten-sweep training-log prefixes."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
STAGE_A=Path('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_action_stage_a_20261102_v1')
EXPECTED_CODE='1bdca9aad668120f028f35e4abb351a664f8ab493d0266d1c9a6781b2ecc5f54'


def summarize(values):
    keys=('loss','rgb_loss','latent_loss','occupancy_loss','feature_loss')
    if not values:raise RuntimeError('Empty training subset')
    return {name:sum(float(row[name]) for row in values)/len(values) for name in keys}


def main():
    spec=importlib.util.spec_from_file_location('_training_prefix_context',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    if binding.sha(STAGE_A/'CODE.sha256')!=EXPECTED_CODE:raise RuntimeError('Stage-A source changed')
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    groups={s:[] for s in range(1,11)};prefix_hashes={};checkpoints={};clip_counts={s:0 for s in range(1,11)}
    for shard in (0,1):
        folder=STAGE_A/f'shard_{shard}';meta=binding.read(folder/'sweep_10.json');checkpoint=folder/'sweep_10.pt'
        if meta['status']!='complete' or binding.sha(checkpoint)!=meta['checkpoint_sha256']:
            raise RuntimeError('Sweep-10 checkpoint incomplete or changed')
        checkpoints[str(shard)]=meta['checkpoint_sha256'];digest=hashlib.sha256()
        with (folder/'train.jsonl').open('rb') as stream:
            for index in range(1890):
                line=stream.readline()
                if not line.endswith(b'\n'):raise RuntimeError('Incomplete log prefix')
                digest.update(line);row=json.loads(line)
                if (row['step']!=index+1 or row['sweep']!=index//189+1 or len(row['windows'])!=4):
                    raise RuntimeError('Training prefix step/sweep/batch mismatch')
                clip_counts[row['sweep']]+=int(row['clip_triggered'])
                groups[row['sweep']].extend(row['windows'])
        prefix_hashes[str(shard)]=digest.hexdigest()
    expected={(s,h) for s in range(168) for h in range(1,10)};sweeps={}
    for sweep,values in groups.items():
        pairs=Counter((row['start'],row['horizon']) for row in values)
        if len(values)!=1512 or set(pairs)!=expected or set(pairs.values())!={1}:
            raise RuntimeError('A sweep did not cover every window/horizon exactly once')
        anchors={str(start):summarize([row for row in values if row['start']==start]) for start in (35,123)}
        sweeps[str(sweep)]=dict(examples=len(values),mean=summarize(values),anchors=anchors,
                                clipped_updates=clip_counts[sweep],total_updates=378)
    output=dict(status='complete',sweeps=sweeps,shard_updates_in_prefix=1890,
      checkpoint_sha256=checkpoints,log_prefix_sha256=prefix_hashes,stage_a_code_sha256=EXPECTED_CODE,
      script_sha256=binding.sha(ROOT/'train_prefix_analysis.py'),
      scope='Training losses observed during each sweep; not a fixed-checkpoint evaluation, render-success metric or held-out result',
      rgb_loss_definition='full-image MAE + fixed hand ROI MAE + (10/9) motion-weighted MAE',
      caution='Parameters change within each sweep; comparisons are training curves on the same finite pair set.')
    with (ROOT/'TRAIN_PREFIX_ANALYSIS.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('PASS: ten complete full-clip training sweeps with exact pair coverage and prefix hashes')


if __name__=='__main__':main()
