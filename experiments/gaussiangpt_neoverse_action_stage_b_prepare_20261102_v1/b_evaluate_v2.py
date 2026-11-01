"""Evaluate one completed shared-decoder/free-code checkpoint on all windows."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import numpy as np
import torch

from b_core import STAGE_A,load_bound_sources
from gpu_runtime import setup_gpu_once

ROOT=Path(__file__).resolve().parent
PARENT_EVAL_SHA='74692f5757622aa150e6617304fbb8786060ec0ceb7f0dd7b105d898cc4d53d7'


def anchor_metrics(rows,store,p):
    from core import color_mask
    result={}
    for start in (35,123):
        sequence=sorted((r for r in rows if r['start']==start and r['horizon']>0),key=lambda r:r['horizon'])
        if len(sequence)!=9:raise RuntimeError('Anchor window incomplete')
        frames=[]
        for row in sequence:
            proxy=row['metrics']['correct']['object_proxy']
            target=store.frames['samples'][row['frame']]['record']['real'].cpu().numpy()
            y,x=np.nonzero(color_mask(target,p['object_proxies'][str(start)]))
            diagonal=float(np.hypot(x.max()-x.min()+1,y.max()-y.min()+1)) if len(x)>=10 else None
            center=proxy['centroid_error_px']
            frames.append(dict(frame=row['frame'],horizon=row['horizon'],target_visible=proxy['target']['valid'],
                prediction_visible=proxy['prediction']['valid'],
                centroid_error_over_target_bbox_diagonal=(center/diagonal if diagonal and center is not None else None),
                axis_error_degrees=proxy['axis_error_degrees'],iou=proxy['iou'],
                target_centroid=proxy['target']['centroid_xy'],prediction_centroid=proxy['prediction']['centroid_xy']))
        visible=[x for x in frames if x['target_visible']]
        retained=sum(x['prediction_visible'] for x in visible)
        center_pass=sum(x['centroid_error_over_target_bbox_diagonal'] is not None and
                        x['centroid_error_over_target_bbox_diagonal']<=.1 for x in visible)
        angles=[x['axis_error_degrees'] for x in visible if x['axis_error_degrees'] is not None]
        run=longest=0
        for x in frames:
            run=run+1 if x['target_visible'] and not x['prediction_visible'] else 0;longest=max(longest,run)
        result[str(start)]=dict(name=p['object_proxies'][str(start)]['name'],visible_target_frames=len(visible),
            retention_fraction=retained/max(len(visible),1),center_within_10pct_fraction=center_pass/max(len(visible),1),
            median_axis_error_degrees=statistics.median(angles) if angles else None,
            maximum_consecutive_missing_frames=longest,frames=frames,
            scope='HSV/ROI proxy, not tracked 3D pose or action success')
    return result


@torch.no_grad()
def warm_initial_codes(posterior,store,starts):
    posterior.eval();values=[]
    for start in starts:
        w=store.get(start);code=posterior([w['initial_memories']])['codes'][0]
        if code.shape!=(9,16) or code.dtype!=torch.float32 or not bool(torch.isfinite(code).all()):
            raise RuntimeError('Invalid repeated-S0 warm Q code')
        values.append(code.detach().cpu())
    return torch.stack(values)


def evaluate(run_root,sweep):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    bundle=load_bound_sources('cpu');binding=bundle['binding'];p=bundle['p'];c=bundle['c']
    for line in (ROOT/'CODE.sha256').read_text().splitlines():
        expected,name=line.split('  ',1)
        if Path(name).name!=name or binding.sha(ROOT/name)!=expected:raise RuntimeError('Stage-B source changed: '+name)
    witness=binding.read(ROOT/'EVAL_TEST_V2.json')
    if (witness['status']!='passed' or witness['driver_sha256']!=binding.sha(ROOT/'b_evaluate_v2.py')
            or witness['evaluator_sha256']!=binding.sha(ROOT/'evaluate_b.py')
            or witness['test_sha256']!=binding.sha(ROOT/'evaluation_tests_v2.py')
            or witness['core_manifest_sha256']!=binding.sha(ROOT/'CODE.sha256')):
        raise RuntimeError('Stage-B evaluator tests changed')
    gpu_test=binding.read(ROOT/'GPU_RUNTIME_TEST.json')
    if (gpu_test['status']!='passed' or gpu_test['runtime_sha256']!=binding.sha(ROOT/'gpu_runtime.py')
            or gpu_test['test_sha256']!=binding.sha(ROOT/'gpu_runtime_tests.py')):
        raise RuntimeError('One-time GPU setup witness changed')
    fm,runtime,store=bundle['fm'],bundle['runtime'],bundle['store']
    run_root=Path(run_root);done=binding.read(run_root/'TRAIN_DONE.json')
    if done['status']!='complete' or sweep not in (5,10,20):raise RuntimeError('Completed fixed-budget B run required')
    path=run_root/f'sweep_{sweep:02d}.pt';meta=binding.read(run_root/f'sweep_{sweep:02d}.json')
    if meta['status']!='complete' or binding.sha(path)!=meta['sha256']:
        raise RuntimeError('Stage-B checkpoint changed')
    item=torch.load(path,map_location='cpu',weights_only=True)
    if (item['schema']!='stage_b_shared_decoder_free_codes_v1' or item['sweep']!=sweep
            or item['starts']!=list(range(168)) or item['decoder_digest']!=meta['decoder_digest']
            or item['source']!=done['source']):raise RuntimeError('Stage-B checkpoint provenance differs')
    decoder=bundle['decoder'];decoder.load_state_dict(item['decoder'],strict=True)
    if runtime.digest(decoder)!=item['decoder_digest'] or fm.frozen_digest(decoder)!=bundle['frozen_digest']:
        raise RuntimeError('Stage-B decoder or frozen initial context changed')
    decoder.requires_grad_(False).eval();posterior=bundle['posterior'].eval()
    parent=Path(p['parent_root'])/'free16/evaluation/EVALUATION.json'
    if binding.sha(parent)!=PARENT_EVAL_SHA:raise RuntimeError('Two-window regression reference changed')
    parent_data=binding.read(parent)
    old_rows=[r for r in parent_data['rows'] if r['horizon']>0]
    if len(old_rows)!=18 or {r['start'] for r in old_rows}!={35,123}:
        raise RuntimeError('Two-window regression reference incomplete')
    old_anchor=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in old_rows)/18
    sweep_root=run_root/f'sweep_{sweep:02d}';sweep_root.mkdir();out=sweep_root/'free_z';out.mkdir()
    phase='initializing';began=time.monotonic()
    binding.write(out/'STARTED.json',dict(pid=__import__('os').getpid(),sweep=sweep,mode='free_z'))
    try:
        setup_gpu_once(p);decoder=decoder.cuda();posterior=posterior.cuda()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        with torch.no_grad():
            window=store.get(35);codes=item['codes'][35].cuda();modified=codes.clone();modified[-1]+=2.5
            first=decoder(window['condition'],codes);second=decoder(window['condition'],modified)
            feature_error=float((first['features'][:-1]-second['features'][:-1]).abs().max())
            occupancy_error=float((first['logits'][:-1]-second['logits'][:-1]).abs().max())
            if max(feature_error,occupancy_error)>2e-5:raise RuntimeError('Decoder causal prefix test failed')
            binding.write(out/'CAUSALITY.json',dict(status='passed',perturbed_horizon=9,checked_horizons=list(range(1,9)),
              feature_max_abs_error=feature_error,occupancy_max_abs_error=occupancy_error,tolerance=2e-5))
            del window,codes,modified,first,second
        phase='extracting_warm_initial_codes';initial=warm_initial_codes(posterior,store,p['starts'])
        source=dict(schema='stage_b_shared_decoder_evaluation_v1',decoder_digest=item['decoder_digest'],
          codes_sha256=meta['sha256'],codec_sha256=p['codec_checkpoint_sha256'],
          cache_manifest_sha256=binding.sha(bundle['paths']['cache_manifest']),
          historical_evaluation_sha256=c['bindings']['evaluation'],
          historical_checkpoint_sha256=c['bindings']['checkpoint'],
          warm_posterior_sha256=p['posterior_checkpoint_sha256'],
          training_source=item['source'],sweep=sweep,quantized=False,
          note='Historical Q metrics use a different decoder; warm repeated-S0 Q is not an independently trained stage-B baseline')
        evaluator=binding.explicit('_stage_b_evaluator',ROOT/'evaluate_b.py')
        local_p=dict(p,evaluation_max_seconds=p['evaluation_max_seconds'])
        phase='evaluating_all_windows'
        evaluator.evaluate(decoder,bridge,store,local_p,out,source,initial,bundle['paths']['evaluation'],path)
        if runtime.digest(decoder)!=item['decoder_digest']:raise RuntimeError('Decoder changed during evaluation')
        result=binding.read(out/'evaluation/EVALUATION.json');future=[r for r in result['rows'] if r['horizon']>0]
        controls=['correct','initial_only','swapped_window_codes','actual_target','historical_qformer']
        horizons={str(h):fm.aggregate([r for r in future if r['horizon']==h],controls)['window_prediction_equal']
                  for h in range(1,10)}
        anchors=anchor_metrics(future,store,p)
        current_anchor=sum(r['metrics']['correct']['metrics']['motion_mae'] for r in future if r['start'] in (35,123))/18
        thresholds=c['anchor_proxy_thresholds']
        def passes(x):
            return (x['visible_target_frames']>0 and x['retention_fraction']>=thresholds['retention_min']
              and x['center_within_10pct_fraction']>=thresholds['center_pass_fraction_min']
              and x['median_axis_error_degrees'] is not None
              and x['median_axis_error_degrees']<=thresholds['median_axis_error_degrees_max']
              and x['maximum_consecutive_missing_frames']<=thresholds['maximum_missing_run'])
        binding.write(out/'STAGE_METRICS.json',dict(status='complete',sweep=sweep,metrics=result['metrics'],
          support=result['support'],by_horizon=horizons,dynamic_anchors=anchors,
          anchor_motion_mae=current_anchor,original_two_window_anchor_motion_mae=old_anchor,
          original_two_window_evaluation_sha256=PARENT_EVAL_SHA,
          preliminary_known_anchor_proxy_pass=all(passes(v) for v in anchors.values()),
          manual_visual_review_required=True,unannotated_other_windows_not_claimed_success=True))
        binding.write(out/'STATUS.json',dict(status='complete',phase='evaluation',native_replay='pending',
                     sweep=sweep,elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        binding.write(out/'FAILED.json',failure);binding.write(out/'STATUS.json',failure,replace=True)
        raise


def replay(run_root,sweep):
    # Do not build or import the state/motion model in the independent native
    # renderer process; only use the read-only source-binding helper.
    spec=importlib.util.spec_from_file_location('_stage_b_replay_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    module=binding.explicit('_stage_b_native_replay',orig/'replay.py')
    module.ROOT=Path(run_root)/f'sweep_{sweep:02d}'
    module.main('free_z')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['evaluate','replay'])
    parser.add_argument('--run-root',required=True);parser.add_argument('--sweep',type=int,required=True)
    args=parser.parse_args()
    if args.phase=='evaluate':evaluate(args.run_root,args.sweep)
    else:replay(args.run_root,args.sweep)
