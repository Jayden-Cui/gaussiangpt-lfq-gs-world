"""Oracle target-support and bounded-feature reachability diagnostics only."""
from pathlib import Path
import math
import subprocess
import time
import traceback
import numpy as np
from PIL import Image
import torch
from stage_context import ROOT,setup,setup_gpu,load_codes,load_decoder,dataset,source_record,assert_frozen,read,write,sha


def png_pair(path,left,right):
    def image(value):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        return (np.clip(value,0,1)*255).round().astype(np.uint8)
    Image.fromarray(np.concatenate([image(left),image(right)],axis=1)).save(path)


def oracle_support(condition,prediction,h,indices):
    """Select the decoder's predicted features at target indices, after prediction."""
    return dict(coordinates=condition.coordinates[indices],features=prediction['features'][h-1,indices])


def main():
    c,helper,orig,p,paths,receipt,evaluation=setup()
    out=ROOT/'diagnostics';out.mkdir();began=time.monotonic();context=dict(phase='initializing')
    write(out/'STATUS.json',dict(status='running',**context))
    try:
        witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['code_sha256']!=sha(ROOT/'CODE.sha256'):
            raise RuntimeError('CPU witness changed')
        setup_gpu(p)
        decoder,modules=load_decoder(c,helper,orig,p,paths,receipt,'cuda')
        fm,data,runtime=modules
        from runtime_bridge import BoundNativeBridge
        from runtime import render
        from support_decoder import select_state
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        store=dataset(modules,p,c);baseline=load_codes(paths,evaluation)['correct']
        std=store.frames['std'].cuda();bound=3*std
        if not torch.allclose(std,decoder.base.output_std,atol=1e-6,rtol=1e-6):
            raise RuntimeError('Feature residual scale differs')
        source=source_record(c,paths,receipt)
        old={(row['start'],row['horizon']):row for row in evaluation['rows'] if row['horizon']>0}
        if len(old)!=1512:raise RuntimeError('Baseline prediction coverage differs')
        for folder in ['snapshots','videos']:(out/folder).mkdir()
        rows=[];total=dict(target_feature_elements=0,out_of_range_elements=0,target_cells=0,
            cells_with_any_out_of_range=0,target_saturated_095=0,target_saturated_099=0,
            candidate_feature_elements=0,candidate_saturated_095=0,candidate_saturated_099=0,
            normalized_lower_bound_sum=0.)
        with torch.no_grad():
            for start in p['starts']:
                context=dict(phase='diagnosing',completed_windows=start)
                if time.monotonic()-began>c['diagnostic_max_seconds']:raise RuntimeError('Diagnostic deadline')
                w=store.get(start);prediction=decoder(w['condition'],baseline[start].cuda())
                selected=start in c['visual_starts']
                if selected:
                    folder=out/'snapshots'/f's{start:03d}';folder.mkdir()
                    for name in ['pred_vs_oracle','source_vs_oracle']:(folder/name).mkdir()
                    sample=store.sample(start,0)
                    initial,_,_=render(bridge,w['states'][0],sample['record'],views=(0,1))
                    png_pair(folder/'pred_vs_oracle/frame_00.png',initial[0],initial[0])
                    png_pair(folder/'source_vs_oracle/frame_00.png',sample['record']['real'],initial[0])
                    del sample,initial
                for h in range(1,10):
                    sample=store.sample(start,h);idx=w['targets']['indices'][h-1]
                    if (idx.numel()!=int(w['targets']['occupancy'][h-1].sum())
                            or idx.unique().numel()!=idx.numel()):
                        raise RuntimeError('Target occupancy index mismatch')
                    actual=w['targets']['features'][h-1]
                    initial=w['condition'].features[idx]
                    difference=(actual-initial).abs()
                    outside=difference>bound[None]+1e-6
                    excess=(difference-bound[None]).clamp_min(0)/std.clamp_min(1e-6)[None]
                    raw=prediction['raw_residual'][h-1]
                    target_tanh=raw[idx].tanh().abs();all_tanh=raw.tanh().abs()
                    stats=dict(target_feature_elements=outside.numel(),out_of_range_elements=int(outside.sum()),
                        target_cells=len(idx),cells_with_any_out_of_range=int(outside.any(-1).sum()),
                        target_saturated_095=int((target_tanh>.95).sum()),target_saturated_099=int((target_tanh>.99).sum()),
                        candidate_feature_elements=all_tanh.numel(),candidate_saturated_095=int((all_tanh>.95).sum()),
                        candidate_saturated_099=int((all_tanh>.99).sum()),normalized_lower_bound_sum=float(excess.sum()))
                    for key,value in stats.items():total[key]+=value
                    # Ground-truth support is used only for this oracle diagnostic, after prediction.
                    oracle=oracle_support(w['condition'],prediction,h,idx)
                    oracle_images,_,_=render(bridge,oracle,sample['record'],views=(0,1))
                    oracle_metrics=data.evaluate_metrics(oracle_images,w,sample,p)
                    previous=old[(start,h)]
                    row=dict(start=start,horizon=h,frame=start+h,metrics={
                        'predicted_support':previous['metrics']['correct'],
                        'oracle_target_support':oracle_metrics,
                        'actual_target':previous['metrics']['actual_target']},
                        reachability=stats)
                    if selected:
                        state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                           prediction['logits'][h-1],p['max_predicted_latent_nodes'])
                        images,_,_=render(bridge,state,sample['record'],views=(0,1))
                        check=data.evaluate_metrics(images,w,sample,p)['metrics']['motion_mae']
                        if abs(check-previous['metrics']['correct']['metrics']['motion_mae'])>1e-5:
                            raise RuntimeError('Baseline rendering does not reproduce prior evaluation')
                        png_pair(folder/f'pred_vs_oracle/frame_{h:02d}.png',images[0],oracle_images[0])
                        png_pair(folder/f'source_vs_oracle/frame_{h:02d}.png',sample['record']['real'],oracle_images[0])
                        del state,images
                    rows.append(row)
                    del sample,idx,actual,initial,difference,outside,excess,raw,target_tanh,all_tanh,oracle,oracle_images
                if selected:
                    for name in ['pred_vs_oracle','source_vs_oracle']:
                        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','5',
                            '-i',str(folder/name/'frame_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',
                            str(out/'videos'/f's{start:03d}_{name}_5fps.mp4')],check=True,timeout=120)
                if start%8==0 or start==167:
                    write(out/'STATUS.json',dict(status='running',phase='diagnosing',completed_windows=start+1,
                          total_windows=168,elapsed_seconds=time.monotonic()-began),replace=True)
                del w,prediction
        if len(rows)!=1512 or {r['frame'] for r in rows}!=set(range(1,177)):
            raise RuntimeError('Incomplete diagnostic coverage')
        controls=['predicted_support','oracle_target_support','actual_target']
        metrics=fm.aggregate(rows,controls)
        horizon={str(h):fm.aggregate([r for r in rows if r['horizon']==h],controls)['window_prediction_equal']
                 for h in range(1,10)}
        ratios=dict(out_of_range_feature_fraction=total['out_of_range_elements']/total['target_feature_elements'],
            cells_with_any_out_of_range_fraction=total['cells_with_any_out_of_range']/total['target_cells'],
            target_tanh_saturated_095_fraction=total['target_saturated_095']/total['target_feature_elements'],
            target_tanh_saturated_099_fraction=total['target_saturated_099']/total['target_feature_elements'],
            candidate_tanh_saturated_095_fraction=total['candidate_saturated_095']/total['candidate_feature_elements'],
            candidate_tanh_saturated_099_fraction=total['candidate_saturated_099']/total['candidate_feature_elements'],
            mean_normalized_unreachable_feature_error_lower_bound=total['normalized_lower_bound_sum']/total['target_feature_elements'])
        assert_frozen(decoder,bridge,runtime,c['decoder_tensor_digest'])
        write(out/'DIAGNOSTIC.json',dict(status='complete',source=source,window_predictions=1512,
            unique_future_frames=176,metrics=metrics,by_horizon=horizon,reachability_counts=total,
            reachability_ratios=ratios,rows=rows,oracle_used_for_training=False,
            note='Oracle selects predicted features at true support after decoder inference; no true target features are supplied to decoder. Feature bound is not an image-space impossibility proof.',
            elapsed_seconds=time.monotonic()-began))
        write(out/'STATUS.json',dict(status='complete',completed_windows=168,total_windows=168,
              elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',context=context,error=repr(error),traceback=traceback.format_exc())
        write(out/'FAILED.json',failure);write(out/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':main()
