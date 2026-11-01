"""Evaluate a completed C posterior against its reviewed frozen teacher."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

from posterior_eval_core import extract_controls,check_prefix_causality,teacher_comparison

ROOT=Path(__file__).resolve().parent
OUTPUTS=Path('/data/CJH/AGS-V6Direct/outputs')
B_ROOT=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
C_ROOT=OUTPUTS/'gaussiangpt_neoverse_stage_c_training_prepare_20261102_v1'
VARIANTS=OUTPUTS/'gaussiangpt_neoverse_posterior_variants_20261102_v2'
STAGE_A=OUTPUTS/'gaussiangpt_neoverse_action_stage_a_20261102_v1'


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def verify_manifest(root,expected):
    manifest=Path(root)/'CODE.sha256'
    if file_sha(manifest)!=expected:raise RuntimeError('Source manifest hash changed: '+str(root))
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name or file_sha(Path(root)/name)!=digest:
            raise RuntimeError('Source file changed: '+name)


@torch.no_grad()
def static_motion_from_codes(decoder,bridge,store,p,correct,static,starts,out):
    """Reuse the exact extracted codes; compare both targets at the same camera."""
    from support_decoder import select_state
    from runtime import render,panel,cpu
    rows=[];folder=out/'static_motion';folder.mkdir()
    for start in starts:
        w=store.get(start);device=w['condition'].coordinates.device
        motion=decoder(w['condition'],correct[start].to(device))
        still=decoder(w['condition'],static[start].to(device))
        for h in (1,3,6,9):
            sample=store.sample(start,h);record=sample['record'];images=[]
            for prediction in (motion,still):
                state=select_state(w['condition'].coordinates,prediction['features'][h-1],
                                   prediction['logits'][h-1],p['max_predicted_latent_nodes'])
                image,_,_=render(bridge,state,record,views=(0,1));images.append(image)
            moving_ref,_,_=render(bridge,w['states'][h],record,views=(0,1))
            static_ref,_,_=render(bridge,w['states'][0],record,views=(0,1))
            difference=(moving_ref[0]-static_ref[0]).abs().mean(-1)
            weight=(difference/.1).clamp(0,1);denominator=weight.sum().clamp_min(1e-6)
            def weighted(pred,target):return float((weight*(pred-target).abs().mean(-1)).sum()/denominator)
            rows.append(dict(start=start,horizon=h,code_rms_difference=float((correct[start]-static[start]).square().mean().sqrt()),
                horizon_code_rms_difference=float((correct[start,h-1]-static[start,h-1]).square().mean().sqrt()),
                motion_code_to_motion_reference=weighted(images[0][0],moving_ref[0]),
                motion_code_to_static_reference=weighted(images[0][0],static_ref[0]),
                static_code_to_static_reference=weighted(images[1][0],static_ref[0]),
                static_code_to_motion_reference=weighted(images[1][0],moving_ref[0]),
                gt_motion_weight_sum=float(weight.sum()),same_initial_coordinates_and_features=True,
                same_target_camera=True,reference_renderer='frozen GaussianGPT codec'))
            panel(folder/f's{start:03d}_h{h:02d}.png',[
                ('MOVING TARGET',cpu(moving_ref[0])),('MOVING CODE',cpu(images[0][0])),
                ('STATIC TARGET',cpu(static_ref[0])),('STATIC CODE',cpu(images[1][0]))])
    return dict(status='complete',rows=rows,source_starts=list(starts),horizons=[1,3,6,9],
      scope='Same S0, original times, camera and frozen GS codec; not cross-source action transfer',
      action_success='requires dynamic and visual interpretation')


def load_trained(run_root,step,receipt_path):
    for root in (B_ROOT,C_ROOT,VARIANTS):
        witness=json.loads((root/'CPU_TESTS.json').read_text(encoding='utf-8'))
        if witness['status']!='passed':raise RuntimeError('Source CPU witness failed: '+str(root))
        verify_manifest(root,witness['code_sha256'])
    sys.path.insert(0,str(B_ROOT));sys.path.insert(0,str(C_ROOT))
    from b_core import load_bound_sources
    from teacher_receipt import validate
    from c_worker import load_training_core
    bundle=load_bound_sources('cpu');binding=bundle['binding'];p=bundle['p'];runtime=bundle['runtime']
    done=binding.read(run_root/'TRAIN_DONE.json');path=run_root/f'step_{step:05d}.pt'
    if done['status']!='complete' or done['updates']!=step or binding.sha(path)!=done['final_checkpoint_sha256']:
        raise RuntimeError('A completed fixed final C checkpoint is required')
    trained=torch.load(path,map_location='cpu',weights_only=True)
    if (trained['schema']!='fixed_teacher_posterior_v1' or trained['step']!=step
            or trained['source']!=done['source'] or trained['variant'] not in ('C0','C1','C2')):
        raise RuntimeError('Posterior checkpoint provenance differs')
    receipt=binding.read(receipt_path);checkpoint=torch.load(receipt['decoder_checkpoint']['path'],map_location='cpu',weights_only=True)
    decoder=bundle['decoder'];decoder.load_state_dict(checkpoint[receipt['decoder_checkpoint'].get('state_key','decoder')],strict=True)
    decoder.requires_grad_(False).eval()
    teacher=validate(receipt_path,decoder,runtime,p['codec_checkpoint_sha256'],binding.sha(bundle['paths']['cache_manifest']))
    if teacher['receipt_sha256']!=trained['source']['teacher_receipt_sha256']:
        raise RuntimeError('C model and supplied teacher receipt differ')
    variants=binding.explicit('_stage_c_evaluation_variants',VARIANTS/'posterior_variants.py')
    code_witness=binding.read(VARIANTS/'CPU_TESTS.json')
    if (code_witness['status']!='passed' or code_witness['code_sha256']!=binding.sha(VARIANTS/'CODE.sha256')):
        raise RuntimeError('Posterior implementation witness changed')
    state=trained['model'];time_features=any(k.startswith('time_projection.') for k in state)
    change_features=any(k.startswith('delta_projection.') for k in state)
    model=variants.ContinuousPosterior(decoder.base,bundle['posterior'],trained['variant'],
                                       include_time=time_features,include_change=change_features)
    model.load_state_dict(state,strict=True);model.eval()
    if runtime.digest(model)!=done['model_digest'] or runtime.digest(decoder)!=teacher['receipt']['decoder_tensor_digest']:
        raise RuntimeError('Trained posterior or fixed decoder tensor digest differs')
    return bundle,teacher,trained,decoder,model,load_training_core()


def evaluate(run_root,step,receipt_path):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    run_root=Path(run_root);receipt_path=Path(receipt_path)
    bundle,teacher,trained,decoder,model,core=load_trained(run_root,step,receipt_path)
    binding=bundle['binding'];runtime=bundle['runtime'];store=bundle['store'];p=bundle['p'];c=bundle['c']
    tests=binding.read(ROOT/'CPU_TESTS.json')
    if tests['status']!='passed':
        raise RuntimeError('Evaluation logic witness changed')
    verify_manifest(ROOT,tests['code_sha256'])
    driver_test=binding.read(ROOT/'DRIVER_TEST_V2.json')
    if (driver_test['status']!='passed' or driver_test['driver_sha256']!=binding.sha(ROOT/'c_evaluate_v2.py')
            or driver_test['test_sha256']!=binding.sha(ROOT/'driver_tests_v2.py')
            or driver_test['core_manifest_sha256']!=binding.sha(ROOT/'CODE.sha256')):
        raise RuntimeError('C evaluation driver witness changed')
    mode=trained['variant'];folder=run_root/f'step_{step:05d}';folder.mkdir();out=folder/mode;out.mkdir()
    binding.write(out/'STARTED.json',dict(pid=__import__('os').getpid(),mode=mode,step=step))
    began=time.monotonic();phase='initializing'
    try:
        binding.setup_gpu(p);decoder=decoder.cuda();model=model.cuda().eval()
        from runtime_bridge import BoundNativeBridge
        bridge=BoundNativeBridge(partial=False,device='cuda').eval()
        phase='extracting_posterior_and_static_codes'
        correct,static=extract_controls(model,store,core.infer_codes,p['starts'])
        if runtime.digest(model)!=done_digest(trained,run_root,binding):raise RuntimeError('Posterior changed during extraction')
        causal=check_prefix_causality(model,store)
        binding.write(out/'CAUSALITY.json',causal)
        phase='same_initial_motion_static'
        static_result=static_motion_from_codes(decoder,bridge,store,p,correct,static,p['diagnostic_starts'],out)
        binding.write(out/'STATIC_CONTROL.json',static_result)
        source=dict(schema='fixed_teacher_posterior_evaluation_v1',decoder_digest=teacher['receipt']['decoder_tensor_digest'],
          codec_sha256=p['codec_checkpoint_sha256'],cache_manifest_sha256=binding.sha(bundle['paths']['cache_manifest']),
          teacher_evaluation_sha256=teacher['receipt']['evaluation']['sha256'],
          teacher_codes_sha256=teacher['receipt']['codes']['sha256'],teacher_receipt_sha256=teacher['receipt_sha256'],
          posterior_checkpoint_sha256=binding.sha(run_root/f'step_{step:05d}.pt'),variant=mode,step=step,quantized=False)
        evaluator=binding.explicit('_stage_c_full_evaluator',ROOT/'evaluate_shared.py')
        phase='evaluating_all_windows'
        evaluator.evaluate(decoder,bridge,store,p,out,source,mode,correct,static,
                           teacher['receipt']['evaluation']['path'],teacher['receipt']['codes']['path'])
        result=binding.read(out/'evaluation/EVALUATION.json');comparison=teacher_comparison(result)
        anchor_test=binding.read(B_ROOT/'EVAL_TEST.json')
        if anchor_test['status']!='passed' or anchor_test['driver_sha256']!=binding.sha(B_ROOT/'b_evaluate.py'):
            raise RuntimeError('Anchor metric source changed')
        sys.path.insert(0,str(B_ROOT));from b_evaluate import anchor_metrics
        anchors=anchor_metrics([r for r in result['rows'] if r['horizon']>0],store,p)
        thresholds=c['anchor_proxy_thresholds']
        def passes(x):
            return (x['visible_target_frames']>0 and x['retention_fraction']>=thresholds['retention_min']
              and x['center_within_10pct_fraction']>=thresholds['center_pass_fraction_min']
              and x['median_axis_error_degrees'] is not None
              and x['median_axis_error_degrees']<=thresholds['median_axis_error_degrees_max']
              and x['maximum_consecutive_missing_frames']<=thresholds['maximum_missing_run'])
        binding.write(out/'STAGE_METRICS.json',dict(status='complete',comparison=comparison,dynamic_anchors=anchors,
          preliminary_known_anchor_proxy_pass=all(passes(v) for v in anchors.values()),
          causal_test='passed',same_initial_motion_static='reported; requires visual interpretation',
          manual_visual_review_required=True,unannotated_other_windows_not_claimed_success=True))
        binding.write(out/'STATUS.json',dict(status='complete',phase='evaluation',native_replay='pending',
                      elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        binding.write(out/'FAILED.json',failure);binding.write(out/'STATUS.json',failure,replace=True)
        raise


def done_digest(trained,run_root,binding):
    done=binding.read(run_root/'TRAIN_DONE.json')
    if trained['source']!=done['source']:raise RuntimeError('Trained model source changed')
    return done['model_digest']


def replay(run_root,step,mode):
    if mode not in ('C0','C1','C2'):raise ValueError('Unknown posterior variant')
    spec=importlib.util.spec_from_file_location('_stage_c_replay_binding',STAGE_A/'stage_context.py')
    binding=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binding;spec.loader.exec_module(binding)
    c,helper,orig,p,paths,receipt,evaluation=binding.setup()
    module=binding.explicit('_stage_c_native_replay',orig/'replay.py')
    module.ROOT=Path(run_root)/f'step_{step:05d}';module.main(mode)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['evaluate','replay'])
    parser.add_argument('--run-root',required=True);parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--receipt');parser.add_argument('--mode',choices=['C0','C1','C2'])
    args=parser.parse_args()
    if args.phase=='evaluate':
        if not args.receipt:parser.error('--receipt required for evaluation')
        evaluate(args.run_root,args.step,args.receipt)
    else:
        if not args.mode:parser.error('--mode required for replay')
        replay(args.run_root,args.step,args.mode)
