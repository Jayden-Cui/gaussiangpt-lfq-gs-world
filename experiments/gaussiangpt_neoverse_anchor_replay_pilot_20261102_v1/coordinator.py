"""Run the predeclared pilot, full evaluation, replay and review assets once."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from PIL import Image
import worker


ROOT=Path(__file__).resolve().parent
OUTPUTS=ROOT.parent
B_PREP=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1'
B_RUN=OUTPUTS/'gaussiangpt_neoverse_action_stage_b_20261102_v1'
AXIS=OUTPUTS/'gaussiangpt_neoverse_visible_axis_reference_20261102_v1'
EIGHT=OUTPUTS/'gaussiangpt_neoverse_eight_window_object_audit_20261102_v1'


def verify_report_dependencies():
    worker.verify_manifest(AXIS)
    if worker.sha(AXIS/'CODE.sha256')!='f2b5ef152256a879d09eec9e7a1077942bda8081558a0fa4fcc246f68d08538c':
        raise RuntimeError('Visible-axis diagnostic version changed')
    spec=importlib.util.spec_from_file_location('_pilot_eight_window_watcher',EIGHT/'watcher.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    previous.validate_sources()
    witness=worker.read(B_PREP/'EVAL_TEST_V2.json')
    if (witness.get('status')!='passed' or witness.get('driver_sha256')!=worker.sha(B_PREP/'b_evaluate_v2.py')
            or witness.get('evaluator_sha256')!=worker.sha(B_PREP/'evaluate_b.py')
            or witness.get('test_sha256')!=worker.sha(B_PREP/'evaluation_tests_v2.py')
            or witness.get('core_manifest_sha256')!=worker.sha(B_PREP/'CODE.sha256')):
        raise RuntimeError('Full evaluator witness changed')


def review_assets(evaluation_dir,out):
    evaluation_dir=Path(evaluation_dir);out=Path(out)
    evaluation=worker.read(evaluation_dir/'EVALUATION.json');replay=worker.read(evaluation_dir/'REPLAY.json')
    if (evaluation.get('status')!='complete' or evaluation.get('window_predictions')!=1512
            or evaluation['source'].get('sweep')!=5 or replay.get('status')!='passed'
            or replay.get('packets')!=104 or replay.get('views')!=208
            or replay.get('packets_sha256')!=evaluation.get('packets_sha256')):
        raise RuntimeError('Complete pilot evaluation and independent replay required')
    out.mkdir();rows=[]
    for start in (0,35,56,84,112,123,140,167):
        canvas=Image.new('RGB',(1792,504));frames=[]
        for i,h in enumerate((0,3,6,9)):
            path=evaluation_dir/'snapshots'/('s%03d'%start)/('frame_%02d.png'%h)
            with Image.open(path) as image:
                if image.size!=(1792,276):raise RuntimeError('Unexpected diagnostic image size')
                canvas.paste(image.crop((0,24,896,276)),((i%2)*896,(i//2)*252))
            frames.append(dict(horizon=h,path=str(path),sha256=worker.sha(path)))
        path=out/('s%03d_contact.png'%start);canvas.save(path)
        rows.append(dict(start=start,file=path.name,sha256=worker.sha(path),frames=frames))
    result=dict(status='complete',sweep=5,evaluation_sha256=worker.sha(evaluation_dir/'EVALUATION.json'),
        native_replay_sha256=worker.sha(evaluation_dir/'REPLAY.json'),starts=[0,35,56,84,112,123,140,167],
        horizons=[0,3,6,9],rows=rows,labels_in_image=False,visual_verdict='not yet reviewed')
    worker.write(out/'CONTACTS.json',result)
    return result


def coordinate(ready_fd=None):
    protocol,_,sampler=worker.verify_sources()
    verify_report_dependencies()
    run=ROOT/'run'
    if run.exists():raise RuntimeError('Pilot run already exists; no automatic retry')
    run.mkdir();began=time.monotonic();phase='initializing'
    worker.write(run/'STARTED.json',dict(pid=os.getpid(),sampler=sampler,quantized=False,
        scope='Five full-base sweeps with anchor replay; not a trained posterior or teacher acceptance'))
    try:
        def child(name,timeout,command,gpu=True,signal=False):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='6' if gpu else '1',
                     MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            with (run/(name+'.log')).open('x',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-B']+command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                worker.write(run/(name+'_PROCESS.json'),dict(pid=proc.pid,gpu=0 if gpu else None,
                    command=command,timeout_seconds=timeout))
                if signal and ready_fd is not None:
                    os.write(ready_fd,b'ready\n');os.close(ready_fd)
                try:status=proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError(name+' exceeded its fixed budget')
            if status!=0:raise RuntimeError(name+' failed with exit code '+str(status))
        training=run/'training';phase='training'
        worker.write(run/'STATUS.json',dict(status='running',phase=phase),replace=True)
        child('train',protocol['max_training_seconds']+60,[str(ROOT/'worker.py'),'--out',str(training)],signal=True)
        done=worker.read(training/'TRAIN_DONE.json')
        if (done.get('status')!='complete' or done.get('updates')!=2115 or done.get('sweeps')!=5
                or done.get('code_shape')!=[168,9,16] or done.get('codec_and_initial_encoder_frozen') is not True
                or done['source']['sampler']!=sampler
                or done['final_checkpoint_sha256']!=worker.sha(training/'sweep_05.pt')):
            raise RuntimeError('Pilot training receipt differs')
        phase='full_evaluation'
        worker.write(run/'STATUS.json',dict(status='running',phase=phase),replace=True)
        child('eval_05',protocol['evaluation_timeout_seconds'],
              [str(B_PREP/'b_evaluate_v2.py'),'evaluate','--run-root',str(training),'--sweep','5'])
        out=training/'sweep_05/free_z';evaluation=out/'evaluation';raw=worker.read(evaluation/'EVALUATION.json')
        if (raw.get('status')!='complete' or raw.get('window_predictions')!=1512
                or raw.get('unique_future_frames')!=176 or raw.get('decoder_future_geometry_input') is not False
                or raw['source']['sweep']!=5 or worker.sha(evaluation/'PACKETS.json')!=raw['packets_sha256']):
            raise RuntimeError('Pilot full evaluation differs')
        phase='native_gs_replay'
        worker.write(run/'STATUS.json',dict(status='running',phase=phase),replace=True)
        child('replay_05',protocol['replay_timeout_seconds'],
              [str(B_PREP/'b_evaluate_v2.py'),'replay','--run-root',str(training),'--sweep','5'])
        replay=worker.read(evaluation/'REPLAY.json')
        if (replay.get('status')!='passed' or replay.get('packets')!=104 or replay.get('views')!=208
                or replay.get('encoder_or_codec_loaded') is not False
                or replay.get('packets_sha256')!=raw['packets_sha256']):
            raise RuntimeError('Pilot independent GS replay differs')
        phase='axis_and_eight_window_reports'
        worker.write(run/'STATUS.json',dict(status='running',phase=phase),replace=True)
        eval_sha=worker.sha(evaluation/'EVALUATION.json')
        child('axis_report',300,[str(AXIS/'axis_audit.py'),'--evaluation-root',str(evaluation),
              '--expected-evaluation-sha256',eval_sha,'--output',str(run/'AXIS_REPORT')],gpu=False)
        baseline=EIGHT/'result_baseline/OBJECT_AUDIT.json'
        child('eight_window_report',300,[str(EIGHT/'audit.py'),'--evaluation-root',str(evaluation),
              '--expected-evaluation-sha256',eval_sha,'--reference-audit',str(baseline),
              '--expected-reference-audit-sha256',worker.sha(baseline),'--output',str(run/'EIGHT_WINDOW_REPORT')],gpu=False)
        assets=review_assets(evaluation,run/'REVIEW_ASSETS')
        pilot_metrics=worker.read(out/'STAGE_METRICS.json')
        old=worker.read(B_RUN/'training/sweep_05/free_z/STAGE_METRICS.json')
        if (pilot_metrics.get('status')!='complete' or pilot_metrics.get('sweep')!=5
                or old.get('status')!='complete' or old.get('sweep')!=5):
            raise RuntimeError('Pilot or baseline stage metrics are incomplete')
        result=dict(status='complete',stage='anchor_replay_pilot',sweep=5,
            train_done_sha256=worker.sha(training/'TRAIN_DONE.json'),evaluation_sha256=eval_sha,
            native_replay_sha256=worker.sha(evaluation/'REPLAY.json'),
            baseline_evaluation_sha256=protocol['baseline_sweep05_evaluation_sha256'],
            pilot_motion_mae=pilot_metrics['metrics']['unique_frame_equal']['correct']['motion_mae'],
            baseline_motion_mae=old['metrics']['unique_frame_equal']['correct']['motion_mae'],
            pilot_anchor_motion_mae=pilot_metrics['anchor_motion_mae'],
            baseline_anchor_motion_mae=old['anchor_motion_mae'],
            original_two_window_anchor_motion_mae=pilot_metrics['original_two_window_anchor_motion_mae'],
            pilot_dynamic_anchors=pilot_metrics['dynamic_anchors'],
            contacts_sha256=worker.sha(run/'REVIEW_ASSETS/CONTACTS.json'),
            axis_report_sha256=worker.sha(run/'AXIS_REPORT/AXIS_DIAGNOSTIC.json'),
            eight_window_report_sha256=worker.sha(run/'EIGHT_WINDOW_REPORT/OBJECT_AUDIT.json'),
            comparison_caveat='Same five full base passes and warm initialization; replay adds 225 B4 updates and changes batch ordering',
            manual_dynamic_review_required=True,action_codec_complete=False,
            scope='One training video; not a validated posterior or cross-video generalization',
            elapsed_seconds=time.monotonic()-began)
        worker.write(run/'SUMMARY.json',result)
        worker.write(run/'STATUS.json',dict(status='complete',phase='awaiting_visual_review',
                     action_codec_complete=False),replace=True)
    except BaseException as error:
        failure=dict(status='failed',phase=phase,error=repr(error),traceback=traceback.format_exc(),
                     no_automatic_retry=True)
        worker.write(run/'FAILED.json',failure);worker.write(run/'STATUS.json',failure,replace=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ready-fd',type=int)
    args=parser.parse_args();coordinate(args.ready_fd)
