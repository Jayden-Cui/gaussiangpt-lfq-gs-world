"""Two bounded independent arms; train -> full evaluation -> fresh GS replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from PIL import Image
from trial_utils import ROOT,OUTPUTS,read,write,sha,verify,require


def run_arm(arm,p):
    folder=ROOT/'run'/arm;folder.mkdir();training=folder/'training'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['arms'][arm]['gpu']),OMP_NUM_THREADS='6',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    phase='train'
    def child(name,command,timeout,gpu=True):
        local=dict(env)
        if not gpu:local.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
        write(folder/'STATUS.json',dict(status='running',phase=name,arm=arm),replace=True)
        with (folder/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-B']+command,cwd=ROOT,env=local,stdout=log,stderr=subprocess.STDOUT)
            write(folder/(name+'_PROCESS.json'),dict(pid=process.pid,command=command,timeout_seconds=timeout,gpu=p['arms'][arm]['gpu'] if gpu else None))
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise RuntimeError(name+' exceeded fixed budget')
        require(code==0,name+' failed, exit '+str(code))
    try:
        child('train',[str(ROOT/'trial_worker.py'),'train','--arm',arm,'--out',str(training)],p['max_training_seconds_per_arm']+60)
        done=read(training/'TRAIN_DONE.json');require(done['additional_updates']==846 and done['cumulative_updates']==2961,'Training budget differs')
        phase='evaluate'
        child('evaluate',[str(ROOT/'evaluate_trial.py'),'evaluate','--run-root',str(training),'--sweep','7'],p['evaluation_timeout_seconds'])
        evaluation=training/'sweep_07/free_z/evaluation';data=read(evaluation/'EVALUATION.json')
        require(data['status']=='complete' and data['window_predictions']==1512 and data['unique_future_frames']==176,'Incomplete full evaluation')
        phase='replay'
        child('replay',[str(ROOT/'evaluate_trial.py'),'replay','--run-root',str(training),'--sweep','7'],p['replay_timeout_seconds'])
        replay=read(evaluation/'REPLAY.json')
        require(replay['status']=='passed' and replay['packets']==104 and replay['views']==208
            and replay['packets_sha256']==data['packets_sha256'] and replay['encoder_or_codec_loaded'] is False,'Native replay incomplete')
        phase='reports';digest=sha(evaluation/'EVALUATION.json')
        axis=OUTPUTS/'gaussiangpt_neoverse_visible_axis_reference_20261102_v1'
        eight=OUTPUTS/'gaussiangpt_neoverse_eight_window_object_audit_20261102_v1';baseline=eight/'result_baseline/OBJECT_AUDIT.json'
        child('axis_report',[str(axis/'axis_audit.py'),'--evaluation-root',str(evaluation),'--expected-evaluation-sha256',digest,
            '--output',str(folder/'AXIS_REPORT')],300,gpu=False)
        child('eight_window_report',[str(eight/'audit.py'),'--evaluation-root',str(evaluation),'--expected-evaluation-sha256',digest,
            '--reference-audit',str(baseline),'--expected-reference-audit-sha256',sha(baseline),'--output',str(folder/'EIGHT_WINDOW_REPORT')],300,gpu=False)
        assets=folder/'REVIEW_ASSETS';assets.mkdir();contacts=[]
        for start in p['evaluation']['diagnostic_starts']:
            canvas=Image.new('RGB',(1792,504));sources=[]
            for i,h in enumerate((0,3,6,9)):
                source=evaluation/'snapshots'/f's{start:03d}'/f'frame_{h:02d}.png'
                with Image.open(source) as im:
                    require(im.size==(1792,276),'Unexpected render panel size')
                    canvas.paste(im.crop((0,24,896,276)),((i%2)*896,(i//2)*252))
                sources.append(dict(horizon=h,sha256=sha(source)))
            dest=assets/f's{start:03d}_contact.png';canvas.save(dest);contacts.append(dict(start=start,file=dest.name,sha256=sha(dest),sources=sources))
        write(assets/'CONTACTS.json',dict(evaluation_sha256=digest,rows=contacts,manual_review_required=True))
        m=read(training/'sweep_07/free_z/STAGE_METRICS.json')
        result=dict(status='complete',arm=arm,object_weight=p['arms'][arm]['object_weight'],additional_updates=846,
            cumulative_updates=2961,evaluation_sha256=digest,replay_sha256=sha(evaluation/'REPLAY.json'),
            motion_mae=m['metrics']['unique_frame_equal']['correct']['motion_mae'],
            h9_motion_mae=m['by_horizon']['9']['correct']['motion_mae'],anchor_motion_mae=m['anchor_motion_mae'],
            dynamic_anchors=m['dynamic_anchors'],checkpoint_sha256=done['final_checkpoint_sha256'],
            initial_training_source=done['source'],manual_review_required=True,teacher_accepted=False)
        write(folder/'SUMMARY.json',result);write(folder/'STATUS.json',dict(status='complete',phase='awaiting_manual_review'),replace=True)
        return result
    except BaseException as error:
        result=dict(status='failed',arm=arm,phase=phase,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(folder/'FAILED.json',result);write(folder/'STATUS.json',result,replace=True);return result


def main(ready_fd):
    p=verify(need_masks=True,need_tests=True)
    a,b=[read(ROOT/'preflight'/arm/'PREFLIGHT.json') for arm in ('control','object')]
    require(a['status']==b['status']=='passed','Both GPU preflights required')
    for key in ('parent_checkpoint_sha256','parent_decoder_digest','parent_optimizer_digest','parent_code_digest',
                'sampler','mask_receipt_sha256','mask_review_sha256','training_parameter_signature','code_sha256'):
        require(a['source'][key]==b['source'][key],'Arms not matched at '+key)
    require(a['source']['code_sha256']==sha(ROOT/'CODE.sha256'),'Preflight code changed')
    run=ROOT/'run';run.mkdir();began=time.monotonic()
    write(run/'STARTED.json',dict(pid=os.getpid(),arms=p['arms'],additional_updates_per_arm=846,protocol_sha256=sha(ROOT/'protocol.json')))
    write(run/'STATUS.json',dict(status='running',phase='paired_training'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_arm,arm,p):arm for arm in ('control','object')}
        if ready_fd is not None:os.write(ready_fd,b'ready\n');os.close(ready_fd)
        results={}
        for future in as_completed(futures):results[futures[future]]=future.result()
    success=all(x['status']=='complete' for x in results.values())
    ratios={k:results['object'][k]/results['control'][k] for k in ('motion_mae','h9_motion_mae','anchor_motion_mae')} if success else None
    write(run/'SUMMARY.json',dict(status='complete' if success else 'failed',arms=results,object_over_control_ratios=ratios,
        elapsed_seconds=time.monotonic()-began,automatic_teacher_acceptance=False,action_codec_complete=False,
        next='Manual dynamic review of both arms; no automatic extra training or Q-Former launch'))
    write(run/'STATUS.json',dict(status='complete' if success else 'failed',phase='awaiting_review' if success else 'inspect_failed_arm'),replace=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ready-fd',type=int);args=parser.parse_args();main(args.ready_fd)
