from runtime_env import configure,environment,PYTHON
configure()
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,os,subprocess,time,traceback
from bindings import ROOT,read,write,sha,require
from worker import verify

def arm_run(arm,p):
    root=ROOT/'run'/arm;root.mkdir();phase='train';gpu=p['arms'][arm]['gpu']
    def child(phase,args,seconds,cpu=False):
        env=environment('' if cpu else gpu)
        write(root/'STATUS.json',dict(status='running',phase=phase,arm=arm),True)
        with (root/(phase+'.log')).open('x') as f:
            command=[PYTHON,'-B']+args;process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            write(root/(phase+'_PROCESS.json'),dict(pid=process.pid,command=command,timeout_seconds=seconds))
            try:code=process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise RuntimeError(phase+' exceeded fixed budget')
        require(code==0,phase+' failed with exit '+str(code))
    try:
        training=root/'training';evaluation=root/'evaluation'
        child('train',[str(ROOT/'worker.py'),'train','--arm',arm,'--out',str(training)],p['training_timeout_seconds']+60)
        phase='evaluate';child(phase,[str(ROOT/'evaluate.py'),'evaluate','--arm',arm,'--training',str(training),'--out',str(evaluation)],p['evaluation_timeout_seconds']+60)
        phase='replay';child(phase,[str(ROOT/'evaluate.py'),'replay','--out',str(evaluation)],p['replay_timeout_seconds']+60)
        phase='videos';child(phase,[str(ROOT/'evaluate.py'),'videos','--out',str(evaluation)],300,cpu=True)
        data=read(evaluation/'EVALUATION.json');replay=read(evaluation/'REPLAY.json')
        require(data['status']=='complete' and data['window_predictions']==1512 and replay['status']=='passed','Incomplete result')
        r=dict(status='complete',arm=arm,config=p['arms'][arm],updates=846,motion_mae=data['metrics']['unique_frame_equal']['motion_mae'],
            h9_motion_mae=data['by_horizon']['9']['motion_mae'],anchor_motion_mae=data['anchors']['motion_mae'],
            yellow_motion_mae=data['by_window']['35']['motion_mae'],green_motion_mae=data['by_window']['123']['motion_mae'],
            evaluation_sha256=sha(evaluation/'EVALUATION.json'),replay_sha256=sha(evaluation/'REPLAY.json'),
            manual_dynamic_review_required=True,teacher_accepted=False)
        write(root/'SUMMARY.json',r);write(root/'STATUS.json',dict(status='complete',phase='awaiting_manual_review'),True);return r
    except BaseException as e:
        r=dict(status='failed',arm=arm,phase=phase,error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(root/'FAILED.json',r);write(root/'STATUS.json',r,True);return r

def main(fd):
    p=verify();require(read(ROOT/'ENV_AGENT_WITNESS.json')['status']=='passed','Independent environment check missing')
    for arm in p['arms']:
        pre=read(ROOT/'preflight'/arm/'PREFLIGHT.json')
        require(pre['status']=='passed' and pre['source']['code_sha256']==sha(ROOT/'CODE.sha256'),'Missing matching preflight')
    require(read(ROOT/'CPU_TESTS.json')['status']=='passed','CPU checks missing')
    (ROOT/'run').mkdir();began=time.monotonic();write(ROOT/'run/STARTED.json',dict(pid=os.getpid(),arms=p['arms'],code_sha256=sha(ROOT/'CODE.sha256')))
    write(ROOT/'run/STATUS.json',dict(status='running',phase='four_matched_capacity_arms'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(arm_run,arm,p):arm for arm in p['arms']}
        if fd is not None:os.write(fd,b'ready\n');os.close(fd)
        results={}
        for future in as_completed(futures):results[futures[future]]=future.result()
    success=all(r['status']=='complete' for r in results.values());ratios=None
    if success:
        ratios={arm:{k:r[k]/results['z16_w64'][k] for k in ('motion_mae','h9_motion_mae','anchor_motion_mae','green_motion_mae')} for arm,r in results.items()}
    write(ROOT/'run/SUMMARY.json',dict(status='complete' if success else 'failed',arms=results,ratios_to_z16_w64=ratios,elapsed_seconds=time.monotonic()-began,
        action_codec_complete=False,manual_dynamic_review_required=True,no_automatic_extension=True))
    write(ROOT/'run/STATUS.json',dict(status='complete' if success else 'failed',phase='awaiting_review'),True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ready-fd',type=int);a=p.parse_args();main(a.ready_fd)
