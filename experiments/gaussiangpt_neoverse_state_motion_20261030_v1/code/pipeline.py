"""One bounded submission; program waits and runs each dependent phase exactly once."""
import argparse,datetime,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from binding import read,write,sha,verify
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def terminate(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=20)

def phase(root,name,jobs,seconds):
    children=[];started=time.monotonic()
    try:
        state={'status':'running','phase':name,'started_utc':now(),'timeout_seconds':seconds,'jobs':[]}
        for label,command,gpu in jobs:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
            log=(root/f'{name}_{label}.log').open('xb')
            p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
            children.append((p,log));state['jobs'].append({'label':label,'gpu':gpu,'pid':p.pid,'command':command})
        write(root/'STATUS.json',state)
        while any(p.poll() is None for p,_ in children):
            if time.monotonic()-started>seconds:raise TimeoutError(name)
            time.sleep(3)
        codes={j['label']:p.returncode for j,(p,_) in zip(state['jobs'],children)}
        write(root/f'phase_{name}.json',{**state,'status':'complete' if all(c==0 for c in codes.values()) else 'partial_failure',
            'exit_codes':codes,'finished_utc':now(),'elapsed_seconds':time.monotonic()-started})
        return codes
    finally:
        for p,log in children:terminate(p);log.close()

def command(sub,script,*args):return [sys.executable,'-u','-B',str(sub/'code'/script),'--root',str(sub),*args]

def main(root):
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'STATUS.json').exists(),'No automatic restart/overwrite'
    started=time.monotonic()
    def interrupted(sig,frame):raise RuntimeError(f'Interrupted by signal{sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        write(root/'STATUS.json',{'status':'running','phase':'validate','pid':os.getpid(),'started_utc':now()})
        verify(root);assert read(root/'witness.json')['status']=='passed'
        cfg=read(root/'protocol.json');limits=cfg['budgets_seconds'];state=root/'state';motion=root/'motion'
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        assert min(int(x) for x in free.split())>=55*1024,free
        write(root/'launch_receipt.json',{'pid':os.getpid(),'started_utc':now(),'protocol_sha256':sha(root/'protocol.json'),
            'source_binding_sha256':sha(root/'source_binding.json'),'frames':177,'pairs':176,'frame_gap':1,
            'state_lfq':False,'motion_fsq':True,'resources':cfg['resources'],'budgets_seconds':limits})
        for name,script,limit in [('state_train','train.py','state_train'),('state_eval','evaluate.py','state_eval')]:
            codes=phase(root,name,[('state',command(state,script),0)],limits[limit]);assert codes['state']==0,(name,codes)
        result_state=read(state/'evaluation.json');assert result_state['api']['passed'] and result_state['gates']['all177_frames']
        assert read(state/'train_done.json')['steps']==3000
        mp=read(motion/'protocol.template.json');mp['source_checkpoint_sha256']=sha(state/'codec_final.pt')
        assert mp['frame_gap']==1 and mp['evaluation']['pairs']==176 and mp['state_quantizer'].startswith('None')
        write(motion/'protocol.json',mp)
        write(motion/'source_binding.json',{'root_binding_sha256':sha(root/'source_binding.json'),
            'protocol_sha256':sha(motion/'protocol.json'),'state_protocol_sha256':sha(state/'protocol.json'),
            'state_checkpoint_sha256':mp['source_checkpoint_sha256'],'state_evaluation_sha256':sha(state/'evaluation.json'),
            'state_quality_passed':result_state['quality_passed'],'state_lfq':False})
        codes=phase(root,'motion_prepare',[('shared',command(motion,'prepare_motion.py'),0)],limits['motion_prepare']);assert codes['shared']==0,codes
        codes=phase(root,'motion_arms',[(a['name'],command(motion,'arm.py','--arm',a['name']),a['gpu']) for a in mp['arms']],limits['motion_train_eval_per_arm'])
        arms={}
        for a in mp['arms']:
            folder=motion/a['name']
            if codes[a['name']]==0:arms[a['name']]=read(folder/'evaluation.json')
            else:
                failure=read(folder/'arm_failure.json') if (folder/'arm_failure.json').exists() else {'error':'See phase log'}
                arms[a['name']]={'status':'failed','quality_passed':False,'failure':failure}
        status='complete' if all(v['status']=='complete' for v in arms.values()) else 'partial_failure'
        result={'status':status,'finished_utc':now(),'elapsed_seconds':time.monotonic()-started,'state':result_state,'arms':arms,
            'quality_passed':result_state['quality_passed'] and any(v['quality_passed'] for v in arms.values()),
            'scope':'Single-source adjacent-frame oracle TRAIN reconstruction; continuous spatialAE, motionFSQ; no stateLFQ or AR'}
        write(root/('COMPLETE.json' if status=='complete' else 'PARTIAL.json'),result)
        write(root/'STATUS.json',{'status':status,'phase':'finished','finished_utc':now(),'state_quality_passed':result_state['quality_passed'],
            'arms':{k:{'status':v['status'],'steps':v.get('training_steps'),'pairs':v.get('pairs'),'quality_passed':v['quality_passed']} for k,v in arms.items()}})
        lines=['# NeoVerse 连续状态编解码与相邻帧运动 Z','',result['scope'],'',f"状态质量门槛：{result_state['quality_passed']}"]
        for name,v in arms.items():
            lines.append(f"- {name}: {v['status']}, 质量门槛 {v['quality_passed']}")
        lines+=['','状态视频四列：真实／Neoverse／体素代表点／连续状态编解码。',
            '运动视频五列：真实目标／保持初始／冻结目标编解码／正确运动Z／错配运动Z。',
            '176个相邻帧对，t→t+1，约29.97FPS；已移除状态LFQ，运动FSQ保留。']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    except BaseException as e:
        v={'status':'failed','phase':'stopped','error':repr(e),'traceback':traceback.format_exc(),'finished_utc':now(),'elapsed_seconds':time.monotonic()-started}
        write(root/'FAILED.json',v);write(root/'STATUS.json',v);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
