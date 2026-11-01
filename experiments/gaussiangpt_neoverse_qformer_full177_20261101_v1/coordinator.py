from concurrent.futures import ThreadPoolExecutor
import os,shutil,subprocess,sys,time,traceback
from io_utils import ROOT,read,write,sha,verify


def available_memory():
    line=next(line for line in open('/proc/meminfo') if line.startswith('MemAvailable:'))
    return int(line.split()[1])*1024


def process(name,mode,timeout,args):
    p=read(ROOT/'protocol.json');gpu=0 if mode=='prepare' else p['gpu_assignment'][mode]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/f'{mode}_{name[:-3]}.log').open('x') as log:
        job=subprocess.Popen([sys.executable,'-B',str(ROOT/name)]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(ROOT/f'{mode}_PROCESS.json',dict(pid=job.pid,mode=mode,phase=name),replace=True)
        try:code=job.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            job.terminate()
            try:job.wait(timeout=15)
            except subprocess.TimeoutExpired:job.kill();job.wait()
            return dict(status='failed',mode=mode,phase=name,error='deadline')
        return dict(status='complete' if code==0 else 'failed',mode=mode,phase=name,exit_code=code)


def run_arm(mode,p):
    first=process('worker.py',mode,p['worker_max_seconds']+p['evaluation_max_seconds']+120,['--mode',mode])
    if first['status']!='complete':return first
    return process('replay.py',mode,p['replay_max_seconds']+30,['--mode',mode])


def summarize(p):
    arms={};initial=[];lines=['# 全177帧：连续Q-Former与受训初态对照','',
        '两组均在全部168个相邻T10窗口训练2000步，B4；原始codec与状态时空encoder冻结。此处是已训练动作阶段的拟合和输入对照，没有未见动作划分。',
        '主表先对同一物理未来帧的不同上下文求平均，再对176帧等权平均。完整1512项窗口预测均保留。','',
        '| 受训组 | 全图MAE | 运动MAE | PSNR | 支撑IoU | 同模型仅初态运动MAE | 换码运动MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in p['modes']:
        out=ROOT/mode;e=read(out/'evaluation/EVALUATION.json');r=read(out/'evaluation/REPLAY.json');t=read(out/'TRAIN_DONE.json')
        if t['steps']!=2000 or r['status']!='passed' or e['window_predictions']!=1512 or e['unique_future_frames']!=176:
            raise RuntimeError('Incomplete full-clip result')
        initial.append(tuple(t['source'][k] for k in ['initial_decoder_digest','initial_posterior_digest','schedule_sha256']))
        m=e['metrics']['unique_frame_equal'];support=e['support']['unique_frame_equal']
        arms[mode]=dict(metrics=e['metrics'],support=e['support'],temporal_segments=e['temporal_segments'],
            code_stats=e['code_stats'],endpoints=[row for row in e['rows'] if row['horizon']==9 and row['start'] in p['diagnostic_starts']],
            train=t,replay={k:v for k,v in r.items() if k!='rows'},input_control_ratios=dict(
                initial_only=m['initial_only']['motion_mae']/max(m['correct']['motion_mae'],1e-12),
                swapped=m['swapped_window_codes']['motion_mae']/max(m['correct']['motion_mae'],1e-12)))
        v=m['correct']
        lines.append(f"| {mode} | {v['mae']:.6f} | {v['motion_mae']:.6f} | {v['psnr']:.3f} | {support['correct']['iou']:.4f} | {m['initial_only']['motion_mae']:.6f} | {m['swapped_window_codes']['motion_mae']:.6f} |")
    if len(set(initial))!=1:raise RuntimeError('Paired initialization or sampling mismatch')
    lines+=['','posterior组的仅初态列是拿掉后续状态的输入消融；initial_only组则从训练开始就只接收初态，两者不能混同。',
        '两组都使用真实未来目标监督训练；监督标签不等于decoder输入。需要结合全窗口视频检查动作和手部伪影。',
        '没有执行LFQ/FSQ、AR或新视频测试；不要求未训练动作表现，不自动进入下一轮。','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,arms=arms,shared_initialization_and_schedule=True,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
        training_frames=177,training_windows=168,unique_future_frames=176,window_predictions_per_arm=1512,
        action_success='pending visual review',scope=p['scope'],next_stage='No automatic LFQ or further run'))


def main():
    began=time.monotonic()
    try:
        p=verify();witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or any(sha(ROOT/name)!=digest for name,digest in witness['sources'].items()):
            raise RuntimeError('Stale CPU witness')
        if available_memory()<p['minimum_host_available_gib']*2**30:raise RuntimeError('Insufficient host RAM')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:raise RuntimeError('Insufficient disk')
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),modes=p['modes'],protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256')))
        write(ROOT/'STATUS.json',dict(status='running',phase='preparing_complete_windows'),replace=True)
        prep=process('prepare.py','prepare',p['preparation_max_seconds']+60,[])
        if prep['status']!='complete':raise RuntimeError('Preparation failed: '+str(prep))
        cache=read(ROOT/'cache/MANIFEST.json');cache_bytes=sum(v['bytes'] for v in cache['windows'].values())+cache['frames']['bytes']
        if available_memory()<2*cache_bytes+8*2**30:raise RuntimeError('RAM unavailable for two complete CPU caches')
        write(ROOT/'STATUS.json',dict(status='running',phase='training',modes=p['modes'],cache_bytes_per_worker=cache_bytes),replace=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_arm,mode,p) for mode in p['modes']];results=[f.result() for f in futures]
        if any(r['status']!='complete' for r in results):raise RuntimeError('Branch failed: '+str(results))
        summarize(p)
        write(ROOT/'STATUS.json',dict(status='complete',steps_per_arm=2000,native_replay='passed',
            training_frames=177,training_windows=168,action_success='pending visual review',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(ROOT/'FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()))
        write(ROOT/'STATUS.json',dict(status='failed',error=repr(e)),replace=True);raise


if __name__=='__main__':main()
