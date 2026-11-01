from concurrent.futures import ThreadPoolExecutor
import os,shutil,subprocess,sys,time,traceback
from io_utils import ROOT,read,write,sha,verify


def run_arm(mode,p):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['gpu_assignment'][mode]),OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    for name,timeout in [('worker.py',p['worker_max_seconds']+p['evaluation_max_seconds']+60),('replay.py',p['replay_max_seconds']+30)]:
        with (ROOT/f'{mode}_{name[:-3]}.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B',str(ROOT/name),'--mode',mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(ROOT/f'{mode}_PROCESS.json',dict(pid=process.pid,mode=mode,phase=name),replace=True)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                return dict(mode=mode,status='failed',phase=name,error='deadline')
            if code:return dict(mode=mode,status='failed',phase=name,exit_code=code)
    return dict(mode=mode,status='complete')


def summarize(p):
    results={};initial_digests=[]
    text=['# 学习未来支撑的动作解码器能力诊断','',
          '两个已知训练窗口；仅初态生成候选，推理时不用未来坐标或未来点数。连续码是联合优化的每窗口变量，未经过 Q-Former/LFQ。',
          '每臂 600 步状态监督 + 600 步混合图像监督，逻辑 B=2；固定最后一步验收。','',
          '| 组别 | 18个未来帧平均 MAE | 平均运动 MAE | 平均支撑 IoU | 换码后运动 MAE |',
          '|---|---:|---:|---:|---:|']
    for mode in p['modes']:
        out=ROOT/mode;e=read(out/'evaluation/EVALUATION.json');r=read(out/'evaluation/REPLAY.json');t=read(out/'TRAIN_DONE.json')
        if r['status']!='passed' or t['steps']!=p['total_steps']:raise RuntimeError('Incomplete arm')
        initial_digests.append(t['source']['common_model_initial_digest'])
        future=[row for row in e['rows'] if row['horizon']>0]
        if len(future)!=18:raise RuntimeError('Future coverage mismatch')
        mean=lambda xs:sum(xs)/len(xs)
        metrics={control:{k:mean([row['metrics'][control]['metrics'][k] for row in future]) for k in ['mae','psnr','motion_mae','object_box_mae','teacher_view1_masked_mae']}
                 for control in ['correct','swapped_window_codes','actual_target']}
        support={control:{k:mean([row['support'][control][k] for row in future]) for k in ['iou','precision','recall','predicted_nodes','target_nodes']}
                 for control in ['correct','swapped_window_codes']}
        result=dict(metrics=metrics,support=support,endpoints=[row for row in future if row['horizon']==9],replay=r,train=t)
        results[mode]=result
        text.append(f"| {mode} | {metrics['correct']['mae']:.6f} | {metrics['correct']['motion_mae']:.6f} | {support['correct']['iou']:.4f} | {metrics['swapped_window_codes']['motion_mae']:.6f} |")
    if len(set(initial_digests))!=1:raise RuntimeError('Arms did not share model initialization')
    text+=['','物体姿态、时间一致性与动作码作用需要结合端点图、10 帧视频和固定码组判断。',
           '支撑/特征监督来自已知 TRAIN 未来状态；这是 decoder 容量与条件依赖测试，不是未来动作预测或跨视频泛化。',
           '候选半径固定为4，是在这两个 TRAIN 窗口覆盖检查后选定；不代表任意运动都能被此候选范围覆盖。','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(text),encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,arms=results,shared_initialization=True,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),action_success='pending visual review',
        next_stage='No automatic Q-Former/LFQ or full-dataset run'))


def main():
    began=time.monotonic()
    try:
        p=verify();check=read(ROOT/'CPU_TESTS.json')
        required=('support_decoder.py','runtime.py','test_decoder.py')
        if check['status']!='passed' or check['sources']!={n:sha(ROOT/n) for n in required}:raise RuntimeError('Stale CPU witness')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:raise RuntimeError('Disk budget unavailable')
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),modes=p['modes'],protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256')))
        write(ROOT/'STATUS.json',dict(status='running',phase='training',modes=p['modes']),replace=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_arm,mode,p) for mode in p['modes']]
            result=[f.result() for f in futures]
        if not all(r['status']=='complete' for r in result):
            write(ROOT/'FAILED.json',dict(status='failed',arms=result));write(ROOT/'STATUS.json',dict(status='failed',arms=result),replace=True);return 1
        summarize(p)
        write(ROOT/'STATUS.json',dict(status='complete',steps_per_arm=p['total_steps'],independent_native_replay='passed',
            action_success='pending visual review',elapsed_seconds=time.monotonic()-began),replace=True);return 0
    except BaseException as e:
        write(ROOT/'COORDINATOR_FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':sys.exit(main())
