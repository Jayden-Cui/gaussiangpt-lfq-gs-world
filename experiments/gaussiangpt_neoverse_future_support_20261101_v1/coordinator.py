"""One bounded submission: two endpoint workers, cold replay, compact summary."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from io_utils import ROOT, read, write, sha, verify


def run_start(start,p):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['gpu_assignment'][str(start)]),
             OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    phases=(('diagnostic.py',p['worker_max_seconds']+60),('replay.py',p['replay_max_seconds']+30))
    outcome=dict(start=start,status='running')
    for script,limit in phases:
        with (ROOT/f's{start:03d}_{Path(script).stem}.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B',str(ROOT/script),'--start',str(start)],
                                     cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(ROOT/f's{start:03d}_PROCESS.json',dict(pid=process.pid,start=start,phase=script),replace=True)
            try:
                code=process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill();process.wait()
                return dict(start=start,status='failed',phase=script,reason='bounded timeout',pid=process.pid)
            if code:
                return dict(start=start,status='failed',phase=script,exit_code=code)
    outcome['status']='complete'
    return outcome


def summarize(p):
    results={}
    previous=read(Path(p['previous_run']['root'])/'SUMMARY.json')
    lines=['# 真实未来坐标上的自由特征诊断','',
           '固定原始 codec；未来坐标已知，初始特征只从初态空间格点转移；每端点有界/无界各 600 步。',
           '与上一轮完整初态坐标的结果比较。预算、损失、codec 和渲染保持一致。',
           '这使用 oracle 未来支撑，不是预测的 occupancy，也不是动作码或 AR 结果。','',
           '| 终帧 | 输入 | MAE | 运动 MAE | PSNR | 颜色区域 IoU | 中心误差 px | 方向误差 ° |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for start in p['starts']:
        out=ROOT/f's{start:03d}';r=read(out/'RESULT.json');replay=read(out/'REPLAY.json')
        if replay['status']!='passed': raise RuntimeError('Replay incomplete')
        old=previous['cases'][str(start)]
        entries=dict(r['references'])
        entries.update({'initial_support_'+m:old['results'][m] for m in p['modes']})
        entries.update({'future_support_'+m:r['results'][m] for m in p['modes']})
        r['previous_initial_support_results']=old['results']
        r['support_comparison']={}
        for mode in p['modes']:
            a,b=old['results'][mode],r['results'][mode]
            r['support_comparison'][mode]=dict(
                motion_mae_relative_reduction=(a['metrics']['motion_mae']-b['metrics']['motion_mae'])/a['metrics']['motion_mae'],
                full_mae_relative_reduction=(a['metrics']['mae']-b['metrics']['mae'])/a['metrics']['mae'],
                object_color_iou_delta=b['object_proxy']['iou']-a['object_proxy']['iou'])
        r['independent_native_replay']=replay
        results[str(start)]=r
        for name,e in entries.items():
            m,c=e['metrics'],e['object_proxy']
            center='NA' if c['centroid_error_px'] is None else f"{c['centroid_error_px']:.2f}"
            angle='NA' if c['axis_error_degrees'] is None else f"{c['axis_error_degrees']:.2f}"
            lines.append(f"| {start+9} | {name} | {m['mae']:.6f} | {m['motion_mae']:.6f} | {m['psnr']:.3f} | {c['iou']:.3f} | {center} | {angle} |")
    lines+=['','坐标改变同时改变了优化参数数量，并需要重新安排初态特征，因此不能把所有差异都归为纯粹坐标效应。',
            '成功拟合可以说明该初始化及表示下可达；优化失败不能证明不存在解。',
            '颜色区域指标只是固定阈值与固定区域的辅助测量，需要人工检查黄色条竖起及绿梁横跨两柱。',
            '终帧图像并不检验中间帧运动、跨视角真实几何或泛化。动作成功结论：等待视觉审查。','']
    with (ROOT/'SUMMARY.md').open('x',encoding='utf-8') as stream: stream.write('\n'.join(lines))
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,protocol_sha256=sha(ROOT/'protocol.json'),
        code_sha256=sha(ROOT/'CODE.sha256'),cases=results,action_success='pending visual review',
        oracle_future_support=True,future_feature_values_used_for_initialization=False,
        next_stage='requires result-based decision; no further experiments automatically submitted'))


def main():
    began=time.monotonic()
    try:
        p=verify()
        witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or witness['sources']!={n:sha(ROOT/n) for n in ('support.py','test_support.py')}:
            raise RuntimeError('CPU regression receipt missing or stale')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:
            raise RuntimeError('Disk budget unavailable')
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),starts=p['starts'],protocol_sha256=sha(ROOT/'protocol.json'),
                                      code_sha256=sha(ROOT/'CODE.sha256')))
        write(ROOT/'STATUS.json',dict(status='running',phase='fitting',starts=p['starts']),replace=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_start,s,p) for s in p['starts']]
            outcomes=[f.result() for f in futures]
        if not all(o['status']=='complete' for o in outcomes):
            write(ROOT/'FAILED.json',dict(status='failed',workers=outcomes,elapsed_seconds=time.monotonic()-began))
            write(ROOT/'STATUS.json',dict(status='failed',workers=outcomes),replace=True)
            return 1
        summarize(p)
        write(ROOT/'STATUS.json',dict(status='complete',steps_per_case=p['steps_per_case'],cases=4,
            independent_native_replay='passed',action_success='pending visual review',
            elapsed_seconds=time.monotonic()-began),replace=True)
        return 0
    except BaseException as error:
        write(ROOT/'COORDINATOR_FAILED.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc()))
        write(ROOT/'STATUS.json',dict(status='failed',error=repr(error)),replace=True)
        raise


if __name__=='__main__': sys.exit(main())
