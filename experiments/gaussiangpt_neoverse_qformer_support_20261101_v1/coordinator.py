import os,shutil,subprocess,sys,time,traceback
from io_utils import ROOT,read,write,sha,verify


def summarize(p):
    out=ROOT/p['modes'][0];e=read(out/'evaluation/EVALUATION.json');r=read(out/'evaluation/REPLAY.json');t=read(out/'TRAIN_DONE.json')
    if r['status']!='passed' or t['steps']!=p['total_steps']:raise RuntimeError('Incomplete result')
    rows=[row for row in e['rows'] if row['horizon']>0]
    if len(rows)!=18:raise RuntimeError('Lost future states')
    mean=lambda values:sum(values)/len(values)
    metrics={name:{key:mean([row['metrics'][name]['metrics'][key] for row in rows])
                  for key in ['mae','psnr','motion_mae','object_box_mae','teacher_view1_masked_mae']}
             for name in p['eval_controls']}
    support={name:{key:mean([row['support'][name][key] for row in rows])
                  for key in ['iou','precision','recall','predicted_nodes','target_nodes']}
             for name in p['eval_controls'] if name!='actual_target'}
    a=metrics['correct'];b=metrics['free_teacher'];limits=p['numeric_acceptance']
    gate_values=dict(mae_ratio_to_free_teacher=a['mae']/b['mae'],motion_mae_ratio_to_free_teacher=a['motion_mae']/b['motion_mae'],
        support_iou_drop=support['free_teacher']['iou']-support['correct']['iou'],
        endpoint_color_iou_drop=max(row['metrics']['free_teacher']['object_proxy']['iou']-row['metrics']['correct']['object_proxy']['iou'] for row in rows if row['horizon']==9),
        initial_only_motion_error_ratio=metrics['initial_only']['motion_mae']/a['motion_mae'],
        swapped_motion_error_ratio=metrics['swapped_window_codes']['motion_mae']/a['motion_mae'])
    gates={key:gate_values[key[:-4]]<=bound if key.endswith('_max') else gate_values[key[:-4]]>=bound for key,bound in limits.items()}
    text=['# Q-Former连续码：冻结空间解码器测试','',
        '两个已知训练窗口，9个64维查询，连续码[2,9,16]；Q-Former按S0..Sh独立前缀读取，未运行LFQ/FSQ或AR。',
        '600步自由码蒸馏 + 600步重建监督；解码器及codec完全冻结。18个未来状态等权平均。','',
        '| 条件 | 全图MAE | 运动MAE | PSNR | 支撑IoU |','|---|---:|---:|---:|---:|']
    for name,m in metrics.items():
        iou=f"{support[name]['iou']:.4f}" if name in support else 'reference'
        text.append(f"| {name} | {m['mae']:.6f} | {m['motion_mae']:.6f} | {m['psnr']:.3f} | {iou} |")
    text+=['','数值门槛：'+str(gates),'','未来输入依赖与重建保持分别报告；两个窗口、单次种子不能证明可迁移动作语义。',
        '科学结果仍需看完整窗口与两端关键帧。没有自动进入LFQ或全量训练。','']
    (ROOT/'SUMMARY.md').write_text('\n'.join(text),encoding='utf-8')
    write(ROOT/'SUMMARY.json',dict(status='complete',engineering_passed=True,metrics=metrics,support=support,
        code_metrics=e['code_metrics'],gate_values=gate_values,numeric_gates=gates,
        reconstruction_retained_numeric=all(value for key,value in gates.items() if key.endswith('_max')),
        future_input_dependence_numeric=all(value for key,value in gates.items() if key.endswith('_min')),
        endpoints=[row for row in rows if row['horizon']==9],train=t,replay=r,
        protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256'),
        action_success='pending visual review',next_stage='No automatic LFQ or full-dataset training'))


def main():
    began=time.monotonic()
    try:
        p=verify();witness=read(ROOT/'CPU_TESTS.json')
        if witness['status']!='passed' or any(sha(ROOT/name)!=digest for name,digest in witness['sources'].items()):
            raise RuntimeError('Stale CPU witness')
        if shutil.disk_usage(ROOT).free<p['minimum_disk_free_gib']*2**30:raise RuntimeError('Insufficient disk')
        mode=p['modes'][0]
        write(ROOT/'STARTED.json',dict(pid=os.getpid(),mode=mode,protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(ROOT/'CODE.sha256')))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['gpu_assignment'][mode]),OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
        for name,timeout,args in [('worker.py',p['worker_max_seconds']+p['evaluation_max_seconds']+60,[]),
                                  ('replay.py',p['replay_max_seconds']+30,['--mode',mode])]:
            with (ROOT/(name[:-3]+'.log')).open('x') as log:
                process=subprocess.Popen([sys.executable,'-B',str(ROOT/name)]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(ROOT/'PROCESS.json',dict(pid=process.pid,phase=name,mode=mode),replace=True)
                write(ROOT/'STATUS.json',dict(status='running',phase=name,mode=mode),replace=True)
                try:code=process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError('Deadline in '+name)
                if code:raise RuntimeError(f'{name} exited {code}')
        summarize(p)
        write(ROOT/'STATUS.json',dict(status='complete',completed_updates=p['total_steps'],native_replay='passed',
            action_success='pending visual review',elapsed_seconds=time.monotonic()-began),replace=True)
    except BaseException as e:
        write(ROOT/'FAILED.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc()))
        write(ROOT/'STATUS.json',dict(status='failed',error=repr(e)),replace=True);raise


if __name__=='__main__':main()
