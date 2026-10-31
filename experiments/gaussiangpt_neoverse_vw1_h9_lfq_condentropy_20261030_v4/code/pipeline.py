"""One bounded submission: H9 train, final causality, all-window evaluation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from contracts import verify_sources

def read(path):return json.loads(path.read_text())
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def main(root):
    started=time.monotonic();p=read(root/'protocol.json');arm=p['arms'][0]['name'];folder=root/arm;folder.mkdir(exist_ok=True)
    verify_sources(root);assert read(root/'witness.json')['status']=='passed'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    write(root/'launch_receipt.json',dict(pid=os.getpid(),started_unix=time.time(),arm=arm,python=sys.executable,
        scope='H9 independentQFormer64 LFQ initial-only jointfuture reconstruction'))
    try:
        for phase,script in [('training','train_motion.py'),('trained_causality','validate_causality.py'),('evaluation','evaluate_motion.py')]:
            write(root/'STATUS.json',dict(status='running',phase=phase,arm=arm,pid=os.getpid(),elapsed_seconds=time.monotonic()-started))
            with (folder/f'{phase}.log').open('xb') as log:
                subprocess.run([sys.executable,'-u','-B',str(root/'code'/script),'--root',str(root),'--arm',arm],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=p['time_limits_seconds'][phase],check=True)
        result=read(folder/'evaluation.json');training=read(folder/'train_done.json')
        summary=dict(status='complete',arm=arm,steps=training['steps'],batch_size=training['batch_size'],horizon=9,
            window_exposures=training['window_exposures'],future_target_exposures=training['future_target_exposures'],
            engineering_passed=result['engineering_passed'],motion_quality_passed=result['quality_passed'],
            all_horizon_quality_passed=result['all_horizon_quality_passed'],state_quality_passed=False,end_to_end_quality_passed=False,
            metrics=result['metrics'],per_horizon=result['per_horizon'],engineering_gates=result['engineering_gates'],quality_gates=result['quality_gates'],
            conditional_entropy_diagnostic=result['conditional_entropy_diagnostic'],
            gain_vs_held_initial_db=result['gain_vs_held_initial_db'],shuffle_hand_drop_db=result['shuffle_hand_drop_db'],
            per_horizon_distinct_ids=result['per_horizon_distinct_ids_valid'],evaluated_windows=result['windows'],
            evaluated_real_future_targets=result['real_target_exposures'],trained_causality=read(folder/'causality.json'),
            elapsed_seconds=time.monotonic()-started,limitations=p['limitations'])
        write(root/'RESULT.json',summary)
        lines=['# VideoWorld1 H9 LFQ 结果','',f"工程通过：{result['engineering_passed']}；运动质量通过：{result['quality_passed']}。",
            f"{training['steps']}步，batch4窗口，每窗口9个未来目标；实际监督{training['future_target_exposures']}次目标。",
            '','|方案|全图PSNR|手部PSNR|','|---|---:|---:|']
        for key in ['held_initial','frozen_target','correct_motion','shuffled_motion','reversed_motion']:
            m=result['metrics'][key];lines.append(f"|{key}|{m['psnr']:.4f}|{m['hand_activity_psnr']:.4f}|")
        lines += ['',f"打乱码后手部PSNR下降{result['shuffle_hand_drop_db']:.4f}dB。",
            '','逐horizon结果见RESULT.json。视频在每9个目标前提供一次真实起始状态，属于后验重建，不是长程AR生成。',
            '冻结状态AE质量门槛仍未通过；结构对齐和工程通过不能替代科学质量证据。']
        (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
        write(root/'STATUS.json',dict(status='complete',phase='complete',arm=arm,engineering_passed=result['engineering_passed'],
            motion_quality_passed=result['quality_passed'],elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started)
        write(root/'FAILED.json',failure);write(root/'STATUS.json',failure);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
