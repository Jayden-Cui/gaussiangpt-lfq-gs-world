"""Single submission: train -> all176-pair evaluation -> report. No retries."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from contracts import verify_sources


def read(path): return json.loads(path.read_text())
def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def main(root):
    started=time.monotonic();protocol=read(root/'protocol.json');config=protocol['arms'][0];arm=config['name']
    verify_sources(root)
    assert read(root/'witness.json')['status']=='passed'
    folder=root/arm;folder.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(config['gpu']),OMP_NUM_THREADS='6',
        MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1')
    write(root/'launch_receipt.json',{'pid':os.getpid(),'arm':arm,'gpu':config['gpu'],
        'python':sys.executable,'started_unix':time.time(),'scope':'diagnostic LFQ-only motion, no state retraining'})
    try:
        for phase,script,budget in [('training','train_motion.py',15000),('evaluation','evaluate_motion.py',9000)]:
            write(root/'STATUS.json',{'status':'running','phase':phase,'arm':arm,'pid':os.getpid(),
                'elapsed_seconds':time.monotonic()-started})
            with (folder/f'{phase}.log').open('xb') as log:
                subprocess.run([sys.executable,'-u','-B',str(root/'code'/script),'--root',str(root),'--arm',arm],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=budget,check=True)
        result=read(folder/'evaluation.json');training=read(folder/'train_done.json')
        old=Path(protocol['source_motion_cache'])/'k4/evaluation.json'
        contextual=read(old) if old.exists() else None
        summary={'status':'complete','arm':arm,'steps':training['steps'],'batch_size':4,
            'pair_exposures':training['pair_exposures'],'evaluated_pairs':result['pairs'],
            'motion_quality_passed':result['quality_passed'],'state_quality_passed':False,
            'end_to_end_quality_passed':False,'gates':result['gates'],'metrics':result['metrics'],
            'gain_vs_held_initial_db':result['gain_vs_held_initial_db'],
            'shuffle_hand_drop_db':result['shuffle_hand_drop_db'],
            'per_query_distinct_ids':result['per_query_distinct_ids'],
            'previous512_fsq_k4_context_only':None if contextual is None else contextual['metrics'],
            'comparison_limit':'Width, quantizer, attention projection/FFN and batch/update schedule changed together; no isolated cause claim.',
            'elapsed_seconds':time.monotonic()-started,'limitations':protocol['limitations']}
        write(root/'RESULT.json',summary)
        lines=['# Native64 LFQ 结果','',f"工程状态：complete；{training['steps']}步，batch4，{training['pair_exposures']}对样本访问。",
            f"运动质量门槛：{result['quality_passed']}；冻结状态AE质量门槛：False。",'',
            '|方案|全图PSNR|手部PSNR|MAE|','|---|---:|---:|---:|']
        for k in ['held_initial','frozen_target','correct_motion','shuffled_motion']:
            m=result['metrics'][k];lines.append(f"|{k}|{m['psnr']:.4f}|{m['hand_activity_psnr']:.4f}|{m['mae']:.6f}|")
        lines += ['',f"打乱Z后的手部PSNR下降：{result['shuffle_hand_drop_db']:.4f} dB。",
            f"每个query在176帧对中的不同ID数：{result['per_query_distinct_ids']}。",'',
            '旧512-FSQ实验仅作背景参考，多项设置同时改变，不能单独归因于LFQ或64维。',
            '这是同一片段的真实目标端点后验重建，不是AR预测或泛化验证。']
        (root/'RESULT.md').write_text('\n'.join(lines)+'\n')
        write(root/'STATUS.json',{'status':'complete','phase':'complete','arm':arm,
            'motion_quality_passed':result['quality_passed'],'end_to_end_quality_passed':False,
            'elapsed_seconds':time.monotonic()-started})
    except BaseException as error:
        write(root/'FAILED.json',{'status':'failed','error':repr(error),'elapsed_seconds':time.monotonic()-started})
        write(root/'STATUS.json',{'status':'failed','arm':arm,'error':repr(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
