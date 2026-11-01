"""Run fixed-teacher paired Q training, then independent native replay once."""
import os
import subprocess
import sys
import traceback

from q_utils import ROOT, read, sha, validate_protocol, verify_self, write


def main():
    phase='source_check'
    try:
        verify_self();p=read(ROOT/'protocol.json');validate_protocol(p)
        write(ROOT/'LAUNCHED.json',{'pid':os.getpid(),'protocol_sha256':sha(ROOT/'protocol.json'),
              'code_sha256':sha(ROOT/'CODE.sha256'),'main_training_signalled':False})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['cuda_visible_device']),OMP_NUM_THREADS='4',
                 MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        for phase,script,timeout in (('training_and_evaluation','worker.py',p['maximum_train_seconds']),
                                     ('independent_native_replay','replay.py',p['maximum_replay_seconds'])):
            with (ROOT/(phase+'.log')).open('x',encoding='utf8') as log:
                run=subprocess.run([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,env=env,
                                   stdout=log,stderr=subprocess.STDOUT,timeout=timeout+60)
            if run.returncode:
                raise RuntimeError(phase+' failed with exit code '+str(run.returncode))
        result=ROOT/'result';summary=read(result/'SUMMARY.json');causal=read(result/'CAUSALITY.json')
        evaluation=read(result/'step_01194/EVALUATION.json');replay=read(result/'step_01194/REPLAY.json')
        if (summary['status']!='complete' or summary['updates']!=1194 or causal['status']!='passed'
                or summary['causality_sha256']!=sha(result/'CAUSALITY.json')
                or evaluation['status']!='complete' or evaluation['step']!=1194
                or replay['status']!='passed' or replay['packets']!=36 or replay['views']!=72
                or replay['packets_sha256']!=evaluation['packets_sha256']
                or replay['encoder_or_codec_loaded'] is not False or replay['producer_pid']==replay['replay_pid']
                or not all(x['passed'] for x in replay['rows'])):
            raise RuntimeError('Paired Q evaluation, causality or native replay failed validation')
        write(ROOT/'DONE.json',{'status':'complete','summary_sha256':sha(result/'SUMMARY.json'),
            'evaluation_sha256':sha(result/'step_01194/EVALUATION.json'),
            'replay_sha256':sha(result/'step_01194/REPLAY.json'),
            'preliminary_numeric_pass':evaluation['summary']['preliminary_numeric_pass'],
            'visual_review':'pending','full_168_teacher_accepted':False,'main_training_signalled':False})
    except BaseException as error:
        write(ROOT/'FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
              'traceback':traceback.format_exc(),'main_training_signalled':False,'no_automatic_retry':True})
        raise


if __name__=='__main__':
    main()
