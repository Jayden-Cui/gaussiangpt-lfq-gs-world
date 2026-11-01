"""Produce once, then run one bounded native-only replay in a fresh process."""
import os
import subprocess
import sys
import traceback

from common import ROOT, read, sha, validate_protocol, verify_self, write


def main():
    phase='source_check'
    try:
        verify_self()
        p=read(ROOT/'protocol.json')
        validate_protocol(p)
        write(ROOT/'LAUNCHED.json',{'pid':os.getpid(),'code_sha256':sha(ROOT/'CODE.sha256'),
              'protocol_sha256':sha(ROOT/'protocol.json'),'main_training_signalled':False})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['cuda_visible_device']),OMP_NUM_THREADS='4',
                 MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        for phase,script,timeout in (('production','produce.py',p['producer_timeout_seconds']),
                                     ('independent_replay','replay.py',p['replay_timeout_seconds'])):
            with (ROOT/(phase+'.log')).open('x',encoding='utf8') as log:
                result=subprocess.run([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=timeout+60)
            if result.returncode:
                raise RuntimeError(phase+' failed with exit code '+str(result.returncode))
        report=ROOT/'result/REPLAY.json'
        replay=read(report)
        if (replay.get('status')!='passed' or replay.get('packets')!=36 or replay.get('views')!=72
                or replay.get('encoder_or_codec_loaded') is not False
                or replay['producer_pid']==replay['replay_pid'] or not all(x['passed'] for x in replay['rows'])):
            raise RuntimeError('Independent replay result failed verification')
        write(ROOT/'DONE.json',{'status':'complete','replay_sha256':sha(report),'packets':36,'views':72,
              'main_training_signalled':False,'model_updates':0})
    except BaseException as error:
        write(ROOT/'FAILED.json',{'status':'failed','phase':phase,'error':repr(error),
            'traceback':traceback.format_exc(),'main_training_signalled':False,'no_automatic_retry':True})
        raise


if __name__=='__main__':
    main()
