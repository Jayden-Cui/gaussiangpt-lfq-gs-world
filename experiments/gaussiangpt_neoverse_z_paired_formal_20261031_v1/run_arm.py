"""Bounded sequential stages for one arm; no retry or cross-arm signals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from codec_binding import require
from formal_common import verify_sources,write_new


def main(mode):
    root=Path(__file__).resolve().parent;p,source=verify_sources(root)
    require(mode in p['arms'],'Unknown arm')
    stages=[('train',[sys.executable,'-B','train.py','--mode',mode],p['training_max_seconds']+300),
            ('evaluate',[sys.executable,'-B','evaluate.py','--mode',mode],p['evaluation_max_seconds']+300),
            ('native_replay',[sys.executable,'-B','replay_native.py','--mode',mode],p['native_replay_max_seconds']+100)]
    started=time.monotonic();stage='initializing'
    try:
        for stage,command,limit in stages:
            print('STAGE',mode,stage,'START',flush=True)
            subprocess.run(command,cwd=root,check=True,timeout=limit)
            print('STAGE',mode,stage,'DONE',flush=True)
        write_new(root/f'ARM_{mode}_DONE.json',dict(status='complete',mode=mode,source=source,
                                                   elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_new(root/f'ARM_{mode}_FAILED.json',dict(status='failed',mode=mode,stage=stage,error=repr(error),
                  traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':main(sys.argv[1])
