import os
from pathlib import Path

DEPLOY=Path('/data/CJH/1')
PYTHON=str(DEPLOY/'bin/python')
def environment(gpu=None):
    env=dict(os.environ,OMP_NUM_THREADS='6',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        XDG_CACHE_HOME=str(DEPLOY/'cache'),TORCH_HOME=str(DEPLOY/'cache/torch'),
        TORCH_EXTENSIONS_DIR=str(DEPLOY/'cache/torch_extensions'),CUDA_CACHE_PATH=str(DEPLOY/'cache/cuda'),
        MPLCONFIGDIR=str(DEPLOY/'cache/matplotlib'),TMPDIR=str(DEPLOY/'tmp'))
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env
def configure():
    os.environ.update(environment())
    for n in ('cache/torch','cache/torch_extensions','cache/cuda','cache/matplotlib','tmp'):(DEPLOY/n).mkdir(parents=True,exist_ok=True)
