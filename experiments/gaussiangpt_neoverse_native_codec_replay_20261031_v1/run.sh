#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
exec >run.log 2>&1
sha256sum -c CODE.sha256 || exit 1
/usr/bin/timeout 3700 /data/CJH/envs/gaussiangpt/bin/python -B produce.py
rc=$?
if [ "$rc" -ne 0 ]; then
  printf '{"stage":"produce","exit_code":%s}\n' "$rc" > PIPELINE_FAILURE.json
  exit "$rc"
fi
/usr/bin/timeout 3700 /data/CJH/envs/gaussiangpt/bin/python -B replay.py
rc=$?
if [ "$rc" -ne 0 ]; then
  printf '{"stage":"replay","exit_code":%s}\n' "$rc" > PIPELINE_FAILURE.json
  exit "$rc"
fi
printf '{"status":"complete","training_updates":0}\n' > PIPELINE_DONE.json
