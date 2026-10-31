#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
exec >preflight.log 2>&1
sha256sum -c PREFLIGHT.sha256 || exit 1
export PYTHONDONTWRITEBYTECODE=1
export Z_ADAPTER_SOURCE_PARENT=/data/CJH/AGS-V6Direct/outputs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
/usr/bin/timeout 180 /data/CJH/envs/gaussiangpt/bin/python -B -m unittest -v test_codec_binding.py test_teacher_binding.py test_paired_step.py test_paired_real_cpu.py
rc=$?
if [ "$rc" -ne 0 ]; then
  printf '{"stage":"cpu_tests","exit_code":%s}\n' "$rc" > PREFLIGHT_WRAPPER_FAILED.json
  exit "$rc"
fi
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0
/usr/bin/timeout 3600 /data/CJH/envs/gaussiangpt/bin/python -B preflight.py
rc=$?
if [ "$rc" -ne 0 ]; then
  printf '{"stage":"gpu_preflight","exit_code":%s}\n' "$rc" > PREFLIGHT_WRAPPER_FAILED.json
  exit "$rc"
fi
printf '{"status":"passed","formal_training":false}\n' > PREFLIGHT_WRAPPER_DONE.json
