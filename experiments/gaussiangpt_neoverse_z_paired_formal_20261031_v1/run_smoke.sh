#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
exec >smoke.log 2>&1
sha256sum -c CODE.sha256 || exit 1
export PYTHONDONTWRITEBYTECODE=1 Z_ADAPTER_SOURCE_PARENT=/data/CJH/AGS-V6Direct/outputs
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/usr/bin/timeout 180 /data/CJH/envs/gaussiangpt/bin/python -B -m unittest -v test_codec_binding.py test_teacher_binding.py test_paired_step.py test_paired_real_cpu.py test_runner_core.py test_compare.py
rc=$?
if [ "$rc" -ne 0 ]; then printf '{"stage":"cpu_tests","exit_code":%s}\n' "$rc" > SMOKE_WRAPPER_FAILED.json; exit "$rc"; fi
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
/usr/bin/timeout 1800 /data/CJH/envs/gaussiangpt/bin/python -B integration_smoke.py
rc=$?
if [ "$rc" -ne 0 ]; then printf '{"stage":"integration","exit_code":%s}\n' "$rc" > SMOKE_WRAPPER_FAILED.json; exit "$rc"; fi
for mode in continuous16 lfq16; do
  /usr/bin/timeout 600 /data/CJH/envs/gaussiangpt/bin/python -B replay_native.py --smoke --mode "$mode"
  rc=$?
  if [ "$rc" -ne 0 ]; then printf '{"stage":"native_replay","exit_code":%s}\n' "$rc" > SMOKE_WRAPPER_FAILED.json; exit "$rc"; fi
done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /data/CJH/envs/gaussiangpt/bin/python -B smoke_finalize.py
