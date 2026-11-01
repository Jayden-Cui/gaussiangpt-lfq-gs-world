#!/usr/bin/env bash
set -euo pipefail
cd /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_qformer_full177_20261101_v1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
sha256sum -c CODE.sha256
exec /data/CJH/envs/gaussiangpt/bin/python -B coordinator.py > coordinator.log 2>&1
