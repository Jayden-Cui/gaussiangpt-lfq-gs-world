#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
exec >bridge_smoke.log 2>&1
sha256sum -c BRIDGE_SMOKE.sha256 || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
exec /usr/bin/timeout 600 /data/CJH/envs/gaussiangpt/bin/python -B bridge_smoke.py
