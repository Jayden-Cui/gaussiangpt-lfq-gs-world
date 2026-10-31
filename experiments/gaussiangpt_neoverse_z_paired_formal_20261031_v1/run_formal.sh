#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
exec >queue.log 2>&1
sha256sum -c CODE.sha256 || exit 1
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /data/CJH/envs/gaussiangpt/bin/python -B coordinator.py
