#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec /data/CJH/envs/gaussiangpt/bin/python -B resume.py coordinate
