#!/usr/bin/env bash
set -euo pipefail
FIGURE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR="$FIGURE_ROOT/caches/mpl_figures" TMPDIR="$FIGURE_ROOT/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$MPLCONFIGDIR"
exec "$FIGURE_ROOT/envs/dtr_forward/bin/python" -B "$FIGURE_ROOT/src/figures/build_all.py" "$@"
