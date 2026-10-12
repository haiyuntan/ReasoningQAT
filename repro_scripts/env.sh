#!/usr/bin/env bash
set -euo pipefail
export RQ_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RQ_ROOT="$(dirname "$RQ_REPO")"
export RQ_MODEL="$RQ_ROOT/models/Qwen3-1.7B"
export RQ_DATA="$RQ_ROOT/datasets"
export HF_HOME="$RQ_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$HF_HOME/modelscope"
export RQ_OPENTHOUGHTS_PARQUET_GLOB="$RQ_DATA/openthoughts3/data/**/*.parquet"
export RQ_FINEWEB_PARQUET="$RQ_DATA/fineweb-edu/sample/10BT/000_00000.parquet"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export PYTHONPATH="$RQ_REPO${PYTHONPATH:+:$PYTHONPATH}"
