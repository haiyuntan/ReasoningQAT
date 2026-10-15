#!/usr/bin/env bash
# Resolve this checkout rather than relying on a particular mount alias.
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export LEGACY="$(dirname "$ROOT")/FedProver"
export BF16_MODEL="$ROOT/models/REAL-Prover-v1-bf16"
export HF_HOME="$ROOT/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
