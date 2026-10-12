#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$RQ_ROOT/env/bitdistiller_qwen3/bin/python" "$RQ_REPO/repro_scripts/r9_bitdistiller_qwen3_w2.py" "$@"
