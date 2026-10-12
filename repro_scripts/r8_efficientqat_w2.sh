#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$RQ_ROOT/env/efficientqat_qwen3/bin/python" "$RQ_REPO/repro_scripts/r8_efficientqat_qwen3_w2.py" "$@"
