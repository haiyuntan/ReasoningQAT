#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$RQ_ROOT/env/bootstrap/bin/python" "$RQ_REPO/repro_scripts/pipeline.py" "$@"
