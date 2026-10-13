#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
campaign_session="${1:-nohup_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RQ_ROOT/logs" "$RQ_ROOT/results" "$RQ_ROOT/outputs"
if [[ -s "$RQ_ROOT/results/background-campaign.pid" ]] && kill -0 "$(cat "$RQ_ROOT/results/background-campaign.pid")" 2>/dev/null; then
    echo 'Campaign already running; stop its process tree before restart.' >&2
    exit 1
fi
if [[ -s "$RQ_ROOT/results/download-background.pid" ]] && kill -0 "$(cat "$RQ_ROOT/results/download-background.pid")" 2>/dev/null; then
    echo 'Downloader already running; refusing a duplicate stream.' >&2
    exit 1
fi
export RQ_DOWNLOAD_WORKERS=1
: > "$RQ_ROOT/results/download-background.pid"
: > "$RQ_ROOT/results/background-campaign.pid"
# Double-fork into a separate session: nohup alone can still be reaped with
# the tool/terminal process group when the launching shell exits.
nohup setsid --fork /bin/bash -c 'echo "$$" > "$1"; exec "$2" "$3"' rq-download \
    "$RQ_ROOT/results/download-background.pid" "$RQ_ROOT/env/bootstrap/bin/python" "$RQ_REPO/repro_scripts/download_ranges.py" \
    >> "$RQ_ROOT/logs/download-serial.log" 2>&1 < /dev/null &
nohup setsid --fork /bin/bash -c 'echo "$$" > "$1"; exec "$2" -u "$3" --session "$4"' rq-campaign \
    "$RQ_ROOT/results/background-campaign.pid" "$RQ_ROOT/env/bootstrap/bin/python" "$RQ_REPO/repro_scripts/background_campaign.py" "$campaign_session" \
    >> "$RQ_ROOT/logs/$campaign_session.nohup.log" 2>&1 < /dev/null &
for ((campaign_attempt=0; campaign_attempt<100; campaign_attempt++)); do
    if [[ -s "$RQ_ROOT/results/background-campaign.pid" && -s "$RQ_ROOT/results/download-background.pid" ]]; then
        kill -0 "$(cat "$RQ_ROOT/results/background-campaign.pid")" 2>/dev/null && \
        kill -0 "$(cat "$RQ_ROOT/results/download-background.pid")" 2>/dev/null && break
    fi
    sleep 0.1
done
kill -0 "$(cat "$RQ_ROOT/results/background-campaign.pid")"
kill -0 "$(cat "$RQ_ROOT/results/download-background.pid")"
echo "Session: $campaign_session"
echo "Campaign PID: $(cat "$RQ_ROOT/results/background-campaign.pid")"
echo "Download PID: $(cat "$RQ_ROOT/results/download-background.pid")"
