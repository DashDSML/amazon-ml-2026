#!/bin/bash
# Memory-guarded launcher for heavy EC2 modelling jobs (CLAUDE.md memory-budget rule).
#
# Runs a command in the background, polls its RSS every POLL_SECS, and kills it
# (SIGTERM, then SIGKILL if it does not exit) the moment RSS crosses CAP_MB --
# well before the kernel OOM-killer would (which fails silently and is only
# diagnosable after the fact via dmesg). Logs every checkpoint plus a clear
# "MEMORY GUARD TRIGGERED" line if it fires, so a run that would have OOM-killed
# instead stops cleanly with an obvious, greppable reason in its own log.
#
# Usage:
#   ./run_with_memcap.sh <cap_mb> <log_file> <command...>
#   ./run_with_memcap.sh 45000 ranker_model.log python ranker_model.py
#
# Exit code of this wrapper is the wrapped command's exit code, unless the
# guard fired, in which case it is 137 (matches SIGKILL convention).

set -u

CAP_MB="$1"; shift
LOG_FILE="$1"; shift
POLL_SECS="${POLL_SECS:-10}"

"$@" >>"$LOG_FILE" 2>&1 &
PID=$!
echo "[memcap] launched pid=$PID cap=${CAP_MB}MB poll=${POLL_SECS}s cmd: $*" | tee -a "$LOG_FILE"

TRIGGERED=0
CHECKS=0
while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL_SECS"
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    [ -z "$RSS_KB" ] && continue   # process just exited between the kill -0 and ps
    RSS_MB=$((RSS_KB / 1024))
    CHECKS=$((CHECKS + 1))
    if [ $((CHECKS % 6)) -eq 0 ]; then  # log a checkpoint roughly once a minute at default poll
        echo "[memcap] pid=$PID rss=${RSS_MB}MB cap=${CAP_MB}MB" | tee -a "$LOG_FILE"
    fi
    if [ "$RSS_MB" -ge "$CAP_MB" ]; then
        echo "[memcap] MEMORY GUARD TRIGGERED: pid=$PID rss=${RSS_MB}MB >= cap=${CAP_MB}MB -- sending SIGTERM" | tee -a "$LOG_FILE"
        kill -TERM "$PID" 2>/dev/null
        sleep 5
        if kill -0 "$PID" 2>/dev/null; then
            echo "[memcap] pid=$PID still alive after SIGTERM, sending SIGKILL" | tee -a "$LOG_FILE"
            kill -KILL "$PID" 2>/dev/null
        fi
        TRIGGERED=1
        break
    fi
done

if [ "$TRIGGERED" -eq 1 ]; then
    echo "[memcap] exiting 137 (guard-killed)" | tee -a "$LOG_FILE"
    exit 137
fi

wait "$PID"
CODE=$?
echo "[memcap] pid=$PID exited normally with code=$CODE" | tee -a "$LOG_FILE"
exit "$CODE"
