#!/bin/bash
# Run a long job so that a dead one is distinguishable from a slow one.
#
# Written after losing five hours to a launch that died silently: the tail of
# its log read exactly like a healthy just-started run, because the last line a
# joblib job prints before its first progress report is the same either way.
#
#   scripts/runjob.sh <logfile> <command...>
#
# Writes alongside <logfile>:
#   <logfile>.status   one line, rewritten while alive:
#                      RUNNING pid=<pid> started=<iso> heartbeat=<iso> elapsed=<s>
#                      (rewritten every 5s)
#                      then DONE/FAILED with the exit code and total elapsed.
#
# So `cat runs/foo.log.status` answers "is this alive?" in one line, and a
# heartbeat older than a minute means it is not.
set -u
log="$1"; shift
status="${log}.status"
start_iso="$(date -Is)"; start_s="$(date +%s)"

"$@" > "$log" 2>&1 &
pid=$!
printf 'RUNNING pid=%s started=%s heartbeat=%s elapsed=0s\ncmd: %s\n' \
    "$pid" "$start_iso" "$start_iso" "$*" > "$status"

while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    now_s="$(date +%s)"
    printf 'RUNNING pid=%s started=%s heartbeat=%s elapsed=%ss\ncmd: %s\n' \
        "$pid" "$start_iso" "$(date -Is)" "$((now_s - start_s))" "$*" > "$status"
done

wait "$pid"; rc=$?
now_s="$(date +%s)"
state=DONE; [ "$rc" -eq 0 ] || state=FAILED
printf '%s pid=%s started=%s finished=%s elapsed=%ss exit=%s\ncmd: %s\n' \
    "$state" "$pid" "$start_iso" "$(date -Is)" "$((now_s - start_s))" "$rc" "$*" \
    > "$status"
