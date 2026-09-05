#!/bin/bash
# One-line liveness for every job started via scripts/runjob.sh.
#
#   scripts/jobs.sh [runs-dir]
#
# A heartbeat older than ~60s on a RUNNING row means the job is dead and the
# status file is stale -- which is the failure this exists to make visible.
set -u
dir="${1:-runs}"
now="$(date +%s)"
shopt -s nullglob
found=0
for f in "$dir"/*.log.status; do
    found=1
    head -1 "$f" | {
        read -r line
        hb="$(printf '%s\n' "$line" | grep -o 'heartbeat=[^ ]*' | cut -d= -f2-)"
        stale=""
        if [ -n "$hb" ]; then
            age=$(( now - $(date -d "$hb" +%s 2>/dev/null || echo "$now") ))
            [ "$age" -gt 60 ] && stale="  <-- STALE ${age}s, job is dead"
        fi
        printf '%-34s %s%s\n' "$(basename "$f" .log.status)" "$line" "$stale"
    }
done
[ "$found" -eq 1 ] || echo "no jobs started via scripts/runjob.sh in $dir/"
