#!/usr/bin/env bash
# What is left of the CPU allowance, and what an array throttle should be.
#
#   bash aire/headroom.sh          # from a laptop, over ssh
#
# The allowance is a total across every running job, not a per-job limit, and it has been
# exceeded three times by sizing one job without counting the others: an 800-task label array
# starved measure-nets for an hour, a 300-task one starved a clock match, and a 24-throttle one
# left no room for a two-core probe. An array's real cost is throttle x cpus-per-task.

set -euo pipefail
REMOTE=${REMOTE:-aire}
CAP=${CAP:-1024}

ssh "$REMOTE" "squeue -u \$USER -t R --format='%.16i %.16j %.6C' | tail -n +2" | \
awk -v cap="$CAP" '
  { used += $3; n += 1; jobs[$2] += $3 }
  END {
    printf "%-20s %s\n", "job name", "cores"
    for (j in jobs) printf "  %-18s %5d\n", j, jobs[j]
    printf "\n  %-18s %5d of %d\n", "total running", used, cap
    printf "  %-18s %5d\n", "headroom", cap - used
    printf "\n  a 160-core clock match needs 160; an array of 32-core tasks can run %d concurrently\n",
           int((cap - used) / 32)
  }'
