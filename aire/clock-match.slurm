#!/usr/bin/env bash
# Play two agents against each other on the platform's real clock.
#
#   sbatch aire/clock-match.slurm                          # working tree vs versions/net-v2
#   B=versions/net-v3 sbatch aire/clock-match.slurm
#   PAIRS=300 sbatch aire/clock-match.slurm
#
# Fixed-node matches answer most questions and are deterministic, which makes them the default.
# They are wrong for anything that changes how long a node takes: a bigger network, a costlier
# evaluation, a change to the pruning that shifts the mix of node types. Those have to be played
# on a clock, because the whole point is that time spent is the currency.
#
# Why this belongs on a cluster, when a laptop can run it. Wall-clock games do not get shorter on
# faster hardware -- the engine spends its allocation whatever the machine -- so the only lever
# is how many run at once. Five workers on a laptop is four and a half hours; thirty here is
# forty-five minutes. Nothing else about the result changes.
#
# Workers are deliberately well under the core count. Each holds two agent processes with one
# thinking at a time, and a node that cannot give a thinking agent a core makes every engine look
# weaker in a way that is not evenly spread across the two sides. Undersubscribing is the whole
# reason the number is trustworthy.
#
# Results are wall-clock, so they are comparable within one machine and not across machines.
# Do not put a number from here beside a number from the laptop and read a difference into it.

#SBATCH --job-name=clock-match
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=aire/logs/clock-match-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

A=${A:-.}
B=${B:-versions/net-v2}
PAIRS=${PAIRS:-198}
TIME_CONTROL=${TIME_CONTROL:-120000,500}
WORKERS=${WORKERS:-30}

if [ ! -e "$B" ]; then
  echo "ERROR: $B does not exist. Snapshot it first with tools/snapshot.py." >&2
  exit 1
fi

echo "node $(hostname), $(nproc) cores, $WORKERS workers"
echo "$A vs $B at $TIME_CONTROL, $PAIRS pairs"
echo

uv run python tools/match.py --a "$A" --b "$B" \
  --time-control "$TIME_CONTROL" --pairs "$PAIRS" --workers "$WORKERS"
