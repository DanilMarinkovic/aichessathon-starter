#!/usr/bin/env bash
# Train several network architectures at once and measure each against the current champion.
#
#   uv run python tools/snapshot.py net-v1        # once, so there is something to beat
#   sbatch --array=0-5 aire/sweep.slurm
#   cat aire/data/sweep-results.txt
#
# This is the job that genuinely needs a cluster. Each task encodes, trains and then plays 200
# games, which is roughly forty minutes; six of them on a laptop is most of an evening, serially,
# and it cannot run two at once without them fighting over cores and memory. Here they are six
# independent nodes and the whole sweep takes as long as one config.
#
# It also answers questions that have so far been settled by argument rather than measurement:
# whether a wider accumulator beats more king buckets at equal parameter count, and whether
# bucketing earns anything at all against the plain 768 control.
#
# Each task works in its own copy of the repository, because the architecture is declared by
# weights/net.npz and two configs sharing a directory would overwrite each other's network.

#SBATCH --job-name=chess-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=aire/logs/sweep-%A_%a.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

# hidden buckets. Task 0 must always be the champion's own architecture. It is the calibration:
# the sweep trains on a reduced budget, so every task carries an offset against the champion,
# and only task 0 measures that offset. Read the results as `task N minus task 0`, never as the
# raw Elo against the champion. The first sweep lost 42 Elo to the budget cut alone, which is
# larger than any architecture effect it found.
#
# These configs test capacity rather than bucketing. The previous sweep compared 128x4 against
# 128x1 at six million positions and they scored identically, so buckets earned nothing there;
# but six million is too little data to distinguish capacity at all, and a wider accumulator
# only pays once there is enough data to fill it. Re-run this after the data grows.
CONFIGS=("128 4" "256 1" "384 1" "512 1" "256 4" "128 1")
TASK=${SLURM_ARRAY_TASK_ID:-0}
read -r HIDDEN BUCKETS <<< "${CONFIGS[$TASK]}"

DATA=${DATA:-$ORIGIN/aire/data/positions.epd}
CONTROL=${CONTROL:-$ORIGIN/versions/net-v1}
LIMIT=${LIMIT:-25000000}
EPOCHS=${EPOCHS:-6}
PAIRS=${PAIRS:-100}
RESULTS="$ORIGIN/aire/data/sweep-results.txt"

for required in "$DATA" "$CONTROL"; do
  if [ ! -e "$required" ]; then
    echo "ERROR: $required is missing." >&2
    echo "Need aire/data/positions.epd and versions/net-v1 (tools/snapshot.py net-v1)." >&2
    exit 1
  fi
done

WORK="${TMPDIR:-/tmp}/sweep-${SLURM_JOB_ID:-0}-$TASK"
rm -rf "$WORK"; mkdir -p "$WORK/weights"
cp "$ORIGIN"/*.py "$WORK/"
cp -r "$ORIGIN/tools" "$ORIGIN/harness" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$WORK/"
cd "$WORK"

echo "task $TASK: HIDDEN=$HIDDEN BUCKETS=$BUCKETS on $(hostname), $(nproc) cores"
echo "working in $WORK"

# Declare the architecture by writing an untrained network of that shape. nnue.py then reads
# its shape from the file exactly as it does on the platform, with no special casing.
uv run python tools/init_net.py --hidden "$HIDDEN" --buckets "$BUCKETS" --out weights/net.npz

# No shared encoding cache: the feature indices depend on the bucket count, so a cache built
# for one architecture is wrong for another, and concurrent tasks would race over the file.
uv run python tools/train.py \
  --data "$DATA" --out weights/net.npz --limit "$LIMIT" --epochs "$EPOCHS"

uv run python tools/verify_net.py \
  --checkpoint weights/net.pt --positions "$DATA" --count 2000

echo
echo "=== playing $PAIRS pairs against $CONTROL ==="
SUMMARY=$(uv run python tools/match.py --a . --b "$CONTROL" \
  --pairs "$PAIRS" --nodes 300000 --workers 32 | tail -20)
echo "$SUMMARY"

SCORE=$(echo "$SUMMARY" | grep -E "^  score" | awk '{print $2}')
ELO=$(echo "$SUMMARY" | grep -E "^  Elo" | sed 's/^  Elo *//')
printf "%-3s hidden=%-4s buckets=%-2s  score=%-9s elo=%s\n" \
  "$TASK" "$HIDDEN" "$BUCKETS" "${SCORE:-?}" "${ELO:-?}" >> "$RESULTS"

cp weights/net.npz "$ORIGIN/aire/data/sweep-${TASK}-h${HIDDEN}-b${BUCKETS}.npz"
echo "kept the network at aire/data/sweep-${TASK}-h${HIDDEN}-b${BUCKETS}.npz"
