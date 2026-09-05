#!/usr/bin/env bash
# Play every network the bullet sweep produced against the current champion.
#
#   sbatch --array=0-6 aire/measure-nets.slurm
#   cat aire/data/net-results.txt
#
# Phase two. The sweep trains candidates in seconds on a GPU; this is the part that actually
# costs time, and it is the reason the two are separate jobs. One task per candidate, each on
# its own node, so seven networks take as long as one.
#
# On the clock, not on a node count. This is the one job where fixed nodes would give an
# actively wrong answer. A wider network evaluates each position better but costs nodes per
# second, and measuring at a fixed node count deletes the cost half of that trade: the widest
# network wins by construction, then searches shallower than a small one under a real clock and
# plays worse. Fixed nodes is right for a pruning rule or a bug fix, where nodes per second does
# not move and determinism makes debugging easy. It is wrong for choosing an architecture.
#
# 120000,500 is the platform's actual control. The agent gets one core and a short clock there,
# which is exactly the regime that punishes an expensive evaluation, so it has to be the regime
# we select in.
#
# Under a clock the engine is no longer deterministic, so unlike a fixed-node run the pairs may
# exceed the 198 openings and still produce new games -- match.py only caps repeats when nodes
# are fixed. 300 pairs is 600 games, about an hour on 28 workers.
#
# 28 workers on 32 cores, not 32. Wall-clock results degrade if the agents cannot get a core
# when they think they have one, and an oversubscribed node makes every engine look weaker in
# a way that is not evenly distributed.
#
# Each task works in its own copy of the repository. The engine loads exactly one network, from
# weights/net.npz, so two tasks sharing a directory would overwrite each other's candidate and
# quietly measure the same network several times.

#SBATCH --job-name=measure-nets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=aire/logs/measure-nets-%A_%a.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

CHAMPION=${CHAMPION:-$ORIGIN/versions/net-v3}
PAIRS=${PAIRS:-300}
TIME_CONTROL=${TIME_CONTROL:-120000,500}
RESULTS="$ORIGIN/aire/data/net-results.txt"

# Sorted so the array index means the same thing on every task. PATTERN selects one sweep's
# output: the data directory accumulates networks from every sweep ever run, and measuring a
# stale one wastes an hour of node time and puts a misleading row in the results file.
PATTERN=${PATTERN:-bullet-*}
mapfile -t CANDIDATES < <(ls "$ORIGIN"/aire/data/${PATTERN}.npz 2>/dev/null | sort -V)
TASK=${SLURM_ARRAY_TASK_ID:-0}

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "ERROR: no aire/data/${PATTERN}.npz. Run a sweep first." >&2
  exit 1
fi
if [ "$TASK" -ge "${#CANDIDATES[@]}" ]; then
  echo "task $TASK: only ${#CANDIDATES[@]} candidates exist, nothing to do"
  exit 0
fi
if [ ! -e "$CHAMPION" ]; then
  echo "ERROR: $CHAMPION is missing." >&2
  echo "Snapshot the uploaded agent on your laptop and rsync it:" >&2
  echo "  uv run python tools/snapshot.py net-v3" >&2
  exit 1
fi

CANDIDATE="${CANDIDATES[$TASK]}"
NAME=$(basename "$CANDIDATE" .npz)

WORK="${TMPDIR:-/tmp}/measure-${SLURM_JOB_ID:-0}-$TASK"
rm -rf "$WORK"; mkdir -p "$WORK/weights"
cp "$ORIGIN"/*.py "$WORK/"
cp -r "$ORIGIN/tools" "$ORIGIN/harness" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$WORK/"
cp "$CANDIDATE" "$WORK/weights/net.npz"
cd "$WORK"

echo "task $TASK: $NAME against $CHAMPION on $(hostname)"

# Calibrate before measuring. eval_scale belongs to a particular set of weights, and searcher.py
# reads its margins in centipawns; an uncalibrated network is measured with the aspiration window
# and futility margins landing on the wrong scale, which is a second change nobody asked for.
uv run python tools/calibrate.py "$ORIGIN/aire/data/shard-1.epd" --limit 4000 --write

SUMMARY=$(uv run python tools/match.py --a . --b "$CHAMPION" \
  --pairs "$PAIRS" --time-control "$TIME_CONTROL" --workers 28 | tail -20)
echo "$SUMMARY"

SCORE=$(echo "$SUMMARY" | grep -E "^  score" | awk '{print $2}')
ELO=$(echo "$SUMMARY" | grep -E "^  Elo" | sed 's/^  Elo *//')
printf "%-16s score=%-9s elo=%s\n" "$NAME" "${SCORE:-?}" "${ELO:-?}" >> "$RESULTS"
echo "done: $NAME ${ELO:-?}"
