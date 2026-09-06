#!/usr/bin/env bash
# Everything worth measuring overnight, launched in one go.
#
#   bash aire/tonight.sh          # run this ON AIRE, from ~/aichessathon-starter
#
# Four independent questions, so four jobs. The two that depend on each other are chained with
# a Slurm dependency rather than left for a human to notice: training the bucketed networks and
# then playing them is one question asked in two stages, and a person asleep cannot submit the
# second stage when the first finishes.
#
# Time limits are worked out, not padded. A clock match is PAIRS * 2 * 290s / WORKERS: 600 pairs
# at 140 workers is about 46 minutes, so 1h10 covers it with margin. measure-nets runs 300 pairs
# on 28 workers, which measured 1h51 on the last run, so 2h30. Asking for more than that is how
# a job that needs an hour queues behind everything on the cluster.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

CHAMP=${CHAMP:-versions/stack2}
if [ ! -d "$CHAMP" ]; then echo "ERROR: $CHAMP is missing; sync it first." >&2; exit 1; fi
echo "measuring everything against $CHAMP"
echo

# 1. Transposition table, 2^22 -> 2^24 entries. At 2.5M nodes a second and 2.3s a move the old
#    table turned over completely inside a single move, so nothing survived to the next.
A=versions/tt24 B="$CHAMP" TIME_CONTROL=120000,500 PAIRS=600 WORKERS=140 \
  sbatch --cpus-per-task=160 --mem=192G --time=01:10:00 aire/clock-match.slurm

# 2. Continuation history. The ordering axis that gave us history gravity at +81.6, done
#    properly: a score per (previous move, reply) pair added to history, rather than the single
#    move at a fixed rank that lost 11 Elo as the countermove heuristic.
A=versions/conthist B="$CHAMP" TIME_CONTROL=120000,500 PAIRS=600 WORKERS=140 \
  sbatch --cpus-per-task=160 --mem=192G --time=01:10:00 aire/clock-match.slurm

# 3. The best schedule-sweep network against the champion. The sweep measured all four against
#    net-v3, which ranked them but never answered whether any of them beats what we ship.
PATTERN=sched-h128b4sb400 CHAMPION="$PWD/$CHAMP" \
  sbatch --array=0-0 --time=02:30:00 aire/measure-nets.slurm

# 4. Output buckets, then the matches that decide them. Capacity the accumulator does not pay
#    for: hidden width is closed by the L2 budget, and these weights are not in that matrix.
TRAIN=$(sbatch --parsable aire/bullet-buckets.slurm)
echo "bucket training is $TRAIN"
# afterok, not afterany: if training fails there are no networks to play and the array would
# spend three nodes discovering that.
PATTERN='obuck-*' CHAMPION="$PWD/$CHAMP" \
  sbatch --parsable --dependency=afterok:"$TRAIN" --array=0-2 --time=02:30:00 \
    aire/measure-nets.slurm

echo
squeue -u "$USER" --format="%.10i %.16j %.9T %.11M %.10L %R"
