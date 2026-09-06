#!/usr/bin/env bash
# Does the network get better if the output layer knows the phase?
#
#   sbatch aire/bullet-buckets.slurm
#   cat aire/data/bucket-sweep.txt
#
# Hidden width is closed. The note at the top of nnue.py has the measurement: every feature
# update reads a row of the accumulator matrix, so BUCKETS x 768 x HIDDEN x 2 bytes has to stay
# inside L2. At 128 wide with four king buckets that is 768KB and a node costs 484ns; at 256 it
# is 1.5MB, it spills to L3, and a node costs 930ns. 192 was measured on the clock and lost 50
# Elo. There is no room to make the network wider.
#
# Output buckets are not in that matrix. One bank is 2 x HIDDEN int16 -- 512 bytes -- and it is
# read once per evaluation rather than once per feature, so eight of them cost four kilobytes
# and nothing in nodes per second. It is the only capacity this architecture has left.
#
# Why phase in particular. The engine's losses are not blunders. A drawn game against a stronger
# opponent went +133 at move 22 and dead level by move 38 with no move losing 100cp: eight small
# inaccuracies converting an advantage. That, and the endgame draws, are what an output layer
# that cannot tell a queen middlegame from a rook ending looks like.
#
# The control is not trained here. bullet rejects MaterialCount<1> outright -- "the output
# bucket type must have more than 1 bucket" -- so a one-bank arm cannot go through this code
# path at all. It does not need to: aire/data/sched-h128b4sb320.npz is already a one-output-
# bucket network over the same 118M positions, the same 320 superbatches, the same width and
# king buckets, from the same trainer. It is copied in below under the sweep's own naming so
# the measurement array picks it up beside the candidates and all three land on one baseline.
#
# On the time limit, which on the gpu partition is not a formality, and which has now been
# wrong in both directions.
#
# Too long first: two hours out of habit, on a partition holding 159 jobs with 92 pending,
# moved the scheduler's estimated start from 06:40 to 18:09. Backfill can only use a gap the
# whole request fits into, so padding an hour to two cost half a day.
#
# Then too short: trimmed to one hour from the schedule sweep's measured 3.70s a superbatch,
# and the second config was killed nine superbatches from the end. That rate was measured
# without output buckets. With them a superbatch takes 5.85s -- 58% slower, because the select
# and the per-position bucket index are real work -- so two configs of 320 is 62 minutes of
# training, not 40. Extrapolating a rate across the one variable the job exists to change is
# how a limit ends up confidently wrong.
#
# 100 minutes: 62 training, about 8 for the control copy, two conversions and two screenings,
# and the rest margin. Re-measure this if OUTPUTS, EPOCHS or the architecture move.

#SBATCH --job-name=bullet-buckets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:40:00
#SBATCH --output=aire/logs/bullet-buckets-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
module load cuda/12.6.2
export CUDA_PATH="${CUDA_DIR:?cuda module did not set CUDA_DIR}"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

HIDDEN=${HIDDEN:-128}
NBUCKETS=${NBUCKETS:-4}
# 320 superbatches: the schedule sweep found 400, 560 and 720 all sat at or below it, so this
# is where the schedule stopped paying and there is no reason to spend longer.
EPOCHS=${EPOCHS:-320}
# The trainer accepts 2, 4 and 8; the count is a Rust const generic, so it is dispatched from a
# named list rather than read straight from the environment. 1 is absent because bullet will
# not build it -- see the note above on where the control comes from instead.
OUTPUTS=${OUTPUTS:-"4 8"}
CONTROL_NET=${CONTROL_NET:-aire/data/sched-h128b4sb320.npz}
# Overridable so the same job can train on a different labelling run. The label-quality
# comparison needs two networks that differ only in which set they saw, and re-editing a
# hardcoded path between two submissions is how they end up differing in something else.
BINARY=${BINARY:-aire/data/positions.data}
TEST_BINARY=${TEST_BINARY:-aire/data/test.data}
# Names the outputs, so two runs over different data do not overwrite each other's networks.
TAG=${TAG:-obuck}
RESULTS="$ORIGIN/aire/data/${TAG}-sweep.txt"
SCREEN=${SCREEN:-aire/data/shard-1.epd}

for required in "$BINARY" "$TEST_BINARY"; do
  if [ ! -s "$required" ]; then
    echo "ERROR: $required is missing. Build both first:" >&2
    echo "  bash aire/build-data.sh" >&2
    exit 1
  fi
done
POSITIONS=$(( $(stat -c %s "$BINARY") / 32 ))
echo "training on $POSITIONS positions, held-out set $(( $(stat -c %s "$TEST_BINARY") / 32 ))"

# Screening installs each candidate at weights/net.npz, the only network the engine will load.
# Put back what was there, or the repository is left holding an arbitrary sweep network.
RESTORE=""
if [ -f weights/net.npz ]; then RESTORE="$(mktemp)"; cp weights/net.npz "$RESTORE"; fi
restore() { [ -n "$RESTORE" ] && cp "$RESTORE" weights/net.npz && echo "restored weights/net.npz"; }
trap restore EXIT

: > "$RESULTS"

# The control, renamed into the sweep's series so PATTERN=obuck-* gathers it with the rest.
# Copied rather than linked: measure-nets calibrates each network in place before playing it,
# and calibration belongs to one set of weights.
if [ -s "$CONTROL_NET" ]; then
  cp "$CONTROL_NET" "aire/data/${TAG}-h128b4ob1.npz"
  printf "%-14s output_buckets=%-3s correlation=%s\n" "h128b4ob1" "1" "control" >> "$RESULTS"
  echo "control: $CONTROL_NET -> aire/data/${TAG}-h128b4ob1.npz"
else
  echo "WARNING: $CONTROL_NET is missing; the sweep will have no one-bucket control." >&2
fi

for OB in $OUTPUTS; do
  NET_ID="h${HIDDEN}b${NBUCKETS}ob${OB}"
  echo
  echo "=============== $NET_ID ==============="

  ( cd aire/bullet-trainer
    HIDDEN="$HIDDEN" BUCKETS="$NBUCKETS" OUTPUT_BUCKETS="$OB" EPOCHS="$EPOCHS" \
      POSITIONS="$POSITIONS" NET_ID="$NET_ID" DATA="$ORIGIN/$BINARY" \
      TEST_DATA="$ORIGIN/$TEST_BINARY" THREADS="${SLURM_CPUS_PER_TASK:-8}" \
      cargo run --release --features cuda )

  CHECKPOINT=$(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* | sort -t- -k2 -n | tail -1)
  KEEP="aire/data/${TAG}-${NET_ID}.npz"
  uv run python tools/from_bullet.py "$CHECKPOINT/raw.bin" \
    --buckets "$NBUCKETS" --output-buckets "$OB" --out "$KEEP"

  cp "$KEEP" weights/net.npz
  SUMMARY=$(uv run python tools/side_bias.py "$SCREEN" --limit 4000 | tail -3)
  echo "$SUMMARY"
  CORRELATION=$(echo "$SUMMARY" | awk '/^  all/ {print $7}')
  printf "%-14s output_buckets=%-3s correlation=%s\n" \
    "$NET_ID" "$OB" "${CORRELATION:-?}" >> "$RESULTS"
done

echo
echo "=== done ==="
cat "$RESULTS"
echo
echo "Correlation only screens for a network that loaded correctly and ranks positions. It has"
echo "never predicted Elo well, so the buckets question is settled on the clock:"
echo "  PATTERN=obuck-\* sbatch --array=0-2 aire/measure-nets.slurm   # control, 4, 8"
