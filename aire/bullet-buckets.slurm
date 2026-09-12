#!/usr/bin/env bash
# Does the network get better if the output layer knows the phase?
#
#   sbatch aire/bullet-buckets.slurm
#   cat aire/data/bucket-sweep.txt
#
# Hidden width is NOT closed, and an earlier version of this note said it was.
#
# The claim was that BUCKETS x 768 x HIDDEN x 2 bytes has to stay inside L2, so 256 wide
# "spills" and is ruled out. That is not what the numbers say. A feature update reads one row,
# HIDDEN int16s, so cost is linear in width: 484ns a node at 128, 930ns at 256, which is 1.92x
# for 2x the width and needs no cache story at all. It is a smooth trade, not a wall, and 192
# losing 50 Elo on the clock was measured on our own material-dominated labels, where the extra
# capacity had nothing to learn. On Stockfish data that trade may well reverse, which is why a
# 256 wide arm is on the cluster rather than ruled out on a spreadsheet.
#
# Output buckets are the capacity that is genuinely free. One bank is 2 x HIDDEN int16 -- 512
# bytes -- read once per evaluation rather than once per feature, so eight of them cost four
# kilobytes and nothing in nodes per second. Width and bucket count both buy capacity too; they
# just have a price, which is what the arms of this sweep and the architecture screen measure.
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

# A binpack run has no separate held-out file in this format; the clock match is the check.
[ "${BINARY##*.}" = "binpack" ] && TEST_BINARY="$BINARY"
for required in "$BINARY" "$TEST_BINARY"; do
  if [ ! -s "$required" ]; then
    echo "ERROR: $required is missing. Build both first:" >&2
    echo "  bash aire/build-data.sh" >&2
    exit 1
  fi
done
# A binpack is compressed and variable width, so its position count cannot be read off the
# file size the way a bulletformat file's can. POSITIONS then has to be given, and it only sets
# how many batches make a superbatch -- an estimate that is too low just means shorter
# superbatches, not less data seen.
case "$BINARY" in
  *.binpack) POSITIONS=${POSITIONS:?POSITIONS must be set for a binpack} ;;
  *)         POSITIONS=$(( $(stat -c %s "$BINARY") / 32 )) ;;
esac
# TARGET_SAMPLES fixes the total number of positions seen rather than the number of passes.
#
# A superbatch is one pass over the data, so two datasets of different sizes trained for the
# same number of superbatches do not see the same amount of data. Comparing a 50M set against a
# 118M set at 320 superbatches each would have the smaller one see 2.4x fewer positions, and it
# would lose on volume while appearing to lose on distribution -- which is the one variable the
# comparison exists to isolate. The control saw 320 x 118,035,504, so that is the number to
# match. Cannot be computed at submission time, because the dataset does not exist yet when the
# job is queued behind the one that builds it.
if [ -n "${TARGET_SAMPLES:-}" ]; then
  EPOCHS=$(( TARGET_SAMPLES / POSITIONS ))
  [ "$EPOCHS" -lt 1 ] && EPOCHS=1
  echo "TARGET_SAMPLES=$TARGET_SAMPLES over $POSITIONS positions -> $EPOCHS superbatches"
fi
# BINARY may be absolute -- large datasets live on $SCRATCH, not under the repository, because
# /users gives 66GB and $SCRATCH gives a terabyte. Prepending $ORIGIN to an absolute path
# produces /users/.../aichessathon-starter//mnt/scratch/... and the loader fails with a bare
# "No such file or directory" three seconds in, having said nothing about which file.
case "$BINARY" in /*) DATA_PATH="$BINARY" ;; *) DATA_PATH="$ORIGIN/$BINARY" ;; esac
case "$TEST_BINARY" in /*) TEST_PATH="$TEST_BINARY" ;; *) TEST_PATH="$ORIGIN/$TEST_BINARY" ;; esac
[ -s "$DATA_PATH" ] || { echo "ERROR: $DATA_PATH is missing or empty" >&2; exit 1; }
echo "training on $POSITIONS positions from $DATA_PATH"

# Screening needs the candidate installed at weights/net.npz, the only network the engine will
# load -- so it happens in a scratch copy of the repository, never in the repository itself.
#
# The previous version installed into the real weights/net.npz and put it back with an EXIT
# trap. That works right up until the job is killed at its wall limit, which is a SIGKILL and
# runs no trap. It happened: a bucketed network was left installed, and every job afterwards
# read a 2-D output layer with code expecting a 1-D one. Three hundred array tasks failed in
# four seconds each with a numba typing error pointing at nnue.py, which is a long way from the
# actual cause. A scratch directory cannot fail that way, and it also lets two of these run at
# once without fighting over the same file.
SCRATCH="${TMPDIR:-/tmp}/screen-${SLURM_JOB_ID:-$$}"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH/weights"
cp "$ORIGIN"/*.py "$SCRATCH/"
cp -r "$ORIGIN/tools" "$ORIGIN/harness" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$SCRATCH/"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

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
  # TAG is part of the name, not just of the output file. Without it every run over different
  # data shares a checkpoint directory, and the newest is not necessarily the one just trained.
  NET_ID="${TAG}-h${HIDDEN}b${NBUCKETS}ob${OB}"
  echo
  echo "=============== $NET_ID ==============="

  ( cd aire/bullet-trainer
    HIDDEN="$HIDDEN" BUCKETS="$NBUCKETS" OUTPUT_BUCKETS="$OB" EPOCHS="$EPOCHS" \
      POSITIONS="$POSITIONS" NET_ID="$NET_ID" DATA="$DATA_PATH" \
      TEST_DATA="$TEST_PATH" THREADS="${SLURM_CPUS_PER_TASK:-8}" \
      cargo run --release --features cuda )

  # Sort on the trailing superbatch number, not on "the second hyphen-separated field" -- the
  # path itself contains hyphens (aire/bullet-trainer/...), so -t- -k2 was sorting on
  # "trainer/checkpoints/..." which is identical for every run. With three checkpoints sharing a
  # NET_ID it silently converted another run's network, and the only reason it was caught is
  # that tools/blindspot.py reported two supposedly different networks with identical error.
  # sed, not awk. The awk version of this line was written from inside a single-quoted ssh
  # command and kept the '\"'\"' escaping that only makes sense there. At the top level of a
  # script that resolves to a double-quoted string, so the shell -- not awk -- expanded $NF,
  # and `set -u` killed the job with "NF: unbound variable". It killed three jobs, every one of
  # them after training had finished, because this line only runs once the GPU work is done.
  #
  # Sort on the trailing superbatch number. Not "the second hyphen-separated field": the path
  # itself contains hyphens (aire/bullet-trainer/...), so -t- -k2 sorted on a field identical
  # for every run, and with three checkpoints sharing a NET_ID it silently converted another
  # run's network.
  CHECKPOINT=$(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* \
    | sed 's/.*-\([0-9][0-9]*\)$/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)
  [ -d "$CHECKPOINT" ] || { echo "ERROR: no checkpoint for $NET_ID" >&2; exit 1; }
  echo "converting $CHECKPOINT"
  KEEP="aire/data/${NET_ID}.npz"
  uv run python tools/from_bullet.py "$CHECKPOINT/raw.bin" \
    --buckets "$NBUCKETS" --output-buckets "$OB" --out "$KEEP"

  # The intermediate checkpoints have done their job once the network is converted, and at
  # this width they are enormous: 218MB each, eight per run. Sixteen runs of that filled a
  # 65GB quota and killed three GPU jobs -- one of them after it had trained for an hour and
  # was writing its output. Keep the last one, in case a resume is wanted, and drop the rest.
  for spent in $(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* 2>/dev/null \
      | sed 's/.*-\([0-9][0-9]*\)$/\1 &/' | sort -k1,1n | head -n -1 | cut -d' ' -f2-); do
    rm -rf "$spent"
  done

  cp "$KEEP" "$SCRATCH/weights/net.npz"
  SUMMARY=$( cd "$SCRATCH" && uv run python tools/side_bias.py "$ORIGIN/$SCREEN" --limit 4000 \
             | tail -3 )
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
