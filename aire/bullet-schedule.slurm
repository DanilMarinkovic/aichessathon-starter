#!/usr/bin/env bash
# How long should the network train? Sweep the schedule length at the architecture that won.
#
#   sbatch aire/bullet-schedule.slurm
#   cat aire/data/schedule-sweep.txt
#
# The architecture question is settled. 128 hidden with 4 king buckets beat every alternative
# measured on the clock: 5 buckets was the same, 1 bucket lost 14 Elo, and 192 hidden lost 50
# because a wider forward pass costs more nodes per second than the better evaluation returns.
# What is not settled is how long to train it. Going from 40 superbatches to 80 was worth about
# 15 Elo on identical data, and nothing yet says where that stops.
#
# Why this is the lever rather than more data. The network is underfitting, not overfitting:
# training on 59M positions ended with a training loss of 0.01164 against a held-out 0.01180,
# and the held-out curve never turned up. More data fixes memorisation, which is not what is
# happening. More passes fix a network that has not finished extracting what it already has,
# which is. When these schedules do start to overfit, that is the signal the 59M set is
# exhausted, and the 120M set now being generated becomes the next lever.
#
# Every run reports a held-out loss, computed on shards deliberately excluded from training.
# Without it the only way to find the overfitting point is an hour-long match per config, which
# is absurd when the training itself takes minutes.

#SBATCH --job-name=bullet-sched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=aire/logs/bullet-sched-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
module load cuda/12.6.2
export CUDA_PATH="${CUDA_DIR:?cuda module did not set CUDA_DIR}"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

HIDDEN=${HIDDEN:-128}
NBUCKETS=${NBUCKETS:-4}
# Superbatch counts. 80 is the current champion, so it is the control as well as a candidate.
SCHEDULES=${SCHEDULES:-"80 120 160 240 320"}
BINARY=aire/data/positions.data
TEST_BINARY=aire/data/test.data
RESULTS="$ORIGIN/aire/data/schedule-sweep.txt"
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
for SB in $SCHEDULES; do
  NET_ID="h${HIDDEN}b${NBUCKETS}sb${SB}"
  echo
  echo "=============== $NET_ID ==============="

  ( cd aire/bullet-trainer
    HIDDEN="$HIDDEN" BUCKETS="$NBUCKETS" EPOCHS="$SB" POSITIONS="$POSITIONS" \
      NET_ID="$NET_ID" DATA="$ORIGIN/$BINARY" TEST_DATA="$ORIGIN/$TEST_BINARY" \
      THREADS="${SLURM_CPUS_PER_TASK:-8}" cargo run --release --features cuda )

  CHECKPOINT=$(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* | sort -t- -k2 -n | tail -1)
  KEEP="aire/data/bullet-${NET_ID}.npz"
  uv run python tools/from_bullet.py "$CHECKPOINT/raw.bin" --buckets "$NBUCKETS" --out "$KEEP"

  cp "$KEEP" weights/net.npz
  SUMMARY=$(uv run python tools/side_bias.py "$SCREEN" --limit 4000 | tail -3)
  echo "$SUMMARY"
  CORRELATION=$(echo "$SUMMARY" | awk '/^  all/ {print $7}')
  printf "%-14s superbatches=%-4s correlation=%s\n" \
    "$NET_ID" "$SB" "${CORRELATION:-?}" >> "$RESULTS"
done

echo
echo "=== done ==="
cat "$RESULTS"
echo
echo "Held-out loss per superbatch is in this log; the schedule to keep is the one where it"
echo "stops falling. Measure the best two or three:"
echo "  sbatch --array=0-4 aire/measure-nets.slurm"
