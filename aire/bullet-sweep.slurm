#!/usr/bin/env bash
# Train several network sizes with bullet on one GPU, and sanity-check each one.
#
#   sbatch aire/bullet-sweep.slurm
#   cat aire/data/bullet-sweep.txt
#
# Phase one of two. This only trains and screens; measuring strength is a separate job on the
# CPU partition, because a 100-pair match would hold a GPU idle for half an hour.
#
# What it is testing. The first bullet network scored -110 Elo against net-v1 despite training
# on 4.6x the data, and the likely reason is that it had a quarter of the parameters: net-v1 is
# 4 king buckets x 768 x 128, which is 393,216 input weights, and the bullet network was 1 x 768
# x 128, or 98,304. The earlier architecture sweep said buckets were worth nothing, but it ran
# at six million positions, where nothing could distinguish capacity at all.
#
# Capacity is added here through the hidden layer rather than through king buckets. Both reach
# the same parameter count -- 768 x 512 is also 393,216 -- but bucketing would require bullet's
# ChessBucketsMirrored layout to agree with nnue.py's _king_buckets square-for-square, and a
# mismatch there is silent: the engine reads a different weight bank than the one trained, loads
# cleanly, and simply plays worse. Widening the hidden layer has no such failure mode.
#
# The cost of width is speed. The forward pass is linear in HIDDEN, so 512 does four times the
# work of 128, about +55ns on a ~500ns node. The weight matrix stays inside L2 either way:
# 768 x 512 x 2 bytes is 786KB, the same as net-v1's four buckets at 128.
#
# Every network is screened against the labelling engine before it is allowed near a match.
# A correlation far below net-v1's 0.96 means the export or the architecture is wrong, and
# there is no point spending an hour of match time finding that out.

# Sized to be backfilled, not to be comfortable. The work is about 35 minutes -- five for the
# conversions, twenty-five training seven networks, five screening them -- and the memory high
# water mark is the data loader's buffer, not the dataset, which is streamed. Asking for two
# hours and 64GB put this behind a four-hour queue on a partition that had idle GPUs.
#SBATCH --job-name=bullet-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=aire/logs/bullet-sweep-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
module load cuda/12.6.2
export CUDA_PATH="${CUDA_DIR:?cuda module did not set CUDA_DIR}"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

EPD=${EPD:-aire/data/positions.epd}
TEXT=aire/data/positions.txt
BINARY=aire/data/positions.data
POSITIONS=${POSITIONS:-59619785}
RESULTS="$ORIGIN/aire/data/bullet-sweep.txt"
SCREEN=${SCREEN:-aire/data/shard-1.epd}

# hidden superbatches. 128/8 is the network that lost, kept as the control: if the rest of the
# sweep does not beat it here, the problem is not capacity and this whole theory is wrong.
CONFIGS=("128 8" "128 40" "256 40" "384 40" "512 40" "512 80" "768 40")

if [ ! -s "$EPD" ]; then
  echo "ERROR: $EPD missing. cat aire/data/shard-*.epd > $EPD" >&2
  exit 1
fi

if [ ! -s "$TEXT" ]; then
  echo "converting to bullet's text format..."
  uv run python tools/to_bullet.py "$EPD" "$TEXT"
fi
if [ ! -s "$BINARY" ]; then
  echo "converting to bulletformat..."
  cargo run --release --manifest-path "$HOME/bullet/crates/utils/Cargo.toml" -- \
    convert --from text --input "$TEXT" --output "$BINARY" --threads "${SLURM_CPUS_PER_TASK:-8}"
fi

# Screening has to install each candidate at weights/net.npz, because that is the only network
# the engine will load. Put back whatever was there before, or the repository is left holding an
# arbitrary sweep network and the next thing anyone runs silently uses it.
RESTORE=""
if [ -f weights/net.npz ]; then
  RESTORE="$(mktemp)"; cp weights/net.npz "$RESTORE"
fi
restore() { [ -n "$RESTORE" ] && cp "$RESTORE" weights/net.npz && echo "restored the original weights/net.npz"; }
trap restore EXIT

: > "$RESULTS"
for config in "${CONFIGS[@]}"; do
  read -r HIDDEN SUPERBATCHES <<< "$config"
  NET_ID="h${HIDDEN}sb${SUPERBATCHES}"
  echo
  echo "=============== $NET_ID ==============="

  ( cd aire/bullet-trainer
    HIDDEN="$HIDDEN" EPOCHS="$SUPERBATCHES" POSITIONS="$POSITIONS" \
      NET_ID="$NET_ID" DATA="$ORIGIN/$BINARY" THREADS="${SLURM_CPUS_PER_TASK:-8}" \
      cargo run --release --features cuda )

  CHECKPOINT=$(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* | sort -t- -k2 -n | tail -1)
  KEEP="aire/data/bullet-${NET_ID}.npz"
  uv run python tools/from_bullet.py "$CHECKPOINT/raw.bin" --out "$KEEP"

  # Screen it. The engine loads weights/net.npz and nothing else, so the candidate has to be
  # installed there to be evaluated at all; it is replaced on every iteration and the last one
  # left behind is meaningless, which is why the keepers are the npz files listed above.
  cp "$KEEP" weights/net.npz
  SUMMARY=$(uv run python tools/side_bias.py "$SCREEN" --limit 4000 | tail -3)
  echo "$SUMMARY"
  CORRELATION=$(echo "$SUMMARY" | awk '/^  all/ {print $6}')
  printf "%-10s hidden=%-4s superbatches=%-3s correlation=%s\n" \
    "$NET_ID" "$HIDDEN" "$SUPERBATCHES" "${CORRELATION:-?}" >> "$RESULTS"
done

echo
echo "=== screening done ==="
cat "$RESULTS"
echo
echo "net-v1 scores 0.9596 on this measure and the network that lost scored 0.9379."
echo "Measure the best few: sbatch --array=0-3 aire/measure-nets.slurm"
