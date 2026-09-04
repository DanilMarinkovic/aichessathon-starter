#!/usr/bin/env bash
# Train the evaluation network with bullet on a GPU node, then hand the weights back to the
# engine's own format and check the engine reproduces them.
#
#   sbatch aire/bullet.slurm
#   HIDDEN=256 EPOCHS=10 sbatch aire/bullet.slurm
#
# Bullet is a trainer and it runs here, offline. Only the weights it produces reach the zip,
# and the agent evaluates them with our own numba code, so nothing of bullet's ships or runs
# inside the submission. That is the same side of the line as labelling with Stockfish.
#
# Three format conversions happen before training, and each one is a place where being wrong
# is silent rather than loud:
#
#   positions.epd   ours: fen|score|result, score side-to-move relative
#     -> to_bullet.py, which negates black-to-move scores
#   positions.txt   bullet's text: FEN | score | result, score white relative
#     -> bullet-utils convert, which flips both back to side-to-move internally
#   positions.data  bulletformat binary
#
# The conversions are cached. Delete aire/data/positions.txt and positions.data to redo them
# after the dataset changes, or the job will happily train on the previous one.

#SBATCH --job-name=chess-bullet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=aire/logs/bullet-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
module load cuda/12.6.2
# bullet's build script reads CUDA_PATH; the Aire module only sets CUDA_DIR.
export CUDA_PATH="${CUDA_DIR:?cuda module did not set CUDA_DIR}"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data weights

HIDDEN=${HIDDEN:-128}
EPOCHS=${EPOCHS:-8}
POSITIONS=${POSITIONS:-50000000}
WDL=${WDL:-0.4}
NET_ID=${NET_ID:-chess}
EPD=${EPD:-aire/data/positions.epd}

echo "node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -s "$EPD" ]; then
  echo "ERROR: $EPD is missing or empty." >&2
  echo "Combine the shards first: cat aire/data/shard-*.epd > $EPD" >&2
  exit 1
fi

TEXT=aire/data/positions.txt
BINARY=aire/data/positions.data

if [ ! -s "$TEXT" ]; then
  echo "converting to bullet's text format (negating black-to-move scores)..."
  uv run python tools/to_bullet.py "$EPD" "$TEXT"
fi

if [ ! -s "$BINARY" ]; then
  echo "converting to bulletformat..."
  cargo run --release --manifest-path "$HOME/bullet/crates/utils/Cargo.toml" -- \
    convert --from text --input "$TEXT" --output "$BINARY" --threads "${SLURM_CPUS_PER_TASK:-8}"
fi

echo
echo "=== training: HIDDEN=$HIDDEN EPOCHS=$EPOCHS WDL=$WDL ==="
cd aire/bullet-trainer
HIDDEN="$HIDDEN" EPOCHS="$EPOCHS" POSITIONS="$POSITIONS" WDL="$WDL" NET_ID="$NET_ID" \
  DATA="$ORIGIN/$BINARY" THREADS="${SLURM_CPUS_PER_TASK:-8}" \
  cargo run --release --features cuda
cd "$ORIGIN"

# bullet writes checkpoints/<net_id>-<superbatch>/. Take the last one by superbatch number.
CHECKPOINT=$(ls -d aire/bullet-trainer/checkpoints/"$NET_ID"-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: bullet wrote no checkpoint under aire/bullet-trainer/checkpoints/" >&2
  exit 1
fi
echo "checkpoint $CHECKPOINT"

# raw.bin, not quantised.bin: floats with no padding, quantised here by the same code path
# that has always produced weights/net.npz, so tests/check_nnue.py still means what it meant.
uv run python tools/from_bullet.py "$CHECKPOINT/raw.bin" --out weights/net.npz

echo
echo "=== does the engine reproduce the network it was given? ==="
uv run python tests/check_nnue.py

echo
echo "trained. measure it before believing it:"
echo "  sbatch aire/measure.slurm --against versions/net-v2"
