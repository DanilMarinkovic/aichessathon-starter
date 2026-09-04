#!/usr/bin/env bash
# Generate the tuning dataset as a Slurm array job.
#
#   sbatch --array=0-51 aire/label.slurm
#   cat aire/data/shard-*.epd > aire/data/positions.epd
#
# Each task takes one node, plays self-play games with our engine, and labels the quiet
# positions with Stockfish. The work is embarrassingly parallel: shards share nothing and each
# writes its own file.
#
# Deliberately a small request: 32 cores for half an hour rather than a whole node for four.
# The work is trivially divisible, and Slurm's backfill scheduler can slot many small short
# jobs into gaps immediately, where a full-node multi-hour reservation waits for a whole node
# to come free for that long. Five of these fit on one node, so 52 shards still saturate the
# machine, they just start now instead of tonight.
#
# 6000 games per task at 50k nodes takes about ten minutes on 32 cores and yields on the order
# of 200k usable positions after the quiet filter, so 52 tasks produce roughly ten million.
#
# You may need to add --partition and --account for your allocation.

#SBATCH --job-name=chess-label
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=aire/logs/label-%A_%a.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs aire/data

export UCI_ENGINE="$HOME/.local/bin/stockfish"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

SHARD=${SLURM_ARRAY_TASK_ID:-0}
# Games are played in-process, so a worker pays the engine's eleven second compile once and
# then plays every game in its batch. That makes the game count per shard worth pushing high:
# at 400 games over 120 workers each worker plays three, and most of the job is compiling.
GAMES=${GAMES:-6000}
WORKERS=${WORKERS:-32}
DEPTH=${DEPTH:-10}
SEED=${SEED:-$((SHARD * 1000 + 1))}

echo "shard $SHARD on $(hostname), $(nproc) cores"

# Fail here, loudly, rather than an hour later inside the labelling phase. The binary was
# built on the login node; this is the first time it runs on the hardware that matters.
#
# The output is captured rather than piped into grep. Under `set -o pipefail`, `grep -q` exits
# as soon as it matches, the closed pipe kills the engine with SIGPIPE, and the pipeline
# inherits that failure, so a perfectly working engine reports as broken.
ENGINE_REPLY=$(printf 'uci\nquit\n' | "$UCI_ENGINE" 2>&1 || true)
case "$ENGINE_REPLY" in
  *uciok*) ;;
  *)
    echo "ERROR: $UCI_ENGINE does not run on $(hostname):" >&2
    echo "$ENGINE_REPLY" | head -3 >&2
    echo "Rebuild it on the login node: bash aire/setup.sh" >&2
    exit 1
    ;;
esac
uv run python tools/label.py \
  --shard "$SHARD" \
  --games "$GAMES" \
  --workers "$WORKERS" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  --out "aire/data/shard-${SHARD}.epd"
