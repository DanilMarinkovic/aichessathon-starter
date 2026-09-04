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
# Rough scale on a 168 core node: 6000 games per task at 50k nodes takes a few minutes and
# yields on the order of 240k usable positions after the quiet filter. Fifty-two tasks produce
# something over ten million positions, which is the right order for a king-bucketed network.
#
# You may need to add --partition and --account for your allocation.

#SBATCH --job-name=chess-label
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=168
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
WORKERS=${WORKERS:-120}
DEPTH=${DEPTH:-10}
SEED=${SEED:-$((SHARD * 1000 + 1))}

echo "shard $SHARD on $(hostname), $(nproc) cores"
uv run python tools/label.py \
  --shard "$SHARD" \
  --games "$GAMES" \
  --workers "$WORKERS" \
  --depth "$DEPTH" \
  --seed "$SEED" \
  --out "aire/data/shard-${SHARD}.epd"
