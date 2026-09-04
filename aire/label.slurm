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
# Rough scale on a 168 core node: 400 games per task at 50k nodes is a few minutes, and yields
# on the order of 10k usable positions after the quiet filter. Fifty tasks therefore produce
# several hundred thousand positions, which is ample for fitting the scalar weights and a
# reasonable start for anything larger.
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
GAMES=${GAMES:-400}
WORKERS=${WORKERS:-120}
DEPTH=${DEPTH:-10}

echo "shard $SHARD on $(hostname), $(nproc) cores"
uv run python tools/label.py \
  --shard "$SHARD" \
  --games "$GAMES" \
  --workers "$WORKERS" \
  --depth "$DEPTH" \
  --out "aire/data/shard-${SHARD}.epd"
