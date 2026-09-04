#!/usr/bin/env bash
# Run the measurement protocol on one standard node.
#
#   sbatch aire/measure.slurm --anchor
#   sbatch aire/measure.slurm --against versions/v1
#
# A standard Aire node is 168 cores and 768GB, so the limit is cores rather than memory: each
# worker holds two agent processes at roughly 250MB, and only one of them thinks at a time.
#
# Results are comparable with laptop results because the protocol is node-limited and the
# engine is deterministic under a node limit. That is not true of a --cost run, which is wall
# clock: those have to stay on one machine to mean anything, so do not mix them across hosts.
#
# You may need to add --partition and --account for your allocation.

#SBATCH --job-name=chess-measure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=aire/logs/measure-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

export UCI_ENGINE="$HOME/.local/bin/stockfish"
# One thread each: the parallelism is across games, not inside them. Threaded engines fighting
# for the same cores would make every result a measurement of the scheduler.
export UCI_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

WORKERS=${WORKERS:-48}
echo "node $(hostname), $(nproc) cores, $WORKERS workers"
uv run python tools/measure.py "$@" --workers "$WORKERS"
