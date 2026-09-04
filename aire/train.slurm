#!/usr/bin/env bash
# Train the evaluation network on a GPU node.
#
#   sbatch aire/train.slurm
#   sbatch aire/train.slurm --limit 4000000 --epochs 30
#
# Training happens offline, here, and only the resulting weights ship. That is the side of the
# line the rules allow: the ban covers running someone else's engine inside the zip, not
# learning from positions one labelled.
#
# Encoding is the slow part and it is single threaded, so --cache is used to do it once; later
# runs over the same data start straight into training.
#
# You may need to add --partition and --account for your allocation. GPU nodes usually need an
# explicit partition, so check `sinfo -s` if this sits in the queue.

#SBATCH --job-name=chess-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=aire/logs/train-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs weights

DATA=${DATA:-aire/data/positions.epd}
if [ ! -s "$DATA" ]; then
  echo "ERROR: $DATA is missing or empty." >&2
  echo "Combine the shards first: cat aire/data/shard-*.epd > $DATA" >&2
  exit 1
fi

echo "node $(hostname), $(nproc) cores"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no gpu visible"
echo "$(wc -l < "$DATA") labelled positions"

uv run python tools/train.py \
  --data "$DATA" \
  --cache aire/data/encoded.npz \
  --out weights/net.npz \
  "$@"

echo
echo "checking the engine reproduces the network it was given..."
uv run python tools/verify_net.py \
  --checkpoint weights/net.pt \
  --positions "$DATA" \
  --count 4000
