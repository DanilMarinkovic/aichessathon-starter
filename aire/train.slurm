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
# Sizing. Each position costs about 268 bytes of index arrays, and the int64 tensors torch wants
# roughly triple that while both copies are live, so budget around 800 bytes per position: 40 GB
# at fifty million, 80 GB at a hundred million. The request below covers a hundred million with
# room to spare. Encoding runs at roughly a million positions a minute, and it is the reason the
# time limit is hours rather than one; training itself is about eleven minutes per epoch at
# fifty million positions on 64 cores.
#
# Runs on CPU, on the default partition, deliberately. This network is small enough that CPU
# training is a matter of minutes, the GPU partition is usually full, and its nodes have 24
# cores against 168 here. If a GPU is ever wanted:
#     sbatch --partition=gpu --gres=gpu:l40s:1 --cpus-per-task=8 aire/train.slurm

#SBATCH --job-name=chess-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
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
