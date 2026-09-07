#!/usr/bin/env bash
# Does king-relative input help? Screened on CPU, because the GPU queue is 382 deep.
#
#   sbatch aire/cpu-arch.slurm
#   cat aire/data/cpu-arch.txt
#
# The question is whether the network's blindness is architectural. Ours is 768x4 -> 128: a
# small input layer feeding a wide hidden one, where the king's position selects one of four
# coarse weight banks. Stockfish's NNUE documentation says the opposite shape is right -- "most
# of the knowledge is stored in the first layer", with HalfKP[40960]->4x2->8->1 as the example --
# and king-relative features are how a network learns that pieces are aimed at a king.
# tools/blindspot.py measures our error at 306cp in exactly those positions.
#
# 32 banks is full resolution: the king is mirrored into files a-d before indexing, so 32 banks
# gives one per distinct king square, making the features (king square, piece, square).
#
# On CPU deliberately. bullet has no CPU backend -- cuda, rocm and metal only -- but tools/
# train.py is PyTorch and falls back to CPU, and it shares nnue.features with the engine so the
# two cannot disagree about what a feature means. It is far too slow to match a GPU run: this
# budget is a small fraction of what the shipped network saw. That is fine, because both arms
# get the *same* budget and the comparison is between architectures, not against the champion.
# A winner here earns GPU time; it does not replace anything on its own.
#
# Both arms also use one output bucket, which bullet cannot build at all -- it rejects
# MaterialCount<1>. Output buckets measured null on the clock, so this is the cleaner control.

#SBATCH --job-name=cpu-arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=aire/logs/cpu-arch-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"
mkdir -p aire/logs aire/data

DATA=${DATA:-aire/data/positions.epd}
LIMIT=${LIMIT:-8000000}
EPOCHS=${EPOCHS:-30}
# "banks:hidden" pairs. Parameter counts are deliberately close so the comparison is about
# where the parameters sit, not how many there are: 4x768x128 = 393k, 32x768x32 = 786k.
#
# A third arm at 32 banks with double the epochs. The screen is biased against the wider input:
# with 32 banks each sees a thirty-second of the data instead of a quarter, and king-relative
# features are correspondingly sparser -- the Stockfish documentation warns that a larger
# feature set "certainly will require more time to converge". If 32:32 loses to 4:128 at equal
# epochs but closes the gap when given twice as many, the architecture is convergence-limited
# rather than worse, and deserves the GPU run at full data. Without this arm a negative result
# cannot distinguish those two.
ARCHES=${ARCHES:-"4:128 32:32 32:32x2"}
RESULTS="$ORIGIN/aire/data/cpu-arch.txt"
SCREEN=${SCREEN:-aire/data/shard-799.epd}

[ -s "$DATA" ] || { echo "ERROR: $DATA missing" >&2; exit 1; }
[ -s "$SCREEN" ] || { echo "ERROR: $SCREEN missing -- needs a held-out shard" >&2; exit 1; }

: > "$RESULTS"
for ARCH in $ARCHES; do
  B=${ARCH%%:*}; H=${ARCH##*:}
  ARM_EPOCHS=$EPOCHS
  case "$H" in *x2) H=${H%x2}; ARM_EPOCHS=$((EPOCHS * 2)); NAME="cpu-b${B}h${H}-long";;
                 *) NAME="cpu-b${B}h${H}";; esac
  echo
  echo "=============== $NAME ==============="
  # Its own directory: train.py reads the architecture from the network already at
  # weights/net.npz, so two arms sharing a tree would train each other's shape.
  W="${TMPDIR:-/tmp}/$NAME-${SLURM_JOB_ID:-$$}"
  rm -rf "$W"; mkdir -p "$W/weights"
  cp "$ORIGIN"/*.py "$W/"
  cp -r "$ORIGIN/tools" "$ORIGIN/harness" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$W/"

  ( cd "$W"
    uv run python tools/init_net.py --hidden "$H" --buckets "$B" --out weights/net.npz
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
      uv run python tools/train.py --data "$ORIGIN/$DATA" --limit "$LIMIT" \
        --epochs "$ARM_EPOCHS" --out weights/net.npz
    uv run python tools/calibrate.py "$ORIGIN/$SCREEN" --limit 4000 --write
    echo "--- blind spot ---"
    uv run python tools/blindspot.py "$ORIGIN/$SCREEN" --limit 12000 | tee /tmp/bs.$$ )

  cp "$W/weights/net.npz" "aire/data/$NAME.npz"
  BLIND=$(grep "material is misleading" /tmp/bs.$$ | awk '{print $(NF-1)}')
  printf "%-18s banks=%-3s hidden=%-4s epochs=%-4s blind-spot-error=%s\n" \
    "$NAME" "$B" "$H" "$ARM_EPOCHS" "${BLIND:-?}" >> "$RESULTS"
  rm -rf "$W" /tmp/bs.$$
done

echo
echo "=== done ==="
cat "$RESULTS"
echo
echo "The champion, on the same held-out shard, scores 306cp in that band. Lower is better."
echo "A clear win here justifies a GPU run at full data; it does not ship on its own."
