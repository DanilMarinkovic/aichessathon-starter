#!/usr/bin/env bash
# Build a training binary shard by shard, instead of concatenating everything first.
#
#   PREFIX=bal OUT=aire/data/bal2 sbatch aire/build-data-streaming.slurm
#
# aire/build-data.sh concatenates every shard into one .epd, rewrites that into one .txt, and
# converts the .txt. For 118M positions those intermediates are about 10GB and 12GB, and the
# quota here has 17GB free -- the build would fail most of the way through, after hours of
# generation.
#
# bulletformat is fixed 32-byte records, so a .data file is just its rows back to back and two
# of them concatenate with `cat`. Converting one shard at a time and appending brings the peak
# from ~26GB of intermediates down to one shard's worth, a few tens of megabytes.
#
# The intermediates also go to $TMPDIR rather than to /users. A compute node turns out to have
# 353GB free there against a 66GB quota on the shared filesystem, and it is node-local NVMe
# where /users is NFS, so this is both safer and faster. Nothing written there survives the
# job, which is exactly right for a file that exists for one conversion.
#
# The held-out shards are converted separately and never appended, so the test set stays made
# of whole games the trainer has not seen any part of.

#SBATCH --job-name=build-stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=aire/logs/build-stream-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

DATA=aire/data
PREFIX=${PREFIX:-bal}
OUT=${OUT:-$DATA/$PREFIX}
HELD_OUT=${HELD_OUT:-8}
REBALANCE=${REBALANCE:-0}

mapfile -t SHARDS < <(ls "$DATA"/${PREFIX}-*.epd 2>/dev/null | sort -V)
if [ "${#SHARDS[@]}" -lt $((HELD_OUT * 4)) ]; then
  echo "ERROR: only ${#SHARDS[@]} ${PREFIX}-*.epd shards" >&2; exit 1
fi
for existing in "$OUT-positions.data" "$OUT-test.data"; do
  [ -s "$existing" ] && { echo "ERROR: $existing exists; move it aside." >&2; exit 1; }
done

TEST_SHARDS=("${SHARDS[@]: -$HELD_OUT}")
TRAIN_SHARDS=("${SHARDS[@]:0:$(( ${#SHARDS[@]} - HELD_OUT ))}")
echo "${#TRAIN_SHARDS[@]} shards for training, ${#TEST_SHARDS[@]} held out"
echo "quota before:"; quota -s | tail -1

FLAGS=""
[ "$REBALANCE" = "1" ] && FLAGS="--rebalance"

SCRATCH="${TMPDIR:-/tmp}/build-${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "intermediates in $SCRATCH ($(df -h "$SCRATCH" | tail -1 | awk '{print $4}') free)"

convert_group() {
  local target="$1"; shift
  : > "$target"
  local n=0
  for shard in "$@"; do
    # shellcheck disable=SC2086
    uv run python tools/to_bullet.py $FLAGS "$shard" "$SCRATCH/rows.txt" >/dev/null
    cargo run --release --quiet --manifest-path "$HOME/bullet/crates/utils/Cargo.toml" -- \
      convert --from text --input "$SCRATCH/rows.txt" --output "$SCRATCH/rows.data" \
      --threads 8 >/dev/null
    cat "$SCRATCH/rows.data" >> "$target"
    rm -f "$SCRATCH/rows.txt" "$SCRATCH/rows.data"
    n=$((n + 1))
    if [ $((n % 50)) -eq 0 ]; then
      echo "  $n/$# shards, $(( $(stat -c %s "$target") / 32 )) positions so far"
    fi
  done
}

echo "converting training shards..."
convert_group "$OUT-positions.data" "${TRAIN_SHARDS[@]}"
echo "converting held-out shards..."
FLAGS="" convert_group "$OUT-test.data" "${TEST_SHARDS[@]}"

echo
echo "train: $(( $(stat -c %s "$OUT-positions.data") / 32 )) positions"
echo "test:  $(( $(stat -c %s "$OUT-test.data") / 32 )) positions"
echo "quota after:"; quota -s | tail -1
