#!/usr/bin/env bash
# Combine the shards into a training set and a genuinely held-out test set.
#
#   bash aire/build-data.sh
#
# Run on a login node after the labelling array finishes. Takes a few minutes; it is all
# streaming text work, no cluster needed.
#
# The held-out set is whole shards, not a random slice of the training file. A shard is one
# array task's self-play games, so holding shards out means the test positions come from games
# the network has never seen any part of. Slicing rows out of the combined file instead would
# leave the test set full of positions from the same games as the training set -- near-duplicate
# positions a few plies apart -- and the validation loss would look far better than the truth.
#
# Quota matters here. The text intermediates are the largest files involved and are useless once
# converted, so they are deleted as soon as the binary exists.

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA=aire/data
HELD_OUT=${HELD_OUT:-8}          # how many shards to reserve for validation
# Which labelling run to build from, and what to call the result. The defaults reproduce the
# original behaviour exactly, so an existing set is never renamed out from under the trainer.
# Set all three together when building a second set: sharing either name with a set already on
# disk overwrites 3.8GB that took a day of cluster time to produce.
PREFIX=${PREFIX:-shard}
TRAIN_OUT=${TRAIN_OUT:-positions}
TEST_OUT=${TEST_OUT:-test}

mapfile -t SHARDS < <(ls "$DATA"/${PREFIX}-*.epd 2>/dev/null | sort -V)
if [ "${#SHARDS[@]}" -lt $((HELD_OUT * 4)) ]; then
  echo "ERROR: only ${#SHARDS[@]} ${PREFIX}-*.epd shards; not enough to hold $HELD_OUT out." >&2
  exit 1
fi
for existing in "$DATA/$TRAIN_OUT.data" "$DATA/$TEST_OUT.data"; do
  if [ -s "$existing" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $existing already exists. Pass different TRAIN_OUT/TEST_OUT, or FORCE=1." >&2
    exit 1
  fi
done

TEST_SHARDS=("${SHARDS[@]: -$HELD_OUT}")
TRAIN_SHARDS=("${SHARDS[@]:0:$(( ${#SHARDS[@]} - HELD_OUT ))}")
echo "${#TRAIN_SHARDS[@]} shards for training, ${#TEST_SHARDS[@]} held out"

echo "combining..."
cat "${TRAIN_SHARDS[@]}" > "$DATA/$TRAIN_OUT.epd"
cat "${TEST_SHARDS[@]}"  > "$DATA/$TEST_OUT.epd"
echo "  train $(wc -l < "$DATA/$TRAIN_OUT.epd") positions"
echo "  test  $(wc -l < "$DATA/$TEST_OUT.epd") positions"

for name in "$TRAIN_OUT" "$TEST_OUT"; do
  echo "converting $name..."
  uv run python tools/to_bullet.py "$DATA/$name.epd" "$DATA/$name.txt"
  cargo run --release --manifest-path "$HOME/bullet/crates/utils/Cargo.toml" -- \
    convert --from text --input "$DATA/$name.txt" --output "$DATA/$name.data" --threads 8
  # The text form is the biggest file here and is dead once the binary exists.
  rm -f "$DATA/$name.txt"
done

echo
ls -la "$DATA/$TRAIN_OUT.data" "$DATA/$TEST_OUT.data"
echo "now: sbatch aire/bullet-schedule.slurm"
