#!/usr/bin/env bash
# Are deeper labels worth more than more labels?
#
#   bash aire/label-quality.sh
#
# The existing set is 118M positions labelled by Stockfish at depth 10. The question is whether
# that depth is the ceiling: the network underfits -- training loss 0.01164 against a held-out
# 0.01180, a gap of 1.4% -- which rules out more data of the same kind, because overfitting is
# the condition more data cures and we do not have it. What it does not rule out is a noisy
# teacher. Some of that loss floor is the label, not the model, and a sharper label lowers it.
#
# The comparison has to be size-matched or it answers a different question. Deep labels are
# expensive, so there will be fewer of them, and a 52M-versus-118M result confounds label depth
# with dataset size -- the one variable we already know the answer for. So the control here is
# the existing depth-10 data truncated to exactly the same number of positions. bulletformat is
# fixed 32-byte records, so truncation is exact and costs nothing.
#
# Both sets keep their own held-out shards, so each network's validation loss is measured
# against data labelled the way it was trained. Comparing a depth-14 network's loss against a
# depth-10 test set would just measure how unlike the two teachers are.

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

DATA=aire/data
SHALLOW=${SHALLOW:-$DATA/positions.data}

if [ ! -s "$SHALLOW" ]; then
  echo "ERROR: $SHALLOW is missing; the depth-10 set is the control." >&2
  exit 1
fi

echo "== building the depth-14 set =="
PREFIX=deep TRAIN_OUT=deep-positions TEST_OUT=deep-test bash aire/build-data.sh

DEEP="$DATA/deep-positions.data"
N=$(( $(stat -c %s "$DEEP") / 32 ))
AVAILABLE=$(( $(stat -c %s "$SHALLOW") / 32 ))
echo
echo "depth-14 set: $N positions"
echo "depth-10 set: $AVAILABLE positions available"
if [ "$N" -gt "$AVAILABLE" ]; then
  echo "ERROR: the deep set is larger than the control; nothing to match against." >&2
  exit 1
fi

echo "== truncating the depth-10 set to match =="
head -c $(( N * 32 )) "$SHALLOW" > "$DATA/shallow-positions.data"
# The control's test set is the depth-10 held-out file, which already exists and is disjoint
# from the training shards, so truncation cannot leak a training position into it.
cp "$DATA/test.data" "$DATA/shallow-test.data"

echo
ls -la "$DATA"/deep-positions.data "$DATA"/deep-test.data \
       "$DATA"/shallow-positions.data "$DATA"/shallow-test.data
echo
echo "Both sets now hold $N training positions. Train one network on each:"
echo "  NET=deep    DATA=$DEEP                        TEST=$DATA/deep-test.data"
echo "  NET=shallow DATA=$DATA/shallow-positions.data TEST=$DATA/shallow-test.data"
echo
echo "then measure them against each other on the clock, which is the only comparison that"
echo "counts -- the two validation losses are computed against different teachers and are not"
echo "comparable to each other."
