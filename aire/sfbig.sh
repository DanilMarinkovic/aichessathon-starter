#!/usr/bin/env bash
# Is 512 wide better than the 256 we are about to ship?
#
# Fixed nodes against sfking directly, because that is the incumbent now -- champion2 is the
# thing sfking already beat by +34 against a fixed Stockfish. A slower candidate that loses at
# equal nodes can be rejected without a timed match; one that wins has to be confirmed on the
# clock, because 512 wide costs nodes per second that fixed nodes hands back for free.
set -euo pipefail
cd "$HOME/aichessathon-starter"
rm -rf versions/sfbig
cp -r versions/champion2 versions/sfbig
cp nnue.py versions/sfbig/nnue.py
cp aire/data/sfbig-h512b32ob8.npz versions/sfbig/weights/net.npz
PATTERN='sfbig-h512b32ob8' bash aire/screen-nets.sh 2>&1 \
  | grep -E 'superbatches|correlation|calibrated|material is|overall'
cp aire/data/sfbig-h512b32ob8.npz versions/sfbig/weights/net.npz
for b in sfking champion2; do
  echo "=== sfbig vs $b at 300000 nodes ==="
  uv run python tools/match.py --a versions/sfbig --b "versions/$b" \
    --nodes 300000 --pairs 198 --workers 30 | tail -5
done
