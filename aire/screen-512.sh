#!/usr/bin/env bash
# Compare the 512x32 variants against each other at fixed nodes.
#
# Fixed nodes is normally the wrong instrument for a network change, because a bigger network
# is slower and equal-node play hands it an advantage it will not have on the clock. That does
# not apply here: every arm is 512 hidden with 32 king buckets, so they all cost the same per
# node and the speed term cancels exactly. What is left is the schedule and the WDL weight,
# which is what these runs vary.
set -euo pipefail
cd "$HOME/aichessathon-starter"
BASE=${BASE:-sfbig}
for tag in sfcos sfwdl02 sfwdl06; do
  net=$(ls aire/data/${tag}-h512b32ob8.npz 2>/dev/null || true)
  [ -n "$net" ] || { echo "skip $tag (no network)"; continue; }
  rm -rf "versions/$tag"; cp -r versions/champion2 "versions/$tag"
  cp nnue.py "versions/$tag/nnue.py"
  cp "$net" "versions/$tag/weights/net.npz"
  PATTERN="${tag}-h512b32ob8" bash aire/screen-nets.sh 2>&1 \
    | grep -E 'correlation|calibrated|overall mean' | sed "s/^/  $tag /"
  cp "$net" "versions/$tag/weights/net.npz"
done
for tag in sfcos sfwdl02 sfwdl06; do
  [ -d "versions/$tag" ] || continue
  echo "=== $tag vs $BASE at 300000 nodes ==="
  uv run python tools/match.py --a "versions/$tag" --b "versions/$BASE" \
    --nodes 300000 --pairs 198 --workers 30 | tail -4
done
