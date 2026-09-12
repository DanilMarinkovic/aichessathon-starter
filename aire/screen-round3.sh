#!/usr/bin/env bash
# Screen the three overnight networks against the shipped one, at fixed nodes.
#
# Valid for sfb32w0 and sfb32w1: both are 512x32, identical in speed to sfwdl02, so the speed
# term cancels exactly and what is left is the WDL weight. For sfhuge it is valid only in one
# direction -- 1024 wide is slower, so equal nodes flatter it; a loss here is a real rejection,
# a win proves nothing until the clock says so.
set -euo pipefail
cd "$HOME/aichessathon-starter"
for tag in sfb32w0 sfb32w1 sfhuge; do
  net=$(ls aire/data/${tag}-h*b32ob8.npz 2>/dev/null | head -1)
  [ -n "$net" ] || { echo "skip $tag"; continue; }
  rm -rf "versions/$tag"; cp -r versions/sfwdl02 "versions/$tag"
  cp nnue.py "versions/$tag/nnue.py"
  cp "$net" "versions/$tag/weights/net.npz"
  PATTERN="$(basename "$net" .npz)" bash aire/screen-nets.sh 2>&1 \
    | grep -E 'calibrated|overall mean' | sed "s/^/  $tag /"
  cp "$net" "versions/$tag/weights/net.npz"
done
for tag in sfb32w0 sfb32w1 sfhuge; do
  [ -d "versions/$tag" ] || continue
  echo "=== $tag vs sfwdl02 at 300000 nodes ==="
  uv run python tools/match.py --a "versions/$tag" --b versions/sfwdl02 \
    --nodes 300000 --pairs 198 --workers 30 | tail -4
done
