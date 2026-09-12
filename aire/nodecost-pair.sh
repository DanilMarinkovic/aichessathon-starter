#!/usr/bin/env bash
# Node cost for two candidates, interleaved in one allocation.
#
#   NETS="versions/sfking versions/sfbig" sbatch ... --wrap='bash aire/nodecost-pair.sh'
#
# Lives in the repository, not in /tmp. A script written to /tmp on the login node does not
# exist on the compute node that runs the job -- they have separate /tmp -- and the job fails in
# one second with "No such file or directory". Anything sbatch needs has to be on the shared
# filesystem.
set -euo pipefail
cd "$HOME/aichessathon-starter"
NETS=${NETS:-"versions/sfking versions/sfbig"}
W="${TMPDIR:-/tmp}/ncpair-$$"; rm -rf "$W"; mkdir -p "$W/weights"
cp ./*.py "$W/"; cp -r tools pyproject.toml uv.lock "$W/"
trap 'rm -rf "$W"' EXIT
for n in $NETS; do
  echo "--- $n"
  cp "$n/weights/net.npz" "$W/weights/net.npz"
  ( cd "$W" && uv run --project "$HOME/aichessathon-starter" python tools/nodecost.py \
      --nodes 5000000 --repeats 1 | grep -E "hidden|ns a node|control:" )
done
