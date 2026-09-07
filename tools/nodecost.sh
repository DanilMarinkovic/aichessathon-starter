#!/usr/bin/env bash
# Sweep network shapes and print what each costs a node.
#
#   bash tools/nodecost.sh                    # the default spread
#   SHAPES="4:128 4:256" bash tools/nodecost.sh
#
# Each shape needs its own process, because nnue.py loads weights/net.npz at import and numba
# compiles the arrays in as constants. Each also needs its own directory, so a sweep never
# touches the repository's real weights -- the trap that once left a bucketed network installed
# and failed three hundred array tasks with a typing error.
#
# The shapes are chosen to separate two explanations of the same two numbers. Holding HIDDEN at
# 128 and moving BUCKETS from 1 to 5 multiplies the weight matrix by five and leaves the row a
# feature update reads unchanged. Holding BUCKETS at 4 and moving HIDDEN doubles both.

set -euo pipefail
cd "$(dirname "$0")/.."
ORIGIN="$PWD"

SHAPES=${SHAPES:-"1:128 2:128 4:128 5:128 4:64 4:192 4:256 5:256"}
NODES=${NODES:-400000}

WORK="${TMPDIR:-/tmp}/nodecost-$$"
rm -rf "$WORK"; mkdir -p "$WORK/weights"
cp "$ORIGIN"/*.py "$WORK/"
cp -r "$ORIGIN/tools" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$WORK/"
trap 'rm -rf "$WORK"' EXIT

# NETS names trained networks to bench instead of generating random ones. Random weights make a
# different search tree -- different move ordering, a different mix of interior and quiescence
# nodes -- and the fraction of a node spent in the accumulator is exactly what that mix decides.
# A shape sweep on random weights answers a question nobody asked.
if [ -n "${NETS:-}" ]; then
  for net in $NETS; do
    echo "--- $(basename "$net") ---"
    cp "$ORIGIN/$net" "$WORK/weights/net.npz"
    ( cd "$WORK" && uv run python tools/nodecost.py --nodes "$NODES" )
    echo
  done
  exit 0
fi

for shape in $SHAPES; do
  buckets=${shape%%:*}
  hidden=${shape##*:}
  ( cd "$ORIGIN" && uv run python tools/init_net.py --hidden "$hidden" --buckets "$buckets" \
      --out "$WORK/weights/net.npz" >/dev/null )
  ( cd "$WORK" && uv run python tools/nodecost.py --nodes "$NODES" )
  echo
done
