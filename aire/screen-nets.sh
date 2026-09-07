#!/usr/bin/env bash
# Calibrate and blind-spot every candidate network, without touching the repository's weights.
#
#   PATTERN='sf*ob8' bash aire/screen-nets.sh
#
# Two minutes a network against two and a half hours for a clock match, so this runs first and
# says which candidates are worth a match at all. It is not a substitute for one: correlation
# and blind-spot error rank evaluations, and Elo is decided by evaluation and speed together.
#
# Each network is screened in its own scratch copy. nnue.py loads weights/net.npz at import and
# numba compiles the arrays in as constants, so one process per network is required anyway, and
# a scratch directory means a killed run can never leave a candidate installed as the engine's
# real network.

set -euo pipefail
cd "$(dirname "$0")/.."
ORIGIN="$PWD"

PATTERN=${PATTERN:-'sf*ob8'}
CALIBRATE_ON=${CALIBRATE_ON:-aire/data/shard-1.epd}
# Held out: the blind-spot number is only comparable across networks if none of them trained on
# these positions, and shard-799 is outside every set we have built.
SCREEN_ON=${SCREEN_ON:-aire/data/shard-799.epd}
LIMIT=${LIMIT:-12000}

for net in $ORIGIN/aire/data/$PATTERN.npz; do
  name=$(basename "$net" .npz)
  work="${TMPDIR:-/tmp}/screen-$name-$$"
  rm -rf "$work"; mkdir -p "$work/weights"
  cp "$ORIGIN"/*.py "$work/"
  cp -r "$ORIGIN/tools" "$ORIGIN/pyproject.toml" "$ORIGIN/uv.lock" "$work/"
  cp "$net" "$work/weights/net.npz"

  echo "=================== $name ==================="
  ( cd "$work"
    uv run python tools/calibrate.py "$ORIGIN/$CALIBRATE_ON" --limit 4000 --write \
      | grep -E "superbatches|correlation|calibrated eval_scale"
    uv run python tools/blindspot.py "$ORIGIN/$SCREEN_ON" --limit "$LIMIT" | tail -12 )
  # Keep the calibrated copy: eval_scale belongs to one set of weights, and a match that plays
  # the uncalibrated file measures the wrong scale.
  cp "$work/weights/net.npz" "$net"
  rm -rf "$work"
  echo
done
