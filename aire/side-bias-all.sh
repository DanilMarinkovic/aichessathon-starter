#!/usr/bin/env bash
# Does the network evaluate one side better than the other, and is it biased?
#
#   bash aire/sh 'cd ~/aichessathon-starter && bash aire/side-bias-all.sh'
#
# Round 59 was level from start to finish by Stockfish's reckoning and our engine read it as
# +150 to +400 for us on every sampled position -- all of them black-to-move. Either the network
# treats the two sides differently, which side_bias.py measures directly by comparing a position
# with its mirror, or it carries an additive offset that calibrate.py's slope fit cannot remove.
# Also prints the mean, which slope and correlation both hide: an offset shows up there and
# nowhere else.
set -euo pipefail
cd "$HOME/aichessathon-starter"
SHARD=${SHARD:-aire/data/shard-799.epd}
W="${TMPDIR:-/tmp}/sb-$$"; rm -rf "$W"; mkdir -p "$W/weights"
cp ./*.py "$W/"; cp -r tools pyproject.toml uv.lock "$W/"
trap 'rm -rf "$W"' EXIT
for v in champion2 sfking sfbig; do
  [ -d "versions/$v" ] || continue
  echo "=== $v"
  cp "versions/$v/weights/net.npz" "$W/weights/net.npz"
  ( cd "$W" && uv run --project "$HOME/aichessathon-starter" python tools/side_bias.py \
      "$HOME/aichessathon-starter/$SHARD" --limit 6000 | tail -5 )
  ( cd "$W" && uv run --project "$HOME/aichessathon-starter" python - "$HOME/aichessathon-starter/$SHARD" <<'PY'
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from tools.side_bias import evaluate, read
fens, reference = read(Path(sys.argv[1]), 6000)
ours = evaluate(fens)
import chess
white = np.array([chess.Board(f).turn == chess.WHITE for f in fens])
for name, mask in (("white to move", white), ("black to move", ~white)):
    o, r = ours[mask], reference[mask]
    print(f"  {name:<14} n {len(o):>5}   our mean {o.mean():+7.1f}   "
          f"reference mean {r.mean():+7.1f}   offset {o.mean() - r.mean():+7.1f}")
o, r = ours, reference
print(f"  {'overall':<14} n {len(o):>5}   our mean {o.mean():+7.1f}   "
      f"reference mean {r.mean():+7.1f}   offset {o.mean() - r.mean():+7.1f}")
PY
  )
done
