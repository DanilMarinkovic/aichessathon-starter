#!/usr/bin/env bash
# Head-to-head at the real time control: each arm against one common base, in one allocation.
#
# Every arm here is the same architecture and the same search code as the base, so the two
# engines cost the same per node and the clock is measuring evaluation alone. Both arms share
# the allocation so machine drift and contention hit them equally -- see the note in
# anchor-pair.sh about the same champion2 reading -55.2 on 160 cores and -66.4 on 88.
set -euo pipefail
cd "$HOME/aichessathon-starter"
export PATH="$HOME/.local/bin:$PATH"
BASE=${BASE:-lazyorder}
PAIRS=${PAIRS:-400}
WORKERS=${WORKERS:-36}
ARMS=${ARMS:-lc0big}
for a in $ARMS; do
  echo "=== $a vs $BASE at 120000ms+500ms, $PAIRS pairs, $WORKERS workers ==="
  uv run python tools/match.py --a "versions/$a" --b "versions/$BASE" \
    --time-control 120000,500 --pairs "$PAIRS" --workers "$WORKERS" | tail -6
done
