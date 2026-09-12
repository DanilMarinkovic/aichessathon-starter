# SEE pruning thresholds — measured, positive, not shipped

Two constants in `searcher.py`. Everything else identical: same network, same `agent.py`.

```python
# searcher.py, capture pruning in the main search (~line 730)
-            and see_ge(states[ply], move, np.int64(-100) * depth) == 0
+            and see_ge(states[ply], move, np.int64(-177) * depth) == 0

# searcher.py, capture filter in quiescence (~line 476)
-        if move_promotion(move) == 0 and see_ge(states[ply], move, np.int64(0)) == 0:
+        if move_promotion(move) == 0 and see_ge(states[ply], move, np.int64(-74)) == 0:
```

Both make the search **less** aggressive: a move now has to be worse before it is pruned, so
more captures get searched. Cost is nodes, not correctness.

## What it measured

| instrument | result |
| --- | --- |
| vs `lazyorder`, clock, 400 pairs | **+13.5 [+2.3, +24.6]**, LOS 99.1% |
| vs Stockfish 12k, 800 pairs | **+65.2 [+54.9, +75.7]** — `lazyorder` reads +65.2, so no external regression |
| vs `singular` (superseded baseline), earlier | +6.8 [-0.4, +14.0], LOS 96.7% |
| failures across 2400 games | none |
| node cost | ~6% fewer nodes/sec, same depth reached in 4 of 5 test positions |

Two independent positive readings against different baselines. Combined estimate roughly
**+9 [+3, +15]** once the winner's-curse correction is applied — six candidates were screened
the same night (`qcheck`, `div18`, `div14`, `div34`, `div40`, `seethv2`) and only this one cleared.

## Why it was not shipped on 11 September

Position-level testing on eight real endgames from lost rounds 94/101/102, at the clock those
moves actually had, showed the depth reached is **1-2 plies shallower in three of them** (r101
mv70: 16 -> 14; r94 mv69: 11 -> 9), 4 plies deeper in one, unchanged in four. Neither build
plays those positions well (current 0/8 best moves, this 1/8). Danil's specific concern was
difficult endgames under time pressure, which is where the 6% node cost bites hardest, and the
aggregate game results cannot rule out a subgroup regression there. With uploads closing at
11:00 the decision was one-way and the expected gain single-digit, so it was left out.

## What would settle it

A head-to-head restricted to endgame starting positions, 400+ pairs, against `lazyorder`. If
that is neutral or positive, the subgroup worry is answered and this should ship. Roughly 2h22
on 40 cores at 36 workers.

The full ready-to-run variant is `aire/searcher-seethresh-v2.py`, built against the searcher at
md5 `3b242e68`. If `searcher.py` has moved on, re-apply the two lines above instead.
