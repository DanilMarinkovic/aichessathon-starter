"""Prove the engine cannot flag, across the whole range of remaining time.

A flag is a lost game, and the time manager is the one place where a change can lose games
without ever playing a bad move. The budget has always been capped at a fraction of what is
left; the extension added for an unsettled root must not widen that cap, only use it. This
calls the agent exactly as the platform does, at remaining times from a full clock down to a
few hundred milliseconds, and checks what it actually spent.

Positions are chosen to include ones where the root move keeps changing, since those are the
only ones the extension applies to and therefore the only ones that could overrun.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Never spend more than this share of the remaining clock on one move. The budget's own cap is
# 0.35; the slack above it covers timer granularity and the last iteration's tail.
SAFE_SHARE = 0.45

import chess  # noqa: E402

import agent  # noqa: E402

POSITIONS = [
    ("opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("sharp", "1r2r1k1/5pp1/p2q3p/bpp5/2P1P3/P2PNN1b/2Q2P2/R1BR2K1 w - - 0 27"),
    ("tactical", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("endgame", "8/5pk1/6p1/8/8/1R6/5PPP/6K1 w - - 0 1"),
]
CLOCKS = [120_000, 60_000, 20_000, 5_000, 2_000, 800, 300, 120]

worst = 0.0
failures = []
print(f"{'position':<10} {'left':>8} {'spent':>8} {'share':>7}")
for name, fen in POSITIONS:
    for left in CLOCKS:
        began = time.monotonic()
        move = agent.get_move(fen, left)
        spent = (time.monotonic() - began) * 1000.0
        share = spent / left
        worst = max(worst, share)
        legal = chess.Move.from_uci(move) in chess.Board(fen).legal_moves
        flag = ""
        if not legal:
            flag = "  ILLEGAL"
            failures.append(f"{name} at {left}ms returned {move}, not legal")
        if share > SAFE_SHARE:
            flag = "  OVER"
            failures.append(f"{name} at {left}ms spent {spent:.0f}ms ({share:.0%})")
        print(f"{name:<10} {left:>8} {spent:>7.0f}ms {share:>6.0%}{flag}")

print()
print(f"worst share of the remaining clock spent on one move: {worst:.0%}")
if failures:
    for line in failures:
        print(f"  FAIL: {line}")
    raise SystemExit(f"{len(failures)} clock failures")
print("clock clean")
