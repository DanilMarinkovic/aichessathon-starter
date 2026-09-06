"""How much memory the agent actually holds, against the platform's 2GB.

Running out of memory loses the game, and the transposition table is the one structure sized
by a constant rather than by the position. This imports the agent exactly as the platform does
and reports the peak resident set after a real search, so a change to TT_BITS is checked
against the limit rather than against arithmetic done in a comment.
"""

import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIMIT_MB = 2048

import agent  # noqa: E402

agent.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 120_000)
agent.get_move("8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1", 120_000)

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"peak resident {peak_mb:.0f} MB of the platform's {LIMIT_MB} MB")
if peak_mb > LIMIT_MB * 0.75:
    raise SystemExit(f"FAIL: {peak_mb:.0f} MB leaves too little headroom under {LIMIT_MB} MB")
print("memory clean")
