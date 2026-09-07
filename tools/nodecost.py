"""What does a node cost, and what actually makes it cost that?

The architecture was frozen at 128 hidden and four king buckets on the strength of a cost model
that said the accumulator matrix has to stay resident in L2 -- BUCKETS x 768 x HIDDEN x 2 bytes
under about a megabyte -- and that 256 hidden "spills". That model and the row-width model both
explain the only two measurements we had (484ns a node at 128 wide, 930ns at 256), because
width moves the matrix and the row together.

They disagree about buckets. BUCKETS multiplies the matrix and leaves the row alone, so:

    cache residency  ->  five buckets is five times the matrix and should cost like a wide net
    row width        ->  five buckets is the same 256-byte row and should cost nothing

This runs a real search at a fixed node count over a spread of positions and reports
nanoseconds per node, so the two can be told apart on a number rather than an argument. Weights
are random unless a trained network is installed: node cost does not depend on what the weights
say, only on how many of them each update touches.

    uv run python tools/nodecost.py --nodes 400000

One process per network, because nnue.py loads weights/net.npz at import and numba compiles the
arrays in as constants. Use tools/nodecost.sh to sweep shapes.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import nnue
import searcher
from position import HASH, from_board
from searcher import NODES, STOP, USE_NNUE
from tools.engine import use_network

# A spread of middlegames and endgames rather than one position, so the answer is not an
# artefact of one branching factor. Kept inline: the point is to compare shapes against each
# other, and a fixed list makes two runs comparable without a data file to keep in step.
POSITIONS = (
    "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8",
    "1r2r1k1/5pp1/p2q3p/bpp5/2P1P3/P2PNN1b/2Q2P2/R1BR2K1 w - - 0 27",
    "r3k2r/pp1n1ppp/2pbpn2/q7/2PP4/2N1PN2/PP2BPPP/R2Q1RK1 w kq - 0 10",
    "8/2p2pk1/1p1p2p1/p2Pn2p/P1P1P2P/1P3PP1/4N1K1/8 w - - 0 30",
    "2rq1rk1/pb2bppp/1p2pn2/8/2BN4/2N1P3/PP3PPP/2RQ1RK1 w - - 0 15",
    "8/8/4kp2/3p1p2/p2P1P2/P4K2/8/8 w - - 0 50",
)


def cost(fen: str, nodes: int) -> tuple[float, int]:
    """Seconds and nodes for one fixed-node search."""
    # reset() deliberately preserves which evaluation is in force, so the caller sets it once.
    # An earlier version set it here, which quietly overrode the control below and made the
    # tool unable to fail its own check.
    searcher.reset()
    state = from_board(chess.Board(fen))
    searcher.STATES[0] = state
    searcher.PATH[:] = 0
    searcher.PATH[0] = np.uint64(state[HASH])
    searcher.CONTROL[STOP] = 0
    start = time.perf_counter()
    searcher.run(64, 0, nodes)
    return time.perf_counter() - start, int(searcher.CONTROL[NODES])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=400_000)
    parser.add_argument("--repeats", type=int, default=3, help="passes over the position list")
    parser.add_argument("--fen", action="append", help="measure these instead of the built-ins")
    parser.add_argument("--each", action="store_true", help="print every position separately")
    arguments = parser.parse_args()
    positions = tuple(arguments.fen) if arguments.fen else POSITIONS

    # Compilation is not part of the measurement. agent.py pays it in the import budget; here it
    # would otherwise land entirely on the first position and make it look several times slower.
    cost(POSITIONS[0], 20_000)

    on = use_network()
    matrix = nnue.BUCKETS * 768 * nnue.HIDDEN * 2
    row = nnue.HIDDEN * 2
    print(f"hidden {nnue.HIDDEN}, {nnue.BUCKETS} king buckets")
    print(f"  matrix {matrix / 1024:.0f}KB, row {row} bytes, "
          f"{nnue.BUCKETS * 768 * nnue.HIDDEN / 1000:.0f}k first-layer weights")

    # Per position, not pooled. A search that ends early -- a mate found, the depth ceiling hit
    # in a simple endgame -- contributes few nodes and little time, and pooling lets those
    # positions set the average. The expensive middlegame is the one that decides a game, and
    # pooling was hiding it: this is why an earlier version of this measurement reported every
    # network shape as costing the same.
    best: dict[str, float] = {}
    reached: dict[str, int] = {}
    for _ in range(arguments.repeats):
        for fen in positions:
            elapsed, searched = cost(fen, arguments.nodes)
            if searched == 0:
                continue
            rate = elapsed / searched
            if fen not in best or rate < best[fen]:
                best[fen] = rate
                reached[fen] = searched

    for fen in positions:
        if arguments.each and fen in best:
            print(f"  {best[fen] * 1e9:6.0f}ns  {reached[fen]:>9,} nodes  {fen}")
    pooled = sum(best.values()) / len(best)
    print(f"  {pooled * 1e9:.0f}ns a node, {1 / pooled / 1e6:.2f}M nodes/s  "
          f"(mean over {len(best)} positions)")

    # The instrument checks itself before its answer is used.
    #
    # This tool's whole job is to report what an evaluation costs, and its first version
    # answered that question for five network shapes without any of them being switched on. The
    # failure was silent because a number came out and the number looked plausible. So: measure
    # the same search again with the hand-crafted evaluation, which is known to cost something
    # different, and refuse to be believed if the two agree. A tool that cannot tell two
    # evaluations apart is not measuring the evaluation.
    if not on:
        return
    searcher.CONTROL[USE_NNUE] = 0
    rates = []
    for fen in positions:
        elapsed, searched = cost(fen, arguments.nodes)
        if searched:
            rates.append(elapsed / searched)
    searcher.CONTROL[USE_NNUE] = 1
    hand = sum(rates) / len(rates)
    shift = abs(pooled - hand) / pooled
    verdict = "ok" if shift > 0.05 else "SUSPECT: the two evaluations measure the same"
    print(f"  control: hand-crafted evaluation {hand * 1e9:.0f}ns, "
          f"{shift * 100:.0f}% apart -- {verdict}")


if __name__ == "__main__":
    main()
