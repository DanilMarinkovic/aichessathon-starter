"""What does a node actually spend its time on?

tools/nodecost.py says a node costs about 650ns at 512 wide, and tools/evalcost.py says the
network accounts for roughly 240ns of that. Nobody has ever measured the other 410ns, which is
why every speed change this week has been aimed at the evaluation -- the only part we had
instrumented. This times the pieces of the search machinery on the same positions, in a jitted
driver, so the next optimisation can be aimed at whatever is actually large.

Each component is timed on real positions reached during a search, not on the root: move
generation at the root of a quiet position is not what a node does on average.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
from numba import int64, njit

import searcher
from movegen import MAX_MOVES, generate
from position import NFIELDS, from_board, in_check, make_move
from searcher import pick_move, score_move

POSITIONS = (
    "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8",
    "1r2r1k1/5pp1/p2q3p/bpp5/2P1P3/P2PNN1b/2Q2P2/R1BR2K1 w - - 0 27",
    "r3k2r/pp1n1ppp/2pbpn2/q7/2PP4/2N1PN2/PP2BPPP/R2Q1RK1 w kq - 0 10",
    "2rq1rk1/pb2bppp/1p2pn2/8/2BN4/2N1P3/PP3PPP/2RQ1RK1 w - - 0 15",
    "8/2p2pk1/1p1p2p1/p2Pn2p/P1P1P2P/1P3PP1/4N1K1/8 w - - 0 30",
)


@njit(int64(int64, searcher.uint64[:, ::1], searcher.int32[:, ::1]), nogil=True, cache=False)
def bench_generate(rounds, states, moves):
    total = 0
    for _ in range(rounds):
        total += generate(states[0], moves[0], 0)
    return total


@njit(int64(int64, searcher.uint64[:, ::1], searcher.int32[:, ::1]), nogil=True, cache=False)
def bench_make(rounds, states, moves):
    total = 0
    count = generate(states[0], moves[0], 0)
    for i in range(rounds):
        total += make_move(states[0], moves[0, i % count], states[1])
    return total


@njit(int64(int64, searcher.uint64[:, ::1]), nogil=True, cache=False)
def bench_incheck(rounds, states):
    total = 0
    for _ in range(rounds):
        total += in_check(states[0])
    return total


@njit(
    int64(int64, searcher.uint64[:, ::1], searcher.int32[:, ::1], searcher.int64[::1]),
    nogil=True,
    cache=False,
)
def bench_order(rounds, states, moves, control):
    """Score every move and pick the best, which is what a node does before its first search."""
    total = 0
    count = generate(states[0], moves[0], 0)
    order = np.zeros((2, MAX_MOVES), dtype=np.int32)
    for _ in range(rounds):
        for index in range(count):
            order[0, index] = np.int32(
                score_move(states[0], moves[0, index], np.int32(0), control)
            )
        total += pick_move(moves, order, 0, 0, count)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=200_000)
    arguments = parser.parse_args()

    states = np.zeros((4, NFIELDS), dtype=np.uint64)
    moves = np.zeros((4, MAX_MOVES), dtype=np.int32)
    control = searcher.CONTROL

    print(f"{'position':>12} {'moves':>6} {'generate':>9} {'make':>7} {'in_check':>9} "
          f"{'order+pick':>11}")
    totals = np.zeros(4)
    for fen in POSITIONS:
        states[0] = from_board(chess.Board(fen))
        count = generate(states[0], moves[0], 0)
        row = []
        for fn, args in (
            (bench_generate, (states, moves)),
            (bench_make, (states, moves)),
            (bench_incheck, (states,)),
            (bench_order, (states, moves, control)),
        ):
            fn(1000, *args)
            best = min(_time(fn, arguments.rounds, args) for _ in range(3))
            row.append(best)
        totals += np.array(row)
        print(f"{fen.split()[0][:12]:>12} {count:>6} {row[0]:>8.1f}ns {row[1]:>6.1f}ns "
              f"{row[2]:>8.1f}ns {row[3]:>10.1f}ns")

    mean = totals / len(POSITIONS)
    print(f"\n{'mean':>12} {'':>6} {mean[0]:>8.1f}ns {mean[1]:>6.1f}ns {mean[2]:>8.1f}ns "
          f"{mean[3]:>10.1f}ns")
    print("\nA node does one generate, one in_check, one order pass, and one make per move it")
    print("actually searches -- so `make` is multiplied by the moves tried, the rest are once.")


def _time(fn, rounds: int, args) -> float:
    start = time.perf_counter_ns()
    fn(rounds, *args)
    return (time.perf_counter_ns() - start) / rounds


if __name__ == "__main__":
    main()
