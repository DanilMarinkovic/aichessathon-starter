"""Where does a node's evaluation time actually go?

Doubling the hidden layer costs about 407ns a node (476ns at 128 wide, 883ns at 256), which
says the accumulator update and the forward pass together are most of a node. That is a lot,
and it is worth knowing which of the two it is before trying to make either faster.

Timed from inside a jitted driver, never from Python. Calling a numba function from the
interpreter costs roughly 230ns of dispatch, which is several times the thing being measured
and would make every variant look identical -- the same mistake that had tools/nodecost.py
reporting five network shapes as the same speed.

    uv run python tools/evalcost.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
from numba import int64, njit

import nnue
from nnue import HIDDEN, advance, forward, new_accumulator, new_cache, output_bucket, refresh
from position import OCC_ALL, from_board

FEN = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8"


@njit(int64(int64, nnue.int16[:, ::1], int64, int64), nogil=True, cache=False)
def bench_forward(rounds: int, accumulator: np.ndarray, side: int, buckets: int) -> int:
    """Sum of `rounds` forward passes. The sum is returned so nothing can be optimised away,
    and the bucket cycles so the result cannot be computed once and reused."""
    total = 0
    for i in range(rounds):
        total += forward(accumulator, (side + i) & 1, i % buckets)
    return total


@njit(
    int64(
        int64,
        nnue.uint64[::1],
        nnue.uint64[::1],
        nnue.int16[:, ::1],
        nnue.int16[:, ::1],
        nnue.int16[:, ::1],
        nnue.uint64[:, ::1],
    ),
    nogil=True,
    cache=False,
)
def bench_advance(
    rounds: int,
    before: np.ndarray,
    after: np.ndarray,
    one: np.ndarray,
    two: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
) -> int:
    """`rounds` accumulator updates, alternating direction so each one has real work to do."""
    total = 0
    for i in range(rounds):
        if i & 1:
            total += advance(before, after, one, two, cache_values, cache_boards)
        else:
            total += advance(after, before, two, one, cache_values, cache_boards)
    return total


def main() -> None:
    board = chess.Board(FEN)
    move = next(m for m in board.legal_moves if not board.is_capture(m))
    after_board = board.copy()
    after_board.push(move)

    before = np.ascontiguousarray(from_board(board))
    after = np.ascontiguousarray(from_board(after_board))
    values, boards = new_cache()
    one, two = new_accumulator(), new_accumulator()
    refresh(before, one, values, boards)
    refresh(after, two, values, boards)
    bucket_count = int(nnue.OUTPUT.shape[0])

    print(f"{HIDDEN} hidden, {nnue.BUCKETS} king buckets, {bucket_count} output buckets")
    rounds = 200_000
    timings = {}
    for name, run in (
        ("forward", lambda n: bench_forward(n, one, 0, bucket_count)),
        ("advance", lambda n: bench_advance(n, before, after, one, two, values, boards)),
    ):
        run(1000)  # compile, and warm the caches the real search would have warm
        # Three passes, fastest kept. A slow pass is the machine doing something else.
        best = min(_time(run, rounds) for _ in range(3))
        timings[name] = best
        print(f"  {name:<8} {best:6.1f}ns a call")

    # 3.3% of accumulator updates are never used, so a node is very nearly one of each.
    print(f"  {'node':<8} {timings['forward'] + timings['advance']:6.1f}ns of evaluation")
    print(f"  output bucket of the start position: {output_bucket(before[OCC_ALL])}")


def _time(run, rounds: int) -> float:
    start = time.perf_counter_ns()
    run(rounds)
    return (time.perf_counter_ns() - start) / rounds


if __name__ == "__main__":
    main()
