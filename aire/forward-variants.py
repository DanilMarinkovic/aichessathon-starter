"""Which formulation of the output layer does LLVM vectorise best?

numba exposes no intrinsics, so we cannot ask for vpdpbusd or vpmaddwd the way Stockfish does.
What we can do is write the loop several ways, measure each, and look at what came out. The
clamp is a saturating pack in hand-written SIMD (packus does min(max(v,0),255) for free); the
dot product wants int16 by int16 into int32. Whether LLVM finds either depends on how the loop
is written, and the only honest way to find out is to compile all of them and look.

Timed from inside a jitted driver, never from Python: dispatch alone costs about 230ns, several
times the thing being measured.
"""

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numba import int16, int32, int64, njit

import nnue
from nnue import HIDDEN, QA, QB

ZERO16 = np.int16(0)
QA16 = np.int16(QA)
OUTPUT = nnue.OUTPUT
OUTPUT_BIAS = nnue.OUTPUT_BIAS
EVAL_SCALE = nnue.EVAL_SCALE


@njit(int32(int16[:, ::1], int64, int64), nogil=True, cache=False)
def current(accumulator, side_to_move, bucket):
    """What ships now: branch-free clamp, int32 multiply, one running sum."""
    total = np.int32(0)
    weights = OUTPUT[bucket]
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        for j in range(HIDDEN):
            clipped = min(max(accumulator[side, j], ZERO16), QA16)
            total += np.int32(clipped) * np.int32(weights[base + j])
    total += OUTPUT_BIAS[bucket]
    return np.int32(total * EVAL_SCALE // (QA * QB))


@njit(int32(int16[:, ::1], int64, int64), nogil=True, cache=False)
def int16_product(accumulator, side_to_move, bucket):
    """Multiply in int16 and widen only the sum.

    The products cannot overflow: clipped is at most QA (255) and the output weights measured
    [-127, 127], so the largest product is 32,385 against int16's 32,767. That is the shape
    vpmaddwd wants -- int16 by int16, accumulated wider -- so it is the formulation with any
    chance of reaching it.
    """
    total = np.int32(0)
    weights = OUTPUT[bucket]
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        for j in range(HIDDEN):
            clipped = min(max(accumulator[side, j], ZERO16), QA16)
            total += np.int32(clipped * weights[base + j])
    total += OUTPUT_BIAS[bucket]
    return np.int32(total * EVAL_SCALE // (QA * QB))


@njit(int32(int16[:, ::1], int64, int64), nogil=True, cache=False)
def four_sums(accumulator, side_to_move, bucket):
    """Four independent accumulators, in case LLVM will not vectorise the reduction itself."""
    a = np.int32(0)
    b = np.int32(0)
    c = np.int32(0)
    d = np.int32(0)
    weights = OUTPUT[bucket]
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        for j in range(0, HIDDEN, 4):
            a += np.int32(min(max(accumulator[side, j], ZERO16), QA16)) * np.int32(
                weights[base + j])
            b += np.int32(min(max(accumulator[side, j + 1], ZERO16), QA16)) * np.int32(
                weights[base + j + 1])
            c += np.int32(min(max(accumulator[side, j + 2], ZERO16), QA16)) * np.int32(
                weights[base + j + 2])
            d += np.int32(min(max(accumulator[side, j + 3], ZERO16), QA16)) * np.int32(
                weights[base + j + 3])
    total = a + b + c + d + OUTPUT_BIAS[bucket]
    return np.int32(total * EVAL_SCALE // (QA * QB))


@njit(int32(int16[:, ::1], int64, int64), nogil=True, cache=False)
def slices(accumulator, side_to_move, bucket):
    """Whole-row numpy expressions, which numba lowers to its own vectorised kernels.

    Allocates two temporaries per call, which is the thing to watch: a heap allocation costs
    tens of nanoseconds and the arithmetic here is only a hundred or so.
    """
    weights = OUTPUT[bucket]
    total = np.int32(0)
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        clipped = np.minimum(np.maximum(accumulator[side], ZERO16), QA16).astype(np.int32)
        total += np.int32((clipped * weights[base:base + HIDDEN].astype(np.int32)).sum())
    total += OUTPUT_BIAS[bucket]
    return np.int32(total * EVAL_SCALE // (QA * QB))


@njit(int64(int64, int16[:, ::1], int64, int64), nogil=True, cache=False)
def drive(rounds, accumulator, side, buckets):
    total = 0
    for i in range(rounds):
        total += current(accumulator, (side + i) & 1, i % buckets)
    return total


def main() -> None:
    values, boards = nnue.new_cache()
    accumulator = nnue.new_accumulator()
    import chess

    from position import from_board
    state = np.ascontiguousarray(from_board(chess.Board(
        "r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 b kq - 0 8")))
    nnue.refresh(state, accumulator, values, boards)
    buckets = int(OUTPUT.shape[0])

    print(f"{HIDDEN} hidden, {buckets} output buckets, weights in "
          f"[{OUTPUT.min()}, {OUTPUT.max()}]")
    reference = None
    for name, fn in (("current", current), ("int16_product", int16_product),
                     ("four_sums", four_sums), ("slices", slices)):
        value = fn(accumulator, 0, 0)
        if reference is None:
            reference = value
        agree = "same" if value == reference else f"DIFFERS ({value} vs {reference})"

        loop = _make_loop(fn)
        loop(1000, accumulator, 0, buckets)
        best = min(_time(loop, accumulator, buckets) for _ in range(3))
        asm = "\n".join(fn.inspect_asm().values())
        width = Counter()
        for line in asm.splitlines():
            for reg, label in (("%zmm", "zmm"), ("%ymm", "ymm"), ("%xmm", "xmm")):
                if reg in line:
                    width[label] += 1
                    break
        packed = sum(1 for line in asm.splitlines()
                     if any(op in line for op in ("vpmaddwd", "vpdpbusd", "vpackuswb")))
        print(f"  {name:<14} {best:6.1f}ns  {agree:<8} {dict(width)}  key instructions {packed}")


def _make_loop(fn):
    """A jitted driver closing over one variant. A default argument would change the
    signature numba is given, which it rejects; a closure does not."""

    @njit(int64(int64, int16[:, ::1], int64, int64), nogil=True, cache=False)
    def loop(rounds, acc, side, nbuckets):
        total = 0
        for i in range(rounds):
            total += fn(acc, (side + i) & 1, i % nbuckets)
        return total

    return loop


def _time(loop, accumulator, buckets, rounds: int = 200_000) -> float:
    start = time.perf_counter_ns()
    loop(rounds, accumulator, 0, buckets)
    return (time.perf_counter_ns() - start) / rounds


if __name__ == "__main__":
    main()
