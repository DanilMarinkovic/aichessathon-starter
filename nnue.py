"""A small perspective network evaluated incrementally: 768 -> 256 -> 1.

The input is one feature per (piece, square), 12 x 64 = 768, built twice: once from White's
point of view and once from Black's, with the board mirrored and the colours swapped. The
search reads the side to move's half first, so the network always sees the position from the
perspective of the player about to move.

Why this shape. A wider second layer is what makes a network expensive: a 32 wide layer over
512 inputs is 8192 multiply-accumulates per leaf, which measured eight times the cost of the
hand written evaluation it would replace. One hidden layer with a linear output is 512, and the
clipped ReLU on the accumulator still makes it non-linear. It measures about the same as the
evaluation it replaces.

Why the accumulator is copied per ply rather than applied and undone. A feature delta is exactly
invertible, so the search could apply it before recursing and subtract it afterwards, keeping
one accumulator. Measured, that is slower: undoing pays the delta a second time on the way out,
576ns against 457ns for copying the previous ply and applying the delta once. So the search
keeps a ply-indexed stack, mirroring what it already does with positions.

Why deltas are computed by diffing bitboards. Castling moves two pieces, en passant removes a
pawn from a square the moving piece never occupies, and promotion changes a piece's type. Each
is a special case, and a wrong one silently corrupts the evaluation rather than crashing.
Comparing the twelve piece bitboards before and after covers all of them with no case analysis:
whatever bits changed are exactly the features that changed.
"""

import numpy as np
from numba import int16, int32, int64, njit, uint64

from bitboards import U0, U1, lsb

HIDDEN = 256
INPUTS = 768

# Quantisation. These three numbers are a contract with the trainer: it scales the accumulator
# weights and biases by QA, the output weights by QB, and clips activations to QA. Inference
# divides both back out. If the trainer and this file ever disagree, the network still runs and
# simply plays worse, which is the failure mode worth being paranoid about.
QA = 255
QB = 64
SCALE = 400

# Placeholder weights for the speed and correctness gate. Training replaces these; the shapes
# and dtypes are the contract between this file and whatever produces them.
_rng = np.random.default_rng(0xC0FFEE)
WEIGHTS = np.ascontiguousarray(_rng.integers(-32, 32, size=(INPUTS, HIDDEN)).astype(np.int16))
BIASES = np.ascontiguousarray(_rng.integers(-32, 32, size=HIDDEN).astype(np.int16))
OUTPUT = np.ascontiguousarray(_rng.integers(-32, 32, size=2 * HIDDEN).astype(np.int16))
OUTPUT_BIAS = np.int32(0)


@njit(int64(int16[:, ::1], int64, int64, int64), nogil=True, cache=False, inline="always")
def _feature(accumulator: np.ndarray, piece: np.int64, square: np.int64, sign: np.int64):
    """Add or subtract one piece-on-square feature from both perspectives.

    White's index is the piece and square as they are. Black's flips the square vertically and
    swaps the piece's colour. With the 0-5 white, 6-11 black ordering the position uses, that
    swap is `(piece + 6) % 12`. It is emphatically not `piece ^ 6`, which maps a white bishop
    to a white queen and sends the black pieces past the end of the table, where numba will
    read out of bounds without complaint.
    """
    white = piece * 64 + square
    black = (piece + 6 if piece < 6 else piece - 6) * 64 + (square ^ 56)
    if sign > 0:
        for j in range(HIDDEN):
            accumulator[0, j] += WEIGHTS[white, j]
            accumulator[1, j] += WEIGHTS[black, j]
    else:
        for j in range(HIDDEN):
            accumulator[0, j] -= WEIGHTS[white, j]
            accumulator[1, j] -= WEIGHTS[black, j]
    return 0


@njit(int64(uint64[::1], int16[:, ::1]), nogil=True, cache=False)
def refresh(state: np.ndarray, accumulator: np.ndarray) -> np.int64:
    """Rebuild both perspectives from scratch. Used at the root and to check the deltas."""
    for side in range(2):
        for j in range(HIDDEN):
            accumulator[side, j] = BIASES[j]
    for piece in range(12):
        board = state[piece]
        while board != U0:
            square = lsb(board)
            board &= board - U1
            _feature(accumulator, piece, square, 1)
    return 0


@njit(int64(uint64[::1], uint64[::1], int16[:, ::1], int64), nogil=True, cache=False)
def apply(before: np.ndarray, after: np.ndarray, accumulator: np.ndarray, sign: np.int64):
    """Move the accumulator across one move, or back again when sign is negative."""
    for piece in range(12):
        changed = before[piece] ^ after[piece]
        if changed == U0:
            continue
        left = changed & before[piece]
        while left != U0:
            square = lsb(left)
            left &= left - U1
            _feature(accumulator, piece, square, -sign)
        arrived = changed & after[piece]
        while arrived != U0:
            square = lsb(arrived)
            arrived &= arrived - U1
            _feature(accumulator, piece, square, sign)
    return 0


@njit(int64(uint64[::1], uint64[::1], int16[:, ::1], int16[:, ::1]), nogil=True, cache=False)
def advance(
    before: np.ndarray, after: np.ndarray, source: np.ndarray, destination: np.ndarray
) -> np.int64:
    """Write the next ply's accumulator from this one, across a single move."""
    for side in range(2):
        for j in range(HIDDEN):
            destination[side, j] = source[side, j]
    apply(before, after, destination, 1)
    return 0


@njit(int32(int16[:, ::1], int64), nogil=True, cache=False)
def forward(accumulator: np.ndarray, side_to_move: np.int64) -> np.int32:
    """Clipped ReLU and the output dot product in one pass, in centipawns.

    The hidden vector is never materialised: clipping and multiplying happen together, and four
    running sums keep the additions independent of each other.
    """
    first = np.int32(0)
    second = np.int32(0)
    third = np.int32(0)
    fourth = np.int32(0)
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        for j in range(0, HIDDEN, 4):
            a = accumulator[side, j]
            b = accumulator[side, j + 1]
            c = accumulator[side, j + 2]
            d = accumulator[side, j + 3]
            if a < 0:
                a = 0
            elif a > QA:
                a = QA
            if b < 0:
                b = 0
            elif b > QA:
                b = QA
            if c < 0:
                c = 0
            elif c > QA:
                c = QA
            if d < 0:
                d = 0
            elif d > QA:
                d = QA
            first += np.int32(a) * np.int32(OUTPUT[base + j])
            second += np.int32(b) * np.int32(OUTPUT[base + j + 1])
            third += np.int32(c) * np.int32(OUTPUT[base + j + 2])
            fourth += np.int32(d) * np.int32(OUTPUT[base + j + 3])
    total = first + second + third + fourth + OUTPUT_BIAS
    return np.int32(total * SCALE // (QA * QB))


def new_accumulator() -> np.ndarray:
    return np.zeros((2, HIDDEN), dtype=np.int16)
