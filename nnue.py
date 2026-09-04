"""A king-bucketed perspective network, evaluated incrementally.

The input is one feature per (piece, square), 12 x 64 = 768, built twice: once from White's
point of view and once from Black's, with the board mirrored vertically and the colours swapped.
The search reads the side to move's half first, so the network always sees the position from the
perspective of the player about to move.

King buckets. The same structure means something different depending on where your king is, and
a single set of weights has to average over both cases. So each perspective selects a bank of
weights from its own king's square, which is the affordable half of the idea Nasu's HalfKP gets
its strength from. The board is mirrored horizontally so the king always sits queenside, halving
the king positions the network must learn about.

Crossing a bucket boundary moves every feature index for that perspective at once, so nothing
carries forward and the accumulator must be rebuilt. The standard remedy, implemented below, is
an accumulator refresh table: each bucket keeps a cached accumulator alongside the bitboards it
was built from, so a crossing costs the difference between two positions rather than a rebuild
from an empty board. Finn Eggers introduced it for Koivisto; Stockfish carries it as
AccumulatorCaches.

It is worth much less here than it is there: the refresh table took a node from 936ns only to
930ns. Not because the updates are slow. numba vectorises them properly, and the inner loop of
_apply compiles to vpaddw on ymm registers, sixteen int16 lanes at a time.

What governs the cost is cache residency. Every feature update reads one row of WEIGHTS, so the
matrix wants to fit in L2:

    BUCKETS x 768 x HIDDEN x 2 bytes  <=  about 1 MB

At HIDDEN 256 with four buckets that is 1.5MB, it spills to L3, and a node costs 930ns. At
HIDDEN 128 with four buckets it is 768KB, it fits, and the same parameter count costs 484ns.
Hence this shape. It also means the bucket layout is free to be chosen for what the network can
learn: a layout a castled king almost never leaves measured the same as one it crosses
constantly, because crossings were never the expense.

Every figure quoted here is nodes per second from a real search. Timing these functions by
calling them from Python does not work: the dispatch alone costs about 230ns, which swamps a
forward pass that actually takes 18ns, and makes everything look the same speed.

BUCKETS = 1 disables bucketing and leaves a plain 768 network. That is not a fallback but a
control: identical code path, so the two can be trained and measured against each other.

Two further decisions, both measured. The accumulator is copied per ply rather than applied and
undone, because undoing pays the delta a second time: 576ns against 457ns. And deltas come from
diffing the twelve piece bitboards rather than from case analysis on the move, because castling
moves two pieces, en passant removes a pawn from a square the mover never occupies, and
promotion changes a piece's type. Each is a special case, and a wrong one corrupts the
evaluation silently rather than crashing.
"""

from pathlib import Path

import numpy as np
from numba import int16, int32, int64, njit, types, uint64

from bitboards import KING, U0, U1, lsb

INPUTS = 768

# The trained network, if one has been built. Everything about its shape comes out of the file
# rather than being repeated here, so the trainer and the engine cannot disagree about the
# architecture. Loading happens before the jitted functions below are defined, because numba
# compiles a global array in as a constant when it compiles the function that reads it, and
# replacing the array afterwards would leave the compiled code reading the old one.
WEIGHTS_FILE = Path(__file__).resolve().parent / "weights" / "net.npz"


def _load() -> dict[str, object]:
    if WEIGHTS_FILE.exists():
        stored = np.load(WEIGHTS_FILE)
        return {
            "hidden": int(stored["hidden"]),
            "buckets": int(stored["buckets"]),
            "qa": int(stored["qa"]),
            "qb": int(stored["qb"]),
            "scale": int(stored["scale"]),
            "weights": np.ascontiguousarray(stored["weights"].astype(np.int16)),
            "biases": np.ascontiguousarray(stored["biases"].astype(np.int16)),
            "output": np.ascontiguousarray(stored["output"].astype(np.int16)),
            "output_bias": np.int32(stored["output_bias"]),
            "trained": True,
        }
    # No network yet. Random weights of the right shape, so every path still compiles and can
    # be timed and tested; the engine keeps using the hand-written evaluation until a real one
    # is loaded, which searcher.CONTROL[USE_NNUE] decides.
    hidden, buckets = 128, 4
    rng = np.random.default_rng(0xC0FFEE)
    return {
        "hidden": hidden,
        "buckets": buckets,
        "qa": 255,
        "qb": 64,
        "scale": 400,
        "weights": np.ascontiguousarray(
            rng.integers(-32, 32, size=(buckets * INPUTS, hidden)).astype(np.int16)
        ),
        "biases": np.ascontiguousarray(rng.integers(-32, 32, size=hidden).astype(np.int16)),
        "output": np.ascontiguousarray(rng.integers(-32, 32, size=2 * hidden).astype(np.int16)),
        "output_bias": np.int32(0),
        "trained": False,
    }


_NET = _load()

HIDDEN = int(_NET["hidden"])
BUCKETS = int(_NET["buckets"])

# Quantisation. These three numbers are a contract with the trainer: it scales the accumulator
# weights and biases by QA, the output weights by QB, and clips activations to QA. Inference
# divides both back out.
QA = int(_NET["qa"])
QB = int(_NET["qb"])
SCALE = int(_NET["scale"])

TRAINED = bool(_NET["trained"])

# One cache entry per perspective, bucket and mirror state.
CACHE_SLOTS = 2 * BUCKETS * 2


def _king_buckets(count: int) -> np.ndarray:
    """Which weight bank a king square selects, after the board is mirrored queenside.

    Corners against centre on the home ranks, which is the distinction that matters most for a
    castled king, then coarser as the king advances. Modelled on the layout Leorik uses.

    Chosen for what the network can learn, not for how rarely a king crosses a boundary.
    Measurement showed crossing frequency does not matter here: a rank-only layout, which a
    castled king almost never leaves, cost 930ns a node against 936ns for this one. What
    matters is the size of the weight matrix, which has to stay inside L2.
    """
    table = np.zeros(64, dtype=np.int64)
    if count == 1:
        return table
    for square in range(64):
        rank, file = divmod(square, 8)
        corner = 0 if file < 2 else 1
        if rank < 2:
            bucket = corner
        elif rank < 4:
            bucket = 2 + corner
        else:
            bucket = 4
        table[square] = min(bucket, count - 1)
    return table


KING_BUCKET = _king_buckets(BUCKETS)

WEIGHTS = _NET["weights"]
BIASES = _NET["biases"]
OUTPUT = _NET["output"]
OUTPUT_BIAS = _NET["output_bias"]


@njit(types.UniTuple(int64, 2)(int64, int64), nogil=True, cache=False, inline="always")
def perspective(king_square: np.int64, flip: np.int64):
    """Where one perspective's weights start, and whether its board is mirrored.

    `flip` is 0 for White and 56 for Black, which puts the square into that perspective's own
    frame before the king is bucketed.
    """
    king = king_square ^ flip
    mirror = 7 if (king & 7) >= 4 else 0
    return KING_BUCKET[king ^ mirror] * INPUTS, mirror


@njit(int64(int64, int64, int64, int64, int64), nogil=True, cache=False, inline="always")
def _index(
    side: np.int64, piece: np.int64, square: np.int64, offset: np.int64, mirror: np.int64
) -> np.int64:
    """The feature index for one piece on one square, from one perspective.

    Black's view flips the square vertically and swaps the piece's colour. With the 0-5 white,
    6-11 black ordering the position uses, that swap is `(piece + 6) % 12`. It is emphatically
    not `piece ^ 6`, which maps a white bishop to a white queen and sends the black pieces past
    the end of the table, where numba reads out of bounds without complaint.
    """
    if side == 0:
        return offset + piece * 64 + (square ^ mirror)
    other = piece + 6 if piece < 6 else piece - 6
    return offset + other * 64 + (square ^ 56 ^ mirror)


@njit(int64(int16[:, ::1], int64, int64, int64), nogil=True, cache=False, inline="always")
def _apply(
    accumulator: np.ndarray, side: np.int64, index: np.int64, sign: np.int64
) -> np.int64:
    if sign > 0:
        for j in range(HIDDEN):
            accumulator[side, j] += WEIGHTS[index, j]
    else:
        for j in range(HIDDEN):
            accumulator[side, j] -= WEIGHTS[index, j]
    return 0


@njit(int64(int64, int64, int64), nogil=True, cache=False, inline="always")
def _slot(side: np.int64, offset: np.int64, mirror: np.int64) -> np.int64:
    return (side * BUCKETS + offset // INPUTS) * 2 + (1 if mirror != 0 else 0)


@njit(
    int64(uint64[::1], int16[:, ::1], int64, int64, int64, int16[:, ::1], uint64[:, ::1]),
    nogil=True,
    cache=False,
)
def refresh_side(
    state: np.ndarray,
    accumulator: np.ndarray,
    side: np.int64,
    offset: np.int64,
    mirror: np.int64,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
) -> np.int64:
    """Rebuild one perspective through the refresh table.

    Starts from whatever this bucket last held rather than from the biases, so the work is the
    difference between two positions that share a bucket instead of a full thirty-two piece
    insertion. The entry is then brought up to date for next time.
    """
    slot = _slot(side, offset, mirror)
    for j in range(HIDDEN):
        accumulator[side, j] = cache_values[slot, j]

    for piece in range(12):
        cached = cache_boards[slot, piece]
        current = state[piece]
        changed = cached ^ current
        if changed == U0:
            continue
        gone = changed & cached
        while gone != U0:
            square = lsb(gone)
            gone &= gone - U1
            _apply(accumulator, side, _index(side, piece, square, offset, mirror), -1)
        arrived = changed & current
        while arrived != U0:
            square = lsb(arrived)
            arrived &= arrived - U1
            _apply(accumulator, side, _index(side, piece, square, offset, mirror), 1)
        cache_boards[slot, piece] = current

    for j in range(HIDDEN):
        cache_values[slot, j] = accumulator[side, j]
    return 0


@njit(
    int64(uint64[::1], int16[:, ::1], int16[:, ::1], uint64[:, ::1]), nogil=True, cache=False
)
def refresh(
    state: np.ndarray,
    accumulator: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
) -> np.int64:
    """Rebuild both perspectives. Used at the root of every search."""
    white_offset, white_mirror = perspective(lsb(state[KING - 1]), 0)
    black_offset, black_mirror = perspective(lsb(state[6 + KING - 1]), 56)
    refresh_side(state, accumulator, 0, white_offset, white_mirror, cache_values, cache_boards)
    refresh_side(state, accumulator, 1, black_offset, black_mirror, cache_values, cache_boards)
    return 0


@njit(
    int64(uint64[::1], uint64[::1], int16[:, ::1], int16[:, ::1], int16[:, ::1], uint64[:, ::1]),
    nogil=True,
    cache=False,
)
def advance(
    before: np.ndarray,
    after: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
) -> np.int64:
    """Write the next ply's accumulator from this one, across a single move."""
    was_white = perspective(lsb(before[KING - 1]), 0)
    now_white = perspective(lsb(after[KING - 1]), 0)
    was_black = perspective(lsb(before[6 + KING - 1]), 56)
    now_black = perspective(lsb(after[6 + KING - 1]), 56)
    white_moved_bucket = was_white != now_white
    black_moved_bucket = was_black != now_black

    for side in range(2):
        for j in range(HIDDEN):
            destination[side, j] = source[side, j]

    # Only the perspectives whose indices are still valid can be carried forward by delta.
    if not (white_moved_bucket and black_moved_bucket):
        white_offset, white_mirror = now_white
        black_offset, black_mirror = now_black
        for piece in range(12):
            changed = before[piece] ^ after[piece]
            if changed == U0:
                continue
            gone = changed & before[piece]
            while gone != U0:
                square = lsb(gone)
                gone &= gone - U1
                if not white_moved_bucket:
                    _apply(destination, 0, _index(0, piece, square, white_offset, white_mirror), -1)
                if not black_moved_bucket:
                    _apply(destination, 1, _index(1, piece, square, black_offset, black_mirror), -1)
            arrived = changed & after[piece]
            while arrived != U0:
                square = lsb(arrived)
                arrived &= arrived - U1
                if not white_moved_bucket:
                    _apply(destination, 0, _index(0, piece, square, white_offset, white_mirror), 1)
                if not black_moved_bucket:
                    _apply(destination, 1, _index(1, piece, square, black_offset, black_mirror), 1)

    if white_moved_bucket:
        refresh_side(after, destination, 0, now_white[0], now_white[1], cache_values, cache_boards)
    if black_moved_bucket:
        refresh_side(after, destination, 1, now_black[0], now_black[1], cache_values, cache_boards)
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


@njit(int64(uint64[::1], int32[::1], int32[::1]), nogil=True, cache=False)
def features(state: np.ndarray, white: np.ndarray, black: np.ndarray) -> np.int64:
    """The active feature indices for both perspectives, and how many there are.

    The trainer calls this rather than reimplementing the indexing, because the one failure
    that cannot be caught by looking at either side alone is the trainer and the engine
    disagreeing about what a feature means. A network fitted to different indices than it is
    evaluated with still runs, and simply plays badly.
    """
    white_offset, white_mirror = perspective(lsb(state[KING - 1]), 0)
    black_offset, black_mirror = perspective(lsb(state[6 + KING - 1]), 56)
    count = 0
    for piece in range(12):
        board = state[piece]
        while board != U0:
            square = lsb(board)
            board &= board - U1
            white[count] = _index(0, piece, square, white_offset, white_mirror)
            black[count] = _index(1, piece, square, black_offset, black_mirror)
            count += 1
    return count


def new_accumulator() -> np.ndarray:
    return np.zeros((2, HIDDEN), dtype=np.int16)


def new_cache() -> tuple[np.ndarray, np.ndarray]:
    """An empty refresh table: every bucket holding the biases and an empty board."""
    values = np.ascontiguousarray(np.tile(BIASES, (CACHE_SLOTS, 1)))
    boards = np.zeros((CACHE_SLOTS, 12), dtype=np.uint64)
    return values, boards
