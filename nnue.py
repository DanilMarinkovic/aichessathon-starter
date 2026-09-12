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

Shape costs nothing here, over any range we would plausibly ship. tools/nodecost.py runs a real
search at a fixed node count on one core of a cluster node and reports:

    banks  hidden    matrix   first-layer weights   ns a node
        4     128     768KB                  393k         401
       32      32     1.5MB                  786k         369
        4     512     3.0MB                 1573k         388
       32     128     6.0MB                 3146k         370
       32     256    12.0MB                 6291k         392

Sixteen times the parameters, no cost. The reason is that the matrix is never the working set.
A feature update reads one row, HIDDEN int16s; within a search the king barely moves, so one or
two banks per perspective are ever selected, and inside a bank only the rows for pieces actually
on the board are touched. That hot set is about sixteen kilobytes and stays in L1 whether the
matrix behind it is 768KB or 12MB. A node is roughly a thousand cycles and the accumulator work
is a few dozen vector operations of it, so widening the layer moves a small percentage.

This note used to say the opposite: that BUCKETS x 768 x HIDDEN x 2 bytes had to stay inside L2
or the network "spills", and that 256 hidden therefore costs 930ns a node against 484ns at 128.
Those two numbers are from different experiments -- 484ns was this network against the 367ns
hand-crafted evaluation it replaced, and 930ns against 936ns was the accumulator refresh table.
Neither was ever a measurement of hidden width. Spliced into one sentence they froze the
architecture at 393k parameters, which is one to two orders of magnitude below a normal NNUE,
and that is the largest structural gap this engine has. Quote nodecost.py, not this paragraph,
and re-measure before believing any shape is unaffordable.

The bucket layout is still free to be chosen for what the network can learn: a layout a castled
king almost never leaves measured the same as one it crosses constantly, because crossings were
never the expense either.

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

import time
from pathlib import Path

import numpy as np
from numba import int16, int32, int64, njit, types, uint64

from bitboards import KING, U0, U1, lsb, popcount

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
            # Absent in networks trained before calibration existed, and they must keep
            # evaluating exactly as they did, so the fallback is the training scale.
            "eval_scale": int(stored["eval_scale"]) if "eval_scale" in stored else
            int(stored["scale"]),
            "weights": np.ascontiguousarray(stored["weights"].astype(np.int16)),
            "biases": np.ascontiguousarray(stored["biases"].astype(np.int16)),
            # A network trained before output buckets existed stores one output vector and a
            # scalar bias. Reshaping both to a leading axis of one makes it a one-bucket
            # network, which is the same arithmetic through the same code path -- so the
            # networks already measured keep evaluating exactly as they did.
            "output": np.ascontiguousarray(
                np.atleast_2d(stored["output"].astype(np.int16))
            ),
            "output_bias": np.ascontiguousarray(
                np.atleast_1d(stored["output_bias"]).astype(np.int32)
            ),
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
        "eval_scale": 400,
        "weights": np.ascontiguousarray(
            rng.integers(-32, 32, size=(buckets * INPUTS, hidden)).astype(np.int16)
        ),
        "biases": np.ascontiguousarray(rng.integers(-32, 32, size=hidden).astype(np.int16)),
        "output": np.ascontiguousarray(
            rng.integers(-32, 32, size=(1, 2 * hidden)).astype(np.int16)
        ),
        "output_bias": np.zeros(1, dtype=np.int32),
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

# The units the evaluation reports in, which is not the same question as the scale it trained
# through. Training fits sigmoid(cp / SCALE), and a network fitted against a blend of the
# reference score and the game result comes out sharper than the reference: net-v1 tracks
# Stockfish with a correlation of 0.96 and a slope of 2.28, so it ranks positions correctly
# and reports them more than twice as large.
#
# That would be harmless -- alpha-beta only compares evaluations -- except that searcher.py is
# full of margins in centipawns, all of them tuned when the evaluation was the hand-written one
# on a classical pawn-is-100 scale. On a 2.28x evaluation the aspiration window of 30 behaves
# like 13 and the futility margin of 80*depth behaves like 35*depth.
#
# So this divides the reported evaluation back onto the scale those constants were written for.
# It is a monotone transform: no position changes its ranking relative to any other, only the
# units change. tools/calibrate.py measures it per network and writes it into the weights file,
# because the number belongs to a particular network and the next one will differ.
EVAL_SCALE = int(_NET["eval_scale"])

TRAINED = bool(_NET["trained"])

# One cache entry per perspective, bucket and mirror state.
CACHE_SLOTS = 2 * BUCKETS * 2

# Clamp bounds as int16, so min/max stay in the accumulator's own type.
ZERO16 = np.int16(0)
QA16 = np.int16(QA)


def _king_buckets(count: int) -> np.ndarray:
    """Which weight bank a king square selects, after the board is mirrored queenside.

    Corners against centre on the home ranks, which is the distinction that matters most for a
    castled king, then coarser as the king advances. Modelled on the layout Leorik uses.

    Chosen for what the network can learn, not for how rarely a king crosses a boundary.
    Measurement showed crossing frequency does not matter here: a rank-only layout, which a
    castled king almost never leaves, measured within 6ns a node of this one. Nor does bank
    count: 32 banks measured the same as 4. See the note at the top.
    """
    table = np.zeros(64, dtype=np.int64)
    if count == 1:
        return table
    if count == 32:
        # One bank per king square over the mirrored half, which makes the feature set
        # (king square, piece, square) rather than (coarse king region, piece, square). This
        # is the layout Stockfish and every HalfKP descendant use, and the one that lets the
        # network say "these pieces are aimed at *that* king" at all.
        #
        # Must match `king_buckets` in aire/bullet-trainer/src/main.rs, which indexes its
        # 32-entry table as rank * 4 + file over the same mirrored half. A layout that differs
        # between trainer and engine loads cleanly and plays like noise.
        for square in range(64):
            rank, file = divmod(square, 8)
            table[square] = rank * 4 + min(file, 3)
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

# Output buckets: one output layer per material count, chosen by how many pieces are left.
#
# This is capacity the accumulator does not pay for at all. What a feature update costs is one
# row of WEIGHTS, HIDDEN int16s, and these weights are not in that matrix: eight buckets add
# 2 x HIDDEN x 8 x 2 bytes, four kilobytes, read once per evaluation rather than once per
# feature. Width is a real trade against nodes per second; this is not.
#
# The bucket must be the one the trainer used, which is bullet's MaterialCount<N>:
#
#     bucket = (popcount(occupied) - 2) / ceil(32 / N)
#
# Integer division, and the -2 is the two kings, which are always on the board. Disagreeing
# with the trainer here selects a bank that was fitted for a different phase: it loads
# cleanly and plays worse, which is why tools/side_bias.py screens correlation afterwards.
OUT_BUCKETS = int(OUTPUT.shape[0])
OUT_DIVISOR = -(-32 // OUT_BUCKETS)


_SIG_output_bucket = int64(uint64)


@njit(nogil=True, cache=False)
def output_bucket(occupied: np.uint64) -> np.int64:
    """Which output bank this position selects. Clamped, because a position reached by a
    promotion the trainer never saw must not index past the end of the array."""
    bucket = (popcount(occupied) - 2) // OUT_DIVISOR
    if bucket < 0:
        return np.int64(0)
    if bucket >= OUT_BUCKETS:
        return np.int64(OUT_BUCKETS - 1)
    return np.int64(bucket)


_SIG_perspective = types.UniTuple(int64, 2)(int64, int64)


@njit(nogil=True, cache=False)
def perspective(king_square: np.int64, flip: np.int64):
    """Where one perspective's weights start, and whether its board is mirrored.

    `flip` is 0 for White and 56 for Black, which puts the square into that perspective's own
    frame before the king is bucketed.
    """
    king = king_square ^ flip
    mirror = 7 if (king & 7) >= 4 else 0
    return KING_BUCKET[king ^ mirror] * INPUTS, mirror


_SIG__index = int64(int64, int64, int64, int64, int64)


@njit(nogil=True, cache=False)
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


_SIG__apply = int64(int16[:, ::1], int64, int64, int64)


@njit(nogil=True, cache=False)
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


_SIG__slot = int64(int64, int64, int64)


@njit(nogil=True, cache=False)
def _slot(side: np.int64, offset: np.int64, mirror: np.int64) -> np.int64:
    return (side * BUCKETS + offset // INPUTS) * 2 + (1 if mirror != 0 else 0)


_SIG_refresh_side = int64(
    uint64[::1],
    int16[:, ::1],
    int64,
    int64,
    int64,
    int16[:, ::1],
    uint64[:, ::1],
)


@njit(nogil=True,
    cache=False)
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


_SIG_refresh = int64(uint64[::1], int16[:, ::1], int16[:, ::1], uint64[:, ::1])


@njit(nogil=True, cache=False)
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


_SIG_advance = int64(
    uint64[::1],
    uint64[::1],
    int16[:, ::1],
    int16[:, ::1],
    int16[:, ::1],
    uint64[:, ::1],
)


@njit(nogil=True,
    cache=False)
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

    # One pass over each perspective, not one per changed feature.
    #
    # The obvious way to write this is a copy followed by a call per feature, each adding or
    # subtracting a weight row. That reads and writes the whole accumulator once per feature:
    # six passes for an ordinary quiet move, eight for a capture. Strong engines do it in one --
    # the accumulator is held in vector registers while every added and removed row is applied,
    # and Stockfish's own note that "going past 256 neurons requires multiple passes over the
    # feature indices as AVX2 doesn't have enough registers" is a statement about exactly this
    # loop. numba has no register control, but the shape carries over: gather the indices first,
    # then write each output element once from the source and the rows that touch it.
    #
    # It matters most at width. Measured per advance() call, the old form cost 82ns at 128
    # hidden and 489ns at 256 -- seven times the cost for twice the arithmetic, because the
    # per-feature loop stops being vectorised somewhere past 160.
    #
    # A legal move changes at most two features per perspective on the way in and two on the
    # way out: castling moves king and rook, a capture-promotion removes the pawn and the
    # captured piece and adds the promoted one. Anything outside that falls back to the general
    # path below, which is correct for any number of changes and simply slower.
    for side in range(2):
        if (white_moved_bucket if side == 0 else black_moved_bucket):
            continue
        offset, mirror = now_white if side == 0 else now_black

        added = 0
        removed = 0
        add0 = np.int64(0)
        add1 = np.int64(0)
        sub0 = np.int64(0)
        sub1 = np.int64(0)
        for piece in range(12):
            changed = before[piece] ^ after[piece]
            if changed == U0:
                continue
            gone = changed & before[piece]
            while gone != U0:
                square = lsb(gone)
                gone &= gone - U1
                index = _index(side, piece, square, offset, mirror)
                if removed == 0:
                    sub0 = index
                elif removed == 1:
                    sub1 = index
                removed += 1
            arrived = changed & after[piece]
            while arrived != U0:
                square = lsb(arrived)
                arrived &= arrived - U1
                index = _index(side, piece, square, offset, mirror)
                if added == 0:
                    add0 = index
                elif added == 1:
                    add1 = index
                added += 1

        if added == 1 and removed == 1:
            plus = WEIGHTS[add0]
            minus = WEIGHTS[sub0]
            for j in range(HIDDEN):
                destination[side, j] = source[side, j] + plus[j] - minus[j]
        elif added == 1 and removed == 2:
            plus = WEIGHTS[add0]
            minus = WEIGHTS[sub0]
            minus_two = WEIGHTS[sub1]
            for j in range(HIDDEN):
                destination[side, j] = source[side, j] + plus[j] - minus[j] - minus_two[j]
        elif added == 2 and removed == 2:
            plus = WEIGHTS[add0]
            plus_two = WEIGHTS[add1]
            minus = WEIGHTS[sub0]
            minus_two = WEIGHTS[sub1]
            for j in range(HIDDEN):
                destination[side, j] = (
                    source[side, j] + plus[j] + plus_two[j] - minus[j] - minus_two[j]
                )
        else:
            # More changes than the specialised forms cover, or none at all. Correct for any
            # move; it just costs a pass per feature the way the whole function used to.
            for j in range(HIDDEN):
                destination[side, j] = source[side, j]
            for piece in range(12):
                changed = before[piece] ^ after[piece]
                if changed == U0:
                    continue
                gone = changed & before[piece]
                while gone != U0:
                    square = lsb(gone)
                    gone &= gone - U1
                    _apply(destination, side, _index(side, piece, square, offset, mirror), -1)
                arrived = changed & after[piece]
                while arrived != U0:
                    square = lsb(arrived)
                    arrived &= arrived - U1
                    _apply(destination, side, _index(side, piece, square, offset, mirror), 1)

    if white_moved_bucket:
        refresh_side(after, destination, 0, now_white[0], now_white[1], cache_values, cache_boards)
    if black_moved_bucket:
        refresh_side(after, destination, 1, now_black[0], now_black[1], cache_values, cache_boards)
    return 0


_SIG_forward = int32(int16[:, ::1], int64, int64)


@njit(nogil=True, cache=False)
def forward(accumulator: np.ndarray, side_to_move: np.int64, bucket: np.int64) -> np.int32:
    """Clipped ReLU and the output dot product in one pass, in centipawns.

    Branch-free, because branches are what stop this vectorising. The obvious way to clamp is
    `if v < 0: v = 0 elif v > QA: v = QA`, which is two conditional jumps for every neuron; an
    earlier version wrote exactly that, unrolled four wide, and cost 200ns a call at 512 hidden
    against a theoretical figure nearer 30. `min(max(v, 0), QA)` lowers to a pair of select
    instructions instead, which the vectoriser turns into packed min and max over sixteen or
    thirty-two lanes at a time.

    The multiply stays in int16: an int16 by int16 product accumulated into int32 is what
    vpmaddwd does in one instruction for a whole vector, and casting each operand to int32
    first, as the previous version did, throws half the lanes away.

    One running sum rather than four. Four independent accumulators break the dependency chain
    when the loop is scalar, but the vectoriser already keeps a vector of partial sums and does
    it better; the manual version mostly gets in its way.
    """
    total = np.int32(0)
    # Taken as a row once rather than indexed two-dimensionally in the inner loop, so the
    # bucket costs one address computation for the whole evaluation instead of one per neuron.
    weights = OUTPUT[bucket]
    for half in range(2):
        side = side_to_move if half == 0 else 1 - side_to_move
        base = half * HIDDEN
        for j in range(HIDDEN):
            clipped = min(max(accumulator[side, j], ZERO16), QA16)
            total += np.int32(clipped) * np.int32(weights[base + j])
    total += OUTPUT_BIAS[bucket]
    return np.int32(total * EVAL_SCALE // (QA * QB))


_SIG_features = int64(uint64[::1], int32[::1], int32[::1])


@njit(nogil=True, cache=False)
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


_COMPILE_PAIRS = (
    (output_bucket, _SIG_output_bucket),
    (perspective, _SIG_perspective),
    (_index, _SIG__index),
    (_apply, _SIG__apply),
    (_slot, _SIG__slot),
    (refresh_side, _SIG_refresh_side),
    (refresh, _SIG_refresh),
    (advance, _SIG_advance),
    (forward, _SIG_forward),
    (features, _SIG_features),
)


def _compile_all(deadline: float | None = None) -> bool:
    """Compile this module's functions, stopping if `deadline` has passed.

    Returns whether it finished. Ordered so the cheap functions land first: whatever
    the init budget can afford does not have to be paid out of the first move's clock.
    """
    for fn, sig in _COMPILE_PAIRS:
        if deadline is not None and time.monotonic() > deadline:
            return False
        fn.compile(sig)
    return True
