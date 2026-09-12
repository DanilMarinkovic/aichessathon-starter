"""Iterative-deepening alpha-beta search.

Every working array is a parameter rather than a global. numba compiles a global array in as
a read-only constant, so anything the search writes to has to be passed in; the arrays are
allocated once here at import and threaded down the recursion.

Time control works around a second numba limitation: jitted code cannot read a clock. The
caller arms a timer thread that writes CONTROL[STOP] = 1, and the search, which releases the
GIL, polls that every 2048 nodes. An iteration that stops early is discarded and the move
from the last finished depth is played.

A transposition entry is packed into one int64 so the table is two arrays instead of five:

    bits 0-17   best move
    bits 18-37  score, offset by 40000 to keep it unsigned
    bits 38-45  depth
    bits 46-47  EXACT, LOWER or UPPER
"""

import numpy as np
from numba import int16, int32, int64, njit, uint64

from bitboards import (
    BISHOP,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    U0,
    U1,
    ZOBRIST_EP,
    ZOBRIST_SIDE,
    popcount,
)
from evaluate import evaluate
from movegen import MAX_MOVES, generate
from nnue import HIDDEN as NNUE_HIDDEN
from nnue import TRAINED as NNUE_TRAINED
from nnue import advance as nnue_advance
from nnue import forward as nnue_forward
from nnue import new_cache as nnue_new_cache
from nnue import output_bucket as nnue_output_bucket
from nnue import refresh as nnue_refresh
from position import (
    EN_PASSANT,
    EP,
    HALFMOVE,
    HASH,
    NFIELDS,
    NO_EP,
    OCC_ALL,
    SIDE,
    in_check,
    make_move,
    move_flag,
    move_from,
    move_promotion,
    move_to,
    piece_on,
)
from see import see_ge

MAX_PLY = 96
MATE = 30000
MATE_THRESHOLD = MATE - 512
INFINITY = 32000

EXACT = 0
LOWER = 1
UPPER = 2

SCORE_OFFSET = 40000

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)

# CONTROL slots.
NODES = 0
STOP = 1
NODE_LIMIT = 2
BEST_MOVE = 3
BEST_SCORE = 4
BEST_DEPTH = 5
# Which evaluation to use. Lives in CONTROL rather than a global because numba freezes a
# global array into compiled code as a read-only constant and folds the branch away.
USE_NNUE = 6
# Raised by the clock thread when the ordinary budget is gone. It is not STOP: it asks the
# search to stop at a sensible boundary rather than wherever it happens to be, which is the
# whole point -- a search cut mid-iteration keeps the previous depth's answer anyway, so the
# time spent on the unfinished iteration bought nothing.
SOFT = 7
# How many completed iterations in a row have agreed on the same root move. Zero means the
# answer just changed, which is exactly when stopping is worst.
STABLE = 8
# Static evaluation correction history.
#
# The network's evaluation of a position is systematically wrong in ways that repeat: the same
# kind of pawn structure gets misjudged the same way every time it appears. The search already
# discovers this -- every node compares what the evaluation said against what searching the
# position actually returned -- and then throws the comparison away. This keeps it.
#
# Written from the chessprogramming wiki's description, following the exponential-moving-average
# form Alexandria uses rather than Stockfish's gravity form, because the update is one line and
# the constants are stated. Introduced in Caissa in October 2023 and since adopted widely.
#
# Two reasons it suits this engine specifically. It corrects the evaluation, which is where our
# remaining error is -- the search is 1 for 7 on measured changes. And it is reported to scale
# up with time control rather than down, which is the opposite of the pruning changes that
# flattered themselves at shallow depth and lost 23 and 33 Elo on a real clock.
#
# Indexed by pawn structure and side, and it lives inside CONTROL rather than as a global.
# numba compiles a global array into a function as read-only, which is why nnue.py can hold its
# weights that way and why this cannot: the table has to be written on every node. The
# alternative is a seventh array threaded through six recursive call sites of otherwise
# identically typed arrays, which is the edit that goes wrong silently. A signature fixes an
# array's dtype and dimensionality, not its length, so CONTROL simply gets longer.
CORR_SIZE = 16384
CORR_MASK = CORR_SIZE - 1
# The table stores centipawns multiplied by GRAIN, so a correction accumulates in fractions of
# a centipawn instead of rounding to nothing. MAX/GRAIN = 64cp is the most it can ever move an
# evaluation, which keeps a bad entry from rewriting the position's assessment outright.
CORR_GRAIN = 256
CORR_MAX = 16384
CORR_WEIGHT_SCALE = 256
# Where the correction table starts inside CONTROL. Side 0 occupies CORR_BASE onwards, side 1
# the CORR_SIZE entries after that.
CORR_BASE = 16

PATH_LIMIT = 2048

TT_KEY = np.zeros(TT_SIZE, dtype=np.uint64)
TT_DATA = np.zeros(TT_SIZE, dtype=np.int64)
STATES = np.zeros((MAX_PLY + 8, NFIELDS), dtype=np.uint64)
MOVES = np.zeros((MAX_PLY + 8, MAX_MOVES), dtype=np.int32)
ORDER = np.zeros((MAX_PLY + 8, MAX_MOVES), dtype=np.int32)
KILLERS = np.zeros((MAX_PLY + 8, 2), dtype=np.int32)
HISTORY = np.zeros((12, 64), dtype=np.int64)
PATH = np.zeros(PATH_LIMIT + MAX_PLY + 8, dtype=np.uint64)
# One slot per ply holding a move the search must pretend does not exist. Used by the singular
# extension below, which asks "is this move the only good one here?" by searching the same node
# again with that move removed. It lives in CONTROL for the same reason the correction table
# does: numba freezes a global array into compiled code as read-only, so anything written during
# a search has to be a parameter, and threading a seventh array through six recursive call sites
# of otherwise identically typed arrays is the edit that goes wrong silently.
EXCLUDED_BASE = CORR_BASE + 2 * CORR_SIZE

# Capture history: a score per (moving piece, target square, captured piece), learned the way
# the quiet history is. SEE already splits captures into winning and losing, which is a stronger
# first cut than most-valuable-victim; this orders within those groups, where SEE says only "not
# losing material" and cannot tell a good capture from a pointless one. Stefan Geschwentner
# introduced it in 2016 and it is standard now, replacing least-valuable-attacker as the
# tiebreak. It lives in CONTROL for the usual reason: numba freezes a global array into compiled
# code read-only, so anything written during a search has to arrive as a parameter.
CAPHIST_BASE = EXCLUDED_BASE + MAX_PLY + 8
CAPHIST_SIZE = 12 * 64 * 6
CONTROL = np.zeros(CAPHIST_BASE + CAPHIST_SIZE, dtype=np.int64)
# The network is the evaluation whenever one is loaded. Set here, at import, and not left for
# each caller: CONTROL is zeros, so the old default was the hand-crafted evaluation, and for
# weeks every tool that drove the search itself -- tools/probe.py, tools/nodecost.py -- silently
# measured that instead of the network. It is how a node-cost sweep came to report five network
# shapes as costing the same to within a few percent: none of them was in use. agent.py still
# sets the same value explicitly, so what ships is unchanged; what changes is that forgetting is
# no longer possible. The hand-crafted evaluation stays reachable by writing a 0 here, which is
# worth keeping: flipping between two evaluations of very different cost is how a measurement
# tool gets checked against an answer that is already known.
CONTROL[USE_NNUE] = 1 if NNUE_TRAINED else 0

# One network accumulator per ply, mirroring how positions are already stacked. numba freezes
# the array's identity into the compiled code and reads its contents at run time, so the search
# can index it as a global without threading it through every signature.
ACCUMULATORS = np.zeros((MAX_PLY + 8, 2, NNUE_HIDDEN), dtype=np.int16)
# The accumulator refresh table: one cached accumulator per bucket, with the bitboards it was
# built from, so a king crossing a bucket boundary costs a small diff instead of a rebuild.
CACHE_VALUES, CACHE_BOARDS = nnue_new_cache()

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)

# History heuristic bounds, using the gravity update every modern engine uses:
#
#     history += bonus - history * abs(bonus) / MAX_HISTORY
#
# The subtracted term is what makes it work. An entry near the ceiling barely moves, one near
# zero moves almost the full bonus, so the table saturates smoothly instead of needing a clamp
# and a periodic halving -- and a cutoff the ordering did not expect teaches it more than one
# it did. Values are bounded to +/-MAX_HISTORY by construction.
#
# Written from the chessprogramming wiki's description rather than invented here. A first
# attempt using a flat `+= depth * depth` with a manual clamp learns which moves are good and
# never which are bad, which is a much weaker signal than it looks.
MAX_HISTORY = 16384

_lmr = np.zeros((64, 64), dtype=np.int64)
for _depth in range(1, 64):
    for _played in range(1, 64):
        _lmr[_depth, _played] = int(0.75 + np.log(_depth) * np.log(_played) / 2.25)
LMR = _lmr


@njit(int64(int32, int32, int64, int64), nogil=True, cache=False, inline="always")
def pack(move: np.int32, score: np.int32, depth: np.int64, flag: np.int64) -> np.int64:
    return (
        np.int64(move & 0x3FFFF)
        | (np.int64(score + SCORE_OFFSET) << 18)
        | (depth << 38)
        | (flag << 46)
    )


@njit(int64(uint64[::1]), nogil=True, cache=False)
def has_pieces(state: np.ndarray) -> np.int64:
    """Whether the side to move has a piece other than pawns, which null move requires."""
    base = np.int64(state[SIDE]) * 6
    heavy = (
        state[base + KNIGHT - 1]
        | state[base + BISHOP - 1]
        | state[base + ROOK - 1]
        | state[base + QUEEN - 1]
    )
    return 1 if heavy != U0 else 0


@njit(int64(uint64[::1]), nogil=True, cache=False)
def material_draw(state: np.ndarray) -> np.int64:
    """King against king, and king and a single minor against king, cannot be won."""
    if (
        state[PAWN - 1]
        | state[6 + PAWN - 1]
        | state[ROOK - 1]
        | state[6 + ROOK - 1]
        | state[QUEEN - 1]
        | state[6 + QUEEN - 1]
    ) != U0:
        return 0
    minors = popcount(
        state[KNIGHT - 1] | state[BISHOP - 1] | state[6 + KNIGHT - 1] | state[6 + BISHOP - 1]
    )
    return 1 if minors <= 1 else 0


@njit(int64(uint64[::1]), nogil=True, cache=False, inline="always")
def pawn_index(state: np.ndarray) -> np.int64:
    """A hash of the pawn structure alone.

    Multiplicative rather than Zobrist: a Zobrist pawn key would have to be maintained by every
    make_move, and this is two multiplies on bitboards the position already holds. The constants
    are the usual odd 64-bit mixers; only their scattering matters, not their provenance.
    """
    mixed = (state[PAWN - 1] * np.uint64(0x9E3779B97F4A7C15)) ^ (
        state[6 + PAWN - 1] * np.uint64(0xC2B2AE3D27D4EB4F)
    )
    return np.int64((mixed >> np.uint64(32)) & np.uint64(CORR_MASK))


@njit(int64(uint64[:, ::1], uint64[::1], int64, int64), nogil=True, cache=False)
def is_repetition(
    states: np.ndarray, path: np.ndarray, ply: np.int64, root_offset: np.int64
) -> np.int64:
    """Whether the position at `ply` has occurred before, in the game or in this line.

    One repetition counts as a draw rather than waiting for a third. Inside a search that is
    the useful convention: it stops the engine shuffling a won position towards a draw, and
    lets it find a perpetual when it is losing.
    """
    here = root_offset + ply
    key = path[here]
    halfmove = np.int64(states[ply, HALFMOVE])
    limit = here - halfmove
    if limit < 0:
        limit = 0
    index = here - 2
    while index >= limit:
        if path[index] == key:
            return 1
        index -= 2
    return 0


@njit(int64(uint64[::1], uint64[::1]), nogil=True, cache=False)
def make_null(state: np.ndarray, out: np.ndarray) -> np.int64:
    for index in range(NFIELDS):
        out[index] = state[index]
    key = state[HASH]
    previous = np.int64(state[EP])
    if previous != NO_EP:
        key ^= ZOBRIST_EP[previous & 7]
        out[EP] = np.uint64(NO_EP)
    out[SIDE] = np.uint64(1 - np.int64(state[SIDE]))
    out[HASH] = key ^ ZOBRIST_SIDE
    out[HALFMOVE] = state[HALFMOVE] + U1
    return 1


@njit(int64(int64, int64, int64), nogil=True, cache=False, inline="always")
def caphist_index(piece: np.int64, target: np.int64, victim: np.int64) -> np.int64:
    """Where a (moving piece, target square, captured piece) triple lives inside CONTROL."""
    return CAPHIST_BASE + (piece * 64 + target) * 6 + victim - 1


@njit(int64(uint64[::1], int32, int32, int64[::1]), nogil=True, cache=False)
def score_move(
    state: np.ndarray, move: np.int32, tt_move: np.int32, control: np.ndarray
) -> np.int64:
    """Order moves so alpha-beta sees the good ones first, which is most of its value."""
    if move == tt_move:
        return 2_000_000
    side = np.int64(state[SIDE])
    victim = PAWN if move_flag(move) == EN_PASSANT else piece_on(state, move_to(move), 1 - side)
    if victim != 0:
        attacker = piece_on(state, move_from(move), side)
        # The victim still dominates the order; capture history replaces the attacker as the
        # tiebreak, divided so a saturated entry is worth about a pawn of victim value and no
        # more -- enough to reorder captures of equal material, never enough to put a capture
        # of a pawn above a capture of a rook.
        learned = control[caphist_index(
            side * 6 + attacker - 1, np.int64(move_to(move)), np.int64(victim)
        )]
        rank = PIECE_VALUE[victim] * 16 - PIECE_VALUE[attacker] + learned // 64
        # Most valuable victim first is a good guess and a bad answer: it rates a pawn taking a
        # defended queen above a clean win of a rook. SEE separates the two, and losing captures
        # drop below the quiet moves rather than being tried first.
        if see_ge(state, move, np.int64(0)) != 0:
            return 1_000_000 + rank
        return 100_000 + rank
    if move_promotion(move) == QUEEN:
        return 900_000
    return 0


@njit(int32(int32[:, ::1], int32[:, ::1], int64, int64, int64), nogil=True, cache=False)
def pick_move(
    moves: np.ndarray, order: np.ndarray, ply: np.int64, start: np.int64, count: np.int64
) -> np.int32:
    """Selection sort one move at a time: a beta cutoff usually arrives before the rest matter."""
    best = start
    for index in range(start + 1, count):
        if order[ply, index] > order[ply, best]:
            best = index
    if best != start:
        move = moves[ply, start]
        moves[ply, start] = moves[ply, best]
        moves[ply, best] = move
        rank = order[ply, start]
        order[ply, start] = order[ply, best]
        order[ply, best] = rank
    return moves[ply, start]


@njit(
    int32(uint64[:, ::1], int16[:, :, ::1], int64[::1], int64),
    nogil=True,
    cache=False,
    inline="always",
)
def score_position(
    states: np.ndarray, accumulators: np.ndarray, control: np.ndarray, ply: np.int64
) -> np.int32:
    """Whichever evaluation is in force. The branch is on a value that never changes mid-search."""
    side = np.int64(states[ply, SIDE])
    if control[USE_NNUE] != 0:
        raw = np.int32(
            nnue_forward(accumulators[ply], side, nnue_output_bucket(states[ply, OCC_ALL]))
        )
    else:
        raw = np.int32(evaluate(states[ply]))
    # Corrected here rather than at each use, so the futility margins, the null-move test and
    # quiescence's stand-pat all read the same number.
    slot_c = CORR_BASE + side * CORR_SIZE + pawn_index(states[ply])
    return raw + np.int32(control[slot_c] // CORR_GRAIN)


@njit(
    int64(uint64[:, ::1], int16[:, :, ::1], int16[:, ::1], uint64[:, ::1], int64[::1], int64),
    nogil=True,
    cache=False,
    inline="always",
)
def push_accumulator(
    states: np.ndarray,
    accumulators: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
    control: np.ndarray,
    ply: np.int64,
) -> np.int64:
    """Carry the accumulator one ply forward, after a move has been made into ply + 1."""
    if control[USE_NNUE] != 0:
        nnue_advance(
            states[ply], states[ply + 1], accumulators[ply], accumulators[ply + 1],
            cache_values, cache_boards,
        )
    return 0


@njit(
    int32(
        uint64[:, ::1],
        int16[:, :, ::1],
        int16[:, ::1],
        uint64[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        int64[::1],
        int64,
        int32,
        int32,
    ),
    nogil=True,
    cache=False,
)
def quiescence(
    states: np.ndarray,
    accumulators: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
    moves: np.ndarray,
    order: np.ndarray,
    control: np.ndarray,
    ply: np.int64,
    alpha: np.int32,
    beta: np.int32,
) -> np.int32:
    """Search captures only, so the evaluation is never taken mid-exchange."""
    control[NODES] += 1
    if (control[NODES] & 2047) == 0 and (
        control[STOP] != 0 or control[NODES] > control[NODE_LIMIT]
    ):
        control[STOP] = 1
        return np.int32(0)

    stand_pat = score_position(states, accumulators, control, ply)
    if ply >= MAX_PLY - 2:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    count = generate(states[ply], moves[ply], 1)
    side = np.int64(states[ply, SIDE])
    for index in range(count):
        order[ply, index] = np.int32(
            score_move(states[ply], moves[ply, index], np.int32(0), control)
        )

    best = stand_pat
    for index in range(count):
        move = pick_move(moves, order, ply, index, count)

        # Delta pruning: if winning the piece outright still falls short of alpha, nothing
        # further down this capture chain will reach it either.
        if move_flag(move) == EN_PASSANT:
            victim = PAWN
        else:
            victim = piece_on(states[ply], move_to(move), 1 - side)
        if (
            victim != 0
            and move_promotion(move) == 0
            and stand_pat + np.int32(PIECE_VALUE[victim] + 200) < alpha
        ):
            continue

        # A capture that loses material does not become good further down the exchange, and
        # searching it costs a node plus its whole subtree. Promotions are exempt: their value
        # is in what arrives on the square, not in what the exchange settles at.
        # Stockfish allows a capture that loses up to 74cp here rather than refusing every
        # losing one. A refusal is only right if the exchange is the whole story, and often the
        # recapture opens a file or removes a defender that the next ply pays for.
        if move_promotion(move) == 0 and see_ge(states[ply], move, np.int64(-74)) == 0:
            continue

        if make_move(states[ply], move, states[ply + 1]) == 0:
            continue
        push_accumulator(states, accumulators, cache_values, cache_boards, control, ply)
        score = -quiescence(
            states, accumulators, cache_values, cache_boards, moves, order, control,
            ply + 1, -beta, -alpha,
        )
        if control[STOP] != 0:
            return np.int32(0)
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


@njit(
    int32(
        uint64[:, ::1],
        int16[:, :, ::1],
        int16[:, ::1],
        uint64[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        int64[:, ::1],
        uint64[::1],
        int64[::1],
        uint64[::1],
        int64[::1],
        int64,
        int64,
        int32,
        int32,
        int64,
        int64,
    ),
    nogil=True,
    cache=False,
)
def negamax(
    states: np.ndarray,
    accumulators: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
    moves: np.ndarray,
    order: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    tt_key: np.ndarray,
    tt_data: np.ndarray,
    path: np.ndarray,
    control: np.ndarray,
    ply: np.int64,
    depth: np.int64,
    alpha: np.int32,
    beta: np.int32,
    root_offset: np.int64,
    can_null: np.int64,
) -> np.int32:
    control[NODES] += 1
    if (control[NODES] & 2047) == 0 and (
        control[STOP] != 0 or control[NODES] > control[NODE_LIMIT]
    ):
        control[STOP] = 1
        return np.int32(0)

    root = ply == 0
    if not root:
        if (
            is_repetition(states, path, ply, root_offset) != 0
            or np.int64(states[ply, HALFMOVE]) >= 100
            or material_draw(states[ply]) != 0
        ):
            return np.int32(0)
        if ply >= MAX_PLY - 8:
            return score_position(states, accumulators, control, ply)

        # Mate distance pruning: a shorter mate found elsewhere makes this subtree irrelevant.
        if np.int32(-MATE + ply) > alpha:
            alpha = np.int32(-MATE + ply)
        if np.int32(MATE - ply - 1) < beta:
            beta = np.int32(MATE - ply - 1)
        if alpha >= beta:
            return alpha

    checked = in_check(states[ply])
    if checked != 0:
        depth += 1
    if depth <= 0:
        return quiescence(
        states, accumulators, cache_values, cache_boards, moves, order, control,
        ply, alpha, beta,
    )

    excluded = np.int32(control[EXCLUDED_BASE + ply])

    key = states[ply, HASH]
    slot = np.int64(key & TT_MASK)
    tt_move = np.int32(0)
    tt_value = np.int32(0)
    tt_flag = np.int64(-1)
    tt_depth = np.int64(0)
    if tt_key[slot] == key:
        entry = tt_data[slot]
        tt_move = np.int32(entry & 0x3FFFF)
        stored_depth = (entry >> 38) & 0xFF
        tt_depth = np.int64(stored_depth)
        tt_flag = np.int64((entry >> 46) & 3)
        tt_value = np.int32(((entry >> 18) & 0xFFFFF) - SCORE_OFFSET)
        if tt_value > MATE_THRESHOLD:
            tt_value -= np.int32(ply)
        elif tt_value < -MATE_THRESHOLD:
            tt_value += np.int32(ply)
        if not root and excluded == 0 and stored_depth >= depth:
            stored = np.int32(((entry >> 18) & 0xFFFFF) - SCORE_OFFSET)
            if stored > MATE_THRESHOLD:
                stored -= np.int32(ply)
            elif stored < -MATE_THRESHOLD:
                stored += np.int32(ply)
            flag = (entry >> 46) & 3
            if flag == EXACT:
                return stored
            if flag == LOWER and stored >= beta:
                return stored
            if flag == UPPER and stored <= alpha:
                return stored

    # Internal iterative reduction: a node the table has never stored a move for is one no
    # earlier search thought worth finishing, so spend a ply less on it. Cheaper than the
    # internal iterative *deepening* it replaced, which searched the node twice to find a move
    # to order by; this simply admits the node is probably not important. Stockfish applies it
    # from depth 6, which is also where the cost of being wrong stops being trivial.
    if depth >= 6 and tt_move == 0:
        depth -= 1

    pv_node = beta - alpha > 1
    static = np.int32(0) if checked != 0 else score_position(states, accumulators, control, ply)

    if not pv_node and checked == 0 and abs(beta) < MATE_THRESHOLD:
        # Reverse futility: so far ahead that conceding a few pawns would still hold beta.
        if depth <= 6 and static - np.int32(80 * depth) >= beta:
            return static
        if can_null != 0 and depth >= 3 and static >= beta and has_pieces(states[ply]) != 0:
            reduction = 2 + depth // 6
            make_null(states[ply], states[ply + 1])
            # A null move leaves every piece where it was, so the bitboard diff is empty and
            # this degenerates to a copy. No special case needed.
            push_accumulator(states, accumulators, cache_values, cache_boards, control, ply)
            path[root_offset + ply + 1] = states[ply + 1, HASH]
            score = -negamax(
                states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                ply + 1, depth - 1 - reduction, np.int32(-beta), np.int32(-beta + 1),
                root_offset, 0,
            )
            if control[STOP] != 0:
                return np.int32(0)
            if score >= beta:
                return beta if abs(score) >= MATE_THRESHOLD else score

    count = generate(states[ply], moves[ply], 0)
    side = np.int64(states[ply, SIDE])
    for index in range(count):
        move = moves[ply, index]
        rank = score_move(states[ply], move, tt_move, control)
        if rank == 0:
            if move == killers[ply, 0]:
                rank = 800_000
            elif move == killers[ply, 1]:
                rank = 700_000
            else:
                piece = side * 6 + piece_on(states[ply], move_from(move), side) - 1
                # Bounded to +/-MAX_HISTORY by the gravity update, so quiet moves always sort
                # below the killers above and the captures scored in score_move.
                rank = history[piece, move_to(move)]
        order[ply, index] = np.int32(rank)

    best_score = np.int32(-INFINITY)
    best_move = np.int32(0)
    played = 0
    original_alpha = alpha

    for index in range(count):
        move = pick_move(moves, order, ply, index, count)
        if move == excluded:
            continue
        quiet = (
            move_promotion(move) == 0
            and move_flag(move) != EN_PASSANT
            and piece_on(states[ply], move_to(move), 1 - side) == 0
        )
        # A capture that loses material is worth searching when the compensation is somewhere
        # in the subtree, and the deeper the search the likelier that is -- so the bar drops
        # with depth rather than being a flat cutoff. Quiescence already refuses every losing
        # capture outright; here, where a whole subtree hangs off the move, the threshold is
        # deliberately looser. Promotions are exempt for the same reason as in quiescence, and
        # nothing is skipped until one move has been played, so a node always returns a move.
        if (
            not pv_node
            and checked == 0
            and depth <= 8
            and played > 0
            and best_score > -MATE_THRESHOLD
            and not quiet
            and move_promotion(move) == 0
            and see_ge(states[ply], move, np.int64(-177) * depth) == 0
        ):
            continue

        # Singular extension: is this the only move holding the position together?
        #
        # The table says this move was good enough to fail high here. If every other move,
        # searched to half depth against a window a little below that score, fails low, the
        # position rests on this one move -- which is what a forcing sequence is, and forcing
        # sequences are exactly where our evaluation is least reliable. A ply spent there is
        # worth more than a ply spent anywhere else.
        #
        # Conditions follow Stockfish: not the root, the move is the table's move, we are not
        # already inside a verification, depth at least six, and the entry is a lower bound
        # with depth within three of ours and a score that is not a mate. The margin is a small
        # multiple of depth, inside the range Stockfish's own formula spans.
        #
        # The verification re-searches this node with the move removed through CONTROL. It
        # cannot recurse, because the whole block is skipped whenever `excluded` is set.
        extension = np.int64(0)
        if (
            not root
            and excluded == 0
            and move == tt_move
            and depth >= 6
            and tt_flag == LOWER
            and tt_depth >= depth - 3
            and abs(tt_value) < MATE_THRESHOLD
        ):
            singular_beta = np.int32(tt_value - np.int32(2 * depth))
            control[EXCLUDED_BASE + ply] = np.int64(move)
            verify = negamax(
                states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                ply, (depth - 1) // 2,
                np.int32(singular_beta - 1), singular_beta, root_offset, 0,
            )
            control[EXCLUDED_BASE + ply] = 0
            if control[STOP] != 0:
                return np.int32(0)
            if verify < singular_beta:
                extension = np.int64(1)
        new_depth = depth - 1 + extension

        if make_move(states[ply], move, states[ply + 1]) == 0:
            continue
        push_accumulator(states, accumulators, cache_values, cache_boards, control, ply)
        path[root_offset + ply + 1] = states[ply + 1, HASH]
        played += 1

        if played == 1:
            score = -negamax(
                states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                ply + 1, new_depth, np.int32(-beta), np.int32(-alpha), root_offset, 1,
            )
        else:
            reduction = np.int64(0)
            if depth >= 3 and played >= 4 and quiet and checked == 0:
                reduction = LMR[min(depth, 63), min(played, 63)]
                if pv_node and reduction > 0:
                    reduction -= 1
                # The table already knows which quiet moves have been causing cutoffs, and the
                # ordering uses it. Using it a second time here says how much to trust the
                # ordering for this move in particular: a move the table likes keeps more of
                # its depth, one it dislikes loses more. Divided so a saturated entry is worth
                # two plies either way, which is the magnitude the engines that do this use.
                # Written out in both directions because floor division on a negative value
                # rounds away from zero, which would penalise a history of -1 and not reward
                # a history of +1.
                stat = history[side * 6 + piece_on(states[ply], move_from(move), side) - 1,
                               move_to(move)]
                if stat > 0:
                    reduction -= stat // (MAX_HISTORY // 2)
                else:
                    reduction += (-stat) // (MAX_HISTORY // 2)
                if reduction > depth - 2:
                    reduction = depth - 2
                if reduction < 0:
                    reduction = 0
            score = -negamax(
                states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                ply + 1, new_depth - reduction, np.int32(-alpha - 1), np.int32(-alpha),
                root_offset, 1,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                    ply + 1, new_depth, np.int32(-alpha - 1), np.int32(-alpha), root_offset, 1,
                )
            if score > alpha and score < beta:
                score = -negamax(
                    states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                    ply + 1, new_depth, np.int32(-beta), np.int32(-alpha), root_offset, 1,
                )

        if control[STOP] != 0:
            return np.int32(0)

        if score > best_score:
            best_score = score
            best_move = move
            if root:
                control[BEST_MOVE] = np.int64(move)
                control[BEST_SCORE] = np.int64(score)
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != move:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        bonus = 300 * depth - 250
                        if bonus > MAX_HISTORY:
                            bonus = MAX_HISTORY
                        piece = side * 6 + piece_on(states[ply], move_from(move), side) - 1
                        entry = history[piece, move_to(move)]
                        history[piece, move_to(move)] = (
                            entry + bonus - entry * bonus // MAX_HISTORY
                        )

                        # The same bonus, negated, for the quiet moves tried before this one.
                        # Rewarding only the winner leaves every loser holding whatever score
                        # it already had, so the table learns which moves are good and never
                        # which are bad. pick_move is a selection sort, so the moves already
                        # tried are exactly moves[ply, 0..index-1], in order; nothing else has
                        # to be recorded to find them.
                        for earlier in range(index):
                            tried = moves[ply, earlier]
                            if (
                                move_promotion(tried) == 0
                                and move_flag(tried) != EN_PASSANT
                                and piece_on(states[ply], move_to(tried), 1 - side) == 0
                            ):
                                loser = (
                                    side * 6
                                    + piece_on(states[ply], move_from(tried), side)
                                    - 1
                                )
                                if loser >= 0:
                                    was = history[loser, move_to(tried)]
                                    history[loser, move_to(tried)] = (
                                        was - bonus - was * bonus // MAX_HISTORY
                                    )
                    else:
                        # The same gravity update for captures, in its own table. Rewarding the
                        # winner alone would leave every other capture holding whatever score it
                        # started with, so the table would learn which captures are good and
                        # never which are pointless -- the failure that cost the quiet history
                        # 81 Elo until it was fixed.
                        bonus = 300 * depth - 250
                        if bonus > MAX_HISTORY:
                            bonus = MAX_HISTORY
                        taken = (
                            PAWN
                            if move_flag(move) == EN_PASSANT
                            else piece_on(states[ply], move_to(move), 1 - side)
                        )
                        if taken != 0:
                            moving = piece_on(states[ply], move_from(move), side)
                            at = caphist_index(
                                side * 6 + moving - 1, np.int64(move_to(move)), np.int64(taken)
                            )
                            was = control[at]
                            control[at] = was + bonus - was * bonus // MAX_HISTORY
                            for earlier in range(index):
                                tried = moves[ply, earlier]
                                victim = (
                                    PAWN
                                    if move_flag(tried) == EN_PASSANT
                                    else piece_on(states[ply], move_to(tried), 1 - side)
                                )
                                if victim == 0:
                                    continue
                                mover = piece_on(states[ply], move_from(tried), side)
                                if mover == 0:
                                    continue
                                spot = caphist_index(
                                    side * 6 + mover - 1,
                                    np.int64(move_to(tried)),
                                    np.int64(victim),
                                )
                                had = control[spot]
                                control[spot] = had - bonus - had * bonus // MAX_HISTORY
                    break

    if played == 0:
        return np.int32(-MATE + ply) if checked != 0 else np.int32(0)

    stored = best_score
    if stored > MATE_THRESHOLD:
        stored += np.int32(ply)
    elif stored < -MATE_THRESHOLD:
        stored -= np.int32(ply)
    if excluded == 0 and (tt_key[slot] != key or depth >= ((tt_data[slot] >> 38) & 0xFF)):
        if best_score >= beta:
            flag = LOWER
        elif best_score > original_alpha:
            flag = EXACT
        else:
            flag = UPPER
        tt_key[slot] = key
        tt_data[slot] = pack(best_move, stored, depth, flag)

    # Record how far off the static evaluation turned out to be, so the next position with this
    # pawn structure starts from a better number.
    #
    # Skipped in three cases, all for the same reason -- the difference would not be the
    # evaluation's fault. In check there is no static evaluation to be wrong. When the best move
    # is a capture the gap is the exchange the search resolved, which is what a search is for.
    # And a mate score is not a quantity this table can average.
    if (
        checked == 0
        and abs(best_score) < MATE_THRESHOLD
        and (
            best_move == 0
            or piece_on(states[ply], move_to(best_move), 1 - side) == 0
        )
    ):
        slot_c = CORR_BASE + side * CORR_SIZE + pawn_index(states[ply])
        entry_c = control[slot_c]
        # Deeper searches are more trustworthy, so they move the entry further, up to half its
        # weight in one update. A shallow node nudges; a deep one nearly replaces.
        weight = depth * depth + 2 * depth + 1
        if weight > 128:
            weight = 128
        updated = (
            entry_c * (CORR_WEIGHT_SCALE - weight)
            + np.int64(best_score - static) * CORR_GRAIN * weight
        ) // CORR_WEIGHT_SCALE
        if updated > CORR_MAX:
            updated = CORR_MAX
        elif updated < -CORR_MAX:
            updated = -CORR_MAX
        control[slot_c] = updated

    return best_score


@njit(
    int64(
        uint64[:, ::1],
        int16[:, :, ::1],
        int16[:, ::1],
        uint64[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        int32[:, ::1],
        int64[:, ::1],
        uint64[::1],
        int64[::1],
        uint64[::1],
        int64[::1],
        int64,
        int64,
    ),
    nogil=True,
    cache=False,
)
def search_position(
    states: np.ndarray,
    accumulators: np.ndarray,
    cache_values: np.ndarray,
    cache_boards: np.ndarray,
    moves: np.ndarray,
    order: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    tt_key: np.ndarray,
    tt_data: np.ndarray,
    path: np.ndarray,
    control: np.ndarray,
    max_depth: np.int64,
    root_offset: np.int64,
) -> np.int64:
    """Deepen until the clock runs out, keeping the best move from the last finished depth."""
    control[NODES] = 0
    control[BEST_MOVE] = 0
    control[BEST_SCORE] = 0
    control[BEST_DEPTH] = 0
    control[SOFT] = 0

    for index in range(killers.shape[0]):
        killers[index, 0] = 0
        killers[index, 1] = 0
    for a in range(12):
        for b in range(64):
            history[a, b] >>= 3

    path[root_offset] = states[0, HASH]

    best_move = np.int64(0)
    best_score = np.int32(0)
    previous_best = np.int64(0)
    stable = np.int64(0)
    control[STABLE] = 0
    for depth in range(1, max_depth + 1):
        window = np.int32(30)
        while True:
            if depth >= 5:
                alpha = np.int32(max(-INFINITY, best_score - window))
                beta = np.int32(min(INFINITY, best_score + window))
            else:
                alpha = np.int32(-INFINITY)
                beta = np.int32(INFINITY)
            score = negamax(
                states, accumulators, cache_values, cache_boards,
                moves, order, killers, history,
                tt_key, tt_data, path, control,
                0, depth, alpha, beta, root_offset, 1,
            )
            if control[STOP] != 0:
                break
            if depth >= 5 and (score <= alpha or score >= beta) and window < 1200:
                window *= np.int32(4)
                continue
            best_score = score
            break

        if control[STOP] != 0:
            break
        best_move = control[BEST_MOVE]
        control[BEST_SCORE] = np.int64(best_score)
        control[BEST_DEPTH] = depth

        # Has the answer settled? An iteration that returns the same root move as the last one
        # is evidence the search has converged; one that changes it is evidence it has not.
        if best_move == previous_best:
            stable += 1
        else:
            stable = 0
        previous_best = best_move
        control[STABLE] = stable

        if abs(best_score) > MATE_THRESHOLD:
            break

        # The ordinary budget is gone. Stop if the answer has held for an iteration; keep going
        # if it just moved, up to the hard limit the clock thread still enforces.
        #
        # This redistributes time rather than adding it. Spending uniformly more was measured at
        # -0.6 Elo, so the average is not the constraint -- where it goes is. A position whose
        # best move is still changing at the budget is precisely the one where another ply pays,
        # and a game was lost to exactly that: the refutation needed depth 15 and the search
        # stopped at 14 with time on the clock.
        if control[SOFT] != 0 and stable >= 1:
            break

    if best_move == 0:
        best_move = control[BEST_MOVE]
    return best_move


def run(max_depth: int, root_offset: int, node_limit: int) -> int:
    """Search the position already loaded into STATES[0]."""
    CONTROL[NODE_LIMIT] = node_limit
    if CONTROL[USE_NNUE] != 0:
        # Everything deeper is reached by delta, so the root is the one place it is built.
        nnue_refresh(STATES[0], ACCUMULATORS[0], CACHE_VALUES, CACHE_BOARDS)
    return int(
        search_position(
            STATES, ACCUMULATORS, CACHE_VALUES, CACHE_BOARDS,
            MOVES, ORDER, KILLERS, HISTORY, TT_KEY, TT_DATA, PATH, CONTROL,
            max_depth, root_offset,
        )
    )


def reset() -> None:
    """Clear everything that must not leak between games. Which evaluation is in force is
    configuration rather than game state, so it survives."""
    evaluation = CONTROL[USE_NNUE]
    TT_KEY.fill(0)
    TT_DATA.fill(0)
    HISTORY.fill(0)
    KILLERS.fill(0)
    # This clears the correction table too, which lives in the tail of CONTROL.
    CONTROL.fill(0)
    CONTROL[USE_NNUE] = evaluation
