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
from numba import int32, int64, njit, uint64

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
from position import (
    EN_PASSANT,
    EP,
    HALFMOVE,
    HASH,
    NFIELDS,
    NO_EP,
    SIDE,
    in_check,
    make_move,
    move_flag,
    move_from,
    move_promotion,
    move_to,
    piece_on,
)

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

PATH_LIMIT = 2048

TT_KEY = np.zeros(TT_SIZE, dtype=np.uint64)
TT_DATA = np.zeros(TT_SIZE, dtype=np.int64)
STATES = np.zeros((MAX_PLY + 8, NFIELDS), dtype=np.uint64)
MOVES = np.zeros((MAX_PLY + 8, MAX_MOVES), dtype=np.int32)
ORDER = np.zeros((MAX_PLY + 8, MAX_MOVES), dtype=np.int32)
KILLERS = np.zeros((MAX_PLY + 8, 2), dtype=np.int32)
HISTORY = np.zeros((12, 64), dtype=np.int64)
PATH = np.zeros(PATH_LIMIT + MAX_PLY + 8, dtype=np.uint64)
CONTROL = np.zeros(8, dtype=np.int64)

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)

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


@njit(int64(uint64[::1], int32, int32), nogil=True, cache=False)
def score_move(state: np.ndarray, move: np.int32, tt_move: np.int32) -> np.int64:
    """Order moves so alpha-beta sees the good ones first, which is most of its value."""
    if move == tt_move:
        return 2_000_000
    side = np.int64(state[SIDE])
    victim = PAWN if move_flag(move) == EN_PASSANT else piece_on(state, move_to(move), 1 - side)
    if victim != 0:
        attacker = piece_on(state, move_from(move), side)
        return 1_000_000 + PIECE_VALUE[victim] * 16 - PIECE_VALUE[attacker]
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
    int32(uint64[:, ::1], int32[:, ::1], int32[:, ::1], int64[::1], int64, int32, int32),
    nogil=True,
    cache=False,
)
def quiescence(
    states: np.ndarray,
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

    stand_pat = np.int32(evaluate(states[ply]))
    if ply >= MAX_PLY - 2:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    count = generate(states[ply], moves[ply], 1)
    side = np.int64(states[ply, SIDE])
    for index in range(count):
        order[ply, index] = np.int32(score_move(states[ply], moves[ply, index], np.int32(0)))

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

        if make_move(states[ply], move, states[ply + 1]) == 0:
            continue
        score = -quiescence(states, moves, order, control, ply + 1, -beta, -alpha)
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
            return np.int32(evaluate(states[ply]))

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
        return quiescence(states, moves, order, control, ply, alpha, beta)

    key = states[ply, HASH]
    slot = np.int64(key & TT_MASK)
    tt_move = np.int32(0)
    if tt_key[slot] == key:
        entry = tt_data[slot]
        tt_move = np.int32(entry & 0x3FFFF)
        stored_depth = (entry >> 38) & 0xFF
        if not root and stored_depth >= depth:
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

    pv_node = beta - alpha > 1
    static = np.int32(0) if checked != 0 else np.int32(evaluate(states[ply]))

    if not pv_node and checked == 0 and abs(beta) < MATE_THRESHOLD:
        # Reverse futility: so far ahead that conceding a few pawns would still hold beta.
        if depth <= 6 and static - np.int32(80 * depth) >= beta:
            return static
        if can_null != 0 and depth >= 3 and static >= beta and has_pieces(states[ply]) != 0:
            reduction = 2 + depth // 6
            make_null(states[ply], states[ply + 1])
            path[root_offset + ply + 1] = states[ply + 1, HASH]
            score = -negamax(
                states, moves, order, killers, history, tt_key, tt_data, path, control,
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
        rank = score_move(states[ply], move, tt_move)
        if rank == 0:
            if move == killers[ply, 0]:
                rank = 800_000
            elif move == killers[ply, 1]:
                rank = 700_000
            else:
                piece = side * 6 + piece_on(states[ply], move_from(move), side) - 1
                rank = history[piece, move_to(move)]
                if rank > 600_000:
                    rank = 600_000
        order[ply, index] = np.int32(rank)

    best_score = np.int32(-INFINITY)
    best_move = np.int32(0)
    played = 0
    original_alpha = alpha

    for index in range(count):
        move = pick_move(moves, order, ply, index, count)
        quiet = (
            move_promotion(move) == 0
            and move_flag(move) != EN_PASSANT
            and piece_on(states[ply], move_to(move), 1 - side) == 0
        )
        if make_move(states[ply], move, states[ply + 1]) == 0:
            continue
        path[root_offset + ply + 1] = states[ply + 1, HASH]
        played += 1

        if played == 1:
            score = -negamax(
                states, moves, order, killers, history, tt_key, tt_data, path, control,
                ply + 1, depth - 1, np.int32(-beta), np.int32(-alpha), root_offset, 1,
            )
        else:
            reduction = np.int64(0)
            if depth >= 3 and played >= 4 and quiet and checked == 0:
                reduction = LMR[min(depth, 63), min(played, 63)]
                if pv_node and reduction > 0:
                    reduction -= 1
                if reduction > depth - 2:
                    reduction = depth - 2
                if reduction < 0:
                    reduction = 0
            score = -negamax(
                states, moves, order, killers, history, tt_key, tt_data, path, control,
                ply + 1, depth - 1 - reduction, np.int32(-alpha - 1), np.int32(-alpha),
                root_offset, 1,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    states, moves, order, killers, history, tt_key, tt_data, path, control,
                    ply + 1, depth - 1, np.int32(-alpha - 1), np.int32(-alpha), root_offset, 1,
                )
            if score > alpha and score < beta:
                score = -negamax(
                    states, moves, order, killers, history, tt_key, tt_data, path, control,
                    ply + 1, depth - 1, np.int32(-beta), np.int32(-alpha), root_offset, 1,
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
                        piece = side * 6 + piece_on(states[ply], move_from(move), side) - 1
                        history[piece, move_to(move)] += depth * depth
                        if history[piece, move_to(move)] > 1_000_000:
                            for a in range(12):
                                for b in range(64):
                                    history[a, b] >>= 1
                    break

    if played == 0:
        return np.int32(-MATE + ply) if checked != 0 else np.int32(0)

    stored = best_score
    if stored > MATE_THRESHOLD:
        stored += np.int32(ply)
    elif stored < -MATE_THRESHOLD:
        stored -= np.int32(ply)
    if tt_key[slot] != key or depth >= ((tt_data[slot] >> 38) & 0xFF):
        if best_score >= beta:
            flag = LOWER
        elif best_score > original_alpha:
            flag = EXACT
        else:
            flag = UPPER
        tt_key[slot] = key
        tt_data[slot] = pack(best_move, stored, depth, flag)

    return best_score


@njit(
    int64(
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

    for index in range(killers.shape[0]):
        killers[index, 0] = 0
        killers[index, 1] = 0
    for a in range(12):
        for b in range(64):
            history[a, b] >>= 3

    path[root_offset] = states[0, HASH]

    best_move = np.int64(0)
    best_score = np.int32(0)
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
                states, moves, order, killers, history, tt_key, tt_data, path, control,
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
        if abs(best_score) > MATE_THRESHOLD:
            break

    if best_move == 0:
        best_move = control[BEST_MOVE]
    return best_move


def run(max_depth: int, root_offset: int, node_limit: int) -> int:
    """Search the position already loaded into STATES[0]."""
    CONTROL[NODE_LIMIT] = node_limit
    return int(
        search_position(
            STATES, MOVES, ORDER, KILLERS, HISTORY, TT_KEY, TT_DATA, PATH, CONTROL,
            max_depth, root_offset,
        )
    )


def reset() -> None:
    """Clear everything that must not leak between games."""
    TT_KEY.fill(0)
    TT_DATA.fill(0)
    HISTORY.fill(0)
    KILLERS.fill(0)
    CONTROL.fill(0)
