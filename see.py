"""Static exchange evaluation: is a capture worth making, without searching it?

Given a move onto a square, both sides keep recapturing there with their cheapest attacker until
one of them declines. SEE plays that sequence out on bitboards and reports the material outcome.
It is not a search; there is no move generation and no recursion, just a swap-off on one square.

The search needs it in two places. Quiescence currently tries every capture, including ones that
simply lose material, and each of those costs a node and its whole subtree. And move ordering
currently sorts captures by MVV-LVA, which says a pawn taking a queen is wonderful without
noticing the queen is defended. Measured over thirty games against the anchor, we played 28
captures that lost material and declined 16 that won it.

The form here is `see_ge`, asking whether the exchange is worth at least a threshold, rather
than computing an exact value. That is what the callers actually need and it allows two early
exits that skip the loop entirely in the common cases: winning the target outright is not enough
to reach the threshold, or losing the moving piece still clears it.

X-rays matter and are handled: when a piece steps off a line, a rook or bishop behind it joins
the exchange, so the attacker set is recomputed against the shrinking occupancy each time.
"""

import numpy as np
from numba import int32, int64, njit, uint64

from bitboards import (
    BISHOP,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    ROOK,
    U0,
    U1,
    WHITE,
    bishop_attacks,
    rook_attacks,
)
from position import (
    EN_PASSANT,
    OCC_ALL,
    OCC_WHITE,
    SIDE,
    move_flag,
    move_from,
    move_promotion,
    move_to,
    piece_on,
)

# Deliberately plain values. SEE decides whether an exchange is worth entering, and piece-square
# subtleties do not survive a sequence of recaptures; the king's value only has to be large
# enough that offering it is never attractive.
VALUE = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)


@njit(uint64(uint64[::1], int64, uint64), nogil=True, cache=False)
def attackers_to(state: np.ndarray, square: np.int64, occupied: np.uint64) -> np.uint64:
    """Every piece of either colour attacking a square, for a given occupancy.

    Occupancy is a parameter rather than read from the position because the exchange removes
    pieces as it goes, and each removal can expose a slider behind it.
    """
    attackers = PAWN_ATTACKS[1, square] & state[PAWN - 1]
    attackers |= PAWN_ATTACKS[0, square] & state[6 + PAWN - 1]
    attackers |= KNIGHT_ATTACKS[square] & (state[KNIGHT - 1] | state[6 + KNIGHT - 1])
    attackers |= KING_ATTACKS[square] & (state[KING - 1] | state[6 + KING - 1])
    diagonal = (
        state[BISHOP - 1] | state[QUEEN - 1] | state[6 + BISHOP - 1] | state[6 + QUEEN - 1]
    )
    attackers |= bishop_attacks(square, occupied) & diagonal
    straight = state[ROOK - 1] | state[QUEEN - 1] | state[6 + ROOK - 1] | state[6 + QUEEN - 1]
    attackers |= rook_attacks(square, occupied) & straight
    return attackers & occupied


@njit(int64(uint64[::1], int32, int64), nogil=True, cache=False)
def see_ge(state: np.ndarray, move: np.int32, threshold: np.int64) -> np.int64:
    """Whether the exchange starting with this move is worth at least `threshold`."""
    origin = move_from(move)
    target = move_to(move)
    side = np.int64(state[SIDE])

    captured = PAWN if move_flag(move) == EN_PASSANT else piece_on(state, target, 1 - side)

    # The promotion has to be counted before the early exit, not after. A pawn taking a rook
    # and promoting is worth the rook plus the difference between a queen and a pawn, and
    # testing the threshold against the rook alone rejects the move at anything above 500.
    gain = VALUE[captured]
    moving = piece_on(state, origin, side)
    promotion = move_promotion(move)
    if promotion != 0:
        gain += VALUE[promotion] - VALUE[PAWN]
        # It is the promoted piece standing on the square, so it is what can be recaptured.
        moving = promotion

    swap = gain - threshold
    if swap < 0:
        # Winning everything on the square still falls short.
        return 0

    swap = VALUE[moving] - swap
    if swap <= 0:
        # Losing the moving piece outright still clears the threshold.
        return 1

    occupied = state[OCC_ALL] ^ (U1 << np.uint64(origin)) ^ (U1 << np.uint64(target))
    if move_flag(move) == EN_PASSANT:
        captured_square = target - 8 if side == WHITE else target + 8
        occupied ^= U1 << np.uint64(captured_square)

    attackers = attackers_to(state, target, occupied)
    result = 1

    while True:
        side = 1 - side
        attackers &= occupied
        mine = attackers & state[OCC_WHITE + side]
        if mine == U0:
            break

        # Recapture with the cheapest piece available; a costlier one only loses more.
        piece = 0
        for candidate in range(1, 7):
            board = mine & state[side * 6 + candidate - 1]
            if board != U0:
                piece = candidate
                break

        if piece == KING:
            # The king may only take if the square is not still defended, so whether this is
            # legal decides who ends up owning the square.
            if (attackers & state[OCC_WHITE + (1 - side)]) != U0:
                break
            result ^= 1
            break

        result ^= 1
        swap = VALUE[piece] - swap
        if swap < result:
            break

        occupied ^= board & (~board + U1)
        if piece in (PAWN, BISHOP, QUEEN):
            diagonal = (
                state[BISHOP - 1] | state[QUEEN - 1] | state[6 + BISHOP - 1] | state[6 + QUEEN - 1]
            )
            attackers |= bishop_attacks(target, occupied) & diagonal
        if piece in (ROOK, QUEEN):
            straight = (
                state[ROOK - 1] | state[QUEEN - 1] | state[6 + ROOK - 1] | state[6 + QUEEN - 1]
            )
            attackers |= rook_attacks(target, occupied) & straight

    return result


@njit(int64(uint64[::1], int32), nogil=True, cache=False)
def see_value(state: np.ndarray, move: np.int32) -> np.int64:
    """The exchange value in centipawns, by binary search over see_ge.

    Only for tests and diagnostics. The search never needs a number, and asking for one costs
    a dozen calls where a threshold test costs one.
    """
    low = -VALUE[QUEEN] - VALUE[ROOK]
    high = VALUE[QUEEN] + VALUE[ROOK]
    while low < high:
        middle = (low + high + 1) // 2
        if see_ge(state, move, middle) != 0:
            low = middle
        else:
            high = middle - 1
    return low
