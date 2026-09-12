"""Pseudo-legal move generation.

Moves are generated pseudo-legally and filtered by `make_move`, which rejects anything that
leaves the mover's king in check. Generating strictly legal moves directly is faster but needs
pin and check-evasion masks to be exactly right; letting make_move be the single authority on
legality keeps one rule in one place.

`captures_only` drives quiescence search: it yields captures, en passant and queen promotions,
and skips quiet moves and castling.
"""

import time

import numpy as np
from numba import int32, int64, njit, uint64

from bitboards import (
    BISHOP,
    FILE_A,
    FILE_H,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    RANK_1,
    RANK_4,
    RANK_5,
    RANK_8,
    ROOK,
    U0,
    U1,
    WHITE,
    bishop_attacks,
    lsb,
    queen_attacks,
    rook_attacks,
)
from position import (
    C1,
    C8,
    CASTLE,
    CASTLE_MOVE,
    D1,
    D8,
    DOUBLE_PUSH,
    E1,
    E8,
    EN_PASSANT,
    EP,
    F1,
    F8,
    G1,
    G8,
    NO_EP,
    NORMAL,
    OCC_ALL,
    OCC_WHITE,
    SIDE,
    encode,
    is_attacked,
)

MAX_MOVES = 256


_SIG_generate = int64(uint64[::1], int32[::1], int64)


@njit(nogil=True, cache=False)
def generate(state: np.ndarray, moves: np.ndarray, captures_only: np.int64) -> np.int64:
    count = 0
    side = np.int64(state[SIDE])
    base = side * 6
    us = state[OCC_WHITE + side]
    them = state[OCC_WHITE + (1 - side)]
    occupied = state[OCC_ALL]
    empty = ~occupied
    targets = them if captures_only != 0 else ~us

    pawns = state[base + PAWN - 1]
    if side == WHITE:
        forward = np.int64(-8)
        single = (pawns << np.uint64(8)) & empty
        double = (single << np.uint64(8)) & empty & RANK_4
        left = (pawns << np.uint64(7)) & ~FILE_H & them
        right = (pawns << np.uint64(9)) & ~FILE_A & them
        left_from = np.int64(-7)
        right_from = np.int64(-9)
        last_rank = RANK_8
    else:
        forward = np.int64(8)
        single = (pawns >> np.uint64(8)) & empty
        double = (single >> np.uint64(8)) & empty & RANK_5
        left = (pawns >> np.uint64(7)) & ~FILE_A & them
        right = (pawns >> np.uint64(9)) & ~FILE_H & them
        left_from = np.int64(7)
        right_from = np.int64(9)
        last_rank = RANK_1

    promoting = single & last_rank
    while promoting != U0:
        target = lsb(promoting)
        promoting &= promoting - U1
        origin = target + forward
        moves[count] = encode(origin, target, QUEEN, NORMAL)
        count += 1
        if captures_only == 0:
            moves[count] = encode(origin, target, ROOK, NORMAL)
            count += 1
            moves[count] = encode(origin, target, BISHOP, NORMAL)
            count += 1
            moves[count] = encode(origin, target, KNIGHT, NORMAL)
            count += 1

    if captures_only == 0:
        quiet = single & ~last_rank
        while quiet != U0:
            target = lsb(quiet)
            quiet &= quiet - U1
            moves[count] = encode(target + forward, target, 0, NORMAL)
            count += 1
        while double != U0:
            target = lsb(double)
            double &= double - U1
            moves[count] = encode(target + forward + forward, target, 0, DOUBLE_PUSH)
            count += 1

    for capture, delta in ((left, left_from), (right, right_from)):
        board = capture
        while board != U0:
            target = lsb(board)
            board &= board - U1
            origin = target + delta
            if (U1 << np.uint64(target)) & last_rank != U0:
                moves[count] = encode(origin, target, QUEEN, NORMAL)
                count += 1
                if captures_only == 0:
                    moves[count] = encode(origin, target, ROOK, NORMAL)
                    count += 1
                    moves[count] = encode(origin, target, BISHOP, NORMAL)
                    count += 1
                    moves[count] = encode(origin, target, KNIGHT, NORMAL)
                    count += 1
            else:
                moves[count] = encode(origin, target, 0, NORMAL)
                count += 1

    ep_square = np.int64(state[EP])
    if ep_square != NO_EP:
        capturers = PAWN_ATTACKS[1 - side, ep_square] & pawns
        while capturers != U0:
            origin = lsb(capturers)
            capturers &= capturers - U1
            moves[count] = encode(origin, ep_square, 0, EN_PASSANT)
            count += 1

    knights = state[base + KNIGHT - 1]
    while knights != U0:
        origin = lsb(knights)
        knights &= knights - U1
        attacks = KNIGHT_ATTACKS[origin] & targets
        while attacks != U0:
            target = lsb(attacks)
            attacks &= attacks - U1
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    bishops = state[base + BISHOP - 1]
    while bishops != U0:
        origin = lsb(bishops)
        bishops &= bishops - U1
        attacks = bishop_attacks(origin, occupied) & targets
        while attacks != U0:
            target = lsb(attacks)
            attacks &= attacks - U1
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    rooks = state[base + ROOK - 1]
    while rooks != U0:
        origin = lsb(rooks)
        rooks &= rooks - U1
        attacks = rook_attacks(origin, occupied) & targets
        while attacks != U0:
            target = lsb(attacks)
            attacks &= attacks - U1
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    queens = state[base + QUEEN - 1]
    while queens != U0:
        origin = lsb(queens)
        queens &= queens - U1
        attacks = queen_attacks(origin, occupied) & targets
        while attacks != U0:
            target = lsb(attacks)
            attacks &= attacks - U1
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    king = state[base + KING - 1]
    king_square = lsb(king)
    attacks = KING_ATTACKS[king_square] & targets
    while attacks != U0:
        target = lsb(attacks)
        attacks &= attacks - U1
        moves[count] = encode(king_square, target, 0, NORMAL)
        count += 1

    if captures_only == 0:
        rights = np.int64(state[CASTLE])
        them_colour = 1 - side
        if side == WHITE:
            if (
                (rights & 0b0001) != 0
                and (occupied & ((U1 << np.uint64(F1)) | (U1 << np.uint64(G1)))) == U0
                and is_attacked(state, E1, them_colour) == 0
                and is_attacked(state, F1, them_colour) == 0
            ):
                moves[count] = encode(E1, G1, 0, CASTLE_MOVE)
                count += 1
            if (
                (rights & 0b0010) != 0
                and (
                    occupied
                    & ((U1 << np.uint64(D1)) | (U1 << np.uint64(C1)) | (U1 << np.uint64(1)))
                )
                == U0
                and is_attacked(state, E1, them_colour) == 0
                and is_attacked(state, D1, them_colour) == 0
            ):
                moves[count] = encode(E1, C1, 0, CASTLE_MOVE)
                count += 1
        else:
            if (
                (rights & 0b0100) != 0
                and (occupied & ((U1 << np.uint64(F8)) | (U1 << np.uint64(G8)))) == U0
                and is_attacked(state, E8, them_colour) == 0
                and is_attacked(state, F8, them_colour) == 0
            ):
                moves[count] = encode(E8, G8, 0, CASTLE_MOVE)
                count += 1
            if (
                (rights & 0b1000) != 0
                and (
                    occupied
                    & ((U1 << np.uint64(D8)) | (U1 << np.uint64(C8)) | (U1 << np.uint64(57)))
                )
                == U0
                and is_attacked(state, E8, them_colour) == 0
                and is_attacked(state, D8, them_colour) == 0
            ):
                moves[count] = encode(E8, C8, 0, CASTLE_MOVE)
                count += 1

    return count


_COMPILE_PAIRS = (
    (generate, _SIG_generate),
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
