"""Tapered evaluation: material, piece-square tables, pawn structure, mobility, king safety.

Scores are centipawns from the side to move's point of view, which is what negamax wants.

Every term is computed twice, once with middlegame weights and once with endgame weights, and
the two are blended by a phase counter that falls as material comes off. Without that, a king
that correctly hides on g1 in the opening still hides there in a king-and-pawn endgame, where
it should be marching up the board.
"""

import time

import numpy as np
from numba import int64, njit, uint64

from bitboards import (
    BISHOP,
    FILES,
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
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
)
from position import OCC_ALL, OCC_WHITE, SIDE

# Piece values, middlegame and endgame. Pawns matter more as the board empties; knights and
# bishops matter less, because open positions favour the pieces that cross them quickly.
MATERIAL_MG = (0, 82, 337, 365, 477, 1025, 0)
MATERIAL_EG = (0, 94, 281, 297, 512, 936, 0)

PHASE_WEIGHT = (0, 0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24

# Tables read a8 first and h1 last, the way a board prints, so they can be checked by eye.
_PAWN_MG = (
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
)  # fmt: skip
_PAWN_EG = (
      0,   0,   0,   0,   0,   0,   0,   0,
    100, 100, 100, 100, 100, 100, 100, 100,
     60,  60,  60,  60,  60,  60,  60,  60,
     35,  35,  35,  35,  35,  35,  35,  35,
     20,  20,  20,  20,  20,  20,  20,  20,
     10,  10,  10,  10,  10,  10,  10,  10,
      5,   5,   5,   5,   5,   5,   5,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
)  # fmt: skip
_KNIGHT = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)  # fmt: skip
_BISHOP = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)  # fmt: skip
_ROOK = (
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
)  # fmt: skip
_QUEEN = (
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
)  # fmt: skip
_KING_MG = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
)  # fmt: skip
_KING_EG = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)  # fmt: skip

_TABLES_MG = (_PAWN_MG, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING_MG)
_TABLES_EG = (_PAWN_EG, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING_EG)


def _square_tables(tables: tuple[tuple[int, ...], ...], material: tuple[int, ...]) -> np.ndarray:
    """Fold material into the square tables, and mirror them for black.

    Index 0-5 is a white piece, 6-11 the black equivalent. Both are scored positively for
    their own side; the caller subtracts. A table entry written for a8 belongs to white's a1,
    so white reads it at `square ^ 56` and black reads it directly.
    """
    out = np.zeros((12, 64), dtype=np.int32)
    for piece_index, table in enumerate(tables):
        value = material[piece_index + 1]
        for square in range(64):
            out[piece_index, square] = value + table[square ^ 56]
            out[piece_index + 6, square] = value + table[square]
    return out


PST_MG = _square_tables(_TABLES_MG, MATERIAL_MG)
PST_EG = _square_tables(_TABLES_EG, MATERIAL_EG)

PHASE = np.array(PHASE_WEIGHT, dtype=np.int64)


def _pawn_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    passed = np.zeros((2, 64), dtype=np.uint64)
    isolated = np.zeros(64, dtype=np.uint64)
    shield = np.zeros((2, 64), dtype=np.uint64)
    for square in range(64):
        rank, file = divmod(square, 8)
        neighbours = 0
        for adjacent in (file - 1, file + 1):
            if 0 <= adjacent < 8:
                neighbours |= int(FILES[adjacent])
        isolated[square] = np.uint64(neighbours)

        span = neighbours | int(FILES[file])
        ahead_white = 0
        ahead_black = 0
        for r in range(rank + 1, 8):
            ahead_white |= 0xFF << (8 * r)
        for r in range(rank):
            ahead_black |= 0xFF << (8 * r)
        passed[0, square] = np.uint64(span & ahead_white)
        passed[1, square] = np.uint64(span & ahead_black)

        own_files = span
        white_shield = 0
        black_shield = 0
        for offset in (1, 2):
            if rank + offset < 8:
                white_shield |= own_files & (0xFF << (8 * (rank + offset)))
            if rank - offset >= 0:
                black_shield |= own_files & (0xFF << (8 * (rank - offset)))
        shield[0, square] = np.uint64(white_shield)
        shield[1, square] = np.uint64(black_shield)
    return passed, isolated, shield


PASSED_MASK, ISOLATED_MASK, SHIELD_MASK = _pawn_masks()

PASSED_BONUS_MG = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int64)
PASSED_BONUS_EG = np.array([0, 10, 20, 40, 70, 120, 180, 0], dtype=np.int64)

DOUBLED_PENALTY = 12
ISOLATED_PENALTY = 16
BISHOP_PAIR = 32
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN_FILE = 10
TEMPO = 10

# Mobility is worth more to a piece that has somewhere to go; a knight on the rim already
# shows up in the square table, so these stay small.
MOBILITY_MG = np.array([0, 0, 4, 5, 2, 1, 0], dtype=np.int64)
MOBILITY_EG = np.array([0, 0, 4, 5, 4, 2, 0], dtype=np.int64)

KING_ATTACK_WEIGHT = np.array([0, 0, 2, 2, 3, 5, 0], dtype=np.int64)


_SIG_evaluate = int64(uint64[::1])


@njit(nogil=True, cache=False)
def evaluate(state: np.ndarray) -> np.int64:
    middlegame = 0
    endgame = 0
    phase = 0

    occupied = state[OCC_ALL]
    white_pawns = state[PAWN - 1]
    black_pawns = state[6 + PAWN - 1]

    for colour in range(2):
        sign = 1 if colour == WHITE else -1
        base = colour * 6
        own_pawns = white_pawns if colour == WHITE else black_pawns
        enemy_pawns = black_pawns if colour == WHITE else white_pawns
        friendly = state[OCC_WHITE + colour]
        enemy_king_square = lsb(state[(1 - colour) * 6 + KING - 1])
        king_zone = KING_ATTACKS[enemy_king_square]
        pressure = 0
        attackers = 0

        for piece in range(1, 7):
            board = state[base + piece - 1]
            phase += PHASE[piece] * popcount(board)
            while board != U0:
                square = lsb(board)
                board &= board - U1
                middlegame += sign * PST_MG[base + piece - 1, square]
                endgame += sign * PST_EG[base + piece - 1, square]

                if piece == KNIGHT:
                    attacks = KNIGHT_ATTACKS[square]
                elif piece == BISHOP:
                    attacks = bishop_attacks(square, occupied)
                elif piece == ROOK:
                    attacks = rook_attacks(square, occupied)
                elif piece == QUEEN:
                    attacks = queen_attacks(square, occupied)
                elif piece == KING:
                    attacks = KING_ATTACKS[square]
                else:
                    attacks = PAWN_ATTACKS[colour, square]

                if piece != PAWN and piece != KING:
                    moves = popcount(attacks & ~friendly)
                    middlegame += sign * MOBILITY_MG[piece] * moves
                    endgame += sign * MOBILITY_EG[piece] * moves

                if piece != KING:
                    hits = popcount(attacks & king_zone)
                    if hits != 0:
                        pressure += KING_ATTACK_WEIGHT[piece] * hits
                        attackers += 1

                if piece == PAWN:
                    file_mask = FILES[square & 7]
                    if (PASSED_MASK[colour, square] & enemy_pawns) == U0:
                        rank = square >> 3 if colour == WHITE else 7 - (square >> 3)
                        middlegame += sign * PASSED_BONUS_MG[rank]
                        endgame += sign * PASSED_BONUS_EG[rank]
                    if (ISOLATED_MASK[square] & own_pawns) == U0:
                        middlegame -= sign * ISOLATED_PENALTY
                        endgame -= sign * ISOLATED_PENALTY
                    if popcount(file_mask & own_pawns) > 1:
                        middlegame -= sign * DOUBLED_PENALTY
                        endgame -= sign * DOUBLED_PENALTY

                if piece == ROOK:
                    file_mask = FILES[square & 7]
                    if (file_mask & own_pawns) == U0:
                        if (file_mask & enemy_pawns) == U0:
                            middlegame += sign * ROOK_OPEN_FILE
                        else:
                            middlegame += sign * ROOK_SEMI_OPEN_FILE

        # A single attacker is a nuisance; three converging on the same king is an attack.
        if attackers > 1:
            middlegame += sign * pressure * (attackers - 1) * 3

        if popcount(state[base + BISHOP - 1]) > 1:
            middlegame += sign * BISHOP_PAIR
            endgame += sign * BISHOP_PAIR

        own_king = lsb(state[base + KING - 1])
        shelter = popcount(SHIELD_MASK[colour, own_king] & own_pawns)
        middlegame += sign * (shelter * 8 - 16)

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (middlegame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    if np.int64(state[SIDE]) != WHITE:
        score = -score
    return score + TEMPO


_COMPILE_PAIRS = (
    (evaluate, _SIG_evaluate),
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
