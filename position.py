"""Position state, move encoding, and making a move.

A position is a flat uint64 array so that numba can pass it around without boxing. The search
uses copy-make: making a move writes a whole new state one ply deeper rather than mutating and
undoing. That costs 160 bytes of copying per node and buys the absence of a whole class of
undo bugs, which is the right trade when an illegal move loses the game.

    0-5    white pawn, knight, bishop, rook, queen, king
    6-11   black, same order
    12-14  white, black and combined occupancy
    15     side to move
    16     castling rights, one bit each: white king, white queen, black king, black queen
    17     en passant target square, or 64 for none
    18     halfmove clock
    19     Zobrist key

A move is packed into an int32:

    bits 0-5    origin square
    bits 6-11   destination square
    bits 12-14  promotion piece type, 0 when the move is not a promotion
    bits 15-17  NORMAL, DOUBLE_PUSH, EN_PASSANT or CASTLE
"""

import chess
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
    ZOBRIST_CASTLE,
    ZOBRIST_EP,
    ZOBRIST_PIECE,
    ZOBRIST_SIDE,
    bishop_attacks,
    lsb,
    rook_attacks,
)

NFIELDS = 20
OCC_WHITE = 12
OCC_BLACK = 13
OCC_ALL = 14
SIDE = 15
CASTLE = 16
EP = 17
HALFMOVE = 18
HASH = 19

NO_EP = 64

NORMAL = 0
DOUBLE_PUSH = 1
EN_PASSANT = 2
CASTLE_MOVE = 3

A1, C1, D1, E1, F1, G1, H1 = 0, 2, 3, 4, 5, 6, 7
A8, C8, D8, E8, F8, G8, H8 = 56, 58, 59, 60, 61, 62, 63

# Moving from or to any of these squares removes the castling right that depends on it.
_CASTLE_MASK = np.full(64, 0b1111, dtype=np.int64)
_CASTLE_MASK[E1] = 0b1100
_CASTLE_MASK[H1] = 0b1110
_CASTLE_MASK[A1] = 0b1101
_CASTLE_MASK[E8] = 0b0011
_CASTLE_MASK[H8] = 0b1011
_CASTLE_MASK[A8] = 0b0111
CASTLE_MASK = _CASTLE_MASK


@njit(int32(int64, int64, int64, int64), nogil=True, cache=False, inline="always")
def encode(origin: np.int64, target: np.int64, promotion: np.int64, flag: np.int64) -> np.int32:
    return np.int32(origin | (target << 6) | (promotion << 12) | (flag << 15))


@njit(int64(int32), nogil=True, cache=False, inline="always")
def move_from(move: np.int32) -> np.int64:
    return np.int64(move & 63)


@njit(int64(int32), nogil=True, cache=False, inline="always")
def move_to(move: np.int32) -> np.int64:
    return np.int64((move >> 6) & 63)


@njit(int64(int32), nogil=True, cache=False, inline="always")
def move_promotion(move: np.int32) -> np.int64:
    return np.int64((move >> 12) & 7)


@njit(int64(int32), nogil=True, cache=False, inline="always")
def move_flag(move: np.int32) -> np.int64:
    return np.int64((move >> 15) & 7)


@njit(int64(uint64[::1], int64, int64), nogil=True, cache=False)
def piece_on(state: np.ndarray, square: np.int64, colour: np.int64) -> np.int64:
    """The piece type a colour has on a square, or 0 when it has none."""
    board = U1 << np.uint64(square)
    base = colour * 6
    for index in range(6):
        if (state[base + index] & board) != U0:
            return index + 1
    return 0


@njit(int64(uint64[::1], int64, int64), nogil=True, cache=False)
def is_attacked(state: np.ndarray, square: np.int64, by: np.int64) -> np.int64:
    """Whether `by` attacks a square. Used for legality, castling and check detection."""
    base = by * 6
    occupied = state[OCC_ALL]
    if (PAWN_ATTACKS[1 - by, square] & state[base + PAWN - 1]) != U0:
        return 1
    if (KNIGHT_ATTACKS[square] & state[base + KNIGHT - 1]) != U0:
        return 1
    if (KING_ATTACKS[square] & state[base + KING - 1]) != U0:
        return 1
    diagonal = state[base + BISHOP - 1] | state[base + QUEEN - 1]
    if (bishop_attacks(square, occupied) & diagonal) != U0:
        return 1
    straight = state[base + ROOK - 1] | state[base + QUEEN - 1]
    if (rook_attacks(square, occupied) & straight) != U0:
        return 1
    return 0


@njit(int64(uint64[::1]), nogil=True, cache=False)
def in_check(state: np.ndarray) -> np.int64:
    side = np.int64(state[SIDE])
    king = lsb(state[side * 6 + KING - 1])
    return is_attacked(state, king, 1 - side)


@njit(int64(uint64[::1], int32, uint64[::1]), nogil=True, cache=False)
def make_move(state: np.ndarray, move: np.int32, out: np.ndarray) -> np.int64:
    """Play a pseudo-legal move into `out`. Returns 0 if it left the mover's king in check.

    Moves are generated pseudo-legally and filtered here, so the caller must treat a 0 as
    "skip this move" rather than as an error.
    """
    for index in range(NFIELDS):
        out[index] = state[index]

    origin = move_from(move)
    target = move_to(move)
    promotion = move_promotion(move)
    flag = move_flag(move)

    side = np.int64(state[SIDE])
    them = 1 - side
    base = side * 6
    other = them * 6

    key = state[HASH]
    origin_bb = U1 << np.uint64(origin)
    target_bb = U1 << np.uint64(target)

    moved = 0
    for index in range(6):
        if (out[base + index] & origin_bb) != U0:
            moved = index + 1
            break

    previous_ep = np.int64(state[EP])
    if previous_ep != NO_EP:
        key ^= ZOBRIST_EP[previous_ep & 7]

    captured = 0
    if flag == EN_PASSANT:
        captured_square = target - 8 if side == WHITE else target + 8
        out[other] &= ~(U1 << np.uint64(captured_square))
        key ^= ZOBRIST_PIECE[other, captured_square]
        captured = PAWN
    elif (out[OCC_WHITE + them] & target_bb) != U0:
        for index in range(6):
            if (out[other + index] & target_bb) != U0:
                out[other + index] &= ~target_bb
                key ^= ZOBRIST_PIECE[other + index, target]
                captured = index + 1
                break

    out[base + moved - 1] &= ~origin_bb
    key ^= ZOBRIST_PIECE[base + moved - 1, origin]
    if promotion != 0:
        out[base + promotion - 1] |= target_bb
        key ^= ZOBRIST_PIECE[base + promotion - 1, target]
    else:
        out[base + moved - 1] |= target_bb
        key ^= ZOBRIST_PIECE[base + moved - 1, target]

    if flag == CASTLE_MOVE:
        if target == G1:
            rook_origin, rook_target = np.int64(H1), np.int64(F1)
        elif target == C1:
            rook_origin, rook_target = np.int64(A1), np.int64(D1)
        elif target == G8:
            rook_origin, rook_target = np.int64(H8), np.int64(F8)
        else:
            rook_origin, rook_target = np.int64(A8), np.int64(D8)
        out[base + ROOK - 1] ^= (U1 << np.uint64(rook_origin)) | (U1 << np.uint64(rook_target))
        key ^= ZOBRIST_PIECE[base + ROOK - 1, rook_origin]
        key ^= ZOBRIST_PIECE[base + ROOK - 1, rook_target]

    rights = np.int64(state[CASTLE])
    key ^= ZOBRIST_CASTLE[rights]
    rights &= CASTLE_MASK[origin] & CASTLE_MASK[target]
    key ^= ZOBRIST_CASTLE[rights]
    out[CASTLE] = np.uint64(rights)

    if flag == DOUBLE_PUSH:
        ep_square = (origin + target) >> 1
        out[EP] = np.uint64(ep_square)
        key ^= ZOBRIST_EP[ep_square & 7]
    else:
        out[EP] = np.uint64(NO_EP)

    if moved == PAWN or captured != 0:
        out[HALFMOVE] = U0
    else:
        out[HALFMOVE] = state[HALFMOVE] + U1

    white = out[0] | out[1] | out[2] | out[3] | out[4] | out[5]
    black = out[6] | out[7] | out[8] | out[9] | out[10] | out[11]
    out[OCC_WHITE] = white
    out[OCC_BLACK] = black
    out[OCC_ALL] = white | black

    out[SIDE] = np.uint64(them)
    key ^= ZOBRIST_SIDE
    out[HASH] = key

    if is_attacked(out, lsb(out[base + KING - 1]), them) != 0:
        return 0
    return 1


@njit(uint64(uint64[::1]), nogil=True, cache=False)
def compute_hash(state: np.ndarray) -> np.uint64:
    """Recompute the Zobrist key from scratch. Only used to check the incremental one."""
    key = U0
    for index in range(12):
        board = state[index]
        while board != U0:
            square = lsb(board)
            key ^= ZOBRIST_PIECE[index, square]
            board &= board - U1
    key ^= ZOBRIST_CASTLE[np.int64(state[CASTLE])]
    if np.int64(state[EP]) != NO_EP:
        key ^= ZOBRIST_EP[np.int64(state[EP]) & 7]
    if np.int64(state[SIDE]) != WHITE:
        key ^= ZOBRIST_SIDE
    return key


_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def from_board(board: chess.Board) -> np.ndarray:
    """Build a state from a python-chess board.

    Parsing goes through python-chess rather than a hand-written FEN reader: it runs once per
    move rather than in the search, and it removes any chance of the engine and the referee
    disagreeing about what the position is.
    """
    state = np.zeros(NFIELDS, dtype=np.uint64)
    for colour_index, colour in enumerate((chess.WHITE, chess.BLACK)):
        for piece_index, piece in enumerate(_PIECE_ORDER):
            state[colour_index * 6 + piece_index] = np.uint64(
                int(board.pieces_mask(piece, colour))
            )
    white = np.uint64(int(board.occupied_co[chess.WHITE]))
    black = np.uint64(int(board.occupied_co[chess.BLACK]))
    state[OCC_WHITE] = white
    state[OCC_BLACK] = black
    state[OCC_ALL] = white | black
    state[SIDE] = np.uint64(0 if board.turn == chess.WHITE else 1)

    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= 0b0001
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= 0b0010
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= 0b0100
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= 0b1000
    state[CASTLE] = np.uint64(rights)

    state[EP] = np.uint64(NO_EP if board.ep_square is None else board.ep_square)
    state[HALFMOVE] = np.uint64(board.halfmove_clock)
    state[HASH] = compute_hash(state)
    return state


def to_uci(move: int) -> str:
    """Render a packed move as UCI, for handing back to the referee."""
    origin = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    text = chess.square_name(origin) + chess.square_name(target)
    if promotion:
        text += chess.piece_symbol(_PIECE_ORDER[promotion - 1]) or ""
    return text
