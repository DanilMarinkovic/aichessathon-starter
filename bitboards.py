"""Bitboard primitives, attack tables and Zobrist keys.

Squares run 0 = a1 to 63 = h8, matching python-chess, so a square index converts straight to
UCI without a translation layer.

Everything here is built during import, inside the 60 second budget. numba freezes global
arrays into the code it emits at compile time, so the tables are constructed at the top of the
module and the jitted functions that read them are defined below: nothing may be rebound after
a reader has compiled.

Bitboards are uint64 throughout. numba follows numpy promotion rules, so mixing a uint64 with
an untyped Python literal yields a float. Every literal in jitted code is wrapped accordingly.
"""

import time

import numpy as np
from numba import int64, njit, uint64

WHITE = 0
BLACK = 1

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

U0 = np.uint64(0)
U1 = np.uint64(1)

MASK64 = 0xFFFF_FFFF_FFFF_FFFF

FILE_A = np.uint64(0x0101_0101_0101_0101)
FILE_H = np.uint64(0x8080_8080_8080_8080)
RANK_1 = np.uint64(0x0000_0000_0000_00FF)
RANK_2 = np.uint64(0x0000_0000_0000_FF00)
RANK_4 = np.uint64(0x0000_0000_FF00_0000)
RANK_5 = np.uint64(0x0000_00FF_0000_0000)
RANK_7 = np.uint64(0x00FF_0000_0000_0000)
RANK_8 = np.uint64(0xFF00_0000_0000_0000)

FILES = np.array([0x0101_0101_0101_0101 << f for f in range(8)], dtype=np.uint64)
RANKS = np.array([0xFF << (8 * r) for r in range(8)], dtype=np.uint64)

ROOK_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DELTAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_DELTAS = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
KING_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _steps(square: int, deltas: tuple[tuple[int, int], ...]) -> int:
    """Single-step attacks from a square, for knights, kings and pawns."""
    rank, file = divmod(square, 8)
    board = 0
    for d_rank, d_file in deltas:
        r, f = rank + d_rank, file + d_file
        if 0 <= r < 8 and 0 <= f < 8:
            board |= 1 << (r * 8 + f)
    return board


def _rays(square: int, occupied: int, deltas: tuple[tuple[int, int], ...]) -> int:
    """Sliding attacks, stopping on and including the first blocker in each direction."""
    rank, file = divmod(square, 8)
    board = 0
    for d_rank, d_file in deltas:
        r, f = rank + d_rank, file + d_file
        while 0 <= r < 8 and 0 <= f < 8:
            board |= 1 << (r * 8 + f)
            if occupied & (1 << (r * 8 + f)):
                break
            r, f = r + d_rank, f + d_file
    return board


def _relevance(square: int, deltas: tuple[tuple[int, int], ...]) -> int:
    """The squares whose occupancy changes a slider's attacks: the ray minus its final square."""
    rank, file = divmod(square, 8)
    board = 0
    for d_rank, d_file in deltas:
        r, f = rank + d_rank, file + d_file
        while 0 <= r < 8 and 0 <= f < 8:
            if not (0 <= r + d_rank < 8 and 0 <= f + d_file < 8):
                break
            board |= 1 << (r * 8 + f)
            r, f = r + d_rank, f + d_file
    return board


def _subsets(mask: int) -> list[int]:
    """Every subset of a mask, by the carry-rippler trick."""
    out: list[int] = []
    subset = 0
    while True:
        out.append(subset)
        subset = (subset - mask) & mask
        if subset == 0:
            break
    return out


KNIGHT_ATTACKS = np.array([_steps(s, KNIGHT_DELTAS) for s in range(64)], dtype=np.uint64)
KING_ATTACKS = np.array([_steps(s, KING_DELTAS) for s in range(64)], dtype=np.uint64)
PAWN_ATTACKS = np.array(
    [
        [_steps(s, ((1, 1), (1, -1))) for s in range(64)],
        [_steps(s, ((-1, 1), (-1, -1))) for s in range(64)],
    ],
    dtype=np.uint64,
)

ROOK_MASK = np.array([_relevance(s, ROOK_DELTAS) for s in range(64)], dtype=np.uint64)
BISHOP_MASK = np.array([_relevance(s, BISHOP_DELTAS) for s in range(64)], dtype=np.uint64)
ROOK_BITS = np.array([bin(int(m)).count("1") for m in ROOK_MASK], dtype=np.int64)
BISHOP_BITS = np.array([bin(int(m)).count("1") for m in BISHOP_MASK], dtype=np.int64)
ROOK_SHIFT = (64 - ROOK_BITS).astype(np.uint64)
BISHOP_SHIFT = (64 - BISHOP_BITS).astype(np.uint64)

_DEBRUIJN = 0x03F7_9D71_B4CB_0A89
DEBRUIJN = np.uint64(_DEBRUIJN)
DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    DEBRUIJN_INDEX[(((1 << _i) * _DEBRUIJN) & MASK64) >> 58] = _i


_SIG_popcount = int64(uint64)


@njit(nogil=True, cache=False)
def popcount(board: np.uint64) -> np.int64:
    x = board
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555_5555_5555_5555))
    x = (x & np.uint64(0x3333_3333_3333_3333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333_3333_3333_3333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F_0F0F_0F0F_0F0F)
    return np.int64((x * np.uint64(0x0101_0101_0101_0101)) >> np.uint64(56))


_SIG_lsb = int64(uint64)


@njit(nogil=True, cache=False)
def lsb(board: np.uint64) -> np.int64:
    """Index of the lowest set bit. Undefined for an empty board, never called with one."""
    isolated = board & (~board + U1)
    return DEBRUIJN_INDEX[np.int64((isolated * DEBRUIJN) >> np.uint64(58))]


_SIG__random = uint64(uint64[:])


@njit(nogil=True, cache=False)
def _random(state: np.ndarray) -> np.uint64:
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


_SIG__find_magic = uint64(uint64, uint64[:], uint64[:], int64, uint64[:])


@njit(nogil=True, cache=False)
def _find_magic(
    mask: np.uint64,
    occupancies: np.ndarray,
    references: np.ndarray,
    bits: np.int64,
    state: np.ndarray,
) -> np.uint64:
    """Search for a multiplier that maps every occupancy of a mask onto a collision-free index.

    Collisions are allowed when both occupancies produce the same attack set, which is what
    makes the tables small enough to be worth building.
    """
    size = 1 << bits
    shift = np.uint64(64 - bits)
    table = np.zeros(size, dtype=np.uint64)
    seen = np.zeros(size, dtype=np.int64)
    epoch = 0
    while True:
        magic = _random(state) & _random(state) & _random(state)
        if popcount((mask * magic) >> np.uint64(56)) < 6:
            continue
        epoch += 1
        good = True
        for i in range(size):
            index = np.int64((occupancies[i] * magic) >> shift)
            if seen[index] != epoch:
                seen[index] = epoch
                table[index] = references[i]
            elif table[index] != references[i]:
                good = False
                break
        if good:
            return magic


def _build_magics(
    masks: np.ndarray, bits: np.ndarray, deltas: tuple[tuple[int, int], ...], width: int
) -> tuple[np.ndarray, np.ndarray]:
    magics = np.zeros(64, dtype=np.uint64)
    table = np.zeros((64, width), dtype=np.uint64)
    state = np.array([0x0123_4567_89AB_CDEF], dtype=np.uint64)
    for square in range(64):
        mask = int(masks[square])
        occupancies = _subsets(mask)
        references = [_rays(square, occupancy, deltas) for occupancy in occupancies]
        magic = _find_magic(
            np.uint64(mask),
            np.array(occupancies, dtype=np.uint64),
            np.array(references, dtype=np.uint64),
            int(bits[square]),
            state,
        )
        magics[square] = magic
        shift = 64 - int(bits[square])
        for occupancy, reference in zip(occupancies, references, strict=True):
            table[square, ((occupancy * int(magic)) & MASK64) >> shift] = reference
    return magics, table


ROOK_MAGIC, ROOK_TABLE = _build_magics(ROOK_MASK, ROOK_BITS, ROOK_DELTAS, 1 << 12)
BISHOP_MAGIC, BISHOP_TABLE = _build_magics(BISHOP_MASK, BISHOP_BITS, BISHOP_DELTAS, 1 << 9)


_SIG_rook_attacks = uint64(int64, uint64)


@njit(nogil=True, cache=False)
def rook_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    index = ((occupied & ROOK_MASK[square]) * ROOK_MAGIC[square]) >> ROOK_SHIFT[square]
    return ROOK_TABLE[square, np.int64(index)]


_SIG_bishop_attacks = uint64(int64, uint64)


@njit(nogil=True, cache=False)
def bishop_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    index = ((occupied & BISHOP_MASK[square]) * BISHOP_MAGIC[square]) >> BISHOP_SHIFT[square]
    return BISHOP_TABLE[square, np.int64(index)]


_SIG_queen_attacks = uint64(int64, uint64)


@njit(nogil=True, cache=False)
def queen_attacks(square: np.int64, occupied: np.uint64) -> np.uint64:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


_zobrist = np.random.default_rng(0x5EED).integers(
    0, 1 << 63, size=(12 * 64) + 16 + 8 + 1, dtype=np.uint64
)
ZOBRIST_PIECE = np.ascontiguousarray(_zobrist[: 12 * 64].reshape(12, 64))
ZOBRIST_CASTLE = np.ascontiguousarray(_zobrist[12 * 64 : 12 * 64 + 16])
ZOBRIST_EP = np.ascontiguousarray(_zobrist[12 * 64 + 16 : 12 * 64 + 24])
ZOBRIST_SIDE = np.uint64(_zobrist[-1])


_COMPILE_PAIRS = (
    (popcount, _SIG_popcount),
    (lsb, _SIG_lsb),
    (_random, _SIG__random),
    (_find_magic, _SIG__find_magic),
    (rook_attacks, _SIG_rook_attacks),
    (bishop_attacks, _SIG_bishop_attacks),
    (queen_attacks, _SIG_queen_attacks),
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
