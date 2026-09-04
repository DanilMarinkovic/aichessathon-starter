"""Validate the magic attack tables against python-chess for random occupancies."""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

start = time.perf_counter()
from bitboards import (  # noqa: E402
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
)

print(f"import built every table in {time.perf_counter() - start:.2f}s")

rng = random.Random(7)
checked = 0
for _ in range(400):
    occupied = 0
    for _ in range(rng.randint(0, 24)):
        occupied |= 1 << rng.randrange(64)
    occupancy = np.uint64(occupied)
    for square in range(64):
        rank = chess.BB_RANK_ATTACKS[square][occupied & chess.BB_RANK_MASKS[square]]
        file = chess.BB_FILE_ATTACKS[square][occupied & chess.BB_FILE_MASKS[square]]
        want_rook = int(rank) | int(file)
        want_bishop = int(chess.BB_DIAG_ATTACKS[square][occupied & chess.BB_DIAG_MASKS[square]])
        got_rook = int(rook_attacks(square, occupancy))
        got_bishop = int(bishop_attacks(square, occupancy))
        got_queen = int(queen_attacks(square, occupancy))
        assert got_rook == want_rook, f"rook {chess.square_name(square)} {got_rook:x} {want_rook:x}"
        assert got_bishop == want_bishop, f"bishop {chess.square_name(square)}"
        assert got_queen == want_rook | want_bishop, f"queen {chess.square_name(square)}"
        checked += 3

for square in range(64):
    assert int(KING_ATTACKS[square]) == int(chess.BB_KING_ATTACKS[square]), "king"
    assert int(KNIGHT_ATTACKS[square]) == int(chess.BB_KNIGHT_ATTACKS[square]), "knight"
    assert int(PAWN_ATTACKS[0][square]) == int(chess.BB_PAWN_ATTACKS[chess.WHITE][square]), "pawn"
    assert int(PAWN_ATTACKS[1][square]) == int(chess.BB_PAWN_ATTACKS[chess.BLACK][square]), "pawn"

for value in (1, 2, 0x8000_0000_0000_0000, 0x0F0F_0F0F_0F0F_0F0F, 0xFFFF_FFFF_FFFF_FFFF):
    assert popcount(np.uint64(value)) == bin(value).count("1"), value
    assert lsb(np.uint64(value)) == (value & -value).bit_length() - 1, value

print(f"{checked:,} slider attack sets match python-chess")
