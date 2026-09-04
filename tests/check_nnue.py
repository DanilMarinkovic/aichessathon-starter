"""Gate 1: the incrementally maintained accumulator must equal a from-scratch rebuild, always.

A wrong delta does not crash. It quietly evaluates a position the network was never trained on,
and the only symptom is that the engine plays worse for no visible reason.

Three things are checked, hardest first:

1. King moves that change bucket or cross the mirror line. Every feature index for that
   perspective moves at once, so the accumulator cannot be carried forward and has to be
   rebuilt. This is the newest code path and the easiest to get silently wrong, so the walk
   below counts how many it actually exercised and fails if the answer is none.
2. Castling, en passant, promotion and capture-promotion, which are the cases a hand-written
   delta would get wrong and which the bitboard diff is supposed to handle without special
   cases.
3. Ordinary play, which catches anything the awkward positions do not.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import nnue
from bitboards import KING
from movegen import MAX_MOVES, generate
from position import NFIELDS, from_board, make_move

TRICKY = (
    # Kings with the whole board to roam, so buckets and the mirror line get crossed often.
    "8/3k4/8/8/8/8/3K4/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/3kK3/8/8/8/8 w - - 0 1",
    # Castling both sides, both colours: the king jumps two files, often changing bucket.
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    # En passant available to White and to Black.
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pPP2/8/PPP3PP/RNBQKBNR b KQkq e3 0 3",
    # Promotions, including capture-promotions.
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "8/1P6/8/8/8/8/6p1/4K2k w - - 0 1",
    # Everything hanging, so captures of every piece type occur.
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
)


def context(state: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """The bucket offset and mirror each perspective is currently using."""
    white = int(np.uint64(state[KING - 1]).item().bit_length() - 1)
    black = int(np.uint64(state[6 + KING - 1]).item().bit_length() - 1)
    return nnue.perspective(white, 0), nnue.perspective(black, 56)


def walk(board: chess.Board, rng: random.Random, plies: int) -> tuple[int, int]:
    """Play a line, keeping an incremental accumulator, checking it at every step."""
    state = np.ascontiguousarray(from_board(board))
    incremental = nnue.new_accumulator()
    expected = nnue.new_accumulator()
    # Separate caches: the incremental path shares one across the whole line, the reference
    # gets a fresh one each time, so a stale or corrupted cache entry shows up as a mismatch.
    live_values, live_boards = nnue.new_cache()
    nnue.refresh(state, incremental, live_values, live_boards)

    moves = np.zeros(MAX_MOVES, dtype=np.int32)
    child = np.zeros(NFIELDS, dtype=np.uint64)
    checked = 0
    rebuilds = 0

    for _ in range(plies):
        legal = []
        count = generate(state, moves, 0)
        for index in range(count):
            if make_move(state, moves[index], child) != 0:
                legal.append(int(moves[index]))
        if not legal:
            break

        move = rng.choice(legal)
        make_move(state, np.int32(move), child)
        if context(state) != context(child):
            rebuilds += 1

        nnue.advance(state, child, incremental, incremental, live_values, live_boards)
        clean_values, clean_boards = nnue.new_cache()
        nnue.refresh(child, expected, clean_values, clean_boards)
        if not np.array_equal(incremental, expected):
            differing = int((incremental != expected).sum())
            raise SystemExit(
                f"accumulator disagrees with a rebuild after move {move:#x}\n"
                f"  line began at {board.fen()}\n"
                f"  context before {context(state)} after {context(child)}\n"
                f"  {differing} of {incremental.size} accumulator entries differ"
            )

        state = np.ascontiguousarray(child.copy())
        checked += 1

    return checked, rebuilds


def main() -> None:
    start = time.perf_counter()
    print(f"BUCKETS={nnue.BUCKETS}, HIDDEN={nnue.HIDDEN}, weights {nnue.WEIGHTS.shape}")

    positions = 0
    rebuilds = 0
    for fen in TRICKY:
        for attempt in range(40):
            done, rebuilt = walk(chess.Board(fen), random.Random(attempt * 7919), 16)
            positions += done
            rebuilds += rebuilt
    print(f"{positions:,} positions from the awkward openings agree")

    rng = random.Random(20260904)
    ordinary = 0
    for game in range(60):
        board = chess.Board()
        for _ in range(rng.randint(0, 12)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if board.is_game_over():
            continue
        done, rebuilt = walk(board, random.Random(game), 60)
        ordinary += done
        rebuilds += rebuilt
    print(f"{ordinary:,} positions from ordinary games agree")

    print(f"{rebuilds:,} of those forced a full rebuild (king changed bucket or mirror)")
    if nnue.BUCKETS > 1 and rebuilds == 0:
        raise SystemExit(
            "no king move crossed a bucket or mirror boundary, so the rebuild path was never "
            "run. The test proves nothing about it; widen the positions."
        )

    print(f"\nnnue accumulator clean in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
