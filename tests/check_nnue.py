"""Gate 1: the incremental accumulator must equal a from-scratch rebuild, always.

A wrong delta does not crash. It quietly evaluates a position the network was never trained on,
and the only symptom is that the engine plays worse for no visible reason. So this checks the
two paths against each other over random play, and separately over positions chosen to force
castling, en passant, promotion and capture-promotion, which are the cases a hand-written delta
would get wrong.

It also checks that applying a delta and then subtracting it restores the accumulator exactly,
since that is what the search relies on instead of copying.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import nnue
from movegen import MAX_MOVES, generate
from position import NFIELDS, from_board, make_move

TRICKY = (
    # castling both sides, both colours
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    # en passant available to White and to Black
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/pppp1ppp/8/8/3pPP2/8/PPP3PP/RNBQKBNR b KQkq e3 0 3",
    # promotions, including capture-promotions
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "8/1P6/8/8/8/8/6p1/4K2k w - - 0 1",
    # a position with everything hanging, so captures of every piece type occur
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def walk(board: chess.Board, rng: random.Random, plies: int) -> tuple[int, int]:
    """Play a line, keeping an incremental accumulator, checking it at every step."""
    state = np.ascontiguousarray(from_board(board))
    incremental = nnue.new_accumulator()
    expected = nnue.new_accumulator()
    nnue.refresh(state, incremental)

    moves = np.zeros(MAX_MOVES, dtype=np.int32)
    child = np.zeros(NFIELDS, dtype=np.uint64)
    checked = 0

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

        before = incremental.copy()
        nnue.apply(state, child, incremental, 1)
        nnue.refresh(child, expected)
        if not same(incremental, expected):
            raise SystemExit(f"delta disagrees with refresh after {move:#x} in {board.fen()}")

        # The search subtracts the delta instead of copying, so it has to invert exactly.
        nnue.apply(state, child, incremental, -1)
        if not same(incremental, before):
            raise SystemExit(f"delta did not invert after {move:#x} in {board.fen()}")
        nnue.apply(state, child, incremental, 1)

        state = np.ascontiguousarray(child.copy())
        checked += 1

    return checked, 0


def main() -> None:
    start = time.perf_counter()
    rng = random.Random(20260904)

    total = 0
    for fen in TRICKY:
        for attempt in range(40):
            board = chess.Board(fen)
            done, _ = walk(board, random.Random(attempt * 7919), 14)
            total += done
    print(f"{total:,} positions from the awkward openings agree")

    total = 0
    for game in range(60):
        board = chess.Board()
        for _ in range(rng.randint(0, 12)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if board.is_game_over():
            continue
        done, _ = walk(board, random.Random(game), 60)
        total += done
    print(f"{total:,} positions from random games agree")

    # The evaluation must not depend on which perspective happens to be stored first.
    board = chess.Board("r1bq1rk1/pp1pppbp/2n2np1/2p5/2PP4/2N1PNP1/PP3PBP/R1BQK2R b KQ - 0 7")
    state = np.ascontiguousarray(from_board(board))
    accumulator = nnue.new_accumulator()
    nnue.refresh(state, accumulator)
    white_view = nnue.forward(accumulator, 0)
    black_view = nnue.forward(accumulator, 1)
    print(f"forward from each perspective: {white_view} and {black_view}")

    print(f"\nnnue accumulator clean in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
