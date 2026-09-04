"""Correctness gate for move generation, make_move and the incremental Zobrist key.

Three checks, cheapest and most localising first:

1. Random playouts, comparing our legal move set against python-chess position by position.
   A mismatch prints the FEN and the exact moves involved, which is what makes a bug findable.
2. The incremental Zobrist key against a from-scratch recomputation after every move.
3. Standard perft node counts, which catch the rare paths playouts miss.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
from numba import int32, int64, njit, uint64

from movegen import MAX_MOVES, generate
from position import NFIELDS, compute_hash, from_board, make_move, to_uci

MAX_DEPTH = 12


@njit(int64(uint64[:, ::1], int32[:, ::1], int64, int64), nogil=True, cache=False)
def perft(states: np.ndarray, moves: np.ndarray, ply: np.int64, depth: np.int64) -> np.int64:
    if depth == 0:
        return 1
    count = generate(states[ply], moves[ply], 0)
    total = 0
    for index in range(count):
        if make_move(states[ply], moves[ply][index], states[ply + 1]) != 0:
            total += perft(states, moves, ply + 1, depth - 1)
    return total


def _stacks() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((MAX_DEPTH + 2, NFIELDS), dtype=np.uint64),
        np.zeros((MAX_DEPTH + 2, MAX_MOVES), dtype=np.int32),
    )


def our_moves(board: chess.Board) -> tuple[set[str], list[np.ndarray]]:
    """Every legal move we generate, as UCI, plus the resulting states."""
    state = from_board(board)
    moves = np.zeros(MAX_MOVES, dtype=np.int32)
    child = np.zeros(NFIELDS, dtype=np.uint64)
    count = generate(state, moves, 0)
    found: set[str] = set()
    children: list[np.ndarray] = []
    for index in range(count):
        if make_move(state, moves[index], child) != 0:
            found.add(to_uci(int(moves[index])))
            children.append(child.copy())
    return found, children


def playouts(games: int, seed: int) -> int:
    rng = random.Random(seed)
    positions = 0
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=False) and board.fullmove_number < 120:
            ours, children = our_moves(board)
            theirs = {move.uci() for move in board.legal_moves}
            if ours != theirs:
                print("MISMATCH at", board.fen())
                print("  we generate and they do not:", sorted(ours - theirs))
                print("  they generate and we do not:", sorted(theirs - ours))
                raise SystemExit(1)
            for child in children:
                if child[19] != compute_hash(np.ascontiguousarray(child)):
                    print("HASH MISMATCH from", board.fen())
                    raise SystemExit(1)
            positions += 1
            board.push(rng.choice(list(board.legal_moves)))
    return positions


CASES = (
    (chess.STARTING_FEN, (20, 400, 8902, 197281, 4865609)),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        (48, 2039, 97862, 4085603),
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", (14, 191, 2812, 43238, 674624)),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        (6, 264, 9467, 422333),
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        (44, 1486, 62379, 2103487),
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        (46, 2079, 89890, 3894594),
    ),
)


def main() -> None:
    start = time.perf_counter()
    positions = playouts(40, 12345)
    print(f"{positions:,} random positions agree with python-chess move for move")

    failures = 0
    nodes = 0
    for fen, expected in CASES:
        states, moves = _stacks()
        states[0] = from_board(chess.Board(fen))
        for depth, want in enumerate(expected, start=1):
            began = time.perf_counter()
            got = int(perft(states, moves, 0, depth))
            elapsed = time.perf_counter() - began
            nodes += got
            status = "ok" if got == want else f"WRONG, expected {want:,}"
            rate = f"{got / elapsed / 1e6:.1f}M nps" if elapsed > 0.01 else ""
            print(f"  depth {depth}: {got:>12,} {status:<24} {rate}")
            if got != want:
                failures += 1
                break
        print(f"  {fen}")

    total = time.perf_counter() - start
    print(f"\n{nodes:,} nodes in {total:.1f}s")
    if failures:
        raise SystemExit(f"{failures} perft positions wrong")
    print("perft clean")


if __name__ == "__main__":
    main()
