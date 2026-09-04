"""Check static exchange evaluation against known positions and against brute force.

The hand-worked cases are the standard ones, and they cover the two failure modes that matter:
a capture that looks good by piece values but loses to a defender, and an x-ray where a slider
behind the capturing piece joins the exchange once it moves.

Hand-picked cases only prove what someone thought to check, so the second half compares SEE
against an exhaustive recapture search over random positions. That search is far too slow to
use in an engine but it is obviously correct, which is what a reference needs to be.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

from movegen import MAX_MOVES, generate
from position import from_board, to_uci
from see import VALUE, see_ge, see_value

PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}

CASES = (
    # A rook takes a defenceless pawn: worth exactly the pawn.
    ("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", 100),
    # A knight takes a pawn defended by another pawn: wins 100, loses 320.
    ("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -220),
    # Nothing on the square at all.
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4", 0),
)


def _gain(position: chess.Board, move: chess.Move) -> int:
    """What one capture is worth: the victim, plus what a promotion upgrades the mover into.

    Forgetting the promotion term is easy and wrong in a specific way: the side promoting is
    credited only a pawn's worth of capture while the opponent is charged for taking a queen.
    """
    victim = position.piece_at(move.to_square)
    value = PIECE_VALUE[victim.piece_type] if victim else 0
    if move.promotion:
        value += PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]
    return value


def brute_force(board: chess.Board, move: chess.Move) -> int:
    """Exhaustive recapture search on one square. Obviously correct, uselessly slow."""
    target = move.to_square

    def best(position: chess.Board) -> int:
        captures = [
            m for m in position.legal_moves
            if m.to_square == target and position.is_capture(m)
        ]
        if not captures:
            return 0
        # Either side may decline to continue the exchange.
        return max(
            0,
            max(_replies(position, capture) for capture in captures),
        )

    def _replies(position: chess.Board, capture: chess.Move) -> int:
        gain = _gain(position, capture)
        position.push(capture)
        value = gain - best(position)
        position.pop()
        return value

    gain = _gain(board, move)
    board.push(move)
    value = gain - best(board)
    board.pop()
    return value


def main() -> None:
    start = time.perf_counter()

    for fen, uci, expected in CASES:
        board = chess.Board(fen)
        state = np.ascontiguousarray(from_board(board))
        move = chess.Move.from_uci(uci)
        packed = None
        moves = np.zeros(MAX_MOVES, dtype=np.int32)
        for index in range(generate(state, moves, 0)):
            if to_uci(int(moves[index])) == uci:
                packed = moves[index]
        assert packed is not None, f"{uci} is not generated in {fen}"
        got = int(see_value(state, packed))
        status = "ok" if got == expected else f"WRONG, expected {expected}"
        print(f"  {uci} in {fen[:34]:<34} see {got:>6}  {status}")
        if got != expected:
            raise SystemExit(1)

    # Random positions, every capture, against brute force.
    rng = random.Random(4242)
    checked = 0
    for _game in range(150):
        board = chess.Board()
        for _ in range(rng.randint(4, 40)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if board.is_game_over():
            continue

        state = np.ascontiguousarray(from_board(board))
        moves = np.zeros(MAX_MOVES, dtype=np.int32)
        count = generate(state, moves, 1)
        for index in range(count):
            uci = to_uci(int(moves[index]))
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves or not board.is_capture(move):
                continue
            if board.is_en_passant(move):
                continue  # brute force below does not model the vacated square
            ours = int(see_value(state, moves[index]))
            theirs = brute_force(board, move)
            if ours != theirs:
                print(f"\nMISMATCH {uci} in {board.fen()}")
                print(f"  see says {ours}, exhaustive recapture says {theirs}")
                raise SystemExit(1)
            checked += 1

    print(f"\n{checked:,} captures agree with an exhaustive recapture search")
    print(f"see clean in {time.perf_counter() - start:.1f}s")
    print(f"piece values {list(VALUE[1:6])}")
    _ = see_ge


if __name__ == "__main__":
    main()
