"""The capture-ordering short-circuit has to be exact, not approximately right.

score_move skips the SEE call whenever the captured piece is worth at least as much as the
capturing one, on the argument that the capturing side can always stop after the first capture,
so the exchange is worth at least victim - attacker and see_ge(move, 0) cannot be false there.
That argument is only worth as much as the code it describes: en passant names a victim that is
not on the target square, a promotion capture changes the attacker's value mid-exchange, and a
king capture is a special case in every engine. So check it against the real see_ge, on real
positions, rather than trusting the reasoning.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

from bitboards import PAWN
from movegen import MAX_MOVES, generate
from position import EN_PASSANT, SIDE, from_board, move_flag, move_from, move_to, piece_on
from searcher import PIECE_VALUE
from see import see_ge


def main() -> None:
    random.seed(11)
    board = chess.Board()
    moves = np.zeros(MAX_MOVES, dtype=np.int32)
    captures = disagreements = 0

    for _ in range(400):
        board.reset()
        for _ in range(80):
            if board.is_game_over():
                break
            state = np.ascontiguousarray(from_board(board))
            count = generate(state, moves, 0)
            side = np.int64(state[SIDE])
            for index in range(count):
                move = moves[index]
                victim = (
                    PAWN
                    if move_flag(move) == EN_PASSANT
                    else piece_on(state, move_to(move), 1 - side)
                )
                if victim == 0:
                    continue
                attacker = piece_on(state, move_from(move), side)
                captures += 1
                # The short-circuit claims this is a winning-or-equal capture without asking.
                # If see_ge disagrees, the ordering changed and the claim is false.
                if PIECE_VALUE[victim] >= PIECE_VALUE[attacker] and see_ge(
                    state, move, np.int64(0)
                ) == 0:
                    disagreements += 1
                    print(f"  MISMATCH {board.fen()} move {move}")
            board.push(random.choice(list(board.legal_moves)))

    print(f"{captures:,} captures checked across 400 games")
    if disagreements:
        raise SystemExit(f"{disagreements} disagreements: the short-circuit is not exact")
    print("capture ordering short-circuit is exact")


if __name__ == "__main__":
    main()
