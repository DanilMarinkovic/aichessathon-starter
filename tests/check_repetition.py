"""Check the engine does not repeat a position it is winning.

This is a regression test for a specific bug that drew three rated games. The engine keeps its
own record of the game, because the platform sends a FEN and nothing else, and that record only
held the positions it was asked to move in. `is_repetition` scans backwards in twos to stay on
one side to move, so starting from a position that arises after our own move it walked a row of
entries that were never written and always reported "no repetition".

The blind spot was exactly the repetitions the engine causes itself. Winning by five pawns in
round 18 it played Rd5, Re5, Rd5 and drew, because the positions it was repeating were ones
where the opponent was to move.

The position below is that game, five moves before the draw, with the opponent's replies played
out as they actually were. Before the fix the engine walks into the repetition twice. It is
driven through `agent.get_move` rather than through the searcher directly, because the record
lives in agent.py and the bug was in how it was written, not in how it was scanned.

Fixed nodes rather than a clock, so the result is the same on any machine.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CHESSATHON_FIXED_NODES", "600000")

import chess

import agent

# Round 18, black to move, winning by about five pawns.
FEN = "6k1/1p4p1/2p1p2p/2P1P3/8/1q3rP1/rBR1Q1KP/8 b - - 3 41"
# What was actually played from there, ours and theirs alternating, ending in the draw claim.
PLAYED = ["f3e3", "e2f2", "e3e5", "c2d2", "e5d5", "d2c2", "d5e5", "c2d2", "e5d5", "d2c2"]


def main() -> None:
    board = chess.Board(FEN)
    ours = board.turn
    repeats = 0
    moves = 0

    for uci in PLAYED:
        if board.turn == ours:
            reply = agent.get_move(board.fen(), 30_000)
            move = chess.Move.from_uci(reply)
            if move not in board.legal_moves:
                raise SystemExit(f"engine returned an illegal move {reply} in {board.fen()}")
            after = board.copy()
            after.push(move)
            moves += 1
            repeated = after.is_repetition(2)
            repeats += repeated
            print(
                f"  ply {board.ply():>3}  engine plays {board.san(move):<6}"
                f"{'  REPEATS a position already reached' if repeated else ''}"
            )
            board.push(chess.Move.from_uci(uci))
        else:
            board.push(chess.Move.from_uci(uci))

    print(f"\n{moves} engine moves from a won position, {repeats} of them repeating")
    if repeats:
        raise SystemExit(
            "the engine walked into a repetition while winning. agent._after is how positions "
            "arising after our own move reach PATH; check it is filled and indexed in step "
            "with agent._history."
        )
    print("repetition avoidance clean")


if __name__ == "__main__":
    main()
