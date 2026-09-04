"""Ask the engine what it thinks about one position, and compare with a reference.

The blunder report says which moves cost the most. This says why: it runs our search over the
same position at a range of node counts and prints what it picks and what it scores, next to
what a reference engine picks. A move that stays wrong as the nodes go up is an evaluation or
pruning problem; one that corrects itself with more nodes was only ever a speed problem.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import searcher
from position import HASH, from_board, to_uci
from searcher import BEST_DEPTH, BEST_SCORE, NODES, STOP

LADDER = (50_000, 200_000, 1_000_000, 5_000_000)


def ours(board: chess.Board, nodes: int) -> tuple[str, int, int, int]:
    searcher.reset()
    state = from_board(board)
    searcher.STATES[0] = state
    searcher.PATH[:] = 0
    searcher.PATH[0] = np.uint64(state[HASH])
    searcher.CONTROL[STOP] = 0
    packed = searcher.run(64, 0, nodes)
    return (
        to_uci(packed) if packed else "----",
        int(searcher.CONTROL[BEST_SCORE]),
        int(searcher.CONTROL[BEST_DEPTH]),
        int(searcher.CONTROL[NODES]),
    )


def reference(path: str, board: chess.Board, depth: int) -> tuple[str, str]:
    process = subprocess.Popen(
        [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(f"position fen {board.fen()}\ngo depth {depth}\n")
        process.stdin.flush()
        score = "?"
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith("info ") and " score " in line:
                parts = line.split()
                kind = parts[parts.index("score") + 1]
                score = f"{kind} {parts[parts.index('score') + 2]}"
            if line.startswith("bestmove"):
                return line.split()[1], score
        return "----", score
    finally:
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one position at increasing depth.")
    parser.add_argument("fen")
    parser.add_argument("--engine", default=os.environ.get("UCI_ENGINE", ""))
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--nodes", type=int, action="append", help="repeat to override the ladder")
    parser.add_argument("--expect", help="the move a reference says is right, for a quick verdict")
    arguments = parser.parse_args()

    board = chess.Board(arguments.fen)
    print(f"{arguments.fen}\n{board.unicode(borders=False, empty_square='.')}\n")
    print(f"{'nodes':>10}  {'move':<6} {'score':>8} {'depth':>6} {'searched':>10}")
    for nodes in arguments.nodes or LADDER:
        move, score, depth, searched = ours(board, nodes)
        print(f"{nodes:>10,}  {move:<6} {score:>8} {depth:>6} {searched:>10,}")

    if arguments.engine:
        best, score = reference(arguments.engine, board, arguments.depth)
        print(f"\nreference at depth {arguments.depth}: {best} ({score})")

    if arguments.expect:
        print(f"expected: {arguments.expect}")


if __name__ == "__main__":
    main()
