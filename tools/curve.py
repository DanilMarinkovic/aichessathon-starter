"""Where did a game's assessment turn, and did our evaluation know?

Walks a PGN and prints, side by side, what a reference engine thinks of each position and what
our network thinks. The gap column is the thing to read: a game lost with a small gap
throughout was lost on merit, and one lost with a growing gap was lost because the evaluation
stopped tracking reality.

Sampled, single-threaded, and at a modest depth on purpose. The first version of this ran four
threads at depth 20 over every position of a 77-move game and was still going an hour later on
a laptop. A curve only has to show where the line turned.

    uv run python tools/curve.py game.pgn --stride 8 --depth 14
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.engine
import chess.pgn

from tools.side_bias import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--stride", type=int, default=8, help="sample every Nth ply")
    parser.add_argument("--depth", type=int, default=14, help="reference engine depth")
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()

    game = chess.pgn.read_game(arguments.pgn.open())
    if game is None:
        raise SystemExit(f"no game in {arguments.pgn}")

    board = game.board()
    rows = [("start", board.fen())]
    for move in game.mainline_moves():
        san, white, number = board.san(move), board.turn == chess.WHITE, board.fullmove_number
        board.push(move)
        rows.append((f"{number}{'.' if white else '...'}{san}", board.fen()))

    picked = [row for index, row in enumerate(rows)
              if index % arguments.stride == 0 or index == len(rows) - 1]
    ours = evaluate([fen for _, fen in picked])

    engine = chess.engine.SimpleEngine.popen_uci(arguments.engine)
    engine.configure({"Threads": arguments.threads, "Hash": 128})
    print(f"{'move':>16} {'reference':>9} {'ours':>7} {'gap':>7}  swing")
    previous = None
    try:
        for (label, fen), our_score in zip(picked, ours, strict=True):
            position = chess.Board(fen)
            if position.is_game_over():
                continue
            info = engine.analyse(position, chess.engine.Limit(depth=arguments.depth))
            reference = info["score"].white().score(mate_score=30000)
            # `ours` is from the side to move; the reference is from White. Put both on White's.
            white_view = int(our_score) if position.turn == chess.WHITE else -int(our_score)
            swing = "" if previous is None else f"{reference - previous:+6d}"
            print(f"{label:>16} {reference:+9d} {white_view:+7d} "
                  f"{reference - white_view:+7d} {swing}")
            previous = reference
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
