"""Where is the evaluation wrong, and by how much, on positions it has never seen.

A network's overall correlation with the labelling engine hides the thing that loses games.
Ours screens at 0.97 and is still wrong by 700cp whenever material and king safety disagree --
because 97% of positions are ones where counting material is nearly enough, and the average is
dominated by them.

So this splits the error by how much of the label material already explains. `residual` is
`label - material`: near zero when the position is decided by who has more wood, large when it
is decided by something else. The last bucket is the engine's blind spot, and the number to
watch is the error there, not the mean.

Run against a held-out shard -- one the trainer never saw -- or it reports memorisation.

    uv run python tools/blindspot.py aire/data/shard-799.epd
    uv run python tools/blindspot.py aire/data/shard-799.epd --weights other/net.npz

Reports mean absolute error per bucket, and the share of positions where the sign is wrong,
which is the failure that actually costs games: not "how far off" but "believed the wrong side
was winning".
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
# Residual bands, in centipawns of |label - material|.
BANDS = ((100, "material explains it"), (300, "some positional"),
         (600, "mostly positional"), (10**9, "material is misleading"))


def material(board: chess.Board) -> int:
    total = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in VALUES.items()
    )
    return total if board.turn == chess.WHITE else -total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", type=Path, help="labelled EPD: fen|score|result")
    parser.add_argument("--limit", type=int, default=20_000)
    arguments = parser.parse_args()

    import nnue
    from position import OCC_ALL, SIDE, from_board

    if not nnue.TRAINED:
        raise SystemExit("no trained network loaded")

    values, boards = nnue.new_cache()
    accumulator = nnue.new_accumulator()
    rows: list[tuple[int, int, int]] = []
    with arguments.positions.open() as handle:
        for line in handle:
            if len(rows) >= arguments.limit:
                break
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            try:
                label = int(parts[1])
            except ValueError:
                continue
            board = chess.Board(parts[0])
            state = np.ascontiguousarray(from_board(board))
            nnue.refresh(state, accumulator, values, boards)
            ours = int(nnue.forward(accumulator, int(state[SIDE]),
                                    nnue.output_bucket(state[OCC_ALL])))
            rows.append((ours, label, material(board)))

    print(f"{len(rows):,} held-out positions, {arguments.positions.name}")
    print()
    print(f"{'residual band':<26} {'n':>7} {'share':>7} {'mean err':>9} {'sign wrong':>11}")
    lower = 0
    for bound, name in BANDS:
        picked = [r for r in rows if lower <= abs(r[1] - r[2]) < bound]
        lower = bound
        if not picked:
            continue
        err = statistics.mean(abs(o - lab) for o, lab, _ in picked)
        # A sign error means we believed the wrong side was better. Only counted where the
        # label is decisive enough for the sign to mean something.
        decisive = [r for r in picked if abs(r[1]) > 100]
        flipped = sum(1 for o, lab, _ in decisive if (o > 0) != (lab > 0))
        share = f"{flipped / len(decisive) * 100:.1f}%" if decisive else "-"
        print(f"  {name:<24} {len(picked):>7,} {len(picked)/len(rows)*100:>6.1f}% "
              f"{err:>8.0f}cp {share:>11}")

    overall = statistics.mean(abs(o - lab) for o, lab, _ in rows)
    decisive = [r for r in rows if abs(r[1]) > 100]
    flipped = sum(1 for o, lab, _ in decisive if (o > 0) != (lab > 0))
    print()
    print(f"  overall mean error {overall:.0f}cp, sign wrong on "
          f"{flipped / len(decisive) * 100:.1f}% of decisive positions")


if __name__ == "__main__":
    main()
