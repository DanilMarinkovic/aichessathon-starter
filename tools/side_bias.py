"""Does the network evaluate black-to-move positions as well as white-to-move ones?

The dataset every network so far was trained on turned out to be 99% white to move, because
the sampler in tools/label.py took every second ply from games that open with white. The
question this answers is what that actually cost, which is not obvious either way.

It is not obvious because the feature encoding is already perspective-based: black's pieces are
mirrored with `square ^ 56` and recoloured, so a black-to-move position and its colour-flipped
white-to-move twin should produce identical features. If that holds, training on one colour
loses much less than the raw 99% suggests, because both colours land in the same feature space.

So there are two questions, and they want different tests.

Symmetry is exact and needs no reference engine. `board.mirror()` flips the board vertically,
swaps the colours and swaps the side to move, producing the same position seen from the other
side. A correct perspective network must return the identical score for both. Any difference at
all is an encoding bug, and the size of it says how much of one.

Strength is statistical and needs a reference. Split labelled positions by side to move and
compare how well the network tracks the reference in each half. If black-to-move positions are
tracked worse, the bias cost real strength; if the two halves match, it did not, whatever the
symmetry test says about elegance.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import nnue
from position import OCC_ALL, SIDE, from_board


def evaluate(fens: list[str]) -> np.ndarray:
    """The engine's static evaluation, in centipawns from the side to move."""
    values, boards = nnue.new_cache()
    accumulator = nnue.new_accumulator()
    out = np.zeros(len(fens), dtype=np.float64)
    for index, fen in enumerate(fens):
        state = np.ascontiguousarray(from_board(chess.Board(fen)))
        nnue.refresh(state, accumulator, values, boards)
        out[index] = float(
            nnue.forward(accumulator, int(state[SIDE]), nnue.output_bucket(state[OCC_ALL]))
        )
    return out


def read(path: Path, limit: int) -> tuple[list[str], np.ndarray]:
    fens: list[str] = []
    scores: list[int] = []
    with path.open() as handle:
        for line in handle:
            if len(fens) >= limit:
                break
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            try:
                scores.append(int(parts[1]))
            except ValueError:
                continue
            fens.append(parts[0])
    return fens, np.array(scores, dtype=np.float64)


def summarise(name: str, ours: np.ndarray, theirs: np.ndarray) -> None:
    """Report slope and correlation, not mean error.

    Mean error is actively misleading here. Split by side to move, the two halves have
    reference means of opposite sign, so any difference in scale shows up as a large positive
    bias in one half and an equally large negative one in the other -- which reads exactly like
    a side-to-move bug and is nothing of the kind. Slope separates the two: a slope near one
    with a high correlation means the network agrees with the reference, and a slope far from
    one means it is tracking the reference correctly on a different scale.
    """
    if len(ours) < 2:
        print(f"  {name:<14} too few positions to say anything")
        return
    correlation = float(np.corrcoef(ours, theirs)[0, 1])
    slope = float(np.polyfit(theirs, ours, 1)[0])
    print(
        f"  {name:<14} n {len(ours):>5}   slope {slope:>5.2f}   "
        f"correlation {correlation:.4f}   spread {ours.std():>6.0f} vs {theirs.std():>5.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the network for a side-to-move bias.")
    parser.add_argument("positions", type=Path, help="labelled EPD: fen|score|result")
    parser.add_argument("--limit", type=int, default=20_000)
    arguments = parser.parse_args()

    fens, reference = read(arguments.positions, arguments.limit)
    if not fens:
        raise SystemExit(f"no usable rows in {arguments.positions}")
    print(f"{len(fens):,} positions from {arguments.positions}\n")

    # Test one: exact mirror symmetry. No reference engine, no statistics, no excuses.
    mirrored = [chess.Board(fen).mirror().fen() for fen in fens]
    ours = evaluate(fens)
    theirs_mirrored = evaluate(mirrored)
    difference = np.abs(ours - theirs_mirrored)
    print("mirror symmetry: eval(position) against eval(same position, colours swapped)")
    print(
        f"  max difference {difference.max():.1f}cp, "
        f"mean {difference.mean():.2f}cp, "
        f"{int((difference > 0.5).sum())} of {len(fens)} differ by more than half a pawn/100"
    )
    if difference.max() < 1.0:
        print("  the encoding is symmetric, so a colour-skewed dataset costs little by itself")
    else:
        print("  NOT symmetric: the feature encoding treats the two colours differently")

    # Test two: does it track the reference equally well for each side to move?
    print("\nagreement with the labelling engine, split by side to move")
    white = np.array([chess.Board(fen).turn == chess.WHITE for fen in fens])
    summarise("white to move", ours[white], reference[white])
    summarise("black to move", ours[~white], reference[~white])
    summarise("all", ours, reference)


if __name__ == "__main__":
    main()
