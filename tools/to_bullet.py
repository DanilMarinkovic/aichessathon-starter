"""Convert our labelled EPD into the plain text layout bullet's converter reads.

Two things differ between the two formats and only one of them is visible.

The separator is cosmetic: we write `fen|score|result`, bullet reads `<FEN> | <score> | <result>`.

The sign is not. Our score is **side to move relative**, because that is what the network is
asked to predict and it keeps the trainer and the engine agreeing about perspective. Bullet's
text format is **white relative**. Feeding ours in unchanged trains the network to predict the
evaluation of whichever side happens to move, which is a sign error on roughly half the rows
and produces a network that looks like it trained and plays like noise. So black-to-move rows
are negated here. The game result is already white relative in both, and passes through.

Side to move is read straight out of the FEN field rather than by constructing a board. At a
hundred million rows the difference between a string split and a full position parse is the
difference between minutes and most of a day, and nothing here needs the position.
"""

import argparse
import sys
from pathlib import Path

RESULTS = {"1": "1.0", "0.5": "0.5", "0": "0.0"}


def convert(source: Path, destination: Path) -> tuple[int, int]:
    """Rewrite one file. Returns rows written and rows skipped."""
    written = 0
    skipped = 0
    with source.open() as reader, destination.open("w") as writer:
        for line in reader:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                skipped += 1
                continue
            fen, score, outcome = parts

            fields = fen.split()
            if len(fields) < 2:
                skipped += 1
                continue

            try:
                centipawns = int(score)
            except ValueError:
                skipped += 1
                continue

            # The one substantive change: ours is from the mover's point of view, bullet's is
            # from white's.
            if fields[1] == "b":
                centipawns = -centipawns

            result = RESULTS.get(outcome)
            if result is None:
                skipped += 1
                continue

            writer.write(f"{fen} | {centipawns} | {result}\n")
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert labelled EPD to bullet's text format.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = convert(arguments.source, arguments.destination)
    print(f"{written:,} rows -> {arguments.destination}", file=sys.stderr)
    if skipped:
        print(f"{skipped:,} rows skipped as malformed", file=sys.stderr)


if __name__ == "__main__":
    main()
