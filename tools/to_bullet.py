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
import random
import sys
from pathlib import Path

RESULTS = {"1": "1.0", "0.5": "0.5", "0": "0.0"}

# How much of each evaluation band to keep, as (upper bound in centipawns, share kept).
#
# The set as generated is dominated by positions already decided by material: on a sampled
# shard, plain material count correlates 0.89 with the label, so 79.7% of the variance the
# network is asked to fit is explained by counting pieces. It duly learned to count pieces. In
# a real loss it valued a position at -544 that Stockfish valued at +455 -- Black was a bishop
# up and being mated -- and agreed to within 8cp again the moment material was restored.
#
# Restricting to |label| <= 300 drops material's share of the variance from 79.7% to 20.0%, but
# throwing the decided positions away entirely is the wrong cure: the network still has to know
# that a queen up is winning, and it can only know that from positions where one side is a
# queen up. So the bands thin rather than exclude. Every kind of position survives; the ones
# where material already gives the answer stop crowding out the ones where it does not.
# Measured on a 40,000-position sample. The shares below take material's share of the variance
# from 79.7% to 54.5% while keeping 42.8% of the rows -- about 50M of the 118M, which is ample
# for a network that is underfitting. Gentler bands were tried: keeping half the 300-600 band
# and a quarter of 600-1000 only reached 68.3%, which is not enough of a change to be worth a
# training run.
KEEP_BANDS = ((300, 1.0), (600, 0.25), (1000, 0.06), (10**9, 0.02))


def keep_share(centipawns: int) -> float:
    """The share of positions at this evaluation to retain."""
    magnitude = abs(centipawns)
    for bound, share in KEEP_BANDS:
        if magnitude <= bound:
            return share
    return KEEP_BANDS[-1][1]


def convert(
    source: Path, destination: Path, rebalance: bool = False, seed: int = 0
) -> tuple[int, int, int]:
    """Rewrite one file. Returns rows written, rows malformed, and rows thinned.

    The last two are counted apart on purpose. A malformed row means the generator produced
    something this cannot read, which is a fault worth noticing; a thinned row is the
    rebalancing doing exactly its job. Reporting them as one number turned a 43% deliberate
    reduction into a log line reading "67,584,926 rows skipped as malformed".
    """
    written = 0
    malformed = 0
    thinned = 0
    # Seeded, so a rebuild of the same shard produces the same set. An unseeded filter would
    # make two runs over identical data differ, and every comparison after that meaningless.
    rng = random.Random(seed)
    with source.open() as reader, destination.open("w") as writer:
        for line in reader:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                malformed += 1
                continue
            fen, score, outcome = parts

            fields = fen.split()
            if len(fields) < 2:
                malformed += 1
                continue

            try:
                centipawns = int(score)
            except ValueError:
                malformed += 1
                continue

            # The one substantive change: ours is from the mover's point of view, bullet's is
            # from white's.
            if fields[1] == "b":
                centipawns = -centipawns

            result = RESULTS.get(outcome)
            if result is None:
                malformed += 1
                continue

            if rebalance and rng.random() >= keep_share(centipawns):
                thinned += 1
                continue

            writer.write(f"{fen} | {centipawns} | {result}\n")
            written += 1
    return written, malformed, thinned


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert labelled EPD to bullet's text format.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--rebalance", action="store_true",
        help="thin the evaluation bands so material stops explaining most of the target",
    )
    parser.add_argument("--seed", type=int, default=0, help="so a rebuild is reproducible")
    arguments = parser.parse_args()

    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    written, malformed, thinned = convert(
        arguments.source, arguments.destination, arguments.rebalance, arguments.seed
    )
    print(f"{written:,} rows -> {arguments.destination}", file=sys.stderr)
    if thinned:
        total = written + thinned
        print(
            f"{thinned:,} rows thinned by rebalancing, {written / total:.1%} of "
            f"{total:,} kept",
            file=sys.stderr,
        )
    if malformed:
        print(f"{malformed:,} rows skipped as malformed", file=sys.stderr)


if __name__ == "__main__":
    main()
