"""Write an untrained network of a chosen shape, so a sweep can train architectures.

`nnue.py` takes its architecture from the weights file, which is what stops the engine and the
trainer ever disagreeing about it. The consequence is that training a different shape needs the
file to say so first, and this writes exactly that: correct shape, correct quantisation fields,
random weights.

Doing it this way rather than with an environment variable keeps the shipped engine unchanged.
nnue.py reads a file, as it always did, and nothing about a local experiment leaks into what a
judge reads or the platform runs.

The weights it writes are noise, so an engine loading one plays terribly until training has
overwritten it. Run this into a scratch copy of the repository, never over a network you intend
to keep.
"""

import argparse
from pathlib import Path

import numpy as np

# These must match nnue.py. They are not imported from it because importing nnue loads whatever
# network already exists, which is the situation this script is here to get around.
QA = 255
QB = 64
SCALE = 400
INPUTS = 768


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an untrained network of a given shape.")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--buckets", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        weights=rng.integers(
            -32, 32, size=(arguments.buckets * INPUTS, arguments.hidden)
        ).astype(np.int16),
        biases=rng.integers(-32, 32, size=arguments.hidden).astype(np.int16),
        output=rng.integers(-32, 32, size=2 * arguments.hidden).astype(np.int16),
        output_bias=np.int32(0),
        hidden=np.int32(arguments.hidden),
        buckets=np.int32(arguments.buckets),
        qa=np.int32(QA),
        qb=np.int32(QB),
        scale=np.int32(SCALE),
    )
    print(
        f"wrote untrained {arguments.hidden}x{arguments.buckets} network to {arguments.out} "
        f"({arguments.buckets * INPUTS * arguments.hidden * 2 / 1024:.0f} KB of weights)"
    )


if __name__ == "__main__":
    main()
