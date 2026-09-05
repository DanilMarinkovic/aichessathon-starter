"""Turn a network bullet trained into the weights file the engine loads.

Bullet writes two files per checkpoint. `quantised.bin` is int16 and padded to 64-byte
multiples, because its reference inference code declares the accumulator `align(64)`.
`raw.bin` is float32, in the same tensor order, with no padding. This reads `raw.bin`.

That choice is deliberate. Reading the quantised file would mean matching bullet's padding and
its integer widths exactly -- it stores the output bias as i16 where we store i32 -- and every
one of those is a silent, plausible-looking way to be wrong. Reading the floats instead lets us
apply the same `round(w * QA)` we already apply to our own PyTorch models, so the quantisation
step stays the one that `tools/verify_net.py` has always checked.

Tensor order comes from the `save_format` list in the bullet trainer, and must stay in step
with it:

    l0w   768 x HIDDEN   feature weights, column-major, so (768, HIDDEN) row-major here
    l0b   HIDDEN         feature bias
    l1w   2 x HIDDEN     output weights, side to move first, then the opponent
    l1b   1              output bias

Column-major `HIDDEN x 768` and row-major `(768, HIDDEN)` are the same bytes: 768 groups of
HIDDEN consecutive values, one group per input feature. That is already the layout `nnue.py`
indexes, so no transpose is involved.

King buckets multiply the first dimension: the weight matrix is (768 * BUCKETS, HIDDEN), laid
out bucket by bucket, which is exactly how nnue.py indexes it via `KING_BUCKET[...] * INPUTS`.
`--buckets` must match the BUCKETS the bullet run used; the size check below catches a mismatch
rather than reshaping into a plausible wrong answer.
"""

import argparse
from pathlib import Path

import numpy as np

# Read from the weights file rather than importing nnue, which would load whatever network
# already exists and fail before this one has been written.
QA = 255
QB = 64
SCALE = 400
INPUTS = 768
# Per hidden neuron: one weight per input row (768 per bucket), a bias, and two output weights.
# Plus a single output bias for the whole network.


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a bullet checkpoint to weights/net.npz")
    parser.add_argument("raw", type=Path, help="path to a checkpoint's raw.bin")
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    parser.add_argument(
        "--buckets", type=int, default=4,
        help="king buckets the trainer used; must match BUCKETS in the bullet run",
    )
    arguments = parser.parse_args()

    values = np.fromfile(arguments.raw, dtype=np.float32)

    # The layout is fully determined by one unknown, so the file size has to agree with a whole
    # number of hidden neurons. If it does not, the tensor order or the architecture differs
    # from what this expects and going further would write a plausible but wrong network.
    per_hidden = INPUTS * arguments.buckets + 3
    if (len(values) - 1) % per_hidden != 0:
        raise SystemExit(
            f"{arguments.raw} holds {len(values):,} floats, which is not "
            f"{per_hidden}*HIDDEN + 1 for {arguments.buckets} buckets. Either --buckets does "
            "not match the BUCKETS the trainer used, or save_format has changed."
        )
    hidden = (len(values) - 1) // per_hidden
    rows = INPUTS * arguments.buckets

    at = 0
    weights = values[at : at + rows * hidden].reshape(rows, hidden)
    at += rows * hidden
    biases = values[at : at + hidden]
    at += hidden
    output = values[at : at + 2 * hidden]
    at += 2 * hidden
    output_bias = float(values[at])

    quantised_weights = np.round(weights * QA).astype(np.int16)
    quantised_biases = np.round(biases * QA).astype(np.int16)
    quantised_output = np.round(output * QB).astype(np.int16)

    saturated = int(
        (np.abs(weights * QA) > 32767).sum() + (np.abs(output * QB) > 32767).sum()
    )
    if saturated:
        print(f"warning: {saturated} weights saturated int16 during quantisation")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        weights=quantised_weights,
        biases=quantised_biases,
        output=quantised_output,
        output_bias=np.int32(round(output_bias * QA * QB)),
        hidden=np.int32(hidden),
        buckets=np.int32(arguments.buckets),
        qa=np.int32(QA),
        qb=np.int32(QB),
        scale=np.int32(SCALE),
    )
    print(
        f"wrote {arguments.out}: {hidden} hidden, {arguments.buckets} buckets, "
        f"{arguments.out.stat().st_size / 1e6:.1f} MB"
    )
    print("check it with: uv run python tests/check_nnue.py")


if __name__ == "__main__":
    main()
