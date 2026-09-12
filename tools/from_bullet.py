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
    l1w   OB x 2 x HIDDEN  output weights, one bank per output bucket, and within a bank the
                           side to move's half first, then the opponent's
    l1b   OB               one output bias per bucket

The trainer saves l1w transposed, so a bank's 2 x HIDDEN weights are contiguous and nnue.py
can take a row. With one bucket that is the same bytes as before output buckets existed, which
is what makes OUTPUT_BUCKETS=1 an exact control rather than an approximate one.

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
    parser.add_argument(
        "--output-buckets", type=int, default=1,
        help="output buckets the trainer used; must match OUTPUT_BUCKETS in the bullet run",
    )
    arguments = parser.parse_args()

    values = np.fromfile(arguments.raw, dtype=np.float32)

    # The layout is fully determined by one unknown, so the file size has to agree with a whole
    # number of hidden neurons. If it does not, the tensor order or the architecture differs
    # from what this expects and going further would write a plausible but wrong network.
    output_buckets = arguments.output_buckets
    per_hidden = INPUTS * arguments.buckets + 1 + 2 * output_buckets
    if (len(values) - output_buckets) % per_hidden != 0:
        raise SystemExit(
            f"{arguments.raw} holds {len(values):,} floats, which is not "
            f"{per_hidden}*HIDDEN + {output_buckets} for {arguments.buckets} king buckets and "
            f"{output_buckets} output buckets. Either --buckets or --output-buckets does not "
            "match the trainer, or save_format has changed."
        )
    hidden = (len(values) - output_buckets) // per_hidden
    rows = INPUTS * arguments.buckets

    at = 0
    weights = values[at : at + rows * hidden].reshape(rows, hidden)
    at += rows * hidden
    biases = values[at : at + hidden]
    at += hidden
    # (2*HIDDEN, OB) read row-major, then transposed to give each bucket a contiguous row.
    # Reading it directly as (OB, 2*HIDDEN) interleaves neurons across buckets: the network
    # loads, quantises and plays, and evaluates like noise. The screening step caught it at a
    # correlation of -0.0667 against the labelling engine, where a sound network scores 0.97.
    # The tell is that adjacent material buckets should hold similar weights -- correlation
    # 0.71 between bucket 0 and 1 under this reading, -0.11 under the other.
    output = (
        values[at : at + output_buckets * 2 * hidden]
        .reshape(2 * hidden, output_buckets)
        .T.copy()
    )
    at += output_buckets * 2 * hidden
    output_bias = values[at : at + output_buckets]

    quantised_weights = np.round(weights * QA).astype(np.int16)
    quantised_biases = np.round(biases * QA).astype(np.int16)
    quantised_output = np.round(output * QB).astype(np.int16)

    saturated = int(
        (np.abs(weights * QA) > 32767).sum() + (np.abs(output * QB) > 32767).sum()
    )
    if saturated:
        print(f"warning: {saturated} weights saturated int16 during quantisation")

    # How far training actually got, taken from the checkpoint directory bullet named
    # `<net_id>-<superbatch>`. It travels with the weights because the alternative is reading it
    # off a training log that lives in a different file, on a different machine, for a job that
    # may have been killed at its wall limit. A network that trained 47 of 380 superbatches and
    # one that trained all 380 are indistinguishable once they are both an .npz, and the second
    # thing anyone does with an .npz is play a two hour match and write down the Elo.
    trained = arguments.raw.resolve().parent.name.rsplit("-", 1)[-1]
    superbatches = int(trained) if trained.isdigit() else 0

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        weights=quantised_weights,
        biases=quantised_biases,
        output=quantised_output,
        output_bias=np.round(output_bias * QA * QB).astype(np.int32),
        hidden=np.int32(hidden),
        buckets=np.int32(arguments.buckets),
        qa=np.int32(QA),
        qb=np.int32(QB),
        scale=np.int32(SCALE),
        superbatches=np.int32(superbatches),
    )
    print(
        f"wrote {arguments.out}: {hidden} hidden, {arguments.buckets} king buckets, "
        f"{output_buckets} output buckets, {arguments.out.stat().st_size / 1e6:.1f} MB"
    )
    print(f"  trained for {superbatches} superbatches (from {arguments.raw.parent.name})")

    # Read it back before claiming success. A .npz written while the filesystem is full comes
    # out the right length with a bad CRC, and numpy only notices when something later tries to
    # load an array from it. That happened on 8 September: a network trained for an hour was
    # written under an exhausted quota, reported as written, and turned out to be unreadable
    # only when a match tried to play it.
    check = np.load(arguments.out)
    for name in ("weights", "biases", "output", "output_bias"):
        _ = check[name].shape
    print("  verified readable")
    print("check it with: uv run python tests/check_nnue.py")


if __name__ == "__main__":
    main()
