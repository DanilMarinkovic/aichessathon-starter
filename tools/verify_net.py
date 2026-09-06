"""Gate 3: prove the engine evaluates the same network the trainer fitted.

This is the failure that has no symptom. A network whose weights were quantised wrongly, or
whose features are indexed differently at inference than during training, loads fine, runs at
full speed, and simply plays worse. There is no exception to catch and no test that fails unless
one is written for exactly this.

So: run the float model from the checkpoint and the engine's integer forward pass over the same
positions, and compare. They cannot agree exactly, because one is float and the other is
quantised integers, but they must agree closely and without bias. A large mean difference means
the scaling is wrong; a large spread means the indexing is.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import torch

import nnue
from position import OCC_ALL, SIDE, from_board
from tools.train import MAX_FEATURES, Network


def engine_scores(fens: list[str]) -> np.ndarray:
    values, boards = nnue.new_cache()
    accumulator = nnue.new_accumulator()
    out = np.zeros(len(fens), dtype=np.float64)
    for index, fen in enumerate(fens):
        state = np.ascontiguousarray(from_board(chess.Board(fen)))
        nnue.refresh(state, accumulator, values, boards)
        out[index] = float(
            nnue.forward(accumulator, int(state[SIDE]), nnue.output_bucket(state[OCC_ALL]))
        )
    # The engine deliberately reports in calibrated units rather than the units it trained
    # through, so undo that before comparing against the model. Without this the check reads a
    # correctly quantised network as a scaling error, which is exactly the alarm it exists to
    # raise and would be a false one. Uncalibrated networks have the two equal and divide by one.
    return out * (nnue.SCALE / nnue.EVAL_SCALE)


def model_scores(model: Network, fens: list[str]) -> np.ndarray:
    white = np.zeros((len(fens), MAX_FEATURES), dtype=np.int64)
    black = np.zeros((len(fens), MAX_FEATURES), dtype=np.int64)
    stm = np.zeros(len(fens), dtype=np.int64)
    buffer_white = np.zeros(MAX_FEATURES, dtype=np.int32)
    buffer_black = np.zeros(MAX_FEATURES, dtype=np.int32)
    for index, fen in enumerate(fens):
        state = np.ascontiguousarray(from_board(chess.Board(fen)))
        count = nnue.features(state, buffer_white, buffer_black)
        white[index, :count] = buffer_white[:count] + 1
        black[index, :count] = buffer_black[:count] + 1
        stm[index] = int(state[SIDE])

    with torch.no_grad():
        probability = model(
            torch.from_numpy(white), torch.from_numpy(black), torch.from_numpy(stm)
        ).numpy()
    # Undo the sigmoid the model trains through, back into centipawns.
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)) * nnue.SCALE


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the engine matches the trained model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the .pt state dict")
    parser.add_argument("--positions", type=Path, required=True, help="an epd from label.py")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=25.0, help="mean absolute cp")
    arguments = parser.parse_args()

    if not nnue.TRAINED:
        raise SystemExit("nnue has no trained weights loaded; train and place weights/net.npz")

    lines = arguments.positions.read_text().splitlines()
    fens = [line.split("|")[0] for line in lines[: arguments.count] if line.strip()]
    print(f"comparing {len(fens):,} positions")

    model = Network()
    model.load_state_dict(torch.load(arguments.checkpoint, map_location="cpu"))
    model.eval()

    theirs = model_scores(model, fens)
    ours = engine_scores(fens)
    difference = ours - theirs

    mean = float(np.mean(difference))
    spread = float(np.std(difference))
    worst = float(np.max(np.abs(difference)))
    correlation = float(np.corrcoef(ours, theirs)[0, 1])

    print(f"  mean difference   {mean:+8.2f} cp   (scaling)")
    print(f"  spread            {spread:8.2f} cp   (indexing and rounding)")
    print(f"  worst case        {worst:8.2f} cp")
    print(f"  correlation       {correlation:8.4f}")

    if correlation < 0.99:
        raise SystemExit(
            "the engine and the model disagree about what these positions are worth. That is a "
            "feature indexing mismatch, not rounding: check that both sides use nnue.features."
        )
    if abs(mean) > arguments.tolerance:
        raise SystemExit(
            f"systematic offset of {mean:+.1f}cp. The quantisation scale is wrong somewhere; "
            "check QA, QB and SCALE agree between tools/train.py and nnue.py."
        )
    print("\nthe engine reproduces the trained network")


if __name__ == "__main__":
    main()
