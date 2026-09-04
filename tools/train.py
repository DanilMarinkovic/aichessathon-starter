"""Train the network, and quantise it into the exact form the engine reads.

Two things about this file matter more than the training itself.

The features come from `nnue.features`, the same jitted function the engine uses, rather than
from a reimplementation here. A trainer and an engine that disagree about what feature 412 means
produce a network that runs perfectly and plays badly, with no error anywhere to find. Sharing
the function makes that disagreement impossible rather than unlikely.

The quantisation matches nnue.QA, QB and SCALE by construction, and `tools/verify_net.py` then
checks the engine reproduces this model's output on held-out positions. Training in float and
shipping integers is the other place networks die silently.

The target blends the reference engine's score with how the game actually ended, which is the
standard objective: the score is dense and available everywhere, the result is what we actually
care about, and the blend keeps the network from inheriting the reference's confident mistakes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np
import torch
from torch import nn

import nnue
from position import from_board

MAX_FEATURES = 32


def encode(
    path: Path, limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn text positions into feature indices, once, so training can run many epochs.

    Padded to a fixed width with -1, which the model maps to a zero row, because a ragged batch
    of index lists cannot be gathered efficiently.
    """
    white = np.full((limit, MAX_FEATURES), -1, dtype=np.int32)
    black = np.full((limit, MAX_FEATURES), -1, dtype=np.int32)
    target = np.zeros(limit, dtype=np.float32)
    # Which half the engine will read first. nnue.forward puts the side to move's accumulator
    # ahead of the opponent's, so the trainer has to order them the same way or it fits a
    # different network than the one that gets played.
    stm = np.zeros(limit, dtype=np.int64)

    buffer_white = np.zeros(MAX_FEATURES, dtype=np.int32)
    buffer_black = np.zeros(MAX_FEATURES, dtype=np.int32)
    kept = 0
    started = time.perf_counter()

    with path.open() as handle:
        for line in handle:
            if kept >= limit:
                break
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            fen, score, outcome = parts
            board = chess.Board(fen)
            state = np.ascontiguousarray(from_board(board))
            count = nnue.features(state, buffer_white, buffer_black)
            if count > MAX_FEATURES:
                continue

            white[kept, :count] = buffer_white[:count]
            black[kept, :count] = buffer_black[:count]

            # Both halves are from the side to move's point of view, which is what the network
            # is asked to predict.
            centipawns = float(score)
            result = float(outcome)
            if board.turn == chess.BLACK:
                result = 1.0 - result
            evaluation = 1.0 / (1.0 + np.exp(-centipawns / nnue.SCALE))
            target[kept] = 0.6 * evaluation + 0.4 * result
            stm[kept] = 0 if board.turn == chess.WHITE else 1
            kept += 1
            if kept % 200_000 == 0:
                rate = kept / (time.perf_counter() - started)
                print(f"  encoded {kept:,} at {rate:,.0f}/s", flush=True)

    return white[:kept], black[:kept], target[:kept], stm[:kept]


class Network(nn.Module):
    """768 x buckets -> HIDDEN per perspective, concatenated, to one output."""

    def __init__(self) -> None:
        super().__init__()
        self.accumulator = nn.EmbeddingBag(
            nnue.BUCKETS * nnue.INPUTS + 1, nnue.HIDDEN, mode="sum", padding_idx=0
        )
        self.bias = nn.Parameter(torch.zeros(nnue.HIDDEN))
        self.output = nn.Linear(2 * nnue.HIDDEN, 1)
        nn.init.uniform_(self.accumulator.weight, -0.05, 0.05)

    def forward(self, white: torch.Tensor, black: torch.Tensor, stm: torch.Tensor):
        # Index 0 is the padding row, so real features are stored shifted up by one.
        us = self.accumulator(white) + self.bias
        them = self.accumulator(black) + self.bias
        # The side to move's half comes first, matching nnue.forward.
        first = torch.where(stm.unsqueeze(1) == 0, us, them)
        second = torch.where(stm.unsqueeze(1) == 0, them, us)
        hidden = torch.cat([first, second], dim=1).clamp(0.0, 1.0)
        return torch.sigmoid(self.output(hidden)).squeeze(1)


def quantise(model: Network, destination: Path) -> None:
    """Write integer weights in the layout nnue.py expects."""
    with torch.no_grad():
        weights = model.accumulator.weight.detach().cpu().numpy()[1:]
        biases = model.bias.detach().cpu().numpy()
        output = model.output.weight.detach().cpu().numpy()[0]
        output_bias = float(model.output.bias.detach().cpu().numpy()[0])

    quantised_weights = np.round(weights * nnue.QA).astype(np.int16)
    quantised_biases = np.round(biases * nnue.QA).astype(np.int16)
    quantised_output = np.round(output * nnue.QB).astype(np.int16)
    quantised_output_bias = np.int32(round(output_bias * nnue.QA * nnue.QB))

    clipped = int(
        (np.abs(weights * nnue.QA) > 32767).sum() + (np.abs(output * nnue.QB) > 32767).sum()
    )
    if clipped:
        print(f"warning: {clipped} weights saturated int16 during quantisation")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        weights=quantised_weights,
        biases=quantised_biases,
        output=quantised_output,
        output_bias=quantised_output_bias,
        hidden=np.int32(nnue.HIDDEN),
        buckets=np.int32(nnue.BUCKETS),
        qa=np.int32(nnue.QA),
        qb=np.int32(nnue.QB),
        scale=np.int32(nnue.SCALE),
    )
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and quantise the evaluation network.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    parser.add_argument("--cache", type=Path, help="save or reuse the encoded features")
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16_384)
    parser.add_argument("--lr", type=float, default=1e-3)
    arguments = parser.parse_args()

    if arguments.cache and arguments.cache.exists():
        print(f"loading encoded features from {arguments.cache}")
        stored = np.load(arguments.cache)
        white, black, target, stm = (
            stored["white"], stored["black"], stored["target"], stored["stm"]
        )
    else:
        print(f"encoding up to {arguments.limit:,} positions from {arguments.data}")
        white, black, target, stm = encode(arguments.data, arguments.limit)
        if arguments.cache:
            arguments.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                arguments.cache, white=white, black=black, target=target, stm=stm
            )
    print(f"{len(target):,} positions, {nnue.BUCKETS} buckets, {nnue.HIDDEN} hidden")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Use every core the job was given. The single-thread rule is a constraint on the
        # platform the agent plays on, not on training, which happens here and offline.
        cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        torch.set_num_threads(cores)
        print(f"training on cpu with {cores} threads")
    else:
        print(f"training on {torch.cuda.get_device_name(0)}")

    # Shift by one so index 0 can be the padding row EmbeddingBag ignores.
    white_tensor = torch.from_numpy(white.astype(np.int64) + 1).to(device)
    black_tensor = torch.from_numpy(black.astype(np.int64) + 1).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    stm_tensor = torch.from_numpy(stm.astype(np.int64)).to(device)

    split = int(len(target) * 0.98)
    model = Network().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=arguments.lr)
    loss_function = nn.MSELoss()

    for epoch in range(arguments.epochs):
        model.train()
        order = torch.randperm(split, device=device)
        total = 0.0
        batches = 0
        for start in range(0, split, arguments.batch):
            index = order[start : start + arguments.batch]
            optimiser.zero_grad()
            prediction = model(white_tensor[index], black_tensor[index], stm_tensor[index])
            loss = loss_function(prediction, target_tensor[index])
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                # Keep weights inside what int16 can hold once scaled by QA and QB.
                model.accumulator.weight.clamp_(-32767 / nnue.QA, 32767 / nnue.QA)
                model.output.weight.clamp_(-32767 / nnue.QB, 32767 / nnue.QB)
            total += float(loss)
            batches += 1

        model.eval()
        with torch.no_grad():
            held = model(
                white_tensor[split:], black_tensor[split:], stm_tensor[split:]
            )
            validation = float(loss_function(held, target_tensor[split:]))
        print(
            f"epoch {epoch + 1:>3}/{arguments.epochs}  train {total / batches:.5f}  "
            f"held out {validation:.5f}",
            flush=True,
        )

    # The float checkpoint is what tools/verify_net.py compares the engine against, so it is
    # saved alongside the quantised network rather than discarded.
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = arguments.out.with_suffix(".pt")
    torch.save(model.state_dict(), checkpoint)
    print(f"wrote {checkpoint}")
    quantise(model, arguments.out)


if __name__ == "__main__":
    main()
