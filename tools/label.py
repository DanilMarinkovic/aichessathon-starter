"""Build a labelled position set for tuning the evaluation.

Positions come from our own engine's self-play, not from a reference engine's games, because
what the evaluation has to get right is the positions our search actually reaches. Labels come
from a reference engine, which the rules allow offline: the ban covers what ships inside the
zip, not what the weights were learned from.

Only quiet positions are kept. A position where the side to move is in check, or where the best
move is a capture, is dominated by a tactic the search will resolve anyway; fitting a static
evaluation to it teaches the wrong thing. This is the standard filter and it matters more than
the volume of data.

Each line of the output is:

    fen | reference score in centipawns, side to move | game result from White, 0/0.5/1

Sharding lets one array task write one file. Concatenate them afterwards.
"""

import argparse
import io
import os
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.pgn

from harness.referee import play_match
from harness.sandbox import local

DEFAULT_OPENINGS = Path(__file__).resolve().parent / "openings.epd"
MATE_SCORE = 30000
SKIP_OPENING_PLIES = 8
SAMPLE_EVERY = 2


def _init(nodes: int) -> None:
    os.environ["CHESSATHON_FIXED_NODES"] = str(nodes)


def _play(job: tuple[str, Path, int]) -> tuple[str, str]:
    fen, agent, _nodes = job
    outcome = play_match(local(agent), local(agent), 600_000, 0, start_fen=fen)
    return outcome.pgn, outcome.result


class Labeller:
    def __init__(self, path: str, depth: int) -> None:
        self.depth = depth
        self._process = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._await("uciok")
        self._send("setoption name Threads value 1")
        self._send("isready")
        self._await("readyok")

    def _send(self, command: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def _await(self, token: str) -> None:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("labeller died")
            if line.startswith(token):
                return

    def score(self, fen: str) -> tuple[int, str]:
        """Centipawns from the side to move, plus the best move."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        value = 0
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("labeller died mid-search")
            if line.startswith("info ") and " score " in line:
                parts = line.split()
                kind = parts[parts.index("score") + 1]
                raw = int(parts[parts.index("score") + 2])
                value = raw if kind == "cp" else (MATE_SCORE - abs(raw)) * (1 if raw > 0 else -1)
            if line.startswith("bestmove"):
                return value, line.split()[1]

    def close(self) -> None:
        try:
            self._send("quit")
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()


def quiet_positions(pgn_text: str) -> Iterator[tuple[str, int]]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= SKIP_OPENING_PLIES and ply % SAMPLE_EVERY == 0 and not board.is_check():
            yield board.fen(), ply
        board.push(move)


def label_game(job: tuple[str, str, str, int]) -> list[str]:
    pgn_text, result, engine_path, depth = job
    outcome = {"white": "1", "draw": "0.5", "black": "0"}.get(result, "0.5")
    labeller = Labeller(engine_path, depth)
    rows: list[str] = []
    try:
        for fen, _ply in quiet_positions(pgn_text):
            score, best = labeller.score(fen)
            board = chess.Board(fen)
            try:
                move = chess.Move.from_uci(best)
            except ValueError:
                continue
            # A position whose best move is a capture is about a tactic, not about the
            # standing features a static evaluation is built from.
            if move in board.legal_moves and board.is_capture(move):
                continue
            if abs(score) > 1500:
                continue
            rows.append(f"{fen}|{score}|{outcome}")
    finally:
        labeller.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled positions for tuning.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=50_000, help="self-play strength")
    parser.add_argument("--depth", type=int, default=10, help="label depth")
    parser.add_argument("--engine", default=os.environ.get("UCI_ENGINE", ""))
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0, help="array task id, offsets openings")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.engine:
        raise SystemExit("pass --engine or set UCI_ENGINE")
    lines = arguments.openings.read_text().splitlines()
    openings = [line.strip() for line in lines if line.strip()]

    agent = arguments.agent.resolve()
    jobs = [
        (openings[(arguments.shard * arguments.games + index) % len(openings)],
         agent, arguments.nodes)
        for index in range(arguments.games)
    ]

    print(f"shard {arguments.shard}: playing {arguments.games} games...", flush=True)
    with ProcessPoolExecutor(
        arguments.workers, initializer=_init, initargs=(arguments.nodes,)
    ) as pool:
        played = list(pool.map(_play, jobs))

    print(f"shard {arguments.shard}: labelling at depth {arguments.depth}...", flush=True)
    label_jobs = [(pgn, result, arguments.engine, arguments.depth) for pgn, result in played]
    with ProcessPoolExecutor(arguments.workers) as pool:
        batches = list(pool.map(label_game, label_jobs))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for batch in batches for row in batch]
    arguments.out.write_text("\n".join(rows) + "\n")
    print(f"shard {arguments.shard}: wrote {len(rows):,} positions to {arguments.out}")


if __name__ == "__main__":
    main()
