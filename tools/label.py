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
import random
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.pgn

DEFAULT_OPENINGS = Path(__file__).resolve().parent / "openings.epd"
MATE_SCORE = 30000
SKIP_OPENING_PLIES = 8
SAMPLE_EVERY = 2


def _selfplay(fen: str, nodes: int, rng: random.Random, random_plies: int) -> tuple[str, str]:
    """Play one game in this process, reusing the already-compiled engine.

    Not through the harness. The harness starts a fresh process per game because the platform
    does, and that fidelity is what makes match.py's measurements honest. Here we only want
    positions, and a process start costs eleven seconds of numba compilation against about two
    seconds of chess, so games are played in-process instead.

    A few random plies are played first. Under a fixed node count the engine is deterministic,
    so without them every game from a given opening would be the same game, and the data would
    be a handful of lines repeated thousands of times.
    """
    import numpy as np

    import searcher
    from position import HASH, from_board, to_uci
    from searcher import STOP

    board = chess.Board(fen)
    for _ in range(random_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(rng.choice(legal))

    searcher.reset()
    history: list[int] = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
        state = from_board(board)
        history.append(int(state[HASH]))
        searcher.STATES[0] = state
        searcher.PATH[:] = 0
        for index, key in enumerate(history):
            searcher.PATH[index] = np.uint64(key)
        searcher.CONTROL[STOP] = 0
        packed = searcher.run(64, len(history) - 1, nodes)
        try:
            move = chess.Move.from_uci(to_uci(packed) if packed else "")
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None and outcome.winner is not None:
        result = "white" if outcome.winner == chess.WHITE else "black"
    elif outcome is not None:
        result = "draw"
    else:
        result = _adjudicate(board)
    return str(chess.pgn.Game.from_board(board)), result


def _adjudicate(board: chess.Board) -> str:
    """Material, the same way the referee settles a game that hits the ply cap."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in values.items()
    )
    if balance > 0:
        return "white"
    if balance < 0:
        return "black"
    return "draw"


def _play_batch(job: tuple[list[str], int, int, int]) -> list[tuple[str, str]]:
    """One worker plays many games, so the engine is compiled once rather than per game."""
    fens, nodes, seed, random_plies = job
    rng = random.Random(seed)
    return [_selfplay(fen, nodes, rng, random_plies) for fen in fens]


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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-plies", type=int, default=4,
                        help="random moves before self-play, to vary the games")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.engine:
        raise SystemExit("pass --engine or set UCI_ENGINE")
    lines = arguments.openings.read_text().splitlines()
    openings = [line.strip() for line in lines if line.strip()]

    # One batch per worker, so each worker pays the engine's compile cost once.
    per_worker = max(1, -(-arguments.games // arguments.workers))
    jobs = []
    for worker in range(arguments.workers):
        first = worker * per_worker
        fens = [
            openings[(arguments.shard * arguments.games + index) % len(openings)]
            for index in range(first, min(first + per_worker, arguments.games))
        ]
        if fens:
            jobs.append((fens, arguments.nodes, arguments.seed + worker, arguments.random_plies))

    print(f"shard {arguments.shard}: playing {arguments.games} games...", flush=True)
    with ProcessPoolExecutor(arguments.workers) as pool:
        played = [game for batch in pool.map(_play_batch, jobs) for game in batch]

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
