"""Find out what kind of mistakes the engine actually makes.

Games tell you that you lost. They do not tell you why. This plays a batch of games, then walks
every position with a strong reference engine and measures how much evaluation each of our moves
threw away. The moves that lose the most, grouped by phase and by kind, are the shortest route
to knowing where the remaining strength is hiding.

The reference engine is used offline, on this machine, purely to label positions. That is the
side of the line the rules allow: the ban covers what ships and runs inside the zip.

One evaluation per position, not two. The score after our move is the same position the
opponent then moves from, so walking the game once and taking consecutive differences gives
every move's loss for half the work.
"""

import argparse
import io
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.pgn

from harness.referee import play_match
from harness.sandbox import local

DEFAULT_OPENINGS = Path(__file__).resolve().parent / "openings.epd"
MATE_SCORE = 30000
# Beyond this the position is decided and a further "mistake" is not informative.
DECIDED_CP = 900
# A score this large is a mate, not an evaluation. Mate scores must never be summed with
# centipawn losses: one of them is worth about 29,000 and drowns a whole game's real errors.
# In the game that exposed this, a single entry was 97% of the reported total and pointed the
# analysis at an endgame that was already lost, while eleven genuine 40-90cp middlegame errors
# -- the actual reason the game was lost -- sat below it looking like rounding.
MATE_THRESHOLD = MATE_SCORE - 1000

# What kind of error a move was. Centipawn losses are comparable to each other and get summed;
# the mate transitions are events, counted rather than added.
ORDINARY = "cp"
ALLOWED_MATE = "allowed mate"
MISSED_MATE = "missed mate"


@dataclass(frozen=True)
class Mistake:
    loss: int
    fen: str
    played: str
    best: str
    phase: str
    ply: int
    was_capture: bool
    best_is_capture: bool
    in_check: bool
    kind: str = ORDINARY


class Reference:
    """A full-strength UCI engine used only to score positions."""

    def __init__(self, path: str, depth: int) -> None:
        self.depth = depth
        self._process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
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

    def _await(self, token: str) -> str:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"reference engine died waiting for {token}")
            if line.startswith(token):
                return line.strip()

    def score(self, board: chess.Board, restrict: str = "") -> tuple[int, str]:
        """Centipawns from White's point of view, plus the engine's preferred move.

        `restrict` limits the search to one root move, which is how a move is priced against
        the best move on equal terms: same root, same depth, same search.
        """
        self._send(f"position fen {board.fen()}")
        limit = f" searchmoves {restrict}" if restrict else ""
        self._send(f"go depth {self.depth}{limit}")
        value = 0
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("reference engine died during search")
            if line.startswith("info ") and " score " in line:
                parts = line.split()
                kind = parts[parts.index("score") + 1]
                raw = int(parts[parts.index("score") + 2])
                value = raw if kind == "cp" else (MATE_SCORE - abs(raw)) * (1 if raw > 0 else -1)
            if line.startswith("bestmove"):
                best = line.split()[1]
                break
        if board.turn == chess.BLACK:
            value = -value
        return value, best

    def close(self) -> None:
        try:
            self._send("quit")
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()


def phase_of(board: chess.Board) -> str:
    pieces = sum(
        len(board.pieces(kind, colour))
        for kind in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for colour in (chess.WHITE, chess.BLACK)
    )
    if pieces >= 11:
        return "opening"
    if pieces >= 5:
        return "middlegame"
    return "endgame"


def analyse_game(job: tuple[str, bool, str, int]) -> list[Mistake]:
    pgn_text, we_are_white, engine_path, depth = job
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []

    reference = Reference(engine_path, depth)
    try:
        board = game.board()
        our_colour = chess.WHITE if we_are_white else chess.BLACK
        sign = 1 if we_are_white else -1
        mistakes: list[Mistake] = []

        for index, move in enumerate(game.mainline_moves()):
            if board.turn != our_colour:
                board.push(move)
                continue

            # Both searches share a root, a depth and a side to move, so the difference
            # between them is the cost of the move rather than the noise between two searches.
            best_value, best = reference.score(board)
            if move.uci() == best:
                # The reference would have played it. Any difference a second search reports
                # is its own noise, not a cost we paid.
                board.push(move)
                continue
            played_value, _ = reference.score(board, restrict=move.uci())
            best_pov = best_value * sign
            played_pov = played_value * sign

            # Classify before measuring. Subtracting a mate score from a centipawn one produces
            # a number in the tens of thousands that means nothing on the centipawn scale.
            if best_pov <= -MATE_THRESHOLD:
                # Every move loses; there was nothing to throw away. Charging for the choice
                # between two forced mates is how a lost endgame comes to dominate a report.
                board.push(move)
                continue
            if played_pov <= -MATE_THRESHOLD:
                kind, loss = ALLOWED_MATE, 0
            elif best_pov >= MATE_THRESHOLD and played_pov < MATE_THRESHOLD:
                kind, loss = MISSED_MATE, 0
            elif best_pov >= MATE_THRESHOLD:
                # Mate either way, just slower. Technique, not a mistake.
                board.push(move)
                continue
            else:
                kind, loss = ORDINARY, best_pov - played_pov

            keep = loss > 0 if kind == ORDINARY else True
            if abs(best_pov) <= DECIDED_CP and keep:
                mistakes.append(
                    Mistake(
                        kind=kind,
                        loss=int(loss),
                        fen=board.fen(),
                        played=move.uci(),
                        best=best,
                        phase=phase_of(board),
                        ply=index,
                        was_capture=board.is_capture(move),
                        best_is_capture=board.is_capture(chess.Move.from_uci(best))
                        if best not in ("(none)", "0000")
                        else False,
                        in_check=board.is_check(),
                    )
                )
            board.push(move)
        return mistakes
    finally:
        reference.close()


def play_games(agent: Path, opponent: Path, openings: list[str], count: int, nodes: int,
               base_ms: int, workers: int) -> list[tuple[str, bool]]:
    jobs = [
        (openings[index % len(openings)], index % 2 == 0, agent, opponent, nodes, base_ms)
        for index in range(count)
    ]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(nodes,)) as pool:
        return list(pool.map(_play_one, jobs))


def _init(nodes: int) -> None:
    if nodes > 0:
        os.environ["CHESSATHON_FIXED_NODES"] = str(nodes)


def _play_one(job: tuple[str, bool, Path, Path, int, int]) -> tuple[str, bool]:
    fen, we_are_white, agent, opponent, _nodes, base_ms = job
    white, black = (agent, opponent) if we_are_white else (opponent, agent)
    outcome = play_match(local(white), local(black), base_ms, 0, start_fen=fen)
    return outcome.pgn, we_are_white


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the engine's most costly mistakes.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, help="agent to play; omit when using --pgn")
    parser.add_argument(
        "--pgn", type=Path, help="analyse games from a file instead of playing new ones"
    )
    parser.add_argument("--colour", choices=("white", "black"), default="white",
                        help="which side we played in --pgn games")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--nodes", type=int, default=300_000)
    parser.add_argument("--depth", type=int, default=14, help="reference engine depth")
    parser.add_argument("--engine", default=os.environ.get("UCI_ENGINE", ""))
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--threshold", type=int, default=100, help="cp loss counted as a mistake")
    arguments = parser.parse_args()

    if not arguments.engine:
        raise SystemExit("pass --engine or set UCI_ENGINE to a reference engine binary")
    lines = arguments.openings.read_text().splitlines()
    openings = [line.strip() for line in lines if line.strip()]

    if arguments.pgn:
        stream = io.StringIO(arguments.pgn.read_text())
        games = []
        while True:
            parsed = chess.pgn.read_game(stream)
            if parsed is None:
                break
            games.append((str(parsed), arguments.colour == "white"))
        print(f"analysing {len(games)} game(s) from {arguments.pgn}, we played "
              f"{arguments.colour}", flush=True)
    else:
        if not arguments.opponent:
            raise SystemExit("pass --opponent to play games, or --pgn to analyse existing ones")
        print(f"playing {arguments.games} games at {arguments.nodes:,} nodes...", flush=True)
        games = play_games(
            arguments.agent.resolve(), arguments.opponent.resolve(), openings,
            arguments.games, arguments.nodes, 600_000, arguments.workers,
        )

    print(f"analysing with {arguments.engine} at depth {arguments.depth}...", flush=True)
    jobs = [(pgn, white, arguments.engine, arguments.depth) for pgn, white in games]
    with ProcessPoolExecutor(arguments.workers) as pool:
        found = list(pool.map(analyse_game, jobs))

    every = [m for batch in found for m in batch]
    # Mate events are not filtered by a centipawn threshold, because they do not have a
    # centipawn size. They are reported on their own terms, above the ranked list.
    mate_events = [m for m in every if m.kind != ORDINARY]
    mistakes = sorted(
        (m for m in every if m.kind == ORDINARY and m.loss >= arguments.threshold),
        key=lambda m: -m.loss,
    )

    if mate_events:
        allowed = sum(1 for m in mate_events if m.kind == ALLOWED_MATE)
        missed = sum(1 for m in mate_events if m.kind == MISSED_MATE)
        print(f"\nforced mates: allowed {allowed}, missed {missed}")
        for mistake in mate_events:
            print(
                f"  {mistake.kind:<12} played {mistake.played:<6} best {mistake.best:<6} "
                f"{mistake.phase:<11} {mistake.fen}"
            )

    if not mistakes:
        print("\nno centipawn mistakes above the threshold")
        return

    total = sum(m.loss for m in mistakes)
    print(f"\n{len(mistakes)} mistakes over {arguments.threshold}cp, {total:,}cp thrown away")

    by_phase: Counter[str] = Counter()
    cost_by_phase: Counter[str] = Counter()
    for mistake in mistakes:
        by_phase[mistake.phase] += 1
        cost_by_phase[mistake.phase] += mistake.loss
    print("\n  phase        count   total cp   mean cp")
    for phase in ("opening", "middlegame", "endgame"):
        count = by_phase.get(phase, 0)
        if count:
            cost = cost_by_phase[phase]
            print(f"  {phase:<12} {count:>5}   {cost:>8,}   {cost // count:>7}")

    checks = sum(1 for m in mistakes if m.in_check)
    captures = sum(1 for m in mistakes if m.was_capture)
    missed_captures = sum(1 for m in mistakes if m.best_is_capture and not m.was_capture)
    print(f"\n  played while in check:        {checks}")
    print(f"  the move played was a capture: {captures}")
    print(f"  best move was a capture we did not play: {missed_captures}")

    print(f"\ntop {arguments.top} by cost:")
    for mistake in mistakes[: arguments.top]:
        print(
            f"  -{mistake.loss:>5}cp  played {mistake.played:<6} best {mistake.best:<6} "
            f"{mistake.phase:<11} {mistake.fen}"
        )


if __name__ == "__main__":
    main()
