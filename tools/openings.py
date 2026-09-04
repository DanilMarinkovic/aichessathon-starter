"""Build the test rig's starting positions from real opening theory.

Rated games start from curated openings, so the measurement set should look like curated
openings: a handful of moves of theory, handed over to the engine at the point a book would
run out. The lines below are mainlines, written as SAN and checked for legality on the way in.

Each line yields two positions, one early and one at its full depth, which doubles the count
and varies how much book the engine is given before it is on its own.

A short search runs over every position as a sanity check. Theory is balanced by construction,
so a line the engine scores far from level is usually a transcription mistake in this file
rather than a discovery about the opening.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

import searcher
from position import HASH, from_board
from searcher import BEST_SCORE, STOP

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "openings.epd"

# Each line is emitted at these depths, and at its full length. More positions raise the
# ceiling on distinct games, which is what sets the smallest Elo change the rig can resolve:
# under fixed nodes the engine is deterministic, so replaying an opening replays the game.
# Lines that share their early moves collapse to the same position at the shallow depths and
# are deduplicated, so this yields fewer positions than lines x depths.
TRUNCATIONS = (8, 12)

LINES: tuple[tuple[str, str], ...] = (
    # --- 1.e4 e5 ---
    ("Ruy Lopez Morphy", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O"),
    ("Ruy Lopez Berlin", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8"),
    ("Ruy Lopez Exchange", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4 Nxd4 c5"),
    ("Ruy Lopez Closed", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d6 h3"),
    ("Ruy Lopez Open", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6"),
    ("Italian Giuoco Piano", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O"),
    ("Two Knights Defence", "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6"),
    ("Evans Gambit", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O"),
    ("Scotch Game", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7"),
    ("Scotch Gambit", "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 c3 Nf6 e5 d5"),
    ("Petroff Defence", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7"),
    ("Philidor Defence", "e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O"),
    ("Four Knights", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6"),
    ("Vienna Game", "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d4 O-O"),
    ("King's Gambit Accepted", "e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6 d4 d6"),
    ("Ponziani", "e4 e5 Nf3 Nc6 c3 Nf6 d4 Nxe4 d5 Ne7"),
    ("Bishop's Opening", "e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6"),
    # --- Sicilian ---
    ("Najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6"),
    ("Najdorf English Attack", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f3 e5 Nb3 Be6 Qd2 Nbd7"),
    ("Dragon", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6"),
    ("Accelerated Dragon", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Nf6 Nc3 d6 Be2 Nxd4 Qxd4 Bg7"),
    ("Scheveningen", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be2 Be7 O-O O-O"),
    ("Sicilian Classical", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6 Qd2 Be7"),
    ("Sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 Bg5 a6 Na3 b5"),
    ("Taimanov", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Bd3 Nf6"),
    ("Kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nf6 O-O d6 c4 g6"),
    ("Rossolimo", "e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 Re1 Nf6 c3 O-O"),
    ("Moscow Variation", "e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 O-O Nc6 c3 Nf6"),
    ("Alapin", "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6"),
    ("Closed Sicilian", "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 Nf6"),
    ("Grand Prix Attack", "e4 c5 Nc3 Nc6 f4 g6 Nf3 Bg7 Bc4 e6 O-O Nge7"),
    # --- French ---
    ("French Winawer", "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7"),
    ("French Classical", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7"),
    ("French Tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6"),
    ("French Advance", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3 Nh6"),
    ("French Rubinstein", "e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6"),
    ("French Exchange", "e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Bd6 O-O O-O"),
    # --- Caro-Kann ---
    ("Caro-Kann Classical", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7"),
    ("Caro-Kann Advance", "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Qb6"),
    ("Caro-Kann Panov", "e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Be7"),
    ("Caro-Kann Exchange", "e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4"),
    ("Caro-Kann Two Knights", "e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6"),
    # --- Other replies to 1.e4 ---
    ("Pirc Classical", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6"),
    ("Pirc Austrian Attack", "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O Bd3 Nc6"),
    ("Modern Defence", "e4 g6 d4 Bg7 Nc3 d6 Nf3 a6 a4 Nf6 Bd3 O-O"),
    ("Alekhine Defence", "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7"),
    ("Scandinavian Qa5", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5"),
    ("Scandinavian Nf6", "e4 d5 exd5 Nf6 d4 Nxd5 Nf3 g6 c4 Nb6 Nc3 Bg7"),
    # --- Queen's Gambit ---
    ("QGD Orthodox", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6"),
    ("QGD Tartakower", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 Nf3 h6 Bh4 O-O e3 b6"),
    ("QGD Lasker", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 Nf3 h6 Bh4 Ne4 Bxe7 Qxe7"),
    ("QGD Exchange", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 c6 Bd3 Nbd7"),
    ("Cambridge Springs", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Nbd7 Nf3 c6 e3 Qa5"),
    ("Queen's Gambit Accepted", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6"),
    ("Slav Defence", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4"),
    ("Semi-Slav Meran", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5"),
    ("Semi-Slav Botvinnik", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 dxc4 e4 b5"),
    ("Chigorin Defence", "d4 d5 c4 Nc6 Nf3 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5"),
    ("London System", "d4 d5 Bf4 Nf6 e3 e6 Nf3 Bd6 Bg3 O-O Bd3 c5"),
    ("Colle System", "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 c3 Nc6 Nbd2 Bd6"),
    ("Torre Attack", "d4 Nf6 Nf3 e6 Bg5 c5 e3 Be7 Nbd2 O-O c3 b6"),
    ("Catalan", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4"),
    # --- Indian defences ---
    ("Nimzo-Indian Rubinstein", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6"),
    ("Nimzo-Indian Classical", "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6"),
    ("Nimzo-Indian Samisch", "d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 e3 O-O"),
    ("Queen's Indian", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O"),
    ("Bogo-Indian", "d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Bxd2+ Qxd2 O-O g3 d5 Bg2 Qe7"),
    ("King's Indian Classical", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6"),
    ("King's Indian Samisch", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 Nh5"),
    ("King's Indian Fianchetto", "d4 Nf6 c4 g6 Nc3 Bg7 Nf3 O-O g3 d6 Bg2 Nbd7 O-O e5"),
    ("King's Indian Four Pawns", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5 d5 e6"),
    ("Grunfeld Exchange", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5"),
    ("Grunfeld Russian", "d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O e4 Bg4"),
    ("Modern Benoni", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7"),
    # A material gambit is poor measuring ground: played from both colours it tends to score
    # 1-1 on the sacrifice rather than on the change under test. Tarrasch instead.
    ("Tarrasch Defence", "d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6 Bg2 Be7"),
    ("Old Indian", "d4 Nf6 c4 d6 Nc3 e5 Nf3 Nbd7 e4 Be7 Be2 O-O"),
    ("Budapest Gambit", "d4 Nf6 c4 e5 dxe5 Ng4 Nf3 Nc6 Bf4 Bb4+ Nbd2 Qe7"),
    # --- Dutch ---
    ("Dutch Leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6"),
    ("Dutch Stonewall", "d4 f5 g3 Nf6 Bg2 e6 Nf3 d5 O-O Bd6 c4 c6"),
    ("Dutch Classical", "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d6"),
    # --- English and flank ---
    ("English Symmetrical", "c4 c5 Nf3 Nf6 Nc3 Nc6 g3 g6 Bg2 Bg7 O-O O-O"),
    ("English Reversed Sicilian", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6"),
    ("English Four Knights", "c4 e5 Nc3 Nf6 Nf3 Nc6 d4 exd4 Nxd4 Bb4 Bg5 h6"),
    ("English vs King's Indian", "c4 g6 Nc3 Bg7 g3 Nf6 Bg2 O-O Nf3 d6 O-O Nc6"),
    ("Anglo-Indian", "c4 Nf6 Nc3 e6 Nf3 d5 d4 Be7 Bg5 O-O e3 h6"),
    ("Reti Opening", "Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7 O-O O-O b3 c5"),
    ("King's Indian Attack", "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5"),
    ("Bird's Opening", "f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5"),
    ("Larsen Attack", "b3 e5 Bb2 Nc6 e3 Nf6 Bb5 Bd6 Na3 O-O"),
)


def _score(board: chess.Board, nodes: int) -> int:
    searcher.reset()
    state = from_board(board)
    searcher.STATES[0] = state
    searcher.PATH[:] = 0
    searcher.PATH[0] = np.uint64(state[HASH])
    searcher.CONTROL[STOP] = 0
    searcher.run(64, 0, nodes)
    return int(searcher.CONTROL[BEST_SCORE])


def _positions(name: str, line: str) -> list[tuple[str, str]]:
    """Replay a line, returning its early and full positions. Raises if the SAN is wrong."""
    board = chess.Board()
    fens: list[tuple[str, str]] = []
    moves = line.split()
    for index, san in enumerate(moves, start=1):
        try:
            board.push_san(san)
        except ValueError as error:
            raise SystemExit(f"{name}: illegal move {san!r} at ply {index}: {error}") from None
        if index in TRUNCATIONS:
            fens.append((f"{name} (ply {index})", board.fen()))
    if len(moves) not in TRUNCATIONS:
        fens.append((f"{name} (ply {len(moves)})", board.fen()))
    return fens


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the opening set used by the test rig.")
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--warn-cp", type=int, default=150)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-check", action="store_true", help="skip the balance sanity check")
    arguments = parser.parse_args()

    seen: set[str] = set()
    kept: list[str] = []
    suspicious: list[tuple[str, int, str]] = []

    for name, line in LINES:
        for label, fen in _positions(name, line):
            signature = " ".join(fen.split(" ")[:4])
            if signature in seen:
                continue
            seen.add(signature)
            kept.append(fen)
            if arguments.no_check:
                continue
            board = chess.Board(fen)
            score = _score(board, arguments.nodes)
            if board.turn == chess.BLACK:
                score = -score
            if abs(score) > arguments.warn_cp:
                suspicious.append((label, score, fen))

    arguments.out.write_text("\n".join(kept) + "\n")
    print(f"wrote {len(kept)} positions from {len(LINES)} lines to {arguments.out}")

    if suspicious:
        print(f"\n{len(suspicious)} positions the engine does not think are level:")
        for label, score, fen in suspicious:
            print(f"  {score:+5d} cp  {label:38s} {fen}")
        print("check these lines for a transcription error before trusting the set")


if __name__ == "__main__":
    main()
