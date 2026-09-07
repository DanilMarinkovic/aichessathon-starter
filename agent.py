"""The submission entrypoint. The platform imports this file and calls get_move.

The engine is a bitboard alpha-beta search compiled by numba: `bitboards` builds the attack
tables, `position` and `movegen` make and generate moves, `evaluate` scores a leaf, and
`searcher` runs iterative deepening. This file owns the two things that touch the outside
world, the clock and the reply.

Time control works around a numba limitation: jitted code cannot read a clock, so a timer
thread flips a flag the search polls. The search runs with the GIL released, which is what
lets that thread run at all.

Whatever the search returns is checked against python-chess before it is sent. A losing
position is recoverable and an illegal move is not, so the last word belongs to the same
library the referee uses.
"""

import os
import threading
import time

import chess
import numpy as np

import nnue
import searcher
from position import HASH, from_board, to_uci
from searcher import (
    BEST_DEPTH,
    BEST_SCORE,
    MAX_PLY,
    NODES,
    PATH_LIMIT,
    SOFT,
    STABLE,
    STOP,
    USE_NNUE,
)

# Wall time the referee charges us that we never see: the JSON round trip, building the board,
# and handing the move back. Measured at a few milliseconds; held well clear of that.
OVERHEAD_MS = 70
MAX_DEPTH = 64
MAX_HISTORY = (PATH_LIMIT - MAX_PLY - 16) // 2

# Search a fixed number of nodes instead of consulting the clock. Only the local test rig sets
# this. A timed game measures the machine as much as the engine: two agents sharing a loaded
# box reach different depths from one run to the next, so an A/B result partly reports which
# process got the core. A fixed node count is deterministic and immune to load, which is what
# makes twenty concurrent games comparable. Unset on the platform, where the clock rules.
FIXED_NODES = int(os.environ.get("CHESSATHON_FIXED_NODES", "0"))

# The increment is worth a large share of the per-move budget, and we are never told it. The
# referee's clock arithmetic gives it away though: what we are handed this move is what was
# left last move, minus what we spent, plus the increment. Measuring it rather than assuming
# 500 ms keeps the same time management honest under the harness's faster controls.
_increment_ms = -1.0
_last_left = -1
_last_spent = 0.0

# Zobrist keys of the game so far. The platform sends a FEN and nothing else, so this is the
# only record of the game there is, and it has to carry both parities.
#
# `_history` holds the positions we were asked to move in, at even offsets in PATH. `_after`
# holds the position each of our chosen moves produced, at the odd offsets between them. A
# repetition scan steps back in twos to stay on one side to move, so with only the even slots
# filled it walks along a row of zeros whenever it starts from an odd ply -- which is every
# position that arises immediately after our own move.
#
# That blind spot cost three drawn games. In two of them the engine was winning by five pawns
# and repeated a position it had itself created two moves earlier, because the position it was
# repeating was one where the opponent was to move, and nothing in this record mentioned it.
# We never see the opponent's turn, but we do choose our own move, so the position it leads to
# is knowable and belongs here.
_history: list[int] = []
_after: list[int] = []

_fallback_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _budget_ms(time_left_ms: int) -> float:
    """How long to think. A flag is a loss, so this stays conservative.

    Dividing the remaining clock rather than spending a constant makes the allocation decay
    on its own: whatever happens, the next move gets a fraction of what is left, and the
    increment keeps that fraction from collapsing.

    The divisor is probably too cautious. The platform's records for seventeen rated games show
    78% of the available time used, one game ending with 76.8 seconds unspent, and -- because
    the allocation only ever decays -- a slowest move of exactly 4.8 seconds in every game, the
    opening allocation. The engine can never think harder about a critical middlegame position
    than about its first move out of book. Replaying the formula over those real game lengths,
    a divisor of 18 would spend 90% and still leave 4.4 seconds in the worst case.

    It stays at 26 until that is measured. More thinking time is not automatically more Elo:
    on the position that lost round 14, twenty times the search picked the same losing move.
    """
    usable = max(1.0, time_left_ms - OVERHEAD_MS)
    increment = max(0.0, _increment_ms)
    budget = usable / 26.0 + increment * 0.6
    return max(1.0, min(budget, usable * 0.35))


def _observe_clock(time_left_ms: int) -> None:
    """Infer the increment from how the clock moved since our last move."""
    global _increment_ms
    if _last_left < 0:
        return
    observed = time_left_ms - _last_left + _last_spent
    if 0.0 <= observed <= 5000.0:
        _increment_ms = observed if _increment_ms < 0 else min(_increment_ms, observed)


def _record(key: int) -> int:
    """Append a position we are to move in, and return its index in PATH."""
    if not _history or _history[-1] != key:
        _history.append(key)
    if len(_history) > MAX_HISTORY:
        # Drop the same number from both, not down to the same length. `_after` always runs one
        # entry behind `_history` -- the move for the current position has not been chosen yet
        # -- so trimming each to 256 would slide them out of step by one, and every repetition
        # test afterwards would compare a position against the wrong ply. The platform caps a
        # game at 300 plies and MAX_HISTORY is 968 of our own moves, so this cannot fire in a
        # rated game; it is right because a silent off-by-one here has no symptom.
        drop = len(_history) - 256
        del _history[:drop]
        del _after[:drop]
    return (len(_history) - 1) * 2


def _record_after(board: chess.Board, move: chess.Move) -> None:
    """Append the position our chosen move produces, so a repetition of it can be seen.

    Trimmed to match `_history`, because the two are indexed in step: entry k of one sits at
    PATH[2k] and entry k of the other at PATH[2k + 1].
    """
    board.push(move)
    try:
        key = int(from_board(board)[HASH])
    finally:
        board.pop()
    while len(_after) >= len(_history):
        _after.pop()
    _after.append(key)


def _fallback(board: chess.Board) -> str:
    """A legal move for when the engine cannot supply one. Grabs the most valuable piece.

    It still records what it played. `_history` and `_after` are indexed in lockstep, so a ply
    that goes unrecorded here would put every later entry on the wrong parity and quietly break
    repetition detection for the rest of the game.
    """
    best = None
    best_value = -1
    for move in board.legal_moves:
        captured = board.piece_at(move.to_square)
        value = _fallback_values[captured.piece_type] if captured else 0
        if value > best_value:
            best_value = value
            best = move
    if best is None:
        return "0000"
    _record_after(board, best)
    return best.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    global _last_left, _last_spent
    started = time.monotonic()
    _observe_clock(time_left_ms)
    try:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return "0000"
        if len(legal) == 1:
            _record(int(from_board(board)[HASH]))
            _record_after(board, legal[0])
            return legal[0].uci()
        try:
            return _think(board, time_left_ms)
        except Exception as error:  # the clock runs through a bug; a legal move does not
            print(f"search failed, falling back: {type(error).__name__}: {error}")
            return _fallback(board)
    finally:
        _last_left = time_left_ms
        _last_spent = (time.monotonic() - started) * 1000.0


def _think(board: chess.Board, time_left_ms: int) -> str:
    state = from_board(board)
    root_offset = _record(int(state[HASH]))

    searcher.STATES[0] = state
    searcher.PATH[: root_offset + MAX_PLY + 8] = 0
    for index, key in enumerate(_history):
        searcher.PATH[index * 2] = np.uint64(key)
    for index, key in enumerate(_after):
        searcher.PATH[index * 2 + 1] = np.uint64(key)

    if FIXED_NODES > 0:
        # The timer stays as a hang guard only; with a sane clock the node limit binds first.
        budget = max(1.0, time_left_ms * 0.5)
        node_limit = FIXED_NODES
    else:
        budget = _budget_ms(time_left_ms)
        # A node ceiling in case the timer thread never fires. At the speeds this engine runs
        # it is several times the nodes the budget can buy, so it never binds on a healthy move.
        node_limit = int(hard_ceiling(budget, time_left_ms) / 1000.0 * 20_000_000) + 100_000
    searcher.CONTROL[STOP] = 0
    searcher.CONTROL[SOFT] = 0

    # Two deadlines rather than one. The soft deadline is the budget the clock allows; reaching
    # it ends the search unless the root move changed on the last completed iteration, in which
    # case the search may run on to the hard deadline. A position still changing its mind is the
    # one where another ply is worth most, and a flat budget spends the same on it as on a
    # position that settled at depth six.
    #
    # The hard deadline never exceeds the safety cap the budget itself already obeys, so the
    # extension cannot cause a flag fall: whatever fraction of the remaining clock was
    # considered safe for one move stays the ceiling.
    hard = hard_ceiling(budget, time_left_ms)
    soft_timer = threading.Timer(budget / 1000.0, _raise_soft)
    hard_timer = threading.Timer(hard / 1000.0, _raise_flag)
    soft_timer.daemon = True
    hard_timer.daemon = True
    began = time.monotonic()
    soft_timer.start()
    hard_timer.start()
    try:
        packed = searcher.run(MAX_DEPTH, root_offset, node_limit)
    finally:
        soft_timer.cancel()
        hard_timer.cancel()
    elapsed = (time.monotonic() - began) * 1000.0

    uci = to_uci(packed) if packed else ""
    move = None
    if uci:
        try:
            candidate = chess.Move.from_uci(uci)
        except ValueError:
            candidate = None
        if candidate is not None and candidate in board.legal_moves:
            move = candidate

    if move is None:
        print(f"engine returned {uci!r}, which is not legal here; falling back")
        return _fallback(board)

    _record_after(board, move)

    nodes = int(searcher.CONTROL[NODES])
    print(
        f"depth {int(searcher.CONTROL[BEST_DEPTH])} score {int(searcher.CONTROL[BEST_SCORE])} "
        f"nodes {nodes} "
        f"time {elapsed:.0f}ms ({nodes / max(elapsed, 1.0):.0f}kn/s) "
        f"budget {budget:.0f}ms hard {hard:.0f}ms stable {int(searcher.CONTROL[STABLE])} "
        f"move {uci}"
    )
    return uci


def hard_ceiling(budget: float, time_left_ms: int) -> float:
    """The longest this move may take once the extension is allowed for."""
    ceiling = min(budget * 2.0, max(1.0, time_left_ms - OVERHEAD_MS) * 0.35)
    return budget if ceiling < budget else ceiling


def _raise_flag() -> None:
    searcher.CONTROL[STOP] = 1


def _raise_soft() -> None:
    """The budget is gone. Decide here whether that ends the search.

    Read the stability counter rather than always deferring to the next iteration boundary. If
    the root move held through the last completed iteration the search has converged and this
    stops it exactly where the old single deadline did -- mid-iteration, keeping the previous
    depth's answer, which is what an unfinished iteration is worth anyway. Only an unsettled
    root defers, and only that case is allowed to run into the extension.

    Deferring unconditionally instead looks similar and is not: a settled position would finish
    whatever iteration it had started, which measured 48% over budget on the opening position.
    That is a uniform increase in time spent, and a uniform increase was already measured at
    -0.6 Elo. The point is to move time between positions, not to spend more everywhere.
    """
    if int(searcher.CONTROL[STABLE]) >= 1:
        searcher.CONTROL[STOP] = 1
    else:
        searcher.CONTROL[SOFT] = 1


def _select_evaluation() -> None:
    """Play with the network if one shipped, and with the hand evaluation otherwise.

    Set before warm-up so whichever path will run is the path numba compiles inside the import
    budget rather than on the first move.
    """
    searcher.CONTROL[USE_NNUE] = 1 if nnue.TRAINED else 0
    print(f"evaluation: {'network' if nnue.TRAINED else 'hand-written'}")


def _warm() -> None:
    """Compile every jitted path inside the 60 second import budget, not on the clock.

    numba compiles per signature on first call, so this has to exercise the real ones: a
    quiet middlegame for the main search, a position with captures hanging for quiescence,
    and enough depth to reach null move and the late move reductions.
    """
    positions = (
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    )
    for fen in positions:
        searcher.STATES[0] = from_board(chess.Board(fen))
        searcher.PATH[0] = searcher.STATES[0, HASH]
        searcher.CONTROL[STOP] = 0
        searcher.run(6, 0, 400_000)
    searcher.reset()
    _history.clear()


_select_evaluation()
_warm()
