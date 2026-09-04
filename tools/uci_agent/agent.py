"""A harness agent that forwards to a local UCI engine. FOR MEASUREMENT ONLY.

THIS IS NOT PART OF THE SUBMISSION AND MUST NEVER BE. Shipping a third party engine, or a
wrapper around one, is an instant disqualification. This file exists so an external engine can
be used as a fixed yardstick on this machine, which the rules do allow: the ban covers what
ships and runs inside the zip, not what you measure yourself against.

It cannot be packaged by accident. `harness/package.py` collects `*.py` at the repository root
and the `weights` directory, and this is neither.

Configured by environment variables, because the harness only ever passes a directory:

    UCI_ENGINE    path to the engine binary (required)
    UCI_ELO       cap the engine's strength at this rating, via UCI_LimitStrength
    UCI_NODES     nodes per move; preferred, because it does not depend on machine load
    UCI_MOVETIME  milliseconds per move, used only when UCI_NODES is unset. Default 100
    UCI_THREADS   engine threads, default 1 to match the platform's single core

Prefer UCI_NODES for anything being measured. A movetime limit is wall clock, so a loaded
machine gets the reference engine less work per move and it plays weaker, which would quietly
move the yardstick between runs. A node limit is load-independent, matching how the agent under
test is held fixed.
"""

import os
import subprocess
import sys

ENGINE = os.environ.get("UCI_ENGINE", "")
ELO = os.environ.get("UCI_ELO", "")
NODES = int(os.environ.get("UCI_NODES", "0"))
MOVETIME = int(os.environ.get("UCI_MOVETIME", "100"))
THREADS = os.environ.get("UCI_THREADS", "1")

if not ENGINE:
    raise RuntimeError("set UCI_ENGINE to the path of a UCI engine binary")

_engine = subprocess.Popen(
    [ENGINE],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)


def _send(command: str) -> None:
    if _engine.stdin is None:
        raise RuntimeError("engine stdin closed")
    _engine.stdin.write(command + "\n")
    _engine.stdin.flush()


def _read_until(token: str) -> str:
    if _engine.stdout is None:
        raise RuntimeError("engine stdout closed")
    while True:
        line = _engine.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited while waiting for {token!r}")
        if line.startswith(token):
            return line.strip()


def _setup() -> None:
    _send("uci")
    _read_until("uciok")
    _send(f"setoption name Threads value {THREADS}")
    if ELO:
        # Strength limiting makes the engine play deliberately imperfectly, which also makes it
        # nondeterministic. That is useful here: it breaks the one-game-per-opening ceiling that
        # a deterministic opponent imposes, so an anchor can be measured over many more games.
        _send("setoption name UCI_LimitStrength value true")
        _send(f"setoption name UCI_Elo value {ELO}")
    _send("isready")
    _read_until("readyok")
    _send("ucinewgame")
    _send("isready")
    _read_until("readyok")
    strength = ELO or "full"
    limit = f"{NODES} nodes" if NODES > 0 else f"{MOVETIME}ms"
    print(f"uci adapter ready: {ENGINE} elo={strength} {limit}", file=sys.stderr)


def get_move(fen: str, time_left_ms: int) -> str:
    _send(f"position fen {fen}")
    if NODES > 0:
        _send(f"go nodes {NODES}")
    else:
        _send(f"go movetime {min(MOVETIME, max(10, time_left_ms // 40))}")
    line = _read_until("bestmove")
    move = line.split()[1]
    if move in ("(none)", "0000"):
        raise RuntimeError("engine reported no legal move")
    return move


_setup()
