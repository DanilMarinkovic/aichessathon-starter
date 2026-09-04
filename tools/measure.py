"""The measurement protocol, fixed so it cannot drift between versions.

The point of this file is that the numbers below are constants. A measurement is only
comparable to an earlier one if nothing about the conditions changed, and the easiest way to
report a fake improvement is to quietly measure the new version under kinder settings: more
nodes, a different opening set, a weaker reference. Putting the protocol in source, rather than
in whichever command line was last typed, is what stops that.

Two questions get asked of every change, because they can disagree:

  anchor   How do we score against a fixed external opponent? This is the absolute yardstick,
           and it is what catches slow drift across many versions.
  head     Is this version better than the one before it? A direct paired match with SPRT is
           far more sensitive than comparing two anchor numbers, so this decides changes.

Anything that alters how much work the evaluation does must also be run with --cost. Fixed
node counts hand new evaluation terms their knowledge for free, because the search is not
charged for the extra time they take. On the clock it is, and changes that pass the first test
and fail the second are common rather than rare.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- the protocol. Changing any of these invalidates comparison with earlier runs. ---
ANCHOR_ELO = 2600
ANCHOR_NODES = 60_000
OUR_NODES = 300_000
ANCHOR_PAIRS = 150
HEAD_SPRT = "0,5"
COST_TIME_CONTROL = "10000,100"
COST_PAIRS = 100
COST_WORKERS = 4
WORKERS = 6


def _run(arguments: list[str], environment: dict[str, str]) -> int:
    print("$ " + " ".join(arguments), flush=True)
    return subprocess.run(arguments, cwd=ROOT, env=environment).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a version under the fixed protocol.")
    parser.add_argument("--agent", default=".")
    parser.add_argument("--against", help="previous version to A/B against, e.g. versions/v1")
    parser.add_argument("--anchor", action="store_true", help="run the external anchor match")
    parser.add_argument("--cost", action="store_true", help="re-check on the clock")
    parser.add_argument("--pairs", type=int, default=0, help="override anchor pairs")
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="parallel games. Safe to raise: fixed-node results do not depend on load",
    )
    arguments = parser.parse_args()
    workers = str(arguments.workers)

    engine = os.environ.get("UCI_ENGINE", "")
    environment = dict(os.environ)
    python = [sys.executable, "tools/match.py"]

    if not (arguments.anchor or arguments.against or arguments.cost):
        raise SystemExit("nothing to do: pass --anchor, --against VERSION, or --cost")

    print(
        f"protocol: agent {OUR_NODES:,} nodes | anchor Stockfish {ANCHOR_ELO} at "
        f"{ANCHOR_NODES:,} nodes | {ANCHOR_PAIRS} pairs"
    )

    failures = 0
    if arguments.anchor:
        if not engine:
            raise SystemExit("set UCI_ENGINE to the reference engine for --anchor")
        environment.update(
            UCI_ELO=str(ANCHOR_ELO), UCI_NODES=str(ANCHOR_NODES), UCI_THREADS="1"
        )
        print("\n=== anchor: absolute yardstick ===")
        failures += _run(
            [*python, "--a", arguments.agent, "--b", "tools/uci_agent",
             "--pairs", str(arguments.pairs or ANCHOR_PAIRS),
             "--nodes", str(OUR_NODES), "--workers", workers],
            environment,
        )

    if arguments.against:
        print("\n=== head to head: is this better than the last one ===")
        failures += _run(
            [*python, "--a", arguments.agent, "--b", arguments.against,
             "--nodes", str(OUR_NODES), "--workers", workers, "--sprt", HEAD_SPRT],
            environment,
        )

    if arguments.cost:
        if not arguments.against:
            raise SystemExit("--cost compares against a previous version; pass --against")
        print("\n=== on the clock: is the knowledge worth the time it costs ===")
        # Deliberately ignores --workers. This run is timed, so crowding the cores would
        # measure the scheduler; it stays at a low fixed parallelism on every machine.
        failures += _run(
            [*python, "--a", arguments.agent, "--b", arguments.against,
             "--pairs", str(COST_PAIRS), "--time-control", COST_TIME_CONTROL,
             "--workers", str(COST_WORKERS)],
            environment,
        )

    raise SystemExit(failures)


if __name__ == "__main__":
    main()
