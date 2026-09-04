"""Play two agents against each other and say whether the difference is real.

Three things make a result trustworthy, and this runs on all of them.

Paired games. Every opening is played twice with the colours swapped, and the pair is scored
as one observation. A pair cancels both the opening's bias and White's advantage, so the
variance that remains is mostly the thing being measured. Reporting per-pair rather than per
game is what makes the error bar honest.

Fixed nodes. A timed game measures the machine as much as the engine: two agents sharing a
loaded box reach different depths from run to run, so part of the result is which process got
the core. Searching a fixed node count is deterministic and load-immune, which is what allows
many games at once. The cost is that it cannot see changes that only affect speed, so anything
touching how fast the engine runs has to be re-checked with --time-control.

Enough games. A 3 game sample says nothing. The summary prints a confidence interval and, with
--sprt, runs a sequential test that stops as soon as the answer is not in doubt.

Because the engine is deterministic under fixed nodes, replaying an opening with the same two
agents replays the same game, so the opening set is the ceiling on distinct games.
"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

DEFAULT_OPENINGS = Path(__file__).resolve().parent / "openings.epd"
FIXED_NODE_BASE_MS = 600_000
SPRT_ALPHA = 0.05
SPRT_BETA = 0.05


def _init(nodes: int) -> None:
    if nodes > 0:
        os.environ["CHESSATHON_FIXED_NODES"] = str(nodes)
    else:
        os.environ.pop("CHESSATHON_FIXED_NODES", None)


Job = tuple[int, str, Path, Path, int, int, int]


def play_pair(job: Job) -> tuple[int, list[float], list[str]]:
    """Play one opening twice, with `a` as White and then as Black."""
    index, fen, a, b, base_ms, increment_ms, ply_cap = job
    scores: list[float] = []
    terminations: list[str] = []
    for a_is_white in (True, False):
        white, black = (a, b) if a_is_white else (b, a)
        outcome = play_match(
            local(white), local(black), base_ms, increment_ms, ply_cap=ply_cap, start_fen=fen
        )
        terminations.append(outcome.termination)
        if outcome.result == "draw" or outcome.result == "void":
            scores.append(0.5)
        elif (outcome.result == "white") == a_is_white:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return index, scores, terminations


def elo(score: float) -> float:
    score = min(max(score, 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def expected(rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating / 400.0))


def statistics_of(pairs: list[float]) -> tuple[float, float, int]:
    """Mean pair score, its standard error, and the count."""
    count = len(pairs)
    if count == 0:
        return 0.5, 0.0, 0
    mean = sum(pairs) / count
    if count < 2:
        return mean, 0.0, count
    variance = sum((value - mean) ** 2 for value in pairs) / (count - 1)
    return mean, math.sqrt(variance / count), count


def log_likelihood_ratio(pairs: list[float], elo0: float, elo1: float) -> float:
    """Normal-approximation GSPRT statistic over paired results."""
    mean, error, count = statistics_of(pairs)
    if count < 2 or error <= 0.0:
        return 0.0
    variance = (error**2) * count
    p0, p1 = expected(elo0), expected(elo1)
    return count * (p1 - p0) * (mean - (p0 + p1) / 2.0) / variance


def summarise(pairs: list[float], games: Counter[str], label: str) -> None:
    mean, error, count = statistics_of(pairs)
    print(f"\n{label}")
    print(f"  {count} pairs, {count * 2} games")
    if count < 2:
        return

    print(f"  score  {mean:.2%}  +-{1.96 * error:.2%}")
    if error <= 0.0:
        los = 1.0 if mean > 0.5 else 0.0 if mean < 0.5 else 0.5
    else:
        los = NormalDist().cdf((mean - 0.5) / error)

    if mean <= 0.0 or mean >= 1.0:
        print("  Elo    saturated: every game went the same way, so the gap is unmeasurable")
    else:
        low, high = mean - 1.96 * error, mean + 1.96 * error
        print(f"  Elo    {elo(mean):+.1f}  [{elo(low):+.1f}, {elo(high):+.1f}]  (95%)")
    print(f"  LOS    {los:.1%} chance A is genuinely better")

    spread = Counter(pairs)
    parts = [
        f"{name} {spread.get(value, 0)}"
        for value, name in ((0.0, "0-2"), (0.25, "0.5"), (0.5, "1-1"), (0.75, "1.5"), (1.0, "2-0"))
    ]
    print("  pairs  " + "  ".join(parts))
    print("  ends   " + ", ".join(f"{name} {n}" for name, n in sorted(games.items())))

    broken = {name: n for name, n in games.items() if name in FAILED_TERMINATIONS}
    if broken:
        detail = ", ".join(f"{name} {n}" for name, n in broken.items())
        print(f"  FAILURES: {detail}  <- fix before trusting any of this")

    if count >= 8 and not 0.1 < mean < 0.9:
        print(
            "\n  This opponent is saturated. A score this lopsided carries almost no information\n"
            "  about a change: everything wins, so everything looks equal. Measure against the\n"
            "  version you are trying to beat, which is your own previous one."
        )


def run_series(
    jobs: list[tuple[int, str, Path, Path, int, int, int]],
    workers: int,
    nodes: int,
    sprt: tuple[float, float] | None,
) -> tuple[list[float], Counter[str], str]:
    bounds = None
    if sprt is not None:
        bounds = (
            math.log(SPRT_BETA / (1.0 - SPRT_ALPHA)),
            math.log((1.0 - SPRT_BETA) / SPRT_ALPHA),
        )

    pairs: list[float] = []
    games: Counter[str] = Counter()
    started = time.monotonic()
    verdict = ""

    with ProcessPoolExecutor(workers, initializer=_init, initargs=(nodes,)) as pool:
        futures = {pool.submit(play_pair, job): job[0] for job in jobs}
        try:
            for done in as_completed(futures):
                _, scores, terminations = done.result()
                pairs.append(sum(scores) / 2.0)
                games.update(terminations)

                mean, error, count = statistics_of(pairs)
                elapsed = time.monotonic() - started
                line = (
                    f"  {count:>4}/{len(jobs)} pairs  score {mean:6.2%}  "
                    f"Elo {elo(mean):+7.1f} +-{1.96 * error * 400:5.1f}  {elapsed:5.0f}s"
                )
                if bounds is not None and sprt is not None:
                    ratio = log_likelihood_ratio(pairs, sprt[0], sprt[1])
                    line += f"  LLR {ratio:+.2f}"
                    if count >= 8 and (ratio <= bounds[0] or ratio >= bounds[1]):
                        verdict = (
                            "SPRT accepts H1: the change is an improvement"
                            if ratio >= bounds[1]
                            else "SPRT accepts H0: no improvement worth keeping"
                        )
                        print(line, flush=True)
                        for future in futures:
                            future.cancel()
                        break
                # Long runs are usually redirected to a file, where Python block-buffers and
                # the progress would not appear until the end.
                print(line, flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted; summarising what finished")
            for future in futures:
                future.cancel()

    return pairs, games, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one agent against others.")
    parser.add_argument("--a", type=Path, default=Path("."), help="the agent under test")
    parser.add_argument(
        "--b",
        type=Path,
        action="append",
        required=True,
        help="an agent to beat; repeat for a gauntlet, which is what catches drift",
    )
    parser.add_argument("--pairs", type=int, default=0, help="0 uses every opening once")
    parser.add_argument("--nodes", type=int, default=200_000, help="fixed nodes per move")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--time-control",
        metavar="BASE_MS,INC_MS",
        help="play on the clock instead of fixed nodes; needed for changes that affect speed",
    )
    parser.add_argument("--sprt", metavar="ELO0,ELO1", help="stop early, e.g. 0,5")
    arguments = parser.parse_args()

    if not arguments.openings.exists():
        raise SystemExit(f"{arguments.openings} does not exist; run tools/openings.py first")
    lines = arguments.openings.read_text().splitlines()
    openings = [line.strip() for line in lines if line.strip()]

    if arguments.time_control:
        base_ms, increment_ms = (int(part) for part in arguments.time_control.split(","))
        nodes = 0
        workers = arguments.workers or max(1, (os.cpu_count() or 2) // 4)
        print("time control mode: results include machine noise, so keep --workers low")
    else:
        base_ms, increment_ms = FIXED_NODE_BASE_MS, 0
        nodes = arguments.nodes
        # Each worker holds two agent processes, and each of those holds its own transposition
        # table, so the ceiling here is memory rather than cores. Raise it with --workers if
        # the box has the RAM.
        workers = arguments.workers or min(6, max(1, (os.cpu_count() or 2) // 2))

    wanted = arguments.pairs or len(openings)
    if wanted > len(openings) and nodes > 0:
        print(
            f"warning: {wanted} pairs asked for but only {len(openings)} openings exist. "
            "Under fixed nodes the engine is deterministic, so repeats replay the same game "
            "and add no information. Add openings instead."
        )
        wanted = len(openings)

    a = arguments.a.resolve()
    mode = f"{nodes:,} nodes/move" if nodes else f"{base_ms}ms+{increment_ms}ms"
    sprt = None
    if arguments.sprt:
        elo0, elo1 = (float(part) for part in arguments.sprt.split(","))
        sprt = (elo0, elo1)
        print(f"SPRT H0 {elo0:+.0f} vs H1 {elo1:+.0f}")

    results: list[tuple[Path, list[float], Counter[str], str]] = []
    for opponent in arguments.b:
        b = opponent.resolve()
        label = f"A={arguments.a}  B={opponent}  {mode}"
        print(f"\n{label}\n{wanted} pairs on {workers} workers")
        jobs = [
            (index, openings[index % len(openings)], a, b, base_ms, increment_ms, arguments.ply_cap)
            for index in range(wanted)
        ]
        pairs, games, verdict = run_series(jobs, workers, nodes, sprt)
        summarise(pairs, games, label)
        if verdict:
            print(f"\n{verdict}")
        results.append((opponent, pairs, games, verdict))

    if len(results) > 1:
        print(f"\n{'gauntlet for ' + str(arguments.a):<34} {'pairs':>6} {'score':>8} "
              f"{'Elo':>9} {'LOS':>7}")
        for opponent, pairs, _, _ in results:
            mean, error, count = statistics_of(pairs)
            if count < 2:
                continue
            rating = "saturated" if not 0.0 < mean < 1.0 else f"{elo(mean):+.1f}"
            los = 1.0 if error <= 0 and mean > 0.5 else 0.0 if error <= 0 else (
                NormalDist().cdf((mean - 0.5) / error)
            )
            print(f"  {opponent!s:<32} {count:>6} {mean:>8.2%} {rating:>9} {los:>7.1%}")
        print(
            "\n  A version that beats the newest opponent but not an older one has not improved,\n"
            "  it has specialised. That is what the gauntlet is for."
        )


if __name__ == "__main__":
    main()
