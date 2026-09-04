"""Freeze the current agent so a later change has something to be measured against.

"Better than my last one" is the only comparison that means anything, and it only works if
the last one still exists to play. A snapshot copies exactly the files the packager would ship,
into a directory the rig can point at:

    uv run python tools/snapshot.py v1
    uv run python tools/match.py --b versions/v1 --sprt 0,5
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "versions"


def _commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    suffix = "+dirty" if dirty.stdout.strip() else ""
    return result.stdout.strip() + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the current agent into versions/<name>.")
    parser.add_argument("name")
    parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    arguments = parser.parse_args()

    destination = VERSIONS / arguments.name
    if destination.exists():
        if not arguments.force:
            raise SystemExit(f"{destination} already exists; pass --force to replace it")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    # The same selection the packager makes: every module at the root, plus weights.
    copied = []
    for source in sorted(ROOT.glob("*.py")):
        shutil.copy2(source, destination / source.name)
        copied.append(source.name)
    weights = ROOT / "weights"
    if weights.is_dir():
        shutil.copytree(weights, destination / "weights")
        copied.append("weights/")

    if "agent.py" not in copied:
        raise SystemExit("no agent.py at the repository root; nothing to snapshot")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (destination / "SNAPSHOT.txt").write_text(f"{arguments.name}\n{stamp}\ncommit {_commit()}\n")
    print(f"snapshot {arguments.name} -> {destination.relative_to(ROOT)}")
    print("  " + ", ".join(copied))
    where = destination.relative_to(ROOT)
    print(f"\nmeasure against it with:\n  uv run python tools/match.py --b {where} --sprt 0,5")


if __name__ == "__main__":
    main()
