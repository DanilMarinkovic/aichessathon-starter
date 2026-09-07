#!/usr/bin/env bash
# Measure against something that is not us.
#
#   A=versions/champion2 NODES=3500 sbatch aire/external-match.slurm
#
# Every other measurement in this repository is self-play: our engine against our engine with
# one change. The intervals are honest and the method is standard, but it has a blind spot that
# matters here. A weakness only costs Elo when the opponent exploits it, and in self-play the
# opponent is equally blind. Our own evaluation is wrong by 700-1000cp whenever material and
# king safety disagree -- and our sparring partner never sets that trap, because it cannot see
# it either.
#
# Three rated losses in a row were the same shape: taking material while an attack landed. A
# fixed external opponent that does see those lines is the only way to measure whether a change
# helps against the field rather than against a mirror.
#
# Stockfish here is a yardstick, never a shipped component: tools/uci_agent is not collected by
# harness/package.py, which takes *.py at the repository root and the weights directory. The
# rules permit an engine for offline measurement; they forbid one inside the zip.
#
# On a node limit rather than a clock. A movetime limit is wall clock, so a loaded node gives
# the reference engine less work and it plays weaker -- the yardstick would move between runs.
# Nodes are load-independent. Our side plays the platform's real 120s+0.5s, because that is the
# regime being selected in, and a faster control flatters exactly the wrong things.

#SBATCH --job-name=external-match
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=aire/logs/external-match-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

A=${A:-versions/champion2}
PAIRS=${PAIRS:-400}
WORKERS=${WORKERS:-140}
TIME_CONTROL=${TIME_CONTROL:-120000,500}
export UCI_ENGINE=${UCI_ENGINE:-$HOME/.local/bin/stockfish}
export UCI_NODES=${NODES:-3500}
export UCI_THREADS=1

[ -x "$UCI_ENGINE" ] || { echo "ERROR: no engine at $UCI_ENGINE" >&2; exit 1; }
[ -e "$A" ] || { echo "ERROR: $A does not exist." >&2; exit 1; }

echo "$A vs stockfish at $UCI_NODES nodes, $TIME_CONTROL, $PAIRS pairs"
echo "node $(hostname), $(nproc) cores, $WORKERS workers"
echo

uv run python tools/match.py --a "$A" --b tools/uci_agent \
  --time-control "$TIME_CONTROL" --pairs "$PAIRS" --workers "$WORKERS"
