#!/usr/bin/env bash
# One-time setup, to be run ON THE LOGIN NODE, because compute nodes have no internet.
#
#   cd $HOME/aichessathon-starter && bash aire/setup.sh
#
# Installs uv (which brings its own Python 3.12, no sudo), syncs the project dependencies,
# and fetches a Stockfish binary matched to the node's instruction set. Stockfish is used
# offline as a sparring partner and a position labeller only. It is installed outside the
# repository and can never enter a submission: harness/package.py collects *.py at the
# repository root and the weights directory, and this is neither.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
export PATH="$BIN:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "syncing dependencies..."
cd "$REPO"
uv sync

# Genoa is Zen 4 and supports AVX-512; the avx512 build is materially faster than avx2 there.
if grep -q avx512f /proc/cpuinfo; then
  BUILD=avx512
elif grep -q avx2 /proc/cpuinfo; then
  BUILD=avx2
else
  BUILD=sse41-popcnt
fi
echo "cpu supports: $BUILD"

if [ ! -x "$BIN/stockfish" ]; then
  echo "fetching stockfish ($BUILD)..."
  URL=$(curl -s https://api.github.com/repos/official-stockfish/Stockfish/releases/latest \
    | grep -oE "\"browser_download_url\": \"[^\"]*stockfish-ubuntu-x86-64-${BUILD}\.tar\"" \
    | cut -d'"' -f4)
  TMP=$(mktemp -d)
  curl -sL "$URL" -o "$TMP/sf.tar"
  tar -xf "$TMP/sf.tar" -C "$TMP"
  cp "$(find "$TMP/stockfish" -maxdepth 1 -type f -executable | head -1)" "$BIN/stockfish"
  chmod +x "$BIN/stockfish"
  rm -rf "$TMP"
fi

echo
echo "stockfish: $("$BIN/stockfish" bench 2>&1 | grep -i 'Nodes/second' || echo unknown)"
echo "warming the numba cache (compiles the engine once)..."
uv run python -c "import agent" >/dev/null 2>&1 && echo "engine imports cleanly"
echo
echo "setup complete. Submit work with: sbatch aire/measure.slurm"
