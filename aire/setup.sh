#!/usr/bin/env bash
# One-time setup, to be run ON THE LOGIN NODE, because compute nodes have no internet.
#
#   cd $HOME/aichessathon-starter && bash aire/setup.sh
#
# Installs uv (which brings its own Python 3.12, no sudo), syncs the project dependencies,
# and builds Stockfish from source for this cluster. Stockfish is used offline as a sparring
# partner and a position labeller only. It is installed outside the repository and can never
# enter a submission: harness/package.py collects *.py at the repository root and the weights
# directory, and this is neither.
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

# Stockfish is built from source rather than downloaded. The official release binaries are
# built on Ubuntu against a newer libstdc++ than this cluster ships, so they die immediately
# with a GLIBCXX version error regardless of the CPU. Building here also targets AVX-512, which
# Aire's Genoa nodes have.
ARCH=${SF_ARCH:-x86-64-avx512}
MARKER="$BIN/.stockfish-build"

LOADED_MODULE=""
if [ ! -x "$BIN/stockfish" ] || [ "$(cat "$MARKER" 2>/dev/null)" != "$ARCH" ]; then
  echo "building stockfish from source (ARCH=$ARCH)..."

  if ! command -v g++ >/dev/null 2>&1; then
    # `module` is a shell function, and a non-interactive script does not always inherit it.
    if ! command -v module >/dev/null 2>&1; then
      for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh "${LMOD_PKG:-}/init/bash"; do
        if [ -r "$init" ]; then
          # shellcheck disable=SC1090
          . "$init" && break
        fi
      done
    fi
    if command -v module >/dev/null 2>&1; then
      for candidate in ${SF_GCC_MODULE:-GCC gcc GCCcore gnu}; do
        if module load "$candidate" >/dev/null 2>&1; then
          LOADED_MODULE="$candidate"
          echo "  loaded module $candidate"
          break
        fi
      done
    fi
  fi

  if command -v g++ >/dev/null 2>&1; then
    echo "  using $(g++ --version | head -1)"
  else
    echo "  no g++ on PATH and no GCC module could be loaded." >&2
    echo "  Find one with: module avail 2>&1 | grep -i gcc" >&2
    echo "  then re-run as: SF_GCC_MODULE=<name> bash aire/setup.sh" >&2
    exit 1
  fi

  SRC="$HOME/.local/src/Stockfish"
  rm -rf "$SRC"
  git clone --depth 1 https://github.com/official-stockfish/Stockfish.git "$SRC"
  # Linked dynamically, on purpose. The GLIBCXX failure came from a binary built elsewhere
  # against a newer libstdc++ than this cluster has; one compiled by the cluster's own
  # compiler needs only the libstdc++ already present on every node. Static linking would
  # avoid that too, but libstdc++.a is a separate package and is not installed here.
  #
  # The exception is if a GCC module had to be loaded above, because then the binary follows
  # that module's libstdc++ and jobs need the module loaded too. Checked for below.
  make -C "$SRC/src" -j"$(nproc)" build ARCH="$ARCH" >/dev/null
  cp "$SRC/src/stockfish" "$BIN/stockfish"
  echo "$ARCH" > "$MARKER"

  if [ -n "$LOADED_MODULE" ]; then
    echo
    echo "NOTE: built with module $LOADED_MODULE, so the binary follows that module's"
    echo "libstdc++. Add this line to aire/*.slurm before the job runs:"
    echo "    module load $LOADED_MODULE"
  fi
fi

echo
echo "stockfish: $("$BIN/stockfish" bench 2>&1 | grep -i 'Nodes/second' || echo unknown)"
echo "warming the numba cache (compiles the engine once)..."
uv run python -c "import agent" >/dev/null 2>&1 && echo "engine imports cleanly"
echo
echo "setup complete. Submit work with: sbatch aire/measure.slurm"
