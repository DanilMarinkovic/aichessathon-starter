#!/usr/bin/env bash
# Install the toolchain bullet needs on Aire, then build it. Run once, on a login node.
#
#   bash aire/bullet-setup.sh
#
# Nothing here needs root. rustup installs into $HOME, and CUDA comes from a module rather than
# a system package, which is why this works on a cluster where we cannot install anything.
#
# Bullet is a trainer. It runs here, offline, and only the weights it produces ever reach the
# zip, so it sits on the allowed side of the line exactly as Stockfish-as-labeller does. Nothing
# from bullet is imported by the agent, and the agent's inference stays our own numba code.

set -euo pipefail
ORIGIN="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ORIGIN"

# 12.6.2 rather than 12.4.1: bullet's CUDA backend tracks recent toolkits, and the L40S nodes
# are Ada, which every 12.x supports.
module load cuda/12.6.2
# bullet's build script reads CUDA_PATH. The Aire module exports CUDA_DIR and CUDADIR but not
# that one, so the CUDA backend fails to build with "CUDA_PATH is not defined" unless it is
# bridged across here.
export CUDA_PATH="${CUDA_DIR:?cuda module did not set CUDA_DIR}"
echo "cuda: $(nvcc --version | tail -2 | head -1)"
echo "CUDA_PATH=$CUDA_PATH"

if ! command -v cargo >/dev/null 2>&1; then
  echo "installing rust into \$HOME..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
echo "cargo: $(cargo --version)"

# Cloned beside the repository rather than inside it, so nothing here can be swept into the
# submission zip by a stray glob. The packager takes root *.py and weights/, but a Rust
# checkout of this size has no business being anywhere near it.
BULLET_DIR="$HOME/bullet"
if [ ! -d "$BULLET_DIR" ]; then
  git clone --depth 1 https://github.com/jw1912/bullet "$BULLET_DIR"
fi
cd "$BULLET_DIR"
echo "bullet at $(git rev-parse --short HEAD)"

# The utilities that convert our text into bulletformat's binary layout.
cargo build --release --package bullet-utils 2>&1 | tail -5

echo
echo "activation functions bullet exposes (the trainer spec has to name one of these):"
grep -rhoE "fn (crelu|screlu|relu|sqrrelu)\b" crates/ | sort -u || true

echo
echo "done. next: cargo run --release --package bullet-utils -- convert --help"
