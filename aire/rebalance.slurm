#!/usr/bin/env bash
# Rebuild the training binary with the evaluation bands thinned.
#
#   sbatch aire/rebalance.slurm
#
# Why. The network learned to count material and little else. Along the losing line of the game
# against FuzzyBot it valued positions at -544, -400, -198 that Stockfish valued at +455, +496,
# +516 -- Black a bishop up and being mated -- then agreed to within 8cp the moment material was
# restored. Stockfish sees the refutation at depth 6; our engine searched to depth 17 and called
# the position level. That is not depth, pruning, or phase.
#
# The cause is in the data. On a 40,000-position sample, plain material count correlates 0.89
# with the Stockfish label, so 79.7% of the variance the network is asked to fit is explained by
# counting pieces. Minimising loss by counting pieces is exactly what it did, and king safety
# lives in the residual with almost no gradient behind it.
#
# This is a real parameter elsewhere -- Stockfish's data generation calls it eval_limit,
# "positions with absolute evaluation higher than this will be skipped". Its default is 32000,
# which skips nothing, because Stockfish's self-play is balanced to begin with. Ours is not: the
# median |eval| here is 518cp, the product of randomised openings and 50k-node self-play that
# swings hard. So the filter that Stockfish does not need, we do.
#
# tools/to_bullet.py thins rather than excludes, because the network still has to learn that a
# queen up is winning and can only learn it from positions where someone is a queen up. Material
# drops from 79.7% of the variance to 54.5%, keeping about 42% of the rows -- some 50M, ample
# for a network that is underfitting.
#
# Converting straight from the existing positions.epd rather than re-concatenating 792 shards,
# and reusing test.data untouched: the held-out set has to stay on the real distribution or it
# stops measuring how the network will do in a game.

#SBATCH --job-name=rebalance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=aire/logs/rebalance-%j.out

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

DATA=aire/data
SOURCE=${SOURCE:-$DATA/positions.epd}
OUT=${OUT:-$DATA/rebal-positions}

for required in "$SOURCE" "$DATA/test.data"; do
  [ -s "$required" ] || { echo "ERROR: $required is missing." >&2; exit 1; }
done
if [ -s "$OUT.data" ]; then
  echo "ERROR: $OUT.data already exists; move it aside or pass a different OUT." >&2
  exit 1
fi

echo "quota before:"; quota -s | tail -1
echo "source: $(wc -l < "$SOURCE") positions"

uv run python tools/to_bullet.py --rebalance "$SOURCE" "$OUT.txt"
cargo run --release --manifest-path "$HOME/bullet/crates/utils/Cargo.toml" -- \
  convert --from text --input "$OUT.txt" --output "$OUT.data" --threads 8
rm -f "$OUT.txt"

echo
echo "rebalanced: $(( $(stat -c %s "$OUT.data") / 32 )) positions"
echo "control:    $(( $(stat -c %s "$DATA/positions.data") / 32 )) positions (unfiltered)"
echo "held-out:   $(( $(stat -c %s "$DATA/test.data") / 32 )) positions, unchanged"
echo
echo "quota after:"; quota -s | tail -1
echo
echo "Train on it with the same job the control used, so the only difference is the data:"
echo "  BINARY=$OUT.data TAG=rebal OUTPUTS=8 EPOCHS=760 sbatch aire/bullet-buckets.slurm"
echo
echo "760 rather than 320 superbatches: a superbatch is one pass, so matching passes would let"
echo "the smaller set see 2.4x fewer positions and lose on volume rather than on distribution."
