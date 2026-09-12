#!/usr/bin/env bash
# Run the forward-pass comparison against a chosen network, in a scratch copy.
set -euo pipefail
cd "$HOME/aichessathon-starter"
NET=${NET:-versions/sfwdl02/weights/net.npz}
W="${TMPDIR:-/tmp}/fv-$$"; rm -rf "$W"; mkdir -p "$W/weights"
cp ./*.py "$W/"; cp -r tools aire pyproject.toml uv.lock "$W/"
cp "$NET" "$W/weights/net.npz"
trap 'rm -rf "$W"' EXIT
cd "$W" && uv run --project "$HOME/aichessathon-starter" python aire/forward-variants.py
