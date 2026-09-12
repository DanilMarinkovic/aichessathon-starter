#!/usr/bin/env bash
# Profile the non-evaluation parts of a node, against a chosen network.
#
# The scratch directory goes under $HOME, not /tmp: the login node's /tmp is not the compute
# node's, and a job whose working directory was created on the login node fails with "couldn't
# change working dir".
set -euo pipefail
cd "$HOME/aichessathon-starter"
NET=${NET:-versions/sfwdl02/weights/net.npz}
W="$HOME/.nodeparts-$$"
rm -rf "$W"; mkdir -p "$W/weights"
cp ./*.py "$W/"; cp -r tools pyproject.toml uv.lock "$W/"
cp "$NET" "$W/weights/net.npz"
trap 'rm -rf "$W"' EXIT
cd "$W" && uv run --project "$HOME/aichessathon-starter" python tools/nodeparts.py
