#!/usr/bin/env bash
# Push everything Aire needs and report what is queued. Safe to re-run.
#
#   bash aire/push.sh
#
# Exists because the login node drops connections, and re-running half a dozen rsync lines by
# hand after an outage is how one of them gets missed and a job trains from stale source.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== source =="
rsync -az nnue.py searcher.py agent.py position.py movegen.py bitboards.py see.py \
  evaluate.py aire:aichessathon-starter/
rsync -az tools/ aire:aichessathon-starter/tools/
# The harness too. It is upstream's, not ours, but it changes -- the 90s init budget and the
# 600-ply draw arrived on 4 and 5 September -- and a cluster copy left behind measures games
# under rules the platform stopped using.
rsync -az harness/ aire:aichessathon-starter/harness/
rsync -az tests/ aire:aichessathon-starter/tests/
rsync -az aire/*.slurm aire/*.sh aire:aichessathon-starter/aire/
rsync -az aire/bullet-trainer/src/ aire:aichessathon-starter/aire/bullet-trainer/src/

echo "== candidates =="
for v in "$@"; do
  [ -d "versions/$v" ] || { echo "no such snapshot: versions/$v" >&2; exit 1; }
  rsync -az --delete "versions/$v" aire:aichessathon-starter/versions/
  echo "  $v"
done

echo "== queue =="
ssh aire 'squeue -u $USER --format="%.10i %.16j %.9T %.11M %.10L"'
