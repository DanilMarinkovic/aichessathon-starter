#!/usr/bin/env bash
# Everything outstanding, in one SSH session.
#
#   bash aire/sh < aire/results.sh
#
# One connection, not five. See [[feedback-one-ssh-connection]]: a day of one-command-per-
# connection cost us cluster access for hours in the middle of the measurements that decide an
# upload, and the fix is to batch reads rather than to make them more carefully.
cd "$HOME/aichessathon-starter"
echo "=== queue ==="
squeue -u "$USER" --format='%.10i %.15j %.9T %.5C %.9M %.9L %R' --sort=+i
echo
echo "=== finished today ==="
sacct -X -S 2026-09-07T12:00 --format=JobID%12,JobName%15,State%12,Elapsed --noheader \
  | grep -Ev '^ *[0-9]+ +(bash|uv) '
echo
echo "=== anchor: champion2 and sfking against the same Stockfish ==="
grep -E '^===|^  score|^  Elo|^  LOS' aire/logs/anchor-pair-7796695.out 2>/dev/null
echo
echo "=== sfbig: 512 wide, 32 buckets ==="
sed 's/\x1b\[[0-9;]*m//g' aire/logs/bullet-buckets-7796735.out 2>/dev/null \
  | grep -E 'hidden |correlation|Saved|Total Training|error' | tail -6
ls -la aire/data/sfbig*.npz 2>/dev/null
