#!/usr/bin/env bash
# Everything the schedule and WDL results imply, submitted together.
set -euo pipefail
cd "$HOME/aichessathon-starter"

# Every version gets the rewritten forward pass. Both sides of a match must run the same engine
# code, and the rewrite helps the wider network more (74ns saved at 512 against 33ns at 256),
# so measuring 512 against 256 on the old code would understate the wide one.
for v in champion2 sfking sfbig sfcos sfwdl02 sfwdl06; do
  [ -d "versions/$v" ] && cp nnue.py "versions/$v/nnue.py"
done
echo "versions updated with the new forward"

# 1. The shipping question: the best network so far against what is uploaded.
A=versions/sfwdl02 B=versions/sfking TIME_CONTROL=120000,500 PAIRS=1000 WORKERS=80 \
  sbatch --parsable --cpus-per-task=88 --mem=100G --time=02:45:00 aire/clock-match.slurm

# 2. The external anchor, both arms in one job so conditions are shared.
ARMS="sfking sfwdl02" PAIRS=400 WORKERS=36 \
  sbatch --parsable --cpus-per-task=40 --mem=48G --time=02:30:00 \
  --job-name=anchor-pair --output=aire/logs/anchor-pair-%j.out --wrap='bash aire/anchor-pair.sh'

# 3. Push the WDL trend further. 0.6 -> 0.4 -> 0.2 is monotonic, so ask where it turns.
for w in 0.1 0.0; do
  BINARY=aire/data/wrongIsRight_nodes5000pv2.binpack POSITIONS=100000000 HIDDEN=512 \
    NBUCKETS=32 OUTPUTS=8 TAG=sfwdl${w#0.} EPOCHS=380 WDL=$w CONTROL_NET= \
    sbatch --parsable --time=01:40:00 aire/bullet-buckets.slurm
done

# 4. Width again, now that forward is 45% cheaper. 1024 was ruled out on a node cost that was
#    itself an artefact of unvectorised code; this is the arm that says whether it still is.
BINARY=aire/data/wrongIsRight_nodes5000pv2.binpack POSITIONS=100000000 HIDDEN=1024 \
  NBUCKETS=32 OUTPUTS=8 TAG=sfhuge EPOCHS=380 WDL=0.2 CONTROL_NET= \
  sbatch --parsable --time=02:30:00 aire/bullet-buckets.slurm

# 5. What the rewritten forward is worth at the node level, on the real networks.
NETS="versions/sfking versions/sfwdl02" sbatch --parsable --cpus-per-task=2 --mem=8G \
  --time=00:30:00 --job-name=nc-pair --output=aire/logs/nc-pair-%j.out \
  --wrap='bash aire/nodecost-pair.sh'

squeue -u "$USER" --format="%.10i %.15j %.9T %.5C %.9L %R" --sort=+i
