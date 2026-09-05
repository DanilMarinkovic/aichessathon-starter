#!/usr/bin/env bash
# Combine the shards and convert them, as a job rather than on a login node.
#
#   sbatch aire/build-data.slurm
#
# The work itself is aire/build-data.sh, which is fine to run by hand. This wrapper exists so
# the step can be chained: the whole pipeline is build -> train the schedule sweep -> measure,
# and a dependency needs a job id to hang off. It also keeps a hundred and twenty million lines
# of text conversion off the login node, which is shared.
#
# One core is enough. Every stage is a single streaming pass -- concatenating, rewriting a line
# at a time, and bullet-utils reading sequentially -- so extra cores would sit idle and a bigger
# request only makes it queue longer.

#SBATCH --job-name=build-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=aire/logs/build-data-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p aire/logs

echo "quota before:"; quota -s | tail -1
bash aire/build-data.sh
echo
echo "quota after:"; quota -s | tail -1
