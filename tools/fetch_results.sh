#!/bin/bash
# Cluster -> Mac: pull results/ into ./results/ (without archive/). EduVPN must be on.
# Overrides: CLUSTER, REMOTE; RSYNC_DRY=1 for a dry run.
# -u: a local file newer than the cluster's (e.g. re-scored by tools/reparse.py) is kept, not overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results
rsync -avu ${RSYNC_DRY:+-n} --exclude /archive/ \
    "${CLUSTER:-martin.janev@10.10.65.6}:${REMOTE:-/home/hpc/users/martin.janev/manu}/results/" results/
