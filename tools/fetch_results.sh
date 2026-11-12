#!/bin/bash
# Cluster -> Mac: pull results/ into ./results/ (without archive/). EduVPN must be on.
# Overrides: CLUSTER, REMOTE; RSYNC_DRY=1 for a dry run.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results
rsync -av ${RSYNC_DRY:+-n} --exclude /archive/ \
    "${CLUSTER:-martin.janev@10.10.65.6}:${REMOTE:-/home/hpc/users/martin.janev/manu}/results/" results/
