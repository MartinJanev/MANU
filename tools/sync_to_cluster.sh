#!/bin/bash
# Mac -> cluster: copy the code (never results/ or data) to the cluster checkout. EduVPN must be on.
# Never deletes anything on the cluster. Overrides: CLUSTER, REMOTE; RSYNC_DRY=1 for a dry run.
set -euo pipefail
cd "$(dirname "$0")/.."

git rev-parse --short HEAD > .manu_commit 2>/dev/null || echo unknown > .manu_commit

# .manu_commit is git-ignored but must reach the cluster; the first matching rule wins.
rsync -av ${RSYNC_DRY:+-n} --include=/.manu_commit --exclude-from=.gitignore \
    --exclude .git --exclude .superpowers/ --exclude /.claude/ --exclude /paper/ \
    ./ "${CLUSTER:-martin.janev@10.10.65.6}:${REMOTE:-/home/hpc/users/martin.janev/manu}/"
