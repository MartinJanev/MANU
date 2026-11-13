#!/bin/bash
#SBATCH --job-name=manu-model
#SBATCH --chdir=/home/hpc/users/martin.janev/manu
#SBATCH --output=results/slurm/%x_%j.out      # %x = job name, %j = job id
#SBATCH --partition=openlab-queue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

# Usage: sbatch -J <model> jobs/run_model.sh <model> <runner args...>
#        DRY_RUN=1 bash jobs/run_model.sh <model> <runner args...>   (print commands only)

set -euo pipefail

# Under SLURM the --chdir header already points at the repo (the script itself
# is a spooled copy); otherwise go to the repo root next to this script.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

MODEL=${1:?Usage: sbatch -J <model> jobs/run_model.sh <model> <runner args...>}
shift

[ -f "models/$MODEL/env.sh" ] || { echo "ERROR: models/$MODEL/env.sh not found" >&2; exit 1; }
[ -f "models/$MODEL/run.py" ] || { echo "ERROR: models/$MODEL/run.py not found" >&2; exit 1; }

# shellcheck disable=SC1090
source "models/$MODEL/env.sh"

DRY_RUN=${DRY_RUN:-0}

# Print the command (one line, arguments expanded) in a dry run, execute it otherwise.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '%s ' "$@"
        echo
    else
        "$@"
    fi
}

export SINGULARITYENV_HF_HOME=/hf_cache
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTHONUSERBASE=/pyuser
export SINGULARITYENV_PYTHONPATH=/manu
SINGULARITYENV_MANU_GIT_COMMIT=$(cat .manu_commit 2>/dev/null || echo unknown)
export SINGULARITYENV_MANU_GIT_COMMIT

if [ "$DRY_RUN" != 1 ]; then
    mkdir -p results/slurm "pyuser/$MODEL" third_party
    echo "Job ${SLURM_JOB_ID:-local} ($MODEL) on $(hostname), start: $(date)"
fi

# EXTRA_BINDS is intentionally unquoted: it holds space-separated "-B src:dst" options.
# shellcheck disable=SC2206
CMD=(singularity exec --nv -B .:/manu -B ./data:/data -B ./results:/results -B ./reference:/reference
     -B ./hf_cache:/hf_cache -B "./pyuser/$MODEL:/pyuser" -B ./third_party:/third_party ${EXTRA_BINDS:-}
     "$SIF" python3 "/manu/models/$MODEL/run.py" "$@")

[ "$DRY_RUN" = 1 ] || nvidia-smi
status=0
run "${CMD[@]}" || status=$?

# The job's packages next to its log (the runner names results/<adapter name>/<split>/ itself), also after a failure
FREEZE_FILE="results/slurm/${SLURM_JOB_NAME:-$MODEL}_${SLURM_JOB_ID:-manual}_freeze.txt"
FREEZE=(singularity exec -B "./pyuser/$MODEL:/pyuser" "$SIF" pip freeze)
if [ "$DRY_RUN" = 1 ]; then
    run "${FREEZE[@]}"
    echo "  > $FREEZE_FILE"
else
    "${FREEZE[@]}" > "$FREEZE_FILE" || echo "WARNING: pip freeze failed" >&2
    echo "End: $(date), exit status $status"
fi
exit "$status"
