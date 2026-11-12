#!/bin/bash
# Login-node setup for one model: image, Python packages, HF weights, git repos.
# Usage: bash jobs/setup_env.sh <model>
#        DRY_RUN=1 bash jobs/setup_env.sh <model>   (print commands only)
# Gated HF repos need HF_TOKEN exported on the login node.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL=${1:?Usage: bash jobs/setup_env.sh <model>}

[ -f "models/$MODEL/env.sh" ] || { echo "ERROR: models/$MODEL/env.sh not found" >&2; exit 1; }

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

if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$(dirname "$SIF")" hf_cache "pyuser/$MODEL" third_party
fi

export SINGULARITYENV_HF_HOME=/hf_cache
export SINGULARITYENV_PYTHONUSERBASE=/pyuser

# EXTRA_BINDS is intentionally unquoted: it holds space-separated "-B src:dst" options.
# shellcheck disable=SC2206
EXEC=(singularity exec --nv -B .:/manu -B ./hf_cache:/hf_cache -B "./pyuser/$MODEL:/pyuser"
      -B ./third_party:/third_party ${EXTRA_BINDS:-} "$SIF")

# 1. Image
if [ "$DRY_RUN" = 1 ] || [ ! -f "$SIF" ]; then
    run singularity pull "$SIF" "$SIF_SOURCE"
fi

# 2. Python packages into pyuser/<model>, torch pinned to the image's versions
PINS=/pyuser/torch_pins.txt
run "${EXEC[@]}" bash -c "pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > $PINS \
&& pip install --user --no-cache-dir -c $PINS -r /manu/models/$MODEL/requirements.txt"

# 3. Weights (offline in jobs, so fetch them here)
for repo in ${HF_REPOS:-}; do
    run "${EXEC[@]}" python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$repo')"
done

# 4. Git repos: url@commit -> third_party/<repo name>
for entry in ${GIT_REPOS:-}; do
    url=${entry%@*}
    commit=${entry##*@}
    name=$(basename "$url" .git)
    dest="third_party/$name"
    [ -d "$dest/.git" ] || run git clone "$url" "$dest"
    run git -C "$dest" checkout "$commit"
done

# 5. Summary
run "${EXEC[@]}" python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
