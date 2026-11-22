#!/bin/bash
# Login node, once: everything the paper's runs download. Checks first, then works in the background, so the
# connection may be closed as soon as it prints STARTED.
#   bash jobs/setup_paper.sh     (asks for the token of an account that accepted the CT-RATE terms)
# It writes results/splits_proposed.json (split "paper"), then downloads about 57 GB: M3D-LaMed Phi-3 (16) and
# Llama-2 (28), 30 more CT-RATE volumes (10), Python packages (3). Safe to run again: finished files are skipped.
#
# The login node's python3 is standard library only (enough for make_split.py). Whatever needs huggingface_hub runs
# in the image with HLIP's packages (huggingface_hub 0.36.2, results/slurm/hlip_2778_freeze.txt).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NEED_GB=60
LOG=results/slurm/setup_paper.log
HUB="singularity exec -B .:/manu -B ./data:/data -B ./results:/results -B ./pyuser/hlip:/pyuser containers/ngc2412.sif"
LABELS=dataset/multi_abnormality_labels/valid_predicted_labels.csv

stop() {
    echo "STOP: $* Send this line to Claude." >&2
    exit 1
}

if [ -z "${HF_TOKEN:-}" ] && [ -t 0 ]; then  # asked for here, so it is neither in the shell history nor in a paste
    read -rsp "Paste your Hugging Face token (it stays invisible), then Enter: " HF_TOKEN
    echo
fi
[ -n "${HF_TOKEN:-}" ] || stop "no HF_TOKEN (needed for the gated CT-RATE volumes)."
export HF_TOKEN SINGULARITYENV_HF_TOKEN="$HF_TOKEN" SINGULARITYENV_PYTHONUSERBASE=/pyuser

mkdir -p results/slurm
echo "Checking the download container (up to a minute) ..."
$HUB python3 -c "import huggingface_hub" \
    || stop "containers/ngc2412.sif with pyuser/hlip cannot import huggingface_hub."
echo "Checking the token against CT-RATE ..."
$HUB python3 -c "import os; from huggingface_hub import get_hf_file_metadata, hf_hub_url; get_hf_file_metadata(hf_hub_url('ibrahimhamamci/CT-RATE', '$LABELS', repo_type='dataset'), token=os.environ['HF_TOKEN'])" \
    || stop "the token cannot read CT-RATE (wrong or expired token, or the terms not accepted with this account)."
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
[ "$free_gb" -ge "$NEED_GB" ] || stop "$free_gb GB free here, $NEED_GB GB needed."

python3 tools/make_split.py --paper --out results/splits_proposed.json

# the token reaches the children through the environment, never through a command line
nohup bash -c "
status=ok
$HUB python3 /manu/data_tools/prezemi_ct_rate.py --split paper --splits /results/splits_proposed.json --data /data || status=FAILED
bash jobs/setup_env.sh m3dlamed_phi3 || status=FAILED
bash jobs/setup_env.sh m3dlamed_llama2 || status=FAILED
echo \"SETUP FINISHED: \$status\"
" > "$LOG" 2>&1 < /dev/null &
pid=$!
sleep 1
echo "STARTED (process $pid, $free_gb GB free). You can log out. Progress: tail $LOG"
