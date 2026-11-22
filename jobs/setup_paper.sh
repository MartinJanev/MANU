#!/bin/bash
# Login node, once: everything the paper's runs download. Checks first, then works in the background, so the
# connection may be closed as soon as it prints STARTED.
#   bash jobs/setup_paper.sh     (asks for the token of an account that accepted the CT-RATE terms)
# It writes results/splits_proposed.json (split "paper"), then downloads about 57 GB: M3D-LaMed Phi-3 (16) and
# Llama-2 (28), 30 more CT-RATE volumes (10), Python packages (3). Safe to run again: finished files are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NEED_GB=60
LOG=results/slurm/setup_paper.log
SPLIT=results/splits_proposed.json

if [ -z "${HF_TOKEN:-}" ] && [ -t 0 ]; then  # asked for here, so it is neither in the shell history nor in a paste
    read -rsp "Paste your Hugging Face token (it stays invisible), then Enter: " HF_TOKEN
    echo
fi
[ -n "${HF_TOKEN:-}" ] || { echo "STOP: no HF_TOKEN (needed for the gated CT-RATE volumes)." >&2; exit 1; }
export HF_TOKEN
python3 -c "import huggingface_hub" \
    || { echo "STOP: the login node's python3 has no huggingface_hub. Send this line to Claude." >&2; exit 1; }
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
if [ "$free_gb" -lt "$NEED_GB" ]; then
    echo "STOP: $free_gb GB free here, $NEED_GB GB needed. Send this line to Claude." >&2
    exit 1
fi

mkdir -p results/slurm
python3 tools/make_split.py --paper --out "$SPLIT"

# the token reaches the children through the environment, never through a command line
nohup bash -c "
status=ok
python3 data_tools/prezemi_ct_rate.py --split paper --splits $SPLIT || status=FAILED
bash jobs/setup_env.sh m3dlamed_phi3 || status=FAILED
bash jobs/setup_env.sh m3dlamed_llama2 || status=FAILED
echo \"SETUP FINISHED: \$status\"
" > "$LOG" 2>&1 < /dev/null &
wait_for=$!
sleep 1
echo "STARTED (process $wait_for, $free_gb GB free). You can log out. Progress: tail $LOG"
