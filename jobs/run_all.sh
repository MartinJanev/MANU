#!/bin/bash
# Login node: submit one stage of the paper's runs.
#   bash jobs/run_all.sh dev      short test of every model and task on the dev patients (replaces older dev results)
#   bash jobs/run_all.sh main     every model on the split paper, then MS FLAIR
#   bash jobs/run_all.sh second   Q&A and prior knowledge for the six full-prompt models
#   DRY_RUN=1 bash jobs/run_all.sh <stage>     print the sbatch commands only
# main and second resume: submitting again runs only what is missing.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE=${1:-}
case "$STAGE" in
    dev|main|second) ;;
    *) echo "Usage: bash jobs/run_all.sh dev|main|second" >&2; exit 1 ;;
esac

# fast first, so a short night still finishes most models
FULL_PROMPT="qwen3vl8b lingshu7b qoqmed7b healthgptpro27b fleming8b medgemma15"
MODELS="hlip m3dlamed_phi3 m3dlamed_llama2 radfm_qna med2e3 braingemma3d $FULL_PROMPT"

if [ "${DRY_RUN:-0}" != 1 ] && ! grep -q "SETUP FINISHED: ok" results/slurm/setup_paper.log 2>/dev/null; then
    echo "STOP: the downloads of Block 1 have not finished OK. Send Claude the output of:" >&2
    echo "  tail -5 results/slurm/setup_paper.log" >&2
    exit 1
fi

submit() {  # job name, model folder, runner arguments
    local name=$1
    shift
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "sbatch -J $name jobs/run_model.sh $*"
    else
        sbatch -J "$name" jobs/run_model.sh "$@"
    fi
}

case "$STAGE" in
dev)
    submit radfm_demo radfm_qna --demo
    for m in $MODELS; do submit "${m}_dev" "$m" --split dev --controls --fresh; done
    for m in $FULL_PROMPT; do submit "${m}_dev_qna" "$m" --split dev --controls --fresh --task qna; done
    for m in $FULL_PROMPT; do
        submit "${m}_dev_prior" "$m" --split dev --controls --fresh --task prior --datasets molab,ms
    done
    ;;
main)
    for m in $MODELS; do submit "${m}_paper" "$m" --split paper --controls; done
    for m in $MODELS; do submit "${m}_paper_flair" "$m" --split paper --controls --datasets ms --ms-flair; done
    ;;
second)
    for m in $FULL_PROMPT; do submit "${m}_paper_qna" "$m" --split paper --controls --task qna; done
    for m in $FULL_PROMPT; do
        submit "${m}_paper_prior" "$m" --split paper --controls --task prior --datasets molab,ms
    done
    ;;
esac
