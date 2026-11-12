#!/bin/bash
#SBATCH --job-name=3d-radfm
#SBATCH --chdir=/home/hpc/users/martin.janev/manu
#SBATCH --output=results/%x_%j.out      # %x = име на job, %j = ID на job
#SBATCH --partition=openlab-queue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail

DATASET=${1:?Употреба: sbatch job_radfm.sh ixi|molab|ms|preview [limit]}

echo "Job $SLURM_JOB_ID ($DATASET) на $(hostname), почеток: $(date)"
nvidia-smi

export HF_HOME=/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUSERBASE=/pyuser


singularity exec --nv \
    -B ./code:/code \
    -B ./data:/data \
    -B ./results:/results \
    -B ./reference:/reference \
    -B ./hf_cache:/hf_cache \
    -B ./pyuser:/pyuser \
    -B ./RadFM/Quick_demo:/radfm \
    containers/radfm.sif \
    python3 /code/run_radfm.py "$DATASET" "${@:2}"

singularity exec -B ./pyuser:/pyuser containers/radfm.sif pip freeze > results/requirements_radfm.txt

echo "Крај: $(date)"
