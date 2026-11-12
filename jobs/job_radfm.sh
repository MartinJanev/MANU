#!/bin/bash
#SBATCH --job-name=3d-radfm
#SBATCH --chdir=/home/hpc/users/martin.janev/manu
#SBATCH --output=results/slurm/%x_%j.out      # %x = име на job, %j = ID на job
#SBATCH --partition=openlab-queue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail

mkdir -p results/slurm results/radfm/full

DATASET=${1:?Употреба: sbatch jobs/job_radfm.sh ixi|molab|ms|preview [limit]}

echo "Job $SLURM_JOB_ID ($DATASET) на $(hostname), почеток: $(date)"
nvidia-smi

export HF_HOME=/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUSERBASE=/pyuser


singularity exec --nv \
    -B ./models/radfm:/code \
    -B ./data:/data \
    -B ./results/radfm/full:/results \
    -B ./reference:/reference \
    -B ./hf_cache:/hf_cache \
    -B ./pyuser/radfm:/pyuser \
    -B ./third_party/RadFM/Quick_demo:/radfm \
    containers/radfm.sif \
    python3 /code/run.py "$DATASET" "${@:2}"

singularity exec -B ./pyuser/radfm:/pyuser containers/radfm.sif pip freeze > results/radfm/full/requirements_radfm.txt

echo "Крај: $(date)"
