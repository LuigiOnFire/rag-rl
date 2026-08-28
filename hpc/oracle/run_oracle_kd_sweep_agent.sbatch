#!/bin/bash
#SBATCH --job-name=oracle_kd_sweep_agent
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a_oracle_kd_sweep_agent.out

if [ -z "$1" ]; then
  echo "Error: You must provide a wandb sweep ID as the first argument."
  echo "Usage: sbatch hpc/run_oracle_kd_sweep_agent.sbatch username/project/sweep_id"
  exit 1
fi

export SWEEP_ID=$1

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting W&B Agent for sweep $SWEEP_ID on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Running W&B Agent for sweep: $SWEEP_ID"
            .venv/bin/wandb agent $SWEEP_ID
        '
