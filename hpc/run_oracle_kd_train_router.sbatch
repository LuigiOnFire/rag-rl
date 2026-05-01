#!/bin/bash
#SBATCH --job-name=oracle_kd_train_router
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/oracle_kd_train_router_%j.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Oracle KD router training on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Optional: pass extra CLI args via ROUTER_TRAIN_ARGS
ROUTER_TRAIN_ARGS=${ROUTER_TRAIN_ARGS:-"--input-csv data/oracle/oracle_training_data.csv --model-name microsoft/deberta-v3-base --output-dir models/router"}

apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Running router training..."
            .venv/bin/python -u oracle_kd/train_router.py $ROUTER_TRAIN_ARGS
        '

echo "Oracle KD router training complete."