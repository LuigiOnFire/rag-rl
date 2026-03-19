#!/bin/bash
#SBATCH --job-name=03_train_sft
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --time=12:00:00            
#SBATCH --output=logs/03_train_sft_%j.out 

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting SFT Training on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 2. The Container Wrap (Pure PyTorch, no Ollama background server!)
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Running Python SFT Training..."
            
            .venv/bin/python -u scripts/03_train_sft.py
            
        '

echo "SFT Training complete."