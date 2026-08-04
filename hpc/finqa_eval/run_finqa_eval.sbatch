#!/bin/bash
#SBATCH --job-name=finqa_thrifty
#SBATCH --nodes=1
#SBATCH --partition=l40s           # Targeting the NVIDIA L40S nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=12:00:00            
#SBATCH --output=logs/%j_finqa_eval.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting FinQA Thrifty Evaluation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Ensure log directory exists
mkdir -p logs

# 2. The Container Wrap
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_finqa.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            # Ensure models are ready
            echo "Checking SLM model availability..."
            ./bin/ollama pull $SLM_MODEL
            
            echo "Running Python Thrifty Evaluation Harness..."
            .venv/bin/python -u scripts/finqa_eval/dry_run.py
            
            echo "Cleaning up Ollama daemon..."
            kill $OLLAMA_PID
        '

echo "FinQA evaluation job complete."