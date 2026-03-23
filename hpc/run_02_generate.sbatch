#!/bin/bash
#SBATCH --job-name=02_generate
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=12:00:00            # Kept your 12-hour limit!
#SBATCH --output=logs/02_trajectory_%j.out 

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Trajectory Generation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 2. The Container Wrap
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_traj.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            # Ensure models are ready (skip if already pulled, but safe to leave)
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL
            
            echo "Running Python Trajectory Generation..."
            # <-- This is the only line that actually changes!
            .venv/bin/python -u scripts/02_generate.py
            
            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "Trajectory generation complete."