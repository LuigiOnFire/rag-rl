#!/bin/bash
#SBATCH --job-name=05_train_grpo
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=24:00:00            # GRPO needs time, keeping your 24hr limit!
#SBATCH --output=logs/05_train_grpo_%j.out 

# 1. Load the secure .env file (Goodbye hardcoded HF_TOKEN!)
export $(grep -v '^#' ~/.env | xargs)

echo "Starting GRPO Training on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 2. The Container Wrap (WITH Ollama background server!)
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_grpo.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            # Ensure environment/judge models are ready
            ./bin/ollama pull $LLM_MODEL
            
            echo "Running Python GRPO Training..."
            .venv/bin/python -u scripts/05_train_grpo.py
            
            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "GRPO Training complete."