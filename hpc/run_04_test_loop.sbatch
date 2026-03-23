#!/bin/bash
#SBATCH --job-name=04_test_loop
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=04:00:00            # 4 hours is usually plenty for evaluation!
#SBATCH --output=logs/04_test_loop_%j.out 

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Evaluation Loop on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 2. The Container Wrap (WITH Ollama background server!)
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_test.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            # Ensure models are ready
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL
            
            echo "Running Python Test Loop..."
            .venv/bin/python -u scripts/04_test_loop.py
            
            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "Evaluation complete."