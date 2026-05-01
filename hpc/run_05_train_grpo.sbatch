#!/bin/bash
#SBATCH --job-name=05_train_grpo
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=48:00:00            
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
            
            # We need to shrink the massive 8GB KV cache so the 32B model fits completely
            # into the VRAM remaining after PyTorch takes its ~14GB slice.
            export OLLAMA_FLASH_ATTENTION=true
            export OLLAMA_KV_CACHE_TYPE=q8_0
            
            # Reserve exactly 18GB of overhead for PyTorch (18 * 1024^3)
            # This leaves 26.4GB on the L40s (44.4GB - 18GB) which perfectly fits 
            # Reserve exactly 14GB for PyTorch (14 * 1024^3)
            export OLLAMA_GPU_OVERHEAD=15032385536
            export OLLAMA_KEEP_ALIVE="5m"
            export PYTORCH_ALLOC_CONF="expandable_segments:True"
            
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