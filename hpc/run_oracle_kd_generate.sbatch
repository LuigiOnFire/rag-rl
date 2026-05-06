#!/bin/bash
#SBATCH --job-name=oracle_kd_generate
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/oracle_kd_generate_%j.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Oracle KD generation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Optional: pass extra CLI args via ORACLE_KD_ARGS
ORACLE_KD_ARGS=${ORACLE_KD_ARGS:-"--datasets hotpot --setting fullwiki --split train"}

# 2. The Container Wrap
apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_oracle_kd.log 2>&1 &
            OLLAMA_PID=$!

            # Wait for the server to wake up
            sleep 15

            # Ensure models are ready (skip if already pulled, but safe to leave)
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL

            echo "Running Oracle KD generator..."
            .venv/bin/python -u oracle_kd/generate_oracle.py $ORACLE_KD_ARGS

            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "Oracle KD generation complete."