#!/bin/bash
#SBATCH --job-name=04_grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --time=24:00:00           
#SBATCH --output=logs/04_grpo_%j.out 

echo "Starting GRPO Training on $HOSTNAME"

export OLLAMA_MODELS="./ollama_models"
./ollama serve > logs/ollama_grpo.log 2>&1 &
OLLAMA_PID=$!
sleep 15 

export HF_TOKEN="your_hf_token_here"

apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/python scripts/05_train_grpo.py

kill $OLLAMA_PID
echo "GRPO Training complete."