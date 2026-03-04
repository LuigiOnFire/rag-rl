#!/bin/bash
#SBATCH --job-name=03_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --time=08:00:00           
#SBATCH --output=logs/03_sft_%j.out 

echo "Starting SFT Training on $HOSTNAME"
# Notice: No Ollama commands here! We just go straight to training.

# Optional: If you use Hugging Face for tracking/downloading base models
export HF_TOKEN="your_hf_token_here" 

apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/python scripts/03_train_sft.py

echo "SFT Training complete."