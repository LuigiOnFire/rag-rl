#!/bin/bash
#SBATCH --job-name=setup_rag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00           # 2 hours is plenty for downloading/installing
#SBATCH --output=logs/setup_%j.out # Saves the terminal output to a log file

echo "Starting setup on compute node: $HOSTNAME"

# 1. Pull the PyTorch image
echo "Pulling Apptainer image..."
apptainer pull docker://pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# 2. Create the virtual environment
echo "Creating .venv..."
apptainer run pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif python -m venv .venv

# 3. Install requirements
echo "Installing dependencies..."
apptainer run pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/pip install -r requirements.txt

echo "Setup complete!"
