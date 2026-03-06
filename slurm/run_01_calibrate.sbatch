#!/bin/bash
#SBATCH --job-name=01_calibrate
#SBATCH --nodes=1
#SBATCH --partition=mi210          # <-- CRITICAL 1: Target the AMD nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # Bumped to 8 for better data loading
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --output=logs/01_calibrate_%j.out 
# (Notice: --time is removed to take advantage of the infinite time rule!)

echo "Starting Calibration on $HOSTNAME"

# 1. Start Ollama (Ollama natively supports AMD ROCm out of the box!)
export OLLAMA_MODELS="./ollama_models"
./ollama serve > logs/ollama_calib.log 2>&1 &
OLLAMA_PID=$!
sleep 15 

# 2. Run Python Script via Apptainer
# <-- CRITICAL 2: Use --rocm instead of --nv
# <-- CRITICAL 3: Use the new AMD .sif file
apptainer run --rocm pytorch_rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.sif .venv/bin/python scripts/01_calibrate.py

# 3. Cleanup
kill $OLLAMA_PID
echo "Calibration complete."