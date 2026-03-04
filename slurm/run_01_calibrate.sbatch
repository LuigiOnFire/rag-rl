#!/bin/bash
#SBATCH --job-name=01_calibrate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1              
#SBATCH --time=04:00:00           
#SBATCH --output=logs/01_calibrate_%j.out 

echo "Starting Calibration on $HOSTNAME"

# 1. Start Ollama
export OLLAMA_MODELS="./ollama_models"
./ollama serve > logs/ollama_calib.log 2>&1 &
OLLAMA_PID=$!
sleep 15 

# 2. Run Python Script via Apptainer
apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/python scripts/01_calibrate.py

# 3. Cleanup
kill $OLLAMA_PID
echo "Calibration complete."