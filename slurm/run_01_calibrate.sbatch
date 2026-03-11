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

# 1. Load the secure .env file from your HOME directory vault
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Calibration on $HOSTNAME"

# 2. Start Ollama (Ollama natively supports AMD ROCm out of the box!)
export OLLAMA_MODELS="./ollama_models"
./bin/ollama serve > logs/ollama_calib.log 2>&1 &
OLLAMA_PID=$!
sleep 15 

# 3. Get the models
echo "Pulling Ollama models..."
./bin/ollama pull $LLM_MODEL
./bin/ollama pull $SLM_MODEL

export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 4. Run Python Script via Apptainer
apptainer run --rocm  \
	--bind /scratch.hpc/$USER:/scratch.hpc/$USER \
	--env-file ~/.env \
	pytorch_rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.sif \
	.venv/bin/python -u scripts/01_calibrate.py

# 4. Cleanup
kill $OLLAMA_PID
echo "Calibration complete."
