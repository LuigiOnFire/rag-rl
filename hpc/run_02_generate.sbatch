#!/bin/bash
#SBATCH --job-name=02_generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --time=12:00:00           
#SBATCH --output=logs/02_trajectory_%j.out 

echo "Starting Trajectory Generation on $HOSTNAME"

export OLLAMA_MODELS="./ollama_models"
./ollama serve > logs/ollama_traj.log 2>&1 &
OLLAMA_PID=$!
sleep 15 

apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/python scripts/02_generate_trajectories.py

kill $OLLAMA_PID
echo "Trajectory generation complete."