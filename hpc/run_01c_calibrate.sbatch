#!/bin/bash
#SBATCH --job-name=01_calibrate
#SBATCH --nodes=1
#SBATCH --partition=l40s        
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              
#SBATCH --output=logs/%j_01_calibrate.out 

# 1. Load the secure .env file from your HOME directory vault
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Calibration on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7
# We set APPTAINERENV_ so these become JAVA_HOME and PATH inside the container
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH


# 2. Run Python Script via Apptainer (Notice the --nv flag and new .sif image!)
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_calib.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            echo "Pulling models..."
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL
            
            echo "Running Python Calibration..."
            .venv/bin/python -u scripts/01c_calibrate.py
            
            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "Calibration complete."
