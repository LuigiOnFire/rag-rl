#!/bin/bash
#SBATCH --job-name=finqa_router_comp
#SBATCH --nodes=1
#SBATCH --partition=l40s           # Targeting the NVIDIA L40S nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=12:00:00            
#SBATCH --output=logs/%j_finqa_comparison.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

# --- JAVA CONFIGURATION (Required for Pyserini/Lucene) ---
export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7
export PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini
mkdir -p $PYSERINI_CACHE

# Inject variables into the Apptainer environment
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH
export APPTAINERENV_PYSERINI_CACHE=$PYSERINI_CACHE

echo "Starting FinQA 3-Way Router Comparison on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Ensure log directory exists
mkdir -p logs

# Verify Java existence before submitting to the scheduler
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JAVA_HOME directory does not exist: $JAVA_HOME"
    exit 1
fi

# 2. The Container Wrap
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_comparison.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for the server to wake up
            sleep 15
            
            # Ensure models are ready for both SLM and LLM actions
            echo "Pulling SLM and LLM models..."
            if [ -n "$SLM_MODEL" ]; then ./bin/ollama pull $SLM_MODEL; fi
            if [ -n "$LLM_MODEL" ]; then ./bin/ollama pull $LLM_MODEL; fi
            
            echo "Running 3-Way Router Comparison Harness..."
            .venv/bin/python -u scripts/finqa_eval/run_comparison.py --num_samples 50
            
            echo "Cleaning up Ollama daemon..."
            kill $OLLAMA_PID
        '

echo "FinQA router comparison job complete."