#!/bin/bash
#SBATCH --job-name=oracle_kd_generate
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j_finqa_generate.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

# --- JAVA CONFIGURATION (Automatic Container Injection) ---
export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7

export PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini
mkdir -p $PYSERINI_CACHE

# We set APPTAINERENV_ so these become JAVA_HOME and PATH inside the container
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH

export APPTAINERENV_PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini
echo "Starting FinQA Trajectory generation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

echo "Dataset: $DATASET_NAME"

# Build the args as an array so offsets like '--offset 100' survive intact.
ARGS=(--split train)

if [[ $# -gt 0 ]]; then
    ARGS+=("$@")
fi

echo "UsingARGS: ${ARGS[*]}"

# Verify Java existence before submitting to the scheduler
if [ ! -d "$JAVA_HOME" ]; then
    echo "ERROR: JAVA_HOME directory does not exist: $JAVA_HOME"
    exit 1
fi

# 2. The Container Wrap
apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            echo "Container ARGS: $*"
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_finqa.log 2>&1 &
            OLLAMA_PID=$!

            # Wait for the server to wake up
            sleep 15

            # Ensure models are ready (skip if already pulled, but safe to leave)
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL

            echo "Running FINQA generator..."
            .venv/bin/python -u scripts/finqa_eval/gen_trajectories.py "$@"

            echo "Cleaning up..."
            kill $OLLAMA_PID
        ' bash "${ARGS[@]}"

echo "FinQA Traj generation complete." 