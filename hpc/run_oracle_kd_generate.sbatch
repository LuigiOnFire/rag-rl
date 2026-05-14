#!/bin/bash
#SBATCH --job-name=oracle_kd_generate
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Oracle KD generation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Parse dataset name from first positional argument (default: hotpotqa)
# Usage: sbatch run_oracle_kd_generate.sbatch [hotpotqa|squad] [extra_args...]
DATASET_NAME="hotpotqa"
if [[ "$1" == "squad" ]] || [[ "$1" == "hotpotqa" ]]; then
    DATASET_NAME="$1"
    shift || true
fi

echo "Dataset: $DATASET_NAME"

# Build ORACLE_KD_ARGS based on dataset with dataset-specific defaults
if [[ $# -gt 0 ]]; then
    # Additional args provided after dataset name (may override defaults)
    ORACLE_KD_ARGS="--dataset-name $DATASET_NAME $*"
else
    # Use dataset-specific defaults (including offset)
    case $DATASET_NAME in
        squad)
            # For now, use same Wikipedia retriever as hotpot
            # (Architecture ready for future --retriever squad_wiki flag)
            # Default offset=0 for squad (no prior offset needed)
            ORACLE_KD_ARGS="--dataset-name squad --setting fullwiki --offset 0"
            ;;
        *)
            # HotpotQA: skip first 6390 to continue from previous run
            ORACLE_KD_ARGS="--dataset-name hotpotqa --setting fullwiki --split train --offset 6390"
            ;;
    esac
fi
export ORACLE_KD_ARGS

echo "Using ORACLE_KD_ARGS: $ORACLE_KD_ARGS"

# 2. The Container Wrap
apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            echo "Container ORACLE_KD_ARGS: $ORACLE_KD_ARGS"
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_oracle_kd.log 2>&1 &
            OLLAMA_PID=$!

            # Wait for the server to wake up
            sleep 15

            # Ensure models are ready (skip if already pulled, but safe to leave)
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL

            echo "Running Oracle KD generator..."
            .venv/bin/python -u oracle_kd/generate_oracle.py $ORACLE_KD_ARGS

            echo "Cleaning up..."
            kill $OLLAMA_PID
        '

echo "Oracle KD generation complete."