#!/bin/bash
#SBATCH --job-name=oracle_kd_generate
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=13:00:00
#SBATCH --output=logs/%j_oracle_kd_generate.out

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

echo "Starting Oracle KD generation on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# Parse dataset name from first positional argument (default: hotpotqa)
# Usage: sbatch run_oracle_kd_generate.sbatch [hotpotqa|squad|nq] [extra_args...]
DATASET_NAME="hotpotqa"
if [[ "$1" == "squad" ]] || [[ "$1" == "hotpotqa" ]] || [[ "$1" == "nq" ]]; then
    DATASET_NAME="$1"
    shift || true
fi

echo "Dataset: $DATASET_NAME"

# Build the oracle args as an array so offsets like '--offset 100' survive intact.
ORACLE_KD_ARGS=(--dataset-name "$DATASET_NAME" --setting fullwiki --execution-mode all_routes --split train)


if [[ $# -gt 0 ]]; then
    ORACLE_KD_ARGS+=("$@")
fi

echo "Using ORACLE_KD_ARGS: ${ORACLE_KD_ARGS[*]}"

# 2. The Container Wrap
apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            echo "Container ORACLE_KD_ARGS: $*"
            export OLLAMA_MODELS="./ollama_models"
            ./bin/ollama serve > logs/ollama_oracle_kd.log 2>&1 &
            OLLAMA_PID=$!

            # Wait for the server to wake up
            sleep 15

            # Ensure models are ready (skip if already pulled, but safe to leave)
            ./bin/ollama pull $LLM_MODEL
            ./bin/ollama pull $SLM_MODEL

            echo "Running Oracle KD generator..."
            .venv/bin/python -u oracle_kd/generate_oracle.py "$@"

            echo "Cleaning up..."
            kill $OLLAMA_PID
        ' bash "${ORACLE_KD_ARGS[@]}"

echo "Oracle KD generation complete."