#!/bin/bash
#SBATCH --job-name=06_eval_checkpoints
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1               
#SBATCH --time=12:00:00            
#SBATCH --output=logs/%j_06_eval_checkpoints.out 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Load the secure .env file
export $(grep -v '^#' ~/.env | xargs)

UNIQUE_PORT=$(( 11000 + (SLURM_JOB_ID % 10000) ))
echo "Starting Checkpoint Evaluation on $HOSTNAME | Assigned Ollama Port: $UNIQUE_PORT"

export HF_HOME=/scratch.hpc/$USER/.cache/huggingface
export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH

export PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini
mkdir -p $PYSERINI_CACHE
export APPTAINERENV_PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini


mkdir -p logs data

# If no arguments were passed to sbatch, evaluate the key v7 checkpoints automatically
if [ $# -eq 0 ]; then
    set -- \
    --datasets hotpot \
    --num_samples 200 \
    --seed 42 \
    --output_json results/eval_v9_cross-dataset_v4.json \
    --checkpoints \
        models/greenrag-grpo-v9-cond-2e5/step_450 \
        models/greenrag-grpo-v9-cond-5e5/step_450 \
        models/greenrag-grpo-v9-cond-5e6/step_450 \
        models/greenrag-grpo-v9-cond-7e7/step_450 \
    --skip_sft \
    --skip_baselines
fi

echo "Passing arguments to evaluation script: $@"

# 2. Container Execution
apptainer exec --nv \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    --env-file ~/.env \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    bash -c '
        echo "Starting Ollama inside container..."
        export OLLAMA_MODELS="./ollama_models"
        export OLLAMA_HOST="127.0.0.1:'"$UNIQUE_PORT"'"
        export OLLAMA_FLASH_ATTENTION=true
        export OLLAMA_KV_CACHE_TYPE=q8_0
        export OLLAMA_GPU_OVERHEAD=12884901888
        export OLLAMA_KEEP_ALIVE="5m"
        export OLLAMA_MAX_LOADED_MODELS=2

        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

        ./bin/ollama serve > logs/ollama_eval_'"$UNIQUE_PORT"'.log 2>&1 &
        OLLAMA_PID=$!
        
        sleep 15
        
        export OLLAMA_HOST="127.0.0.1:'"$UNIQUE_PORT"'"
        ./bin/ollama pull $LLM_MODEL
        
        # Strip all SLURM variables before Python launches
        for var in $(compgen -v | grep "^SLURM_"); do
            unset "$var"
        done

        echo "Running Python Checkpoint Evaluation with args: $@"
        .venv/bin/python -u scripts/06_eval_checkpoints.py "$@"
        
        echo "Cleaning up Ollama..."
        kill $OLLAMA_PID
    ' bash "$@"

echo "Checkpoint Evaluation complete."