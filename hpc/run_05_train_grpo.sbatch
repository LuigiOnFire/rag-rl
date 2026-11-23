#!/bin/bash
#SBATCH --job-name=05_train_grpo
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=48:00:00            
#SBATCH --output=logs/%j_05_train_grpo.out 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Load the secure .env file
set -a
source ~/.env
set +a

UNIQUE_PORT=$(( 11000 + (SLURM_JOB_ID % 10000) ))
echo "Starting GRPO Training on $HOSTNAME | Assigned Ollama Port: $UNIQUE_PORT"

export HF_HOME=/scratch.hpc/$USER/.cache/huggingface
export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH


export PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini
mkdir -p $PYSERINI_CACHE
export APPTAINERENV_PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini


# If no arguments are passed to sbatch, default to launching the v10 mixed-dataset run
if [ $# -eq 0 ]; then
    set -- \
        --run_name greenrag-grpo-v10-mixed-5e5 \
        --output_dir models/greenrag-grpo-v10-mixed-5e5 \
        --joule_lambda 5.0e-5 \
        --learning_rate 1.0e-6 \
        --kl_coef 0.0015 \
        --gradient_accum 8 \
        --datasets hotpot,squad,nq
fi

ARGS="$@"
echo "Passing arguments to training script: $ARGS"

# 2. Container Execution
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"
export OLLAMA_MODELS="./ollama_models"
export OLLAMA_FLASH_ATTENTION=true
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_GPU_OVERHEAD=12884901888
export OLLAMA_KEEP_ALIVE="5m"
export OLLAMA_MAX_LOADED_MODELS=2

./bin/ollama serve > "logs/ollama_grpo_${UNIQUE_PORT}.log" 2>&1 &
OLLAMA_PID=$!
sleep 15

./bin/ollama pull "$LLM_MODEL"

apptainer exec --nv \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    --env-file ~/.env \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    .venv/bin/python -u scripts/05_train_grpo.py $ARGS

kill "$OLLAMA_PID"

echo "GRPO Training complete."