#!/bin/bash
#SBATCH --job-name=05_sweep_agent
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1              
#SBATCH --time=48:00:00       
#SBATCH --array=1-2
#SBATCH --output=logs/05_sweep_agent_%A_%a.out 

if [ -z "$1" ]; then
  echo "Error: You must provide a wandb sweep ID as the first argument."
  echo "Usage: sbatch hpc/run_05_sweep_agent.sbatch username/project/sweep_id"
  exit 1
fi

export SWEEP_ID=$1

# 1. Load the secure .env file (Goodbye hardcoded HF_TOKEN!)
export $(grep -v '^#' ~/.env | xargs)

echo "Starting W&B Agent for sweep $SWEEP_ID on $HOSTNAME"
export HF_HOME=/scratch.hpc/$USER/.cache/huggingface

# 2. The Container Wrap (WITH Ollama background server!)
apptainer exec --nv  \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Starting Ollama inside container..."
            export OLLAMA_MODELS="./ollama_models"
            
            export OLLAMA_FLASH_ATTENTION=true
            export OLLAMA_KV_CACHE_TYPE=q8_0
            export OLLAMA_GPU_OVERHEAD=15032385536
            export OLLAMA_KEEP_ALIVE="5m"
            export PYTORCH_ALLOC_CONF="expandable_segments:True"

            export OLLAMA_HOST="127.0.0.1:$((11434 + SLURM_ARRAY_TASK_ID))"
            
            ./bin/ollama serve > logs/ollama_grpo_sweep_${SLURM_ARRAY_TASK_ID}.log 2>&1 &  
            
            OLLAMA_PID=$!
            
            sleep 15
            ./bin/ollama pull $LLM_MODEL
            
            echo "Running Wandb Agent for Sweep: $SWEEP_ID"
            source .venv/bin/activate
            export GRPO_RUN_ID="sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            export WANDB_NAME="$GRPO_RUN_ID"
            wandb agent $SWEEP_ID
            
            echo "Cleaning up..."
            kill $OLLAMA_PID
        '
