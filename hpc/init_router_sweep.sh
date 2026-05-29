#!/bin/bash
# Initialize a W&B sweep for router training on the HPC.
# Uses srun to get a short-lived compute node and register the sweep inside the container.

# 1. Load environment variables (like HF_TOKEN, WANDB_API_KEY)
export $(grep -v '^#' ~/.env | xargs)

if [[ "$1" == "deberta" ]]; then
    MODEL_NAME="deberta"
elif [[ "$1" == "bert" ]]; then
    MODEL_NAME="base_bert"
else
    echo "Usage: sbatch init_router_sweep.sh [deberta|bert]"
    exit 1
fi

echo "Requesting a brief 5-minute compute job to register the router sweep with W&B..."

# 2. Execute W&B sweep inside the Apptainer using 'srun'
srun -p l40s -N 1 -n 1 -c 2 -t 00:05:00 --mem 8G \
  apptainer exec \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    --env-file ~/.env \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    .venv/bin/wandb sweep oracle_kd/sweep_router_$MODEL_NAME.yaml

echo "Done! Copy your sweep ID above."
