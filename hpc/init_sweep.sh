#!/bin/bash
# A lightweight script to initialize a W&B sweep on the HPC.
# Since the login node doesn't have our Python environment or Apptainer loaded natively,
# we use 'srun' to briefly grab a compute node and initialize the sweep inside our container.

# 1. Load environment variables (like HF_TOKEN, WANDB_API_KEY)
export $(grep -v '^#' ~/.env | xargs)

echo "Requesting a brief 5-minute compute job to register the sweep with weights and biases..."

# 2. Execute W&B sweep inside the Apptainer using 'srun'
srun -p l40s -N 1 -n 1 -c 2 -t 00:05:00 --mem 8G \
  apptainer exec \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    --env-file ~/.env \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    .venv/bin/wandb sweep sweep.yaml

echo "Done! Copy your sweep ID above."
