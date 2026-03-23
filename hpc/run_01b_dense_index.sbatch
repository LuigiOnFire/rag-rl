#!/bin/bash
#SBATCH --job-name=01b_dense_index
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/01b_dense_index_%j.out 

export $(grep -v '^#' ~/.env | xargs)

apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Building GPU-based FAISS dense index..."
            .venv/bin/python -u scripts/01b_build_dense_index.py
            echo "Dense index successfully saved!"
        '
