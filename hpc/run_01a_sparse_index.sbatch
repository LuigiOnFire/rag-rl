#!/bin/bash
#SBATCH --job-name=01a_sparse_index
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/01a_sparse_index_%j.out 

export $(grep -v '^#' ~/.env | xargs)

apptainer exec \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Building CPU-based BM25 sparse index..."
            .venv/bin/python -u scripts/01a_build_sparse_index.py
            echo "Sparse index successfully saved!"
        '
