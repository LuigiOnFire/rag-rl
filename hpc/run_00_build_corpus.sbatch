#!/bin/bash
#SBATCH --job-name=00_build_corpus
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/00_build_corpus_%j.out 

export $(grep -v '^#' ~/.env | xargs)

apptainer exec \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -c '
            echo "Running CPU-based corpus fetcher..."
            .venv/bin/python -u scripts/00_build_corpus.py
            echo "Corpus successfully downloaded and prepared!"
        '
