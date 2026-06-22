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

# Java / Pyserini environment variables
export JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-21.0.4+7
export APPTAINERENV_JAVA_HOME=$JAVA_HOME
export APPTAINERENV_PATH=$JAVA_HOME/bin:$PATH
export APPTAINERENV_PYSERINI_CACHE=/scratch.hpc/$USER/.cache/pyserini

# Define index paths
INDEX_DIR="data/indices/fullwiki_index"

# 1. Logic for SQuAD (Prebuilt - nothing to build)
if [[ "$1" == "squad" ]] || [[ "$1" == "squad_wiki" ]]; then
    echo "Using prebuilt index for SQuAD/DPR. Skipping build."
    exit 0
fi

# 2. Logic for Fullwiki (Custom build)
# Safety check: Look for the Lucene directory, not a .pkl file
if [ -d "$INDEX_DIR" ] && [ "$(ls -A $INDEX_DIR)" ]; then
    echo "Sparse index directory $INDEX_DIR already exists and is not empty. Skipping."
    exit 0
fi

echo "Building Lucene index for fullwiki..."
apptainer exec --nv \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    bash -lc ".venv/bin/python -u scripts/01a_build_sparse_index.py --corpus-type fullwiki"