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

CORPUS_TYPE="fullwiki"
if [[ "$1" == "squad" ]] || [[ "$1" == "squad_wiki" ]]; then
    CORPUS_TYPE="squad_wiki"
elif [[ "$1" == "hotpot" ]] || [[ "$1" == "fullwiki" ]]; then
    CORPUS_TYPE="fullwiki"
fi

if [[ "$CORPUS_TYPE" == "squad_wiki" ]]; then
    OUT_SPARSE="data/meta/retriever_sparse_squad_wiki.pkl"
else
    OUT_SPARSE="data/meta/retriever_sparse_fullwiki.pkl"
fi

if [[ -f "$OUT_SPARSE" ]]; then
    echo "Skipping sparse index build; $OUT_SPARSE already exists."
    exit 0
fi

apptainer exec \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -lc "echo \"Building CPU-based BM25 sparse index for $CORPUS_TYPE...\"; .venv/bin/python -u scripts/01a_build_sparse_index.py --corpus-type $CORPUS_TYPE; echo \"Sparse index successfully saved!\""
