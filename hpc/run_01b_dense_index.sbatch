#!/bin/bash
#SBATCH --job-name=01b_dense_index
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/01b_dense_index_%j.out 

export $(grep -v '^#' ~/.env | xargs)

CORPUS_TYPE="fullwiki"
if [[ "$1" == "squad" ]] || [[ "$1" == "squad_wiki" ]]; then
    CORPUS_TYPE="squad_wiki"
elif [[ "$1" == "hotpot" ]] || [[ "$1" == "fullwiki" ]]; then
    CORPUS_TYPE="fullwiki"
fi

if [[ "$CORPUS_TYPE" == "squad_wiki" ]]; then
    OUT_FAISS="data/meta/retriever_dense_squad_wiki.faiss"
else
    OUT_FAISS="data/meta/retriever_dense_fullwiki.faiss"
fi

if [[ -f "$OUT_FAISS" ]]; then
    echo "Skipping dense index build; $OUT_FAISS already exists."
    exit 0
fi

apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -lc "echo \"Building GPU-based FAISS dense index for $CORPUS_TYPE...\"; .venv/bin/python -u scripts/01b_build_dense_index.py --corpus-type $CORPUS_TYPE; echo \"Dense index successfully saved!\""
