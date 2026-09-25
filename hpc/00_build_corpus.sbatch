#!/bin/bash
#SBATCH --job-name=00_build_corpus
#SBATCH --nodes=1
#SBATCH --partition=l40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/00_build_corpus_%j.out 

export $(grep -v '^#' ~/.env | xargs)

CORPUS_TYPE="fullwiki"
if [[ "$1" == "squad" ]]; then
    CORPUS_TYPE="squad_wiki"
elif [[ "$1" == "hotpot" ]]; then
    CORPUS_TYPE="fullwiki"
fi

if [[ "$CORPUS_TYPE" == "squad_wiki" ]]; then
    OUT_CORPUS="data/meta/squad_wiki_corpus.pkl"
else
    OUT_CORPUS="data/meta/fullwiki_corpus.pkl"
fi

if [[ -f "$OUT_CORPUS" ]]; then
    echo "Skipping corpus build; $OUT_CORPUS already exists."
    exit 0
fi

apptainer exec \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -lc "echo \"Running CPU-based corpus fetcher for $CORPUS_TYPE...\"; .venv/bin/python -u scripts/00_build_corpus.py --corpus-type $CORPUS_TYPE; echo \"Corpus successfully downloaded and prepared!\""
