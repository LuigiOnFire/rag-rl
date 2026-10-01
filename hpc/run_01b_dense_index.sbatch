#!/bin/bash
#SBATCH --job-name=01b_dense_index
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_01b_dense_index.out 

export $(grep -v '^#' ~/.env | xargs)

# ---------------------------------------------------------
# Argument 1: Corpus Selection (default: fullwiki)
# ---------------------------------------------------------
CORPUS_ARG="${1:-fullwiki}"
case "$CORPUS_ARG" in
    squad|squad_wiki)
        CORPUS_TYPE="squad_wiki"
        ;;
    hotpot|fullwiki)
        CORPUS_TYPE="fullwiki"
        ;;
    *)
        echo "Unknown corpus: $CORPUS_ARG. Use 'fullwiki' or 'squad_wiki'."
        exit 1
        ;;
esac

# ---------------------------------------------------------
# Argument 2: Index Topology (default: ivf)
# Options: flat, ivf, hnsw
# ---------------------------------------------------------
INDEX_TYPE="${2:-ivf}"
case "$INDEX_TYPE" in
    flat|ivf|hnsw)
        ;;
    *)
        echo "Unknown index type: $INDEX_TYPE. Choices: 'flat', 'ivf', 'hnsw'."
        exit 1
        ;;
esac

# ---------------------------------------------------------
# Output Path Configuration
# ---------------------------------------------------------
# Flat retains the legacy filename if needed, or appends the suffix
if [[ "$INDEX_TYPE" == "flat" ]]; then
    OUT_FAISS="data/meta/retriever_dense_${CORPUS_TYPE}.faiss"
else
    OUT_FAISS="data/meta/retriever_dense_${CORPUS_TYPE}_${INDEX_TYPE}.faiss"
fi

if [[ -f "$OUT_FAISS" ]]; then
    echo "Skipping dense index build; $OUT_FAISS already exists."
    exit 0
fi

# ---------------------------------------------------------
# Execution via Apptainer
# ---------------------------------------------------------
apptainer exec --nv \
    --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
    --env-file ~/.env \
    pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
    bash -lc "
        echo \"Building GPU-based FAISS index ($INDEX_TYPE) for $CORPUS_TYPE...\"
        .venv/bin/python -u scripts/01b_build_dense_index.py \
            --corpus-type \"$CORPUS_TYPE\" \
            --index-type \"$INDEX_TYPE\" \
            --out-faiss \"$OUT_FAISS\"
        echo \"Index successfully saved to $OUT_FAISS!\"
    "