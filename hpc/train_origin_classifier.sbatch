#!/bin/bash
#SBATCH --job-name=train_origin_classifier
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_origin_classifier_%j.out

set -euo pipefail

export $(grep -v '^#' ~/.env | xargs)
export HF_HOME="/scratch.hpc/$USER/rag-rl/.cache/huggingface"

INPUT_CSV="${1:-data/oracle/origin_training_data.csv}"
OUTPUT_DIR="models/router_origin"

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Origin CSV not found at $INPUT_CSV. Generating it now..."
    apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -lc ".venv/bin/python -u oracle_kd/prep_origin_data.py --output-csv '$INPUT_CSV'"
fi

echo "Training origin classifier from $INPUT_CSV"

apptainer exec --nv \
        --bind /scratch.hpc/$USER:/scratch.hpc/$USER \
        --env-file ~/.env \
        pytorch_2.4.0-cuda12.1-cudnn9-devel.sif \
        bash -lc ".venv/bin/python -u oracle_kd/train_router.py --input-csv '$INPUT_CSV' --model-name microsoft/deberta-v3-base --output-dir '$OUTPUT_DIR' --epochs 25 --routing-strategy origin"

echo "Origin classifier training complete."
