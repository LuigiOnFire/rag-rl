#!/bin/bash
#SBATCH --job-name=setup_rag
#SBATCH --nodes=1
#SBATCH --partition=mi210
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm.%N.%j.out

echo "Starting AMD ROCm setup on $HOSTNAME"

# Point cache to the massive 120TB drive
export APPTAINER_CACHEDIR=/scratch.hpc/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch.hpc/$USER/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Pulling ROCm Apptainer image..."
apptainer pull docker://rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4

echo "Creating .venv..."
apptainer run --rocm pytorch_rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.sif python -m venv .venv

echo "Installing dependencies..."
apptainer run --rocm pytorch_rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.sif .venv/bin/pip install -r requirements.txt

# Clean up the cache so you don't clutter the shared drive
rm -rf $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Setup complete!"
