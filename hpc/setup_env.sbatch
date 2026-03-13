#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --nodes=1
#SBATCH --partition=l40s           # <-- Targeting the idle NVIDIA nodes!
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out

echo "Starting NVIDIA environment setup on $HOSTNAME"

# 1. Point cache to scratch so we don't blow up your home directory quota
export APPTAINER_CACHEDIR=/scratch.hpc/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch.hpc/$USER/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# 2. Pull the modern PyTorch/CUDA image
echo "Pulling NVIDIA Apptainer image..."
apptainer pull docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 3. Create the virtual environment using the new container
echo "Creating .venv..."
apptainer run --nv pytorch_2.4.0-cuda12.1-cudnn9-devel.sif python -m venv .venv

# 4. Install your Python packages inside the containerized venv
echo "Installing dependencies..."
apptainer run --nv pytorch_2.4.0-cuda12.1-cudnn9-devel.sif .venv/bin/pip install -r requirements.txt

# 5. Clean up the Apptainer temp files
rm -rf $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Setup complete! You are ready to run the pipeline."
