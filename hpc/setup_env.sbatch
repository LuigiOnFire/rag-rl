#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --nodes=1
#SBATCH --partition=l40s 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_%j.out

echo "Starting NVIDIA environment setup on $HOSTNAME"

# 1. Paths
export JAVA_DIR=/scratch.hpc/$USER/.local/java
export APPTAINER_CACHEDIR=/scratch.hpc/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch.hpc/$USER/.apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $JAVA_DIR


# # 3. Pull the modern PyTorch/CUDA image
# echo "Pulling NVIDIA Apptainer image..."
# apptainer pull --force docker://pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# # 4. Create the virtual environment using the new container
# echo "Creating .venv..."
# # We pass --env to ensure the container knows where Java is
# apptainer run --nv --env JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-11.0.2 --env PATH=/scratch.hpc/$USER/.local/java/jdk-11.0.2/bin:$PATH pytorch_2.4.0-cuda12.1-cudnn9-devel.sif python -m venv .venv

# 5. Install dependencies
echo "Installing dependencies..."
# Pass the Java env variables here too!
apptainer run --nv --env JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-11.0.2 --env PATH=/scratch.hpc/$USER/.local/java/jdk-11.0.2/bin:$PATH pytorch_2.4.0-cuda12.1-cudnn9-devel.sif .venv/bin/pip install -r requirements.txt
apptainer run --nv --env JAVA_HOME=/scratch.hpc/$USER/.local/java/jdk-11.0.2 --env PATH=/scratch.hpc/$USER/.local/java/jdk-11.0.2/bin:$PATH pytorch_2.4.0-cuda12.1-cudnn9-devel.sif .venv/bin/pip install pyserini

# 6. Clean up
rm -rf $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Setup complete! You are ready to run the pipeline."
