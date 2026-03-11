#!/bin/bash

echo "🚀 Initializing RAG-RL Workspace on AMD HPC..."

# 1. Create necessary directories
echo "📁 Setting up directories..."
mkdir -p logs
mkdir -p ollama_models

# 2. Download Ollama if it doesn't exist
if [ ! -f "ollama" ]; then
    echo "🦙 Downloading Ollama binary for AMD Linux..."
    curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst
    tar -xf ollama.tar.zst
    rm ollama.tar.zst
else
    echo "✅ Ollama binary already exists."
fi

# 3. Scaffold the secure .env vault in the user's home directory
if [ ! -f ~/.env ]; then
    echo "🔐 Creating secure ~/.env template..."
    echo "# RAG-RL Configuration" > ~/.env
    echo "LLM_MODEL=llama3:8b" >> ~/.env
    echo "SLM_MODEL=hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest" >> ~/.env
    echo "HF_TOKEN=put_your_token_here" >> ~/.env
    chmod 600 ~/.env
    echo "⚠️  ACTION REQUIRED: Please run 'nano ~/.env' to add your HF_TOKEN!"
else
    echo "✅ Secure ~/.env file already exists in home directory."
fi

# 4. Submit the container build job
if [ ! -f "pytorch_rocm6.1_ubuntu22.04_py3.10_pytorch_2.4.sif" ]; then
    echo "🐳 Submitting Apptainer setup job to Slurm..."
    sbatch setup_amd.sbatch
    echo "⏳ Please wait for the setup job to finish before running the pipeline."
else
    echo "✅ Apptainer container already exists."
fi

echo "🎉 Workspace initialization triggered! Use 'squeue -u \$USER' to monitor."
