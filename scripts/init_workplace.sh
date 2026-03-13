#!/bin/bash
echo "🚀 Initializing RAG-RL Workspace for NVIDIA..."

# 1. Create necessary directories
echo "📁 Setting up directories..."
mkdir -p logs ollama_models data

# 2. Download Ollama if it doesn't exist
if [ ! -f "bin/ollama" ]; then
    echo "🦙 Downloading Ollama binary for Linux..."
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
    echo "✅ Secure ~/.env file already exists."
fi

# 4. Submit the heavy lifting to Slurm!
echo "🐳 Submitting Apptainer and Python setup job to Slurm..."
sbatch hpc/setup_env.sbatch

echo "🎉 Workspace initialization triggered! Use 'squeue -u $USER' to monitor the setup."
