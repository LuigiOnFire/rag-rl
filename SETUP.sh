# !/bin/bash
# script for setting up the environment on a new machine
apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif python -m venv .venv
apptainer run --nv pytorch_2.3.1-cuda11.8-cudnn8-runtime.sif .venv/bin/pip install -r requirements.txt