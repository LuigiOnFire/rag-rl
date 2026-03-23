#!/bin/bash

echo "Submitting ThriftyRAG Pipeline to Slurm..."

# 1. Ensure logs directory exists so Slurm doesn't crash
mkdir -p logs

# Step 00: Build Corpus
echo "Submitting 00_build_corpus..."
JOB00=$(sbatch --parsable hpc/run_00_build_corpus.sbatch)
echo "Job ID: $JOB00"

# Step 01a: Build Sparse Index
echo "Submitting 01a_sparse_index (Waiting for $JOB00)..."
JOB01A=$(sbatch --parsable --dependency=afterok:$JOB00 hpc/run_01a_sparse_index.sbatch)
echo "Job ID: $JOB01A"

# Step 01b: Build Dense Index
echo "Submitting 01b_dense_index (Waiting for $JOB00)..."
JOB01B=$(sbatch --parsable --dependency=afterok:$JOB00 hpc/run_01b_dense_index.sbatch)
echo "Job ID: $JOB01B"

# Step 01c: Calibrate
echo "Submitting 01c_calibrate (Waiting for indexes $JOB01A and $JOB01B)..."
JOB01C=$(sbatch --parsable --dependency=afterok:$JOB01A:$JOB01B hpc/run_01c_calibrate.sbatch)
echo "Job ID: $JOB01C"

# Step 02: Generate Trajectories
echo "Submitting 02_generate (Waiting for $JOB01C)..."
JOB02=$(sbatch --parsable --dependency=afterok:$JOB01C hpc/run_02_generate.sbatch)
echo "Job ID: $JOB02"

# Step 03: Train SFT
echo "Submitting 03_train_sft (Waiting for $JOB02)..."
JOB03=$(sbatch --parsable --dependency=afterok:$JOB02 hpc/run_03_train_sft.sbatch)
echo "Job ID: $JOB03"

# Step 04: Test Loop
echo "Submitting 04_test_loop (Waiting for $JOB03)..."
JOB04=$(sbatch --parsable --dependency=afterok:$JOB03 hpc/run_04_test_loop.sbatch)
echo "Job ID: $JOB04"

# Step 05: Train GRPO
echo "Submitting 05_train_grpo (Waiting for $JOB04)..."
JOB05=$(sbatch --parsable --dependency=afterok:$JOB04 hpc/run_05_train_grpo.sbatch)
echo "Job ID: $JOB05"

echo "All jobs submitted!"
