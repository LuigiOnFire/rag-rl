#!/bin/bash

echo "Submitting RLHF Pipeline to Slurm..."

# 1. Ensure logs directory exists so Slurm doesn't crash
mkdir -p logs

# 2. Submit Job 1 and capture its ID
echo "Submitting 01_calibrate..."
JOB1=$(sbatch --parsable jobs/run_01_calibrate.sbatch)
echo "Calibration Job ID: $JOB1"

# 3. Submit Job 2, waiting for Job 1
echo "Submitting 02_trajectory (Waiting for $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/run_02_trajectory.sbatch)
echo "Trajectory Job ID: $JOB2"

# 4. Submit Job 3, waiting for Job 2
echo "Submitting 03_sft (Waiting for $JOB2)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 jobs/run_03_sft.sbatch)
echo "SFT Job ID: $JOB3"

# 5. Submit Job 4, waiting for Job 3
echo "Submitting 04_grpo (Waiting for $JOB3)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 jobs/run_04_grpo.sbatch)
echo "GRPO Job ID: $JOB4"

echo "Pipeline successfully queued! Use 'squeue -u \$USER' to monitor."