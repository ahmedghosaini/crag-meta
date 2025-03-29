#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --job-name=evaluation             # A nice readable name of your job, to see it in the queue
#SBATCH --nodes=1                         # Number of nodes to request
#SBATCH --cpus-per-task=8                 # Number of CPUs to request
#SBATCH --gpus=2                         # Number of GPUs to request


module load mamba

# Define your custom variable
MY_VAR="md"

# Activate your environment, you have to create it first
mamba activate default

# Your job script goes below this line
python local_evaluation.py
