#!/bin/bash
#SBATCH --job-name=jupyter_notebook_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3             # Number of GPU's
#SBATCH --time=02:00:00
#SBATCH --output=notebook_output.log
#SBATCH --export=ALL

# Load Python environment (modify based on your setup)
conda activate base
source activate base  # If using Conda

# Get the notebook name from arguments
NOTEBOOK_NAME=$1  # First command-line argument

# Run Jupyter notebook without displaying
jupyter nbconvert --to notebook --execute $NOTEBOOK_NAME --output "executed_$NOTEBOOK_NAME"

