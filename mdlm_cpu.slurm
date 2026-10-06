#!/bin/bash
#SBATCH --job-name=mdlm-cpu
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH -A punim2198

# CPU-only Jupyter for analysis (crop drawing, CSV analysis, AMBER scoring); run commands in it with
#   srun --jobid=<this job> --overlap --ntasks=1 bash -c '...'

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

jupyter notebook --no-browser --port=8890 --ip=0.0.0.0
