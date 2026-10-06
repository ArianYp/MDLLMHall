#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition=gpu-l40s-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH -A punim2198 

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

jupyter notebook --no-browser --port=8889 --ip=0.0.0.0

#python eval_hallusionbench.py --generate \
#    --output results/hallusionbench/mmada_predictions.json


python eval_hallusionbench.py \
    --score results/hallusionbench/mmada_predictions.json

my-job-stats -a -n -s