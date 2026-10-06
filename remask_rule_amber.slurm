#!/bin/bash
#SBATCH --job-name=remask-amber
#SBATCH --nodes 1
#SBATCH --partition gpu-deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/remask_full/slurm-%j.out

# MMaDA on the full AMBER-g set (ids 1-1004) with image-contrast remasking
# (remask_decoding.py --mode rule), then the official AMBER-g scorer.
# Resubmitting continues from the last saved caption.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/remask_full
PRED=$OUT/mmada_rule_predictions.json

python remask_decoding.py \
    --mode rule \
    --ids-files $OUT/ids_all.json \
    --output $PRED \
    --start-step 15 \
    --attention-threshold 0.12567252350541264 \
    --max-triggers 2 \
    --max-extra-steps 16

bash benchmarks/amber/evaluate_amber_g.sh \
    --predictions $PRED \
    --metrics-out $OUT/amber_g_metrics.txt

my-job-stats -a -n -s
