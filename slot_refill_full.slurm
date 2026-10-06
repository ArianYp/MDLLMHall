#!/bin/bash
#SBATCH --job-name=cropveto-full
#SBATCH --nodes 1
#SBATCH --partition gpu-deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/slot_refill_full/slurm-%j.out

# CLIP trigger (rank_full < 0.9) + tight-crop check (veto if crop rank >= 0.95) + zoom refill
# (slot_refill_study.py) on the full AMBER-g set (ids 1-1004), then the official AMBER-g scorer.
# Resubmitting continues from the last saved caption.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/slot_refill_full
PRED=$OUT/clip_cropveto_zoom/predictions.json

python slot_refill_study.py \
    --trigger clip --clip-threshold 0.9 --crop-veto 0.95 --refill zoom \
    --ids-files results/amber_g/remask_full/ids_all.json \
    --output $PRED

bash benchmarks/amber/evaluate_amber_g.sh \
    --predictions $PRED \
    --metrics-out $OUT/clip_cropveto_zoom/amber_g_metrics.txt

my-job-stats -a -n -s
