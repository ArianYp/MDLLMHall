#!/bin/bash
#SBATCH --job-name=slot-cropveto
#SBATCH --nodes 1
#SBATCH --partition gpu-deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/hallu_study/rag/slot_refill/slurm-cropveto-%j.out

# CLIP trigger with a second check on the tight crop (drop the flag if the word's CLIP rank on the crop
# is >= 0.95), zoom refill, on the 144 study captions; then random triggers at the matched rate, and the
# official AMBER-g scorer against oracle_zoom and clip_zoom from slot_refill_study.slurm.
# Resubmitting continues from the last saved caption of each condition.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/hallu_study/rag/slot_refill

python slot_refill_study.py --trigger clip --clip-threshold 0.9 --crop-veto 0.95 --refill zoom \
    --output $OUT/clip_cropveto_zoom/predictions.json
python slot_refill_study.py --trigger random --refill zoom \
    --random-rate-from $OUT/clip_cropveto_zoom/predictions_events.json --output $OUT/random_cropveto_zoom/predictions.json

python eval_remask.py --output-dir $OUT/eval_cropveto --conditions \
    oracle_zoom=$OUT/oracle_zoom/predictions.json clip_zoom=$OUT/clip_zoom/predictions.json \
    clip_cropveto_zoom=$OUT/clip_cropveto_zoom/predictions.json random_cropveto_zoom=$OUT/random_cropveto_zoom/predictions.json

my-job-stats -a -n -s
