#!/bin/bash
#SBATCH --job-name=slot-refill
#SBATCH --nodes 1
#SBATCH --partition gpu-deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/hallu_study/rag/slot_refill/slurm-%j.out

# slot_refill_study.py on the 144 study captions (84 hallucinating + 60 clean):
# triggers oracle / clip / random (matched to clip's rate) x refills plain / zoom,
# then the official AMBER-g scorer per subset (eval_remask.py).
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

for refill in plain zoom; do
    python slot_refill_study.py --trigger oracle --refill $refill --output $OUT/oracle_$refill/predictions.json
    python slot_refill_study.py --trigger clip --refill $refill --clip-threshold 0.9 --output $OUT/clip_$refill/predictions.json
done
for refill in plain zoom; do
    python slot_refill_study.py --trigger random --refill $refill \
        --random-rate-from $OUT/clip_$refill/predictions_events.json --output $OUT/random_$refill/predictions.json
done

python eval_remask.py --output-dir $OUT/eval --conditions \
    oracle_plain=$OUT/oracle_plain/predictions.json oracle_zoom=$OUT/oracle_zoom/predictions.json \
    clip_plain=$OUT/clip_plain/predictions.json clip_zoom=$OUT/clip_zoom/predictions.json \
    random_plain=$OUT/random_plain/predictions.json random_zoom=$OUT/random_zoom/predictions.json

my-job-stats -a -n -s
