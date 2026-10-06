#!/bin/bash
#SBATCH --job-name=remask-neighbour
#SBATCH --nodes 1
#SBATCH --partition gpu-deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/remask/slurm-neighbour-%j.out

# Neighbour rule (remask_decoding.py --mode neighbour) on the study subsets
# (ids.json + ids_clean.json), then a random control at the rate the rule
# actually fired, then eval_remask.py for all conditions.
# Resubmitting continues from the last saved caption.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/remask
COMMON="--start-step 16 --attention-threshold 0.12567252350541264 --max-triggers 2 --max-extra-steps 16"

python remask_decoding.py --mode neighbour --output $OUT/neighbour.json $COMMON

RATE=$(python -c "
import json
c = json.load(open('$OUT/neighbour_events.json'))['captions']
print(sum(x['n_triggers'] for x in c) / sum(x['n_eligible'] for x in c))")
echo "matched random rate: $RATE"

python remask_decoding.py --mode random --random-rate $RATE --output $OUT/random_neighbour.json $COMMON

python eval_remask.py --conditions \
    rule=$OUT/rule.json random=$OUT/random.json \
    neighbour=$OUT/neighbour.json random_neighbour=$OUT/random_neighbour.json

my-job-stats -a -n -s
