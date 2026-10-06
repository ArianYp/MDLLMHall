#!/bin/bash
# Next-steps item 4: matched-rate random+zoom control on all 1004 AMBER-g captions for clip+cropveto+zoom
# (rate = fired / eligible of its full run, 733 / 6892 = 0.106), then the official scorer and paired counts.
# Run inside the GPU job: srun --jobid=<job> --overlap --ntasks=1 bash random_zoom_full.sh. Rerunning resumes.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/slot_refill_full
PRED=$OUT/random_zoom/predictions.json

python slot_refill_study.py --trigger random --refill zoom \
    --random-rate-from $OUT/clip_cropveto_zoom/predictions_events.json \
    --ids-files results/amber_g/remask_full/ids_all.json \
    --output $PRED

bash benchmarks/amber/evaluate_amber_g.sh --predictions $PRED --metrics-out $OUT/random_zoom/amber_g_metrics.txt
python $OUT/score_partial.py $OUT/random_zoom final_eval
