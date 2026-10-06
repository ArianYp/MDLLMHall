#!/bin/bash
# Next-steps item 2: degrade the oracle+zoom detector on the 144 study captions to see whether recall or precision matters more.
#   half_recall   oracle keeps each flag with p = 0.5                                      (recall ~0.5, precision 1)
#   prec40        oracle + random flags on non-hallucinated eligible words at 0.26          (recall 1, precision ~0.4)
#   half_prec40   keep 0.5 + false rate 0.13, roughly CLIP's full-set operating point       (recall ~0.5, precision ~0.4)
# Rates from oracle_zoom/predictions_events.json: 160 hallucinated of 1085 eligible (925 not).
# Then the official scorer against baseline, oracle_zoom, clip_cropveto_zoom and random_cropveto_zoom.
# Run inside the GPU job: srun --jobid=<job> --overlap --ntasks=1 bash degraded_oracle.sh. Rerunning resumes.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/hallu_study/rag/slot_refill

python slot_refill_study.py --trigger oracle --refill zoom --oracle-keep 0.5 \
    --output $OUT/oracle_half_recall_zoom/predictions.json
python slot_refill_study.py --trigger oracle --refill zoom --oracle-false-rate 0.26 \
    --output $OUT/oracle_prec40_zoom/predictions.json
python slot_refill_study.py --trigger oracle --refill zoom --oracle-keep 0.5 --oracle-false-rate 0.13 \
    --output $OUT/oracle_half_prec40_zoom/predictions.json

python eval_remask.py --output-dir $OUT/eval_degraded --conditions \
    oracle_zoom=$OUT/oracle_zoom/predictions.json \
    oracle_half_recall_zoom=$OUT/oracle_half_recall_zoom/predictions.json \
    oracle_prec40_zoom=$OUT/oracle_prec40_zoom/predictions.json \
    oracle_half_prec40_zoom=$OUT/oracle_half_prec40_zoom/predictions.json \
    clip_cropveto_zoom=$OUT/clip_cropveto_zoom/predictions.json \
    random_cropveto_zoom=$OUT/random_cropveto_zoom/predictions.json
