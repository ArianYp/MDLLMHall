#!/bin/bash
# Oracle trigger + zoom refill (slot_refill_study.py) on the 202 baseline-hallucinating AMBER-g captions
# that the study-set run (hallu_study/rag/slot_refill/oracle_zoom) did not cover. Rerunning resumes.
# Launched inside the Jupyter job: srun --jobid=<job> --overlap --ntasks=1 bash results/amber_g/oracle_zoom_full/run_missing.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/oracle_zoom_full
python slot_refill_study.py \
    --trigger oracle --refill zoom \
    --ids-files $OUT/ids_missing.json \
    --output $OUT/missing/predictions.json
echo "finished exit=$? $(date)"
