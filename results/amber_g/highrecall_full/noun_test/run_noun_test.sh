#!/bin/bash
# Small-set test (AMBER-g ids 1-100) of the noun constraints on the span_r2 settings of run_highrecall_full.sh:
#   --noun-trigger      only act on a flagged word that NLTK tags NN* in context (the scorer's test)
#   --refill-vocab noun the word-first refill must put a noun at the flagged position
# Compared with span_r2 on the same ids (same A100 job, so captions match until the first differing trigger).
# Resumable. Run inside the A100 job:  srun --jobid=<job> --overlap --ntasks=1 bash results/amber_g/highrecall_full/noun_test/run_noun_test.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/highrecall_full/noun_test
echo "=== decoding ($(date))"
python slot_refill_study.py \
    --trigger clip --clip-threshold 0.99 --caption-aware --crop-not-top1 --remove-sinks \
    --refill zoom --refill-order word-first --span-radius 2 \
    --noun-trigger --refill-vocab noun \
    --ids-files $OUT/ids_1_100.json \
    --output $OUT/noun_r2/predictions.json
echo "=== comparing ($(date))"
python $OUT/compare_noun_refill.py 2>&1 | grep -v -E "Warning|similarity\(" | tee $OUT/compare_noun_refill.txt
echo "=== done ($(date))"
