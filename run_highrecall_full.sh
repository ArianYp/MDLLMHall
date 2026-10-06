#!/bin/bash
# High-recall CLIP trigger on the full AMBER-g set (ids 1-1004), then the official scorer, for span radius 1 and 2:
#   flag if the caption-aware full-image CLIP rank < 0.99 (synonyms and already-committed caption objects removed from the vocabulary),
#   keep the flag unless the word is CLIP's top remaining word on its sink-free tight crop,
#   refill = zoom crop (sinks removed), word-first, remask pos-r .. pos+r.
# Study-set operating point (offline, caption_aware_veto.txt): 92% of hallucinated nouns flagged, 28% of grounded ones.
# Resumable: rerunning continues from the last saved caption of each run.
# Run inside the A100 job:  srun --jobid=<job> --overlap --ntasks=1 bash run_highrecall_full.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/highrecall_full
for R in 1 2; do
    PRED=$OUT/span_r$R/predictions.json
    echo "=== span radius $R: decoding ($(date))"
    python slot_refill_study.py \
        --trigger clip --clip-threshold 0.99 --caption-aware --crop-not-top1 --remove-sinks \
        --refill zoom --refill-order word-first --span-radius $R \
        --ids-files results/amber_g/remask_full/ids_all.json \
        --output $PRED
    echo "=== span radius $R: scoring ($(date))"
    bash benchmarks/amber/evaluate_amber_g.sh --predictions $PRED --metrics-out $OUT/span_r$R/amber_g_metrics.txt
    cat $OUT/span_r$R/amber_g_metrics.txt
done
echo "=== done ($(date))"
