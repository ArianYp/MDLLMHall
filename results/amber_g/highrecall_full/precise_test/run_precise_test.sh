#!/bin/bash
# Small-set test (AMBER-g ids 1-100): a lower-false-positive CLIP detector with the span_r2 refill of run_highrecall_full.sh
# (zoom, word-first, sinks removed, span +-2; no noun options). Detector = caption-aware full-image rank < T1 AND
# caption-aware rank on the sink-free tight crop < T2 (--crop-veto drops the flag if crop rank >= T2).
# Offline study-set points (caption_aware_veto.txt): A 0.97/0.97 -> 103/155 caught, 103/895 FP; B 0.90/0.95 -> 70 / 32;
# current high-recall (0.99, crop not top-1) -> 142 / 264.
# Configs: name, T1, T2, span radius. A (span 2) done 2026-10-06; B was stopped at 2/100 to make room for A3 (span 3).
# Compared with span_r2 on the same ids (same A100 job). Resumable. Run inside the A100 job:
#   srun --jobid=<job> --overlap --ntasks=1 bash results/amber_g/highrecall_full/precise_test/run_precise_test.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/highrecall_full/precise_test
IDS=results/amber_g/highrecall_full/noun_test/ids_1_100.json
for cfg in "A3 0.97 0.97 3"; do
    set -- $cfg
    echo "=== $1: full < $2 AND crop < $3, span ±$4, decoding ($(date))"
    mkdir -p $OUT/$1
    python slot_refill_study.py \
        --trigger clip --clip-threshold $2 --caption-aware --crop-veto $3 --remove-sinks \
        --refill zoom --refill-order word-first --span-radius $4 \
        --ids-files $IDS \
        --output $OUT/$1/predictions.json
done
echo "=== done ($(date))"
