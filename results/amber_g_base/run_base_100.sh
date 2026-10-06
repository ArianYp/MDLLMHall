#!/bin/bash
# MMaDA-8B-Base on AMBER-g ids 1-100, to compare with MixCoT:
#   1) download the Base weights into cache/huggingface (the only online step)
#   2) baseline with the official runner (same schedule as the MixCoT baseline)
#   3) the method exactly as the MixCoT noun test (results/amber_g/highrecall_full/noun_test/run_noun_test.sh):
#      high-recall CLIP trigger (caption-aware rank < 0.99, sink-free crop not top-1), zoom word-first refill,
#      span +-2, sinks removed, --noun-trigger --refill-vocab noun
#   4) compare_models_100.py: official scores and noun counts of both models, baseline and method
# Resumable. Run inside the A100 job:  srun --jobid=<job> --overlap --ntasks=1 bash results/amber_g_base/run_base_100.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

MODEL=Gen-Verse/MMaDA-8B-Base
OUT=results/amber_g_base

echo "=== download $MODEL ($(date))"
HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
export HF_HUB_OFFLINE=1

echo "=== baseline ($(date))"
python benchmarks/amber/run_mmada_amber_g.py --model $MODEL --start-id 1 --end-id 100 --resume \
    --output $OUT/mmada_base_predictions_1-100.json

echo "=== method ($(date))"
python slot_refill_study.py --model $MODEL \
    --trigger clip --clip-threshold 0.99 --caption-aware --crop-not-top1 --remove-sinks \
    --refill zoom --refill-order word-first --span-radius 2 \
    --noun-trigger --refill-vocab noun \
    --ids-files results/amber_g/highrecall_full/noun_test/ids_1_100.json \
    --output $OUT/noun_r2/predictions.json

echo "=== compare ($(date))"
python $OUT/compare_models_100.py 2>&1 | grep -v -E "Warning|similarity\(" | tee $OUT/compare_models_100.txt
echo "=== done ($(date))"
