#!/bin/bash
# Full AMBER-g (ids 1-1004), MixCoT: span_r2 settings of run_highrecall_full.sh plus the noun constraints of
# noun_test/run_noun_test.sh (--noun-trigger --refill-vocab noun). Same detector as span_r1 / span_r2.
# Seeded with the 100 noun_test captions (same A100 job, same settings), then resumed. Afterwards: official scorer,
# paired counts vs the baseline (score_partial.py), noun diffs (analyze_span.py), false-flag outcomes.
# Resumable. Run inside the A100 job:  srun --jobid=<job> --overlap --ntasks=1 bash results/amber_g/highrecall_full/run_noun_full.sh

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg
export HF_HUB_OFFLINE=1

module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/highrecall_full/span_r2_noun
PRED=$OUT/predictions.json
mkdir -p $OUT
if [ ! -f $PRED ]; then
    cp results/amber_g/highrecall_full/noun_test/noun_r2/predictions.json $PRED
    cp results/amber_g/highrecall_full/noun_test/noun_r2/predictions_events.json $OUT/predictions_events.json
fi
echo "=== decoding ($(date))"
python slot_refill_study.py \
    --trigger clip --clip-threshold 0.99 --caption-aware --crop-not-top1 --remove-sinks \
    --refill zoom --refill-order word-first --span-radius 2 \
    --noun-trigger --refill-vocab noun \
    --ids-files results/amber_g/remask_full/ids_all.json \
    --output $PRED
echo "=== scoring ($(date))"
bash benchmarks/amber/evaluate_amber_g.sh --predictions $PRED --metrics-out $OUT/amber_g_metrics.txt
cat $OUT/amber_g_metrics.txt
F="Warning|similarity\("
echo "=== paired counts ($(date))"
python results/amber_g/slot_refill_full/score_partial.py $OUT final_eval 2>&1 | grep -v -E "$F"
python results/amber_g/highrecall_full/analyze_span.py $OUT 2>&1 | grep -v -E "$F" > /dev/null
echo "=== false-flag outcomes ($(date))"
python results/amber_g/highrecall_full/grounded_flag_outcomes.py $OUT 2>&1 | grep -v -E "$F" | tee $OUT/grounded_flag_outcomes.txt
echo "=== done ($(date))"
