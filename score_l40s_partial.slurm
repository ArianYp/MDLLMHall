#!/bin/bash
#SBATCH --job-name=score-l40s
#SBATCH --nodes 1
#SBATCH --partition sapphire,cascade
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -A punim2198
#SBATCH --output=results/amber_g/hallu_study/rag/slot_refill/eval_l40s_partial/slurm-%j.out

# CPU only: official AMBER-g scores + paired counts for the oracle runs on the study captions.
# oracle_zoom_l40s was preempted at 88/144 (84 hallucinating + 4 clean), and eval_remask.py keeps only ids
# present in every condition, so this scores all 84 hallucinating captions but only 4 clean ones.

set -eo pipefail
cd /data/gpfs/projects/punim2198/Aryan/MDLLM
module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate

OUT=results/amber_g/hallu_study/rag/slot_refill
python eval_remask.py --output-dir $OUT/eval_l40s_partial --conditions \
    oracle_plain=$OUT/oracle_plain/predictions.json \
    oracle_zoom=$OUT/oracle_zoom/predictions.json \
    oracle_zoom_l40s=$OUT/oracle_zoom_l40s/predictions.json \
    oracle_zoom_nosink=$OUT/oracle_zoom_nosink/predictions.json
