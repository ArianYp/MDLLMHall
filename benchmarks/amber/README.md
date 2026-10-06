# AMBER-g for MMaDA

AMBER-g is the generative subset of [AMBER](https://github.com/junyangwang0410/AMBER) (ids 1 through 1004). Each example asks `Describe this image.` The discriminative subset is not part of this pipeline.

Paper: [An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation](https://arxiv.org/abs/2311.07397)

Official image archive: https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing

Captions are produced by the existing `mmada_infer.generate_answer` path (`MMadaModelLM.mmu_generate`). Scores are produced only by the cloned official `inference.py --evaluation_type g`. Metric definitions, prompts, annotations, synonyms, safe words, and the similarity threshold stay in the official repository.

## Metric direction

| Metric | Direction |
| --- | --- |
| CHAIR | lower is better |
| Cover | higher is better |
| Hal | lower is better |
| Cog | lower is better |

## Setup

From `MDLLM/`:

```bash
bash benchmarks/amber/download_amber.sh
```

If the image archive is already downloaded:

```bash
bash benchmarks/amber/download_amber.sh /path/to/amber_images.zip
```

`gdown` is optional. When it is installed, the script above downloads the official Drive archive. When it is not, the script prints the Drive link and exits after preparing the scorer and query file.

Scorer dependencies, matching the official README plus the NLTK resources `inference.py` loads:

```bash
bash benchmarks/amber/setup_scorer_deps.sh
```

That runs:

```bash
pip install -U spacy nltk tqdm
python -m spacy download en_core_web_lg
python benchmarks/amber/setup_scorer_deps.py
```

On this project’s Spartan environment, load the same Python module used by `mdlm.slurm` before activating `virtualenv/`:

```bash
module load GCCcore/11.3.0 Python/3.11.3
source virtualenv/bin/activate
```

## Smoke test (5 images, no aggregate score)

```bash
bash benchmarks/amber/run_amber_g_mmada.sh \
  --start-id 1 \
  --end-id 5 \
  --image-dir data/amber/images \
  --output results/amber_g/mmada_predictions_smoke.json
```

This checks JSON format for ids 1..5. It does not call the official scorer, because a 5-example score is not comparable to full AMBER-g.

## Full AMBER-g

```bash
bash benchmarks/amber/run_amber_g_mmada.sh \
  --config third_party/MMaDA/configs/mmada_demo.yaml \
  --image-dir data/amber/images \
  --output results/amber_g/mmada_predictions.json
```

`--config` is optional and only supplies the MMaDA and VQ checkpoint paths. Generation defaults follow the reported MMaDA AMBER setting: semi-autoregressive decoding, 128 new tokens, block size 32, and 64 denoising steps. `mmu_generate` receives 64 as the total step budget and divides it across the four blocks. Temperature is 0, cfg scale is 0, remasking is `low_confidence`, seed is 0, and resolution is 512. Override them with `--max-new-tokens`, `--steps`, `--block-length`, `--temperature`, `--cfg-scale`, `--remasking`, `--resolution`, and `--seed`.

A checkpoint already on disk is `--model` or `--checkpoint`. The default checkpoint is `Gen-Verse/MMaDA-8B-MixCoT` and the default VQ model is `showlab/magvitv2`.

Resume a partial file with `--resume`. Replace it with `--overwrite`.

## Standalone official scoring

```bash
bash benchmarks/amber/evaluate_amber_g.sh \
  --predictions results/amber_g/mmada_predictions.json \
  --amber-repo third_party/AMBER
```

That is equivalent to:

```bash
cd third_party/AMBER
python inference.py \
  --inference_data /absolute/path/to/mmada_predictions.json \
  --evaluation_type g
```

## Outputs

- Predictions: `results/amber_g/mmada_predictions.json`
- Provenance, separate from the official file: `results/amber_g/amber_g_metadata.json`
- Official scorer stdout, including CHAIR, Cover, Hal, and Cog: `results/amber_g/amber_g_metrics.txt`

The prediction file is only:

```json
[
  {"id": 1, "response": "..."},
  {"id": 2, "response": "..."}
]
```
