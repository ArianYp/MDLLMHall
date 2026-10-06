# MDLLM hallucination project — notes

Last updated 2026-10-06 15:15. Claude Code transcripts are disabled on this cluster
(managed setting `CLAUDE_CODE_SKIP_PROMPT_HISTORY=1`), so this file is the record.

## Currently running (check first)

Nothing. The high-recall full-set runs (span ±1, ±2) finished at 03:18 on 2026-10-06, and the noun-constrained follow-up was
cancelled at 746/1004 on 2026-10-06 15:09 because it was worse (results in "Full-set run: high-recall CLIP trigger" below).

- 2026-10-07 10:00 (sacct): no jobs running or queued. 32261020 (A100) and 32354148 (CPU) hit their time limits on 2026-10-06 at 16:48 and 22:09.
  32342851 and 32343837 started (A100) but only ran an idle Jupyter server, and were cancelled at 22:25. 32342648 and 32372943 were cancelled before starting.
  The detector A full run below has **not** been started.
- 2026-10-06 15:37–16:18: lower-false-positive detector A tested on ids 1–100 (span ±2 and ±3); it keeps most of the gain at almost
  no Cover cost. See "Lower-false-positive detector on ids 1–100" below.
- Next: detector A + span ±2 on all 1004, with a matched-rate random+zoom control (fire rate ≈ 17%), on an A100 (job 32261020 ends 16:47 today).

## Setup

- Model `Gen-Verse/MMaDA-8B-MixCoT`, VQ `showlab/magvitv2`, 512 px (1024 image tokens).
- Greedy, 128 new tokens, 64 steps, block length 32 → 4 semi-autoregressive blocks × 16 steps,
  2 low-confidence commits per step (`MMadaModelLM.mmu_generate`).
- Benchmark: AMBER generative (AMBER-g, 1004 images), official `third_party/AMBER/inference.py`.
- Baseline, full AMBER-g (`results/amber_g/amber_g_metrics.txt`): **CHAIR 7.5, Cover 48.4, Hal 28.4, Cog 2.3**.

## Code map

| file | what it does |
|---|---|
| `mmada_infer.py`, `run_mmada.py` | load MMaDA, build image+prompt ids, baseline generation |
| `trace_mmada_steps.py`, `trace_amber_steps.py` | per-step commit traces |
| `trace_amber_image_ablation.py` | p(token \| image) vs p(token \| no image) on the same pre-commit state |
| `trace_amber_image_attention.py` | `AttentionRecorder`: per-layer attention mass / entropy / map over image tokens |
| `trace_amber_batch.py` | all of the above for a list of ids → `hallu_study/traces*/<id>.json` |
| `amber_noun_labels.py` | AMBER-g per-noun labels (hallucinated / grounded / ignored) |
| `analyze_hallu_study.py` | hallucinated vs grounded noun features, AUROCs, grouped-CV logistic regression |
| `remask_decoding.py` | the method (`--mode rule`), the `random` control, and `baseline` |
| `eval_remask.py` | official scores + per-noun counts for remask runs on the study subsets |
| `remask_rule_amber.slurm` | rule run on all 1004 ids, then the official scorer |
| `rag_retrieve.py` | attended-region crops of one caption's nouns → CLIP top-k from COCO train (`hallu_study/rag/<id>/`) |
| `rag_mmada_context.py` | retrieved captions as MMaDA prompt context: full regeneration and single-noun slot re-prediction |
| `rag_slot_decoding.py` | during decoding: on a watched word's commit, remask it ± neighbours and refill with its tight-region retrieved captions in the prompt |
| `slot_refill_study.py` | on a committed object word flagged by oracle / CLIP / random, remask ± neighbours and refill (plain or zoom) |
| `rag_clip_relation.py` | CLIP noun-text vs full image / crops / retrieved images, AUROCs over the study nouns |

## The method (`remask_decoding.py --mode rule`)

From global step 15 on, for every token at the step it is committed (EOT excluded):
fire if **p(token | no image) > p(token | image)** on the same pre-commit state **and**
the mean late-layer (19–31) attention mass on the image ≥ 0.1257 (median over
content tokens, `hallu_study/attention_threshold.json`). On a fire, the token and its
neighbours (pos−1, pos+1) are masked again. ≤2 triggers per position, and ≤16 extra steps per block
at the normal commit rate.

## Analysis findings (`results/amber_g/hallu_study/`)

Study set: 84 captions that hallucinate at baseline (`ids.json`, drawn from the first 729 ids)
plus 20 clean captions (`ids_clean_20.json`). There are 796 object nouns, 155 of them hallucinated. Each
feature is taken at the commit of the noun's first token. The numbers below are from `with_clean_20/summary.json`.

**Clean set expanded to 60 (2026-10-02).** I re-ran `select_clean.py 60` (seed 0, pool 717). The result is a superset of the old 20, and none of the ids overlap `ids_test.json`.
The 40 new ids were traced in the Jupyter job 32074045 via `srun --overlap` (`batch_clean_60.log`), and every trace reproduced the saved baseline caption.
`with_clean/` is now 84 + 60 = 144 captions with 1050 nouns, 155 of them hallucinated (895 grounded, 381 of those in clean captions). The 20-caption version is kept in `with_clean_20/`.
Combined CV AUROC: decoding order 0.773, all 0.777, attention 0.595, confidence + image removal 0.581, confidence only 0.551, so the conclusions are unchanged.
The remask tables below still use the old 20 clean captions.
AUROC > 0.5 means the feature is higher for hallucinated nouns. The 95% CIs are bootstrapped over captions.

| feature | AUROC | 95% CI | hallucinated mean | grounded mean |
|---|---|---|---|---|
| step (commit step) | **0.662** | 0.62–0.71 | 22.9 | 16.8 |
| relative_position | **0.660** | 0.62–0.70 | 0.58 | 0.42 |
| context_before (±3 neighbours already committed) | **0.630** | 0.58–0.68 | 0.54 | 0.43 |
| late_top5 (top-5 patch share, L19–31) | **0.623** | 0.55–0.69 | 0.142 | 0.107 |
| repeat_mention | **0.352** | 0.31–0.40 | 0.20 | 0.50 |
| entropy_image | 0.572 | 0.52–0.62 | 1.87 | 1.63 |
| p_image | 0.416 | 0.37–0.47 | 0.55 | 0.62 |
| dlogp = log p(img) − log p(no img) | 0.445 | 0.39–0.50 | +0.84 | +1.29 |
| p_no_image | 0.509 | 0.46–0.56 | 0.43 | 0.43 |
| kl(img ‖ no img) | 0.492 | 0.44–0.54 | 0.83 | 1.02 |
| late_image_mass | 0.458 | 0.41–0.50 | 0.153 | 0.161 |

Combined (logistic regression, 5-fold CV grouped by caption): decoding order **0.775**, all features 0.768,
attention 0.601, confidence + image removal 0.566, confidence only 0.554.

What this says:

1. **Late commitment.** Hallucinated nouns are committed later. The hallucination rate is 0.11 for nouns committed at steps 1–16,
   0.29 at steps 17–32 and 0.27 at steps 33–48. **This is mostly position in the caption, though.** In semi-AR decoding,
   later blocks are both later steps and later text. From `position_control.py`:
   position alone 0.646, position + repeat 0.764, + context_before 0.775. Step-within-block alone gives 0.570,
   and adding it to position + repeat gives 0.760, so it adds nothing.
2. **There is a real "fill-in" effect beyond position.** `context_before` stays predictive within position
   thirds (0.68 / 0.55 / 0.63). A noun committed after its neighbours, i.e. filled in to fit the
   surrounding text, is more likely hallucinated.
3. **First mentions are the risk.** A noun whose lemma was already mentioned is rarely hallucinated (AUROC 0.35).
4. **Removing the image barely separates the groups.** Hallucinated nouns still gain from the image on average
   (dlogp +0.84), and p_no_image is identical in both groups. Only 36% of hallucinated nouns have dlogp < 0.
   Caption 163 (`traces/image_ablation_amber_163_sentence_order.md`) shows this:
   eggs +4.72, bowl +2.80, apples +1.20 are all *raised* by the image. These hallucinations are not the
   language prior overriding the image. They are image-driven but wrong.
5. **Attention:** total image mass gives no signal, and hallucinated nouns are slightly *lower*. Concentration does help:
   hallucinated nouns put more of their attention on a few patches (top-5 share, 0.62–0.64).

### Commit confidence, hallucinated vs grounded (2026-10-02, `with_clean/nouns.csv`, 144 captions: 84 hallucinating + 60 clean)

`p_image` = p(committed token | image) at the commit step of the noun's first token, which is the decoder's own confidence.

| | n | mean | median | p10 | p25 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| hallucinated | 155 | 0.547 | 0.524 | 0.275 | 0.388 | 0.718 | 0.855 |
| grounded | 895 | 0.611 | 0.595 | 0.311 | 0.426 | 0.798 | 0.953 |
| grounded, clean captions only | 381 | 0.605 | 0.593 | 0.308 | 0.436 | 0.787 | 0.952 |

The AUROC for "low confidence ⇒ hallucinated" is 0.577 (0.584 with 20 clean captions). The hallucination rate by bin is 0.21 for < 0.3, 0.16 for [0.3, 0.7), 0.14 for [0.7, 0.9) and 0.07 for ≥ 0.9.
The distributions overlap almost completely, and only very high confidence is informative (≥ 0.9: 6.5% of hallucinated vs 15.3% of grounded nouns).
For late nouns (step ≥ 16, where the zoom gate acts) there is essentially no gap: mean 0.552 vs 0.581, median 0.554 vs 0.555.
For first mentions it is larger: 0.530 vs 0.612. Grounded nouns have the same confidence in hallucinating and clean captions (0.616 vs 0.605).

### Testing the language-prior assumption (dlogp < 0 ⇒ not looking at the image ⇒ hallucination)

From `with_clean/nouns.csv` (796 nouns):

- **Part 1 holds.** dlogp and late-layer image mass are correlated (Spearman +0.50). Nouns with dlogp < 0 have mean mass 0.137, versus 0.168 for nouns with dlogp > 0.
- **Part 2 is weak.** Nouns with dlogp < 0 are hallucinated at a rate of 0.24, versus 0.18 for dlogp > 0. 64% of hallucinated nouns have dlogp > 0 (image-driven).
- Almost all dlogp < 0 nouns sit in [−1, 0), so the no-image prediction wins only narrowly.
- **The attention gate in the rule contradicts the assumption.** Language-prior tokens should have *low* mass, but the rule requires high mass.
  Among dlogp < 0 nouns the hallucination rate is 0.24 at low and at high mass. The safest cell is dlogp > 0 with high mass (0.16).

## Remask results

On the study subsets (`results/amber_g/remask/eval_report.json`). The random control remasks each eligible token with
p = 0.1374, which matches the rule's trigger rate (780 / 5676 eligible tokens).

| subset / condition | CHAIR | Cover | Hal | Cog | halluc. nouns |
|---|---|---|---|---|---|
| hallucinating (84) baseline | 22.3 | 43.5 | 98.8 | 21.3 | 155 |
| hallucinating rule | 18.3 | 43.5 | 76.2 | 14.3 | 121 |
| hallucinating random | 17.6 | 44.8 | 79.8 | 16.2 | 119 |
| clean (20) baseline | 0.0 | 50.5 | 0.0 | 0.0 | 0 |
| clean rule | 0.8 | 51.5 | 5.0 | 1.0 | 1 |
| clean random | 1.5 | 52.4 | 10.0 | 1.0 | 2 |

**The rule is not better than random remasking at the same rate.** Most of the gain on the hallucinating
subset is likely regression to the mean. Those captions were selected *because* they hallucinated, so
any perturbation of a greedy trajectory tends to help.

Why the rule underperforms (noun level, baseline traces):

- It fires on 14% of hallucinated nouns versus 4–5% of grounded ones. Precision is 0.41 against a 0.19 base rate.
  So it has some signal, but recall is low.
- 71 of the 117 hallucinated nouns committed at step ≥ 15 have dlogp ≥ 0, so the rule can never fire on them.
- The attention-mass gate points the wrong way: hallucinated nouns have slightly lower mass.
- It acts on every token, including function words, so most triggers are not on object nouns.

Full run: `remask_full/` (job 31876377) finished at 22:07 on all 1004 captions (`remask_full/amber_g_metrics.txt`):
rule **CHAIR 7.3, Cover 48.4, Hal 28.7, Cog 2.1** vs baseline 7.5 / 48.4 / 28.4 / 2.3. So it is a wash, as the 872-caption snapshot below showed.

### Neighbour rule (`--mode neighbour`, 2026-10-01, run inside the Jupyter job 31927297)

From step 16 on, a committed token is remasked with pos−1 and pos+1 when **both neighbours were committed at an
earlier step** (it is filled in between them) **and** its late-layer image mass is ≥ 0.1257. There is no image ablation.
Offline on baseline traces: fires on 3.8% of tokens, noun precision 0.53 (old rule 0.41), recall 13%.
Script: `remask_neighbour_subset.slurm`. Log: `remask/neighbour_run.log`. Outputs: `remask/neighbour.json`, `remask/random_neighbour.json`.
In the real run it fired at 5.5% (276 / 5008). The random control was matched to that (actual rate 5.7%).

| condition (official scorer) | hallucinating (84) CHAIR / Hal / Cog | halluc. nouns | grounded | clean (20) CHAIR / Hal | clean halluc. / grounded |
|---|---|---|---|---|---|
| baseline | 22.3 / 98.8 / 21.3 | 155 | 521 | 0.0 / 0.0 | 0 / 120 |
| rule (13.7%) | 18.3 / 76.2 / 14.3 | 121 | 526 | 0.8 / 5.0 | 1 / 118 |
| random (13.7%) | 17.6 / 79.8 / 16.2 | 119 | 541 | 1.5 / 10.0 | 2 / 123 |
| neighbour (5.5%) | 20.1 / 84.5 / 17.2 | 137 | 530 | 1.6 / 5.0 | 2 / 119 |
| random (5.7%) | 20.6 / 83.3 / 16.7 | 140 | 523 | 0.0 / 0.0 | 0 / 112 |

**The neighbour rule is no better than random at the same rate (137 vs 140 hallucinated nouns).** Random remasking at
5.7% removes about as much as at 13.7%. On captions selected for hallucinating, almost any perturbation helps, so this
subset cannot separate good rules from bad. Future rules should be tested on unselected held-out ids (730–1004) with a
matched random control, and detection (offline noun AUROC) kept separate from correction (an oracle remask of the
labelled hallucinated nouns, to bound what remasking can fix).
### Rule on ids 1–872 (partial full run, scored 2026-10-01)

Snapshot of `remask_full/mmada_rule_predictions.json` at 872 captions, with the baseline restricted to the same ids.
Both were scored with the official `third_party/AMBER/inference.py --evaluation_type g`. Files are in `remask_full/partial_eval/`.

| ids 1–872 | CHAIR | Cover | Hal | Cog |
|---|---|---|---|---|
| baseline | 7.4 | 47.9 | 28.2 | 2.3 |
| rule | 7.2 | 47.8 | 28.6 | 2.0 |

Paired counts from `AmberLabeler`:

- Text changed in 595/872 captions. Hallucinated nouns went 448 → 435 and grounded nouns 5378 → 5355.
- Per caption: 69 have fewer hallucinations, 732 the same, 71 more.
- On the 246 captions that hallucinate at baseline, hallucinated nouns went 448 → 390. On the 626 clean captions they went 0 → 45, and 41 of those captions now hallucinate.

**On the full set the rule is a wash.** The gain on hallucinating captions is cancelled by new
hallucinations in clean ones, which is the regression-to-the-mean pattern the random control on the study subset suggested.

A full-set random control was started and then stopped at ~11 captions (`remask_full/random_full.log`).
Its partial output, `remask_full/mmada_random_predictions.json`, should be ignored or deleted.

### Caption 163 probes (2026-10-02, finished baseline caption, run in the Jupyter job)

The image shows a woman holding a **lime** to her face, broccoli on the left, and a plate with a **half melon** and a lime on the right.
AMBER truth is wall, table, broccoli, lemon, melon, plate, person. The baseline hallucinates cup, drink, bowl, eggs and apples.
The scripts are in `/tmp` on the login node (`ayazdanparas_{probe,patch,isolate,zoom}163*.py`). The figures are in `hallu_study/traces/`:
`patch_ablation_163.png`, `isolate_163.png`, `zoom_163.png`.

- **Text remask (mask the noun in the finished caption, image present):** the model puts every hallucinated noun back.
  Masking all 5 at once returns the identical caption. Banning the word swaps in another wrong one (mug, soup, olives).
  Grounded nouns are re-predicted at 0.98–0.998 and hallucinated ones at 0.29–0.93, which may be a detector.
- **Attention:** eggs and bowl attend to the half melon, apples to the limes, and cup/drink to the hand and the lime at her mouth.
  The hallucinations are misreadings of real objects, not inventions.
- **Patch ablation (remove the top-k attended patches):** eggs falls 0.69 → 0.01 at k=5 and bowl 0.29 → 0.007 at k=20.
  Broccoli (grounded) falls 0.998 → 0.09, so these hallucinations depend on the image as causally as a grounded noun.
  Cup does not move (0.93), so it comes from text and pose. Random patches change nothing.
- **Isolation (keep only the attended patches, mean fill):** apples rises (0.94 at k=100) and bowl survives (≈0.2–0.5).
  Eggs dies (→ soup 0.78), so eggs needs the melon *and* the scene. Melon and lime stay at 0.000 everywhere.
- **Zoom (crop the attended cluster from the 5685×3790 original, upscale to 512):** at the melon, eggs is gone
  (slot gives cereal/nuts, i.e. the seeds). The caption names "a green lime" and "watermelon", and the loose crop gives "two halves of a melon".
  The hand lime is still "Apple" (0.87, lime 0.06). Broccoli is unchanged (0.99). Cup stays 0.87 in the caption slot
  even on a lime close-up, because the text "holding a white ___ to her mouth" decides it.
- So the types are: **resolution-limited misreading** (eggs, fixable by zooming in), **stable confusion** (lime → apple, not fixed),
  and **text completion** (cup). Remasking fixes none of them. An idea to test: attention-guided zoom-and-verify on candidate nouns.

### Zoom second pass (`--mode zoom`, full AMBER-g, job 31951930 submitted 2026-10-02)

From step 16 on, a committed token with confidence < 0.3 triggers a zoom pass. The crop is centred on the connected cluster of its top-30 late-layer
attended patches around the peak. It is a square 1.5× the cluster, at least 8 patches wide, cut from the original photo and upscaled to 512.
The token and its committed neighbours are re-predicted greedily with the crop in place of the image, and decoding then continues with the full image.
There is at most 1 pass per position. Offline, confidence < 0.3 covers 12.6% of late tokens, but it barely separates the groups
(12% of hallucinated vs 10% of grounded nouns), so this mostly tests whether the zoomed pass itself helps.
Script: `remask_zoom_amber.slurm`. Outputs go to `results/amber_g/remask_zoom/` (`amber_g_metrics.txt` when done). Smoke tests are in `remask_zoom/smoke/`.
In the smoke test, caption 163 lost eggs and apples but gained "a few oranges". Caption 102 fixed "A few is walking" → "A man is walking".
A random control at the matched rate is still needed before any gain can be claimed.

Result (job 31951930 completed 04:13 on 2026-10-02, 2h52m, all 1004 captions, official scorer, `remask_zoom/amber_g_metrics.txt`).
It triggered on 3748 / 42597 eligible committed tokens (8.8%).

| full AMBER-g (1004) | CHAIR | Cover | Hal | Cog |
|---|---|---|---|---|
| baseline | 7.5 | 48.4 | 28.4 | 2.3 |
| rule (remask) | 7.3 | 48.4 | 28.7 | 2.1 |
| zoom | 7.6 | 47.3 | 29.1 | 2.4 |

**The zoom pass is slightly worse than baseline on every metric** (Cover −1.1, Hal +0.7). Single-run differences this small
are probably noise. A confidence < 0.3 gate does not single out hallucinated nouns, so on its own the zoom does not help.

### Attention-guided retrieval (RAG) from COCO train, caption 163 (2026-10-02, `rag_retrieve.py`)

Bank: `../hallucination-attack/clip_embeddings.pt`, 117,266 COCO train2017 images (`indices` = COCO image ids), OpenCLIP ViT-H-14
`laion2b_s32b_b79k`, 1024-d, unnormalised; images were resized to 336×336 before the CLIP preprocess. COCO is the shared copy at
`/data/gpfs/datasets/COCO`. The query is encoded with the same weights via `transformers` (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`).
Self-check: re-embedding bank images gives cosine 0.97–0.99 to the stored vectors with the 336 resize (0.93–0.96 without), so it is the same model.
For every noun in `with_clean/nouns.csv` for the caption, the noun's `late_map` from the baseline trace gives a `zoom_crop` of the original photo
(top-30 cluster). There are three sizes: super tight = 1.0× the cluster with a side of at least 4 patches, tight = 1.5× with at least 8, and
loose = 3× with at least 8. Several tight crops are 947 px only because of the 8-patch minimum. The top-10 COCO images are copied to
`hallu_study/rag/163/<label>_<pos>_<word>/<super_tight|tight|loose>/` with `retrieved.json` (captions) and `sheet.jpg`. The run log is `rag/rag_163.log`.
support = fraction of the top-10 whose COCO captions mention the noun.

| noun | label | support super tight / tight / loose | what the retrieved captions say |
|---|---|---|---|
| cup | hallucinated | 0.0 / 0.0 / 0.0 | woman holding an apple, red apple, hand |
| drink | hallucinated | 0.0 / 0.0 / 0.0 | apple, woman, holding, eating |
| eggs | hallucinated | 0.0 / 0.0 / 0.0 | fruit, bowl, plate, apples, melon; super tight: fruit, bananas |
| bowl | hallucinated | 0.3 / 0.4 / 0.3 | fruits in bowls, plate, melon |
| apples | hallucinated | 0.3 / 0.4 / 0.6 | apples and a cantaloupe in a dish; super tight: fruit, watermelon, bananas |
| broccoli | grounded | 1.0 / 1.0 / 1.0 | broccoli (similarity 0.86 super tight, 0.78 tight) |
| plate | grounded | 0.5 / 0.7 / 0.3 | fruit on a plate |
| woman (pos 4) | grounded | 0.8 / 0.9 / 0.9 | woman with long red hair |
| table (pos 13) | grounded | 0.1 / 0.2 / 0.4 | super tight (593 px) retrieves toilet seats |
| table (pos 36) | grounded | 0.6 / 0.5 / 0.5 | |
| woman (pos 60) | grounded | 0.0 / 0.0 / 0.0 | its attention is on the plate, not on her |

Super tight sharpens a real object (broccoli similarity 0.78 → 0.86) and lowers apple support to 0.3. Below about 600 px it loses context,
though: table (pos 13) retrieves toilet seats, and the melon rind pulls in bananas.

- **As a check, retrieval rejects the text-completion and resolution errors.** Cup, drink and eggs get support 0. Their regions retrieve
  "hand holding fruit" and "melon / fruit on a plate", i.e. what is actually there.
- **It does not catch the stable confusion.** Lime → apple is shared by CLIP: the lime regions retrieve apples (support 0.4–0.6). Bowl gets
  partial support because COCO fruit photos are often in bowls.
- **As a correction source, the captions point the right way.** For the melon region they say fruit / melon / cantaloupe, and for the hand
  region "holding an apple". Neither names a lime.
- **The threshold is not clean.** Support < 0.2 flags 3/5 hallucinated nouns, but also the woman at pos 60 (attention elsewhere) and one
  table crop. This is one caption, so it is a sanity check, not a result.

### Retrieved captions as MMaDA context, caption 163 (2026-10-02, `rag_mmada_context.py`, `rag/163/mmada_context.json`)

Captions are put before the query as "Captions of similar images retrieved from a database (they may not exactly match this image): - … Describe this image."
The image is present throughout. Log: `rag/mmada_context_163.log`.

**Regenerating the whole caption** (baseline schedule, AmberLabeler):

| context | hallucinated | caption |
|---|---|---|
| none (= saved baseline) | cup, drink, bowl, eggs, apples | |
| full image top-5 | phone, bowl ×2 | "eating a green phone … bowl containing green olives" |
| 5 random COCO captions (control) | apple | "holding a green apple to her mouth … a plate with a sandwich cut in half" |
| all noun regions, top-2 each (13–14 captions) | apples, bowl / bowl, apples / knife, apple | copies "getting her hair cut with a pair of black scissors" from the woman region |
| hallucinated regions only (oracle), any crop | apple | copies "A woman with blue fingernails gripping an apple." verbatim |

**Context makes the model copy, not correct.** Random captions lower the hallucination count as much as retrieved ones (perturbation again).
Region captions are copied wholesale (haircut and scissors), and the oracle context turns the answer into one retrieved caption. Coverage collapses.

**Slot re-prediction** (finished baseline caption, mask one noun, one forward pass): p(original token) with each context.

| noun | label | none | full top-5 | random-5 | super tight | tight | loose |
|---|---|---|---|---|---|---|---|
| woman, table ×2, plate, broccoli, woman | grounded | 0.94–1.00 | 0.93–1.00 | 0.93–1.00 | 0.96–1.00 | 0.93–1.00 | 0.92–1.00 |
| cup | hallucinated | 0.93 | 0.90 | 0.93 | 0.84 | 0.81 | 0.87 |
| drink | hallucinated | 0.56 | 0.55 | 0.74 | 0.69 | 0.69 | 0.64 |
| bowl | hallucinated | 0.29 | 0.74 | 0.60 | 0.77 | 0.88 | 0.85 |
| eggs | hallucinated | 0.69 | 0.59 | 0.45 | 0.32 | **0.08** (→ bananas, oranges) | 0.35 (→ fruit) |
| apples | hallucinated | 0.80 | **0.16** (→ peas, olives) | 0.37 | 0.45 | 0.81 | 0.95 |

Grounded nouns do not move under any context. Hallucinated ones do, but not always toward the truth: bowl is *reinforced* by fruit-in-bowl
captions, and random captions also shake eggs and apples. Stability under context perturbation may itself be a detector (cf. the text-remask probe).

### Retrieval-context second pass inside decoding, caption 163 (2026-10-02, `rag_slot_decoding.py`, `rag/163/rag_slot/`)

The baseline schedule is used. When a watched token is committed (the baseline's hallucinated nouns cup, drink, bowl, eggs, apples), the steps are:
1. take its tight crop from the live late-layer attention and retrieve the top-5 COCO images with CLIP;
2. mask the token and its committed neighbours again;
3. refill them one at a time with the 5 first captions before the query;
4. continue with the normal prompt.

There is at most one trigger per position. The control uses 5 random COCO captions. Log: `rag/rag_slot_163.log`.

| context | event | p(word) at commit → with context | remasked → refilled |
|---|---|---|---|
| tight retrieval | step 13, pos 20 "cup" | 0.52 → 0.00 (apple 0.93) | white cup to → **green apple** to |
| tight retrieval | step 26, pos 49 "apples" | 0.47 → 0.35 (bananas 0.35, oranges 0.27) | of apples . → of **bananas** . |
| random captions | step 13, pos 20 "cup" | 0.52 → 0.01 (apple 0.55, object 0.26) | white cup to → green **object** to |
| random captions | step 25, pos 50 "bowl" | 0.38 → 0.43 | a bowl → a bowl |

Captions retrieved for cup: "A woman with blue fingernails gripping an apple." / "A hand holding a half eaten red apple." / "A green apple being sliced with a knife." /
"Close-up picture of an apple, with a knife in front." / "a female wearing black a sink and a bed".
Captions retrieved for apples: "apples and a cantaloupe in a dish on a table" / "some fruits are sitting on a table in bowls." /
"A plate that is covered with bannannas and oranges." / "A white plate topped with sliced up fruit." / "An apple sitting on a plate next to a knife and spoon."

The final captions:
- tight: "…holding a green apple to her mouth, possibly about to take a bite. On the table, there are various fruits, including a broccoli broccoli
  and a couple of bananas…". Hallucinated: apple, bananas (baseline 5). Lost: plate.
- random: "…holding a green object to her mouth, possibly a it. a broccoli broccoli. … a plate of food … a bowl is also visible…". Hallucinated: bowl.
  This is disfluent.

- **Fixing cup does not need the retrieved content.** Any prepended captions drop p(cup) from 0.52 to ≈ 0, and apple is the top candidate even with random captions.
  So the image already says "round green fruit". Cup only survives in the baseline because of the sentence frame.
- **Retrieval gives a fluent, concrete replacement** (green apple, closer to the true lime) where random context gives "object" and broken text.
  But the replacement is limited by COCO's vocabulary: there are no lime captions, so we get apple and then bananas.
- After the cup fix the trajectory changes and drink, bowl and eggs are never written. The count drops 5 → 2, but per AMBER the result is still not clean.

**Caption 84** (same script, log `rag/rag_slot_84.log`, outputs `rag/84/rag_slot/`).
The scene is a boy in a bath pouring water from a yellow cup, with a purple shampoo bottle on the tiled edge.
AMBER truth: wall, shampoo, child, bath, toy, cup, water, bottle.
Baseline: "…washing his hands under a faucet. He is standing in the sink, holding the soap bar… a purple towel hanging on the wall". Hallucinated: faucet, sink, soap, bar, towel.
The photo is only 333×500, so the tight crop is 83 px (the 8-patch minimum). For faucet, sink and soap it is the same box, the top of the **shampoo bottle**.
All tight events therefore retrieve the same 5 captions: "A tube of cream sitting on top of a bathroom sink." / "there is a small bottle of hand soap on the sink" /
"A bathroom sink with a soap dispenser on it." / "Cluttered white shelf with tootbrush, toothpaste and hairbrush." / "Deoderant, a razor, some rocks and a vase sitting on a shelf."

| context | event | p(word) at commit → with context | remasked → refilled |
|---|---|---|---|
| tight | step 12, pos 21 faucet | 0.47 → 0.13 (soap 0.36, water 0.32) | a faucet . → a faucet . |
| tight | step 15, pos 28 sink | 0.55 → 0.04 | the sink , → front of a |
| tight | step 16, pos 30 sink | 0.42 → **0.51** | a sink , → a sink , |
| random | step 12, pos 21 faucet | 0.47 → 0.30 | kept |
| random | step 15, pos 28 sink | 0.55 → 0.06 (water 0.32, bathtub 0.14) | the sink , → a **bathtub filled** |
| random | step 19, pos 39 soap | 0.21 → 0.26 | kept |

- tight: "…standing in front of a sink, … a pair of purple flip flop on the floor". Hallucinated: faucet, sink, floor. Grounded: boy ×2.
- random: "…standing in a bathtub filled with water, and the hands are covered in soap suds…". Hallucinated: faucet, soap. Grounded: boy, bathtub, water, boy, bath.

**Here retrieval is worse than random captions.** The attended region is a bottle, COCO's bottle-in-bathroom photos are captioned with sinks, and
the retrieved text re-installs "sink" one position later. Random context frees the slot, and the image fills it correctly (bathtub, water).
Across 163 and 84, the useful part of the intervention is remask + any prompt perturbation, not the retrieved content. Low-resolution AMBER photos
also make attention crops tiny and blurry.

**Zoom refill instead of captions** (`rag_slot_decoding.py --contexts zoom`, outputs `rag/<id>/rag_slot_zoom/`, logs `rag/rag_slot_zoom_<id>.log`).
The trigger is the same (watched word committed). The refill pass puts the tight crop, upscaled to 512, in place of the photo, with the normal prompt.
Decoding then continues with the full photo.

| caption | event | p(word) at commit → zoomed | refilled |
|---|---|---|---|
| 163 | step 13 cup | 0.52 → 0.00 (apple 0.53, "close" 0.37, ball 0.04, **lime 0.01**) | white cup to → green apple to |
| 163 | step 26 apples | 0.47 → 0.15 (oranges 0.42, bananas 0.37) | of apples . → of oranges . |
| 84 | step 12 faucet | 0.47 → 0.13 (soap 0.46) | a faucet . → purple soap . |
| 84 | step 15 sink | 0.45 → 0.01 | a sink , → front of a |
| 84 | step 16 sink | 0.55 → 0.21 | kept |
| 84 | step 19 soap | 0.36 → 0.03 | soap → his |
| 84 | step 29 soap | 0.44 → 0.81 | kept |
| 84 | step 47 sink | 0.46 → 0.29 (bathroom 0.55) | the sink . → the bathroom . |

The final captions:
- 163: "…holding a green apple to her mouth … various fruits, including a broccoli broccoli and a couple of oranges". AMBER counts only apple as hallucinated.
  "orange" is in AMBER's `safe_words.txt` (it is also a colour), so oranges is ignored, though it is still wrong. In practice that makes 2 wrong, as with tight captions.
- 84: "…washing his hands with purple soap. He is standing in front of a sink, and is holding a yellow sponge … a pair of sandals on the floor".
  Hallucinated: soap, sink, sponge, soap, floor. No better than baseline.
  The 83-px bottle crop reads as "purple soap", and the yellow cup becomes a "sponge".

All caption-163 and 84 variants (AMBER-counted hallucinated nouns; baseline 5 and 5):

| | tight captions | random captions | zoom crop |
|---|---|---|---|
| 163 | 2 (apple, bananas) | 1 (bowl), broken text | 1 (apple) + oranges, ignored by AMBER |
| 84 | 3 (faucet, sink, floor) | 2 (faucet, soap), correct "bathtub filled with water" | 5 (soap ×2, sink, sponge, floor) |

The zoom pass helps on the high-resolution 163 photo, where lime even enters the top-5. It does nothing on the 333×500 photo 84, where the crop is an
upscaled 83-px blur. No variant is consistently better than the random-caption control.

### Slot refill on the full study set (job 32082608, submitted 2026-10-02 15:15, `slot_refill_study.py` / `.slurm`)

The 144 study captions (84 hallucinating + 60 clean) are decoded with the baseline schedule. A committed token is *eligible* if it decodes to a whole
word whose lemma is an AMBER object word (not a global safe word). Triggers:
- **oracle**: hallucinated by the AMBER rule for that image, applied online, so it also catches new hallucinations after the trajectory diverges;
- **clip**: CLIP rank_full < 0.9. On the study nouns that gives recall 0.54, a 7% false-flag rate on grounded nouns and precision 0.56. It was chosen on the same set;
- **random**: matched to clip's fired / eligible rate.

On a trigger the token and its committed neighbours are masked and refilled one at a time (most confident first) in a second pass:
- **plain**: same image and prompt;
- **zoom**: the tight crop in place of the image.

Decoding then continues normally, with at most one trigger per position. Every event logs the oracle label, so detection precision can be read from any run.
Outputs: `rag/slot_refill/<trigger>_<refill>/predictions.json` (+ `_events.json`), scored by `eval_remask.py --output-dir rag/slot_refill/eval`.
Log: `rag/slot_refill/slurm-32082608.out`. Resubmitting resumes. A smoke test on ids 2 and 4 is in `rag/slot_refill/smoke/`.

Partial scores (`rag/slot_refill/eval_partial/`, official scorer, hallucinating 84). Baseline: 155 hallucinated nouns, CHAIR 22.3, Cover 43.5.
- oracle+plain: 123, CHAIR 18.7;
- **oracle+zoom: 102, CHAIR 15.3, Cover 45.0, grounded 521 → 548**;
- clip+plain: 138, CHAIR 20.6.

- **clip+zoom: 132, CHAIR 19.7, Hal 86.9, Cog 16.9, but Cover 42.1 and grounded 518** (fewer / same / more = 21 / 58 / 5). Clean: 1 hallucinated noun
  (CHAIR 0.3, Cover 48.9). Of its 166 triggers, 98 hit hallucinated words (35 of those changed) and 68 grounded ones (27 changed, e.g. ground → carpet,
  path → "ripples", dog → ","). So the false flags are what cost Cover. The zoom only helps fully when detection is right.

**Random controls (matched rate ≈ 16% of eligible object words; precision 0.18, i.e. the base rate):**

| condition | hallucinating 84: CHAIR / Cover / Hal / Cog | hallu nouns | grounded | clean 60: hallu nouns (CHAIR) |
|---|---|---|---|---|
| baseline | 22.3 / 43.5 / 98.8 / 21.3 | 155 | 521 | 0 (0.0) |
| random+plain | 21.9 / 43.3 / 92.9 / 19.9 | 149 | 511 | 1 (0.3) |
| clip+plain | 20.6 / 42.9 / 94.0 / 19.4 | 138 | 511 | 2 (0.5) |
| random+zoom | 19.4 / 44.4 / 82.1 / 16.7 | **130** | 524 | **4** (1.0) |
| clip+zoom | 19.7 / 42.1 / 86.9 / 16.9 | **132** | 518 | **1** (0.3) |
| oracle+zoom | 15.3 / 45.0 / 78.6 / 13.8 | 102 | 548 | 0 (0.0) |

Direct hits: random+zoom acts on only 31 hallucinated words (12 changed) and 140 grounded (52 changed), yet hallucinating captions still drop 155 → 130.
**Most of the zoom gain is perturbation, not correction.** Changing any word re-routes the rest of the caption, which helps captions selected for hallucinating
(regression to the mean again) and hurts clean ones (random+zoom gives 4 new hallucinations on clean, clip+zoom 1).
- **Zoom refill: CLIP is no better than random on the hallucinating set** (132 vs 130), and better only on clean (1 vs 4). Summed over both subsets it is a tie (133 vs 134).
- **Plain refill: CLIP beats random** (138 vs 149), because the plain refill rarely changes a word, so only direct hits matter.
- **Oracle+zoom (102) is far ahead of everything**, so detection quality does matter. CLIP's 0.59 precision is not enough to beat the perturbation effect.
These are single runs, and differences of a few nouns are within noise.

Clean (60): oracle runs are untouched, and clip+plain adds 2 hallucinated nouns. Online CLIP precision is 0.62 (plain) / 0.59 (zoom).
The word itself changes on 24% (plain) vs 41% (zoom) of oracle triggers.

Efficiency (`bench_slot_refill.py`, A100-PCIE-40GB on the Jupyter node, median of 5 captions, `rag/slot_refill/efficiency.json`):
- **costs:** a baseline caption takes 9.03 s (64 forwards, 135 ms each). The attention recorder adds 0.59 s (+6.5%). The zoom crop + VQ takes 56 ms per trigger.
  **CLIP image embedding + rank takes 16 ms per caption (0.2%)**;
- **triggers:** there are about 1.2 per caption, each refilling about 2.6 tokens, i.e. about 3 extra forwards per caption;
- **extra cost per caption:** clip+plain +0.40 s (+4.5%), clip+zoom +1.09 s (+12%), oracle+zoom +1.03 s (+11.4%). The recorder is most of the zoom
  overhead, and it could be cut by recording only layers 19–31 and only at steps that commit an eligible word;
- **detection:** 16% of eligible committed object words are hallucinated (oracle fired 174 / 1076). CLIP's precision of 0.62 is about 4× that rate,
  and it catches about 60% of what the oracle catches (105 vs 174 true hits).

### What separates CLIP's false flags from real hallucinations (2026-10-02, `rag_flag_filter.py`, `rag/clip_relation/flag_filter.json`)

Baseline-trace features within the 148 nouns with rank_full < 0.9 (83 hallucinated, 65 grounded, 80 captions). AUROC is for higher ⇒ hallucinated, with a 95% CI over captions.
- **Language prior: nothing.** p_no_image 0.54 (0.42–0.65), dlogp 0.47, KL 0.44, no-image top-1 0.56. Confidence: p_image 0.52.
- **Attention: some.** late_top5 **0.68** (0.59–0.76; hallucinated 0.147 vs grounded 0.098), late_image_entropy 0.35 (i.e. hallucinated more focused), late_image_mass 0.41.
  The false flags attend diffusely, and the hallucinations attend to one spot (the misread object).
- **Decoding order: some.** step 0.63, relative_position 0.63, context_before 0.60.
- **CLIP on crops / retrieval: some.** support_full 0.31 (i.e. real objects are confirmed by the retrieved captions), rank_tight 0.37 (the crop sees the real object).
- Grouped-CV logistic regression: attention 0.69, decoding order 0.62, CLIP crops 0.62, all MMaDA 0.68, everything 0.69, language prior 0.44.

**The false flags are mostly background words** (`inspect_false_flags.py`): tree 9, wall 6, water 5, sky 4, ground 4, grass 4, road 3, bush 2, court 2…
"a photo of a {word}" is a poor prompt for scene "stuff". The flagged hallucinations are mostly objects: people 17, sun 7, mouse 7, person 5, table 4.
Some false flags are AMBER similarity artefacts where CLIP is right. In caption 241 the image has a **cat** and "dog" is labelled grounded via spaCy dog ≈ cat.

Two-stage filters on the flagged set (precision / recall over all 155 hallucinated nouns; CLIP alone is 0.56 / 0.535):

| second stage | precision | recall | grounded removed | hallucinated lost |
|---|---|---|---|---|
| drop if support_full ≥ 0.3 | 0.66 | 0.46 | 29 / 65 | 12 / 83 |
| drop if late_top5 < 0.06 | 0.61 | 0.45 | 21 | 13 |
| drop if rank_tight ≥ 0.95 | 0.66 | 0.48 | 26 | 8 |
| stricter CLIP instead (rank_full < 0.85) | 0.57 | 0.41 | 17 | 19 |
| **never flag background words** | **0.77** | 0.46 | 44 | 12 (sun 7, street 2…) |
| background words only if rank_full < 0.5 | 0.72 | 0.48 | 36 | 8 |
| never flag background + drop if rank_tight ≥ 0.95 | **0.84** | 0.42 | 53 | 18 |

Caveat: the background list (COCO-Stuff-style, includes "sun") was written *after* seeing these flags, and all thresholds were chosen on the same nouns.
Overall, background words are 30% of nouns, with a hallucination rate of 0.114 vs 0.162 for objects.

**Crop check only (no background list)**, `rag/clip_relation/crop_veto_sweep.py`: keep a CLIP flag only if the word's CLIP rank on a crop is below a cutoff.

| veto on | cutoff | precision | recall | grounded removed / 65 | hallucinated lost / 83 | fire rate |
|---|---|---|---|---|---|---|
| none (CLIP alone) | | 0.561 | 0.535 | 0 | 0 | 0.141 |
| super tight | 0.95 | 0.619 | 0.471 | 20 | 10 | 0.112 |
| **tight** | **0.95** | **0.658** | **0.484** | **26** | **8** | 0.109 |
| tight | 0.98 | 0.603 | 0.510 | 13 | 4 | 0.125 |
| loose | 0.95 | 0.652 | 0.471 | 26 | 10 | 0.107 |

**Why not the crop alone?** (`rag/clip_relation/compare_detectors.py`, all 1050 nouns, 155 hallucinated)

| detector | flagged | hallucinated | grounded | precision | recall |
|---|---|---|---|---|---|
| full < 0.9 | 148 | 83 | 65 | 0.56 | 0.54 |
| tight crop < 0.9 | 293 | 83 | 210 | 0.28 | 0.54 |
| full < 0.9 + tight veto ≥ 0.95 | 114 | 75 | 39 | 0.66 | 0.48 |
| *equal budget 114:* full only | 114 | 65 | 49 | 0.57 | 0.42 |
| *equal budget 114:* tight crop only | 114 | 28 | 86 | 0.25 | 0.18 |
| *equal budget 148:* tight crop only | 148 | 36 | 112 | 0.24 | 0.23 |

AUROC is 0.860 for the full image vs 0.765 / 0.709 / 0.775 for the tight / super tight / loose crops.
At the operating points, nouns flagged by the crop but not the full image are 196 with only 22 hallucinated (precision 0.11). The crop shows what the model attended to.
A hallucination is a misreading of a real attended object, so its crop really looks like the word, and a grounded noun whose attention is elsewhere gets a crop rank near 0.
**The crop fails as a detector but works as a veto.** When the crop confirms the word, the object is probably there.

The tight crop at 0.95 is chosen. It is the same crop the zoom refill uses, so it costs one extra CLIP call per flag.
`slot_refill_study.py --crop-veto 0.95` applies this during decoding and logs vetoed / vetoed_oracle_hallucinated.
**Job 32089240** (`slot_refill_cropveto.slurm`, submitted 17:05) runs clip+cropveto+zoom and a matched random+zoom on the 144 captions, scored into `rag/slot_refill/eval_cropveto/`.

**Crop-veto snapshot** (136 / 144 captions, `rag/slot_refill/eval_partial_cropveto/`):
- hallucinating 84: clip+cropveto+zoom has 136 hallucinated nouns (CHAIR 20.2, Cover 41.9) vs clip+zoom 132 and random+zoom 130;
- clean 52: 0 new hallucinated nouns vs 1 (clip+zoom) and 4 (random+zoom);
- online precision is 0.68 vs 0.59, with grounded triggers 41 vs 68. Hallucinated words changed: 34 in both. The veto dropped 32 flags (23 grounded, 9 hallucinated).
So the veto makes the method safer (clean captions are untouched), but it fixes no more hallucinations. The fixes are limited by the refill (41% change rate even for the oracle).

**Crop-veto final** (job 32089240, `rag/slot_refill/eval_cropveto/eval_report.json`). The random control is matched to the veto's fire rate (0.12, precision 0.18).

| condition | hallucinating 84: CHAIR / Cover / Hal / Cog | hallu nouns | clean 60: new hallu nouns (CHAIR, Cover) |
|---|---|---|---|
| baseline | 22.3 / 43.5 / 98.8 / 21.3 | 155 | 0 (0.0, 49.5) |
| clip+zoom | 19.7 / 42.1 / 86.9 / 16.9 | 132 | 1 (0.3, 48.9) |
| clip+cropveto+zoom | 20.2 / 41.9 / 90.5 / 17.7 | 136 | **0** (0.0, 49.2) |
| random+zoom @ 12% | 20.1 / 44.0 / 88.1 / 17.9 | 135 | 3 (0.8, 49.8) |
| oracle+zoom | 15.3 / 45.0 / 78.6 / 13.8 | 102 | 0 (0.0, 49.5) |

Online: 131 triggers, precision 0.687; 32 vetoed (9 of them hallucinated).
**On hallucinating captions the veto run ties with its matched random control (136 vs 135), with lower Cover (41.9 vs 44.0).
Its only advantage is on clean captions (0 vs 3 new hallucinations).** Over all 144 it is 136 vs 138 hallucinated nouns.

**Full AMBER-g run.** It was first submitted as job 32092603, which was cancelled while still pending. At the user's request it was restarted at **18:51 inside the Jupyter job
32078532 (gpgpu092) as step 32078532.13**, launched from a detached tmux session `cropveto-full` on **spartan-login3**
(`srun --jobid=32078532 --overlap --ntasks=1 bash slot_refill_full.slurm`). Log: `results/amber_g/slot_refill_full/jupyter_run.log`.
It needs ≈ 3 h (≈ 9.9 s / caption), and the Jupyter job ends ≈ 00:03 on 2026-10-03. If it is cut off, rerunning the same command resumes.
Reattach with `ssh spartan-login3; tmux attach -t cropveto-full`.
Script: `slot_refill_full.slurm`.

Snapshot at 20:00, ids 1–407 (`slot_refill_full/score_partial.py` → `clip_cropveto_zoom/partial_eval/partial_report.json`, official scorer, baseline on the same ids):

| ids 1–407 | CHAIR | Cover | Hal | Cog | hallu nouns | grounded nouns |
|---|---|---|---|---|---|---|
| baseline | 8.2 | 47.5 | 31.0 | 2.6 | 226 | 2462 |
| clip+cropveto+zoom | **7.5** | 46.1 | 30.2 | 2.6 | **202** | 2420 |

- The text changed in 116 / 407 captions. Per caption: 30 fewer, 357 same, 20 more hallucinations.
- Baseline-hallucinating captions (126): 226 → 180. Baseline-clean captions (281): 0 → 22, and 14 of them now hallucinate.
- Triggers: 295 / 2754 eligible (10.7%), **precision 0.42** (vs 0.69 on the study set; the base rate is lower here). Changed: 58 hallucinated, **90 grounded** words. 114 were vetoed (14 hallucinated).
- The net is −24 hallucinated nouns (−11%) for −42 grounded (Cover −1.4), better than the rule run (−13 over 872 captions). The same pattern holds, though:
  gains on hallucinating captions are partly cancelled by new hallucinations in clean ones. **No full-set random control yet**, so the gain cannot be attributed to CLIP. clip (rank_full < 0.9) + tight-crop veto (≥ 0.95) + zoom refill on ids 1–1004,
then `benchmarks/amber/evaluate_amber_g.sh`. Outputs: `results/amber_g/slot_refill_full/clip_cropveto_zoom/` (`amber_g_metrics.txt` when done). There is no random control on the full set yet.
Compare with baseline 7.5 / 48.4 / 28.4 / 2.3, rule 7.3 / 48.4 / 28.7 / 2.1 and zoom (confidence gate) 7.6 / 47.3 / 29.1 / 2.4.

**Result** (step 32078532.13 COMPLETED 21:44 on 2026-10-02, 2h52m, exit 0, all 1004 captions, `clip_cropveto_zoom/amber_g_metrics.txt`).
Paired counts: `score_partial.py clip_cropveto_zoom final_eval` → `clip_cropveto_zoom/final_eval/partial_report.json` (the 20:00 snapshot stays in `partial_eval/`).

| full AMBER-g (1004) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded nouns |
|---|---|---|---|---|---|---|
| baseline | 7.5 | 48.4 | 28.4 | 2.3 | 523 | 6141 |
| rule (remask) | 7.3 | 48.4 | 28.7 | 2.1 | | |
| zoom (confidence < 0.3) | 7.6 | 47.3 | 29.1 | 2.4 | | |
| clip+cropveto+zoom | 7.4 | 47.1 | 28.1 | 2.5 | 503 | 6054 |

- The text changed in 261 / 1004 captions. Per caption: 61 fewer, 899 same, 44 more hallucinations.
- Baseline-hallucinating captions (285): 523 → 452. Baseline-clean captions (719): 0 → 51, and 28 of them now hallucinate.
- Triggers: 733 / 6892 eligible (10.6%), precision 0.41. Changed: 124 hallucinated, 221 grounded words. 276 were vetoed (38 hallucinated).
- **Net −20 hallucinated nouns (−3.8%) for −87 grounded (Cover −1.3), Cog +0.2.** Ids 1–407 gave −24, so ids 408–1004 gave +4: the early gain did not hold.
  This is within noise of baseline and the rule run, and still has no full-set random control. On the full set, false flags (precision 0.41, 221 grounded words changed)
  and new hallucinations in clean captions cancel most of the fixes.

### Oracle+zoom on the full AMBER-g set (2026-10-02, `results/amber_g/oracle_zoom_full/`)

This is the upper bound for the zoom refill with perfect detection. The baseline has 285 hallucinating captions (AmberLabeler on `mmada_predictions.json`), 83 of them in the study set.
The other 202 were run with oracle+zoom as step 32078532.17 inside the Jupyter job (tmux `oraclezoom-full`, 22:05–22:41, `run_missing.sh`, `jupyter_run.log`).
That run fired 345 / 1611 eligible words (21%).
`merge_and_score.py` builds the 1004 predictions from three sources:
- the study-set oracle+zoom run (144 captions, including the 60 decoded clean ones);
- the new run (202);
- the baseline text for the other 658 clean captions.

The baseline fill is an approximation. The online oracle also fires on verbs that are AMBER object words ("can", "watches", "drawing"), which the official scorer drops by POS.
On the 60 study clean captions it fired 7 times, changed 3 texts and added 0 hallucinations.
Report: `report.json`.

| full AMBER-g (1004) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded nouns |
|---|---|---|---|---|---|---|
| baseline | 7.5 | 48.4 | 28.4 | 2.3 | 523 | 6141 |
| clip+cropveto+zoom | 7.4 | 47.1 | 28.1 | 2.5 | 503 | 6054 |
| **oracle+zoom** | **4.6** | **48.9** | **20.4** | **1.8** | **314** | **6229** |

- Hallucinated nouns: 523 → 314 (−40%). Grounded nouns: 6141 → 6229 (+88). Per caption: 126 fewer, 866 same, 12 more.
- 505 oracle triggers, of which the word changed in 235 (47%). The refill fixes about half of what it is pointed at, and the rest of the gain comes from re-routing the caption.
- **Correction is not the bottleneck; detection is.** The same refill with CLIP detection (precision 0.41 on the full set) gives −20.

**Why oracle+zoom works and clip+cropveto+zoom does not** (event logs of both full-set runs; "hallucinated" = online oracle label):

| | oracle+zoom | clip+cropveto+zoom |
|---|---|---|
| hallucinating 285: triggers on hallucinated words | 498 | 230 |
| hallucinating 285: captions where it reached a hallucination | 279 | 154 |
| hallucinating 285: hallucinated words changed | 232 (47%) | 102 (44%) |
| hallucinating 285: triggers on grounded words (changed) | 0 | 124 (60) |
| hallucinating 285: hallucinated nouns (baseline 523) | 314 (−209) | 452 (−71) |
| clean 719: triggers on grounded words (changed) | 0 | 311 (161) |
| clean 719: new hallucinated nouns | 0 | +51 |

- **The refill is equally good in both runs** (it changes 44–47% of the hallucinated words it is pointed at). The whole gap is the detector.
- **Recall:** CLIP reaches less than half the hallucinations and never touches 131 of the 285 hallucinating captions. This is about 140 of the 189-noun gap.
- **False flags:** CLIP flags 435 grounded words (mostly background: tree, wall, sky) and changes 221 of them, which costs Cover and adds 51 hallucinations to clean captions.
  Precision falls from 0.69 (study) to 0.41 (full) because the full set's base rate is lower: 72% of captions are clean, against 42% in the study set.
- **The oracle is self-correcting and metric-aligned.** It uses AMBER's own labels online, so it also catches new wrong words written after a refill (51 of its triggers were on words absent from the baseline caption).
  It never touches a word the scorer counts as correct, so Cover goes up. It is an upper bound, not a method.

### CLIP text–image similarity as a detector (2026-10-02, `rag_clip_relation.py`, `rag/clip_relation/`)

All 1050 nouns (155 hallucinated) of the 144 study captions. t = CLIP text embedding of "a photo of a {lemma}." (same ViT-H-14).
`rank` = percentile of the noun among 418 object words (AMBER relation.json words + study lemmas) for that image, which removes word bias.
`support` = fraction of the top-10 retrieved COCO images whose captions mention the noun. AUROC is for lower ⇒ hallucinated, with a 95% CI bootstrapped over captions.

| view | sim | rank | ret (mean over top-10) | support | top-1 image–image sim |
|---|---|---|---|---|---|
| full image | 0.824 | **0.860** (0.81–0.90) | 0.782 | **0.861** (0.82–0.91) | 0.497 |
| loose crop | 0.743 | 0.775 | 0.734 | 0.776 | 0.558 |
| tight crop | 0.726 | 0.765 | 0.708 | 0.760 | 0.581 |
| super tight crop | 0.700 | 0.709 | 0.643 | 0.678 | 0.565 |

Checks (offline on `nouns_clip.csv`):
- hallucinating captions only: rank_full 0.866;
- first mentions: 0.849;
- step ≥ 16: 0.870;
- **within the same lemma** (41 lemmas with both labels, 594 nouns): 0.777.

Grouped 5-fold CV logistic regression: decoding order (position, repeat, context_before) 0.786, CLIP full rank + support 0.871, both 0.872, and all CLIP views 0.884.

- **The full image is the best view, far better than any MMaDA-internal feature** (best so far 0.775). Adding the decoding-order features to it gains nothing (0.871 → 0.872).
- **The attended crop is worse than the full image**, and the tighter the crop, the worse. This fits the caption-163 finding: hallucinations are misreadings
  of the attended real object, so the crop really does look like a cup or an egg to CLIP as well.
- Retrieval confidence (top-1 image–image similarity) carries no signal.
- Caption 163 is a hard case even here: rank_full is 0.83–0.96 for its hallucinated nouns versus 0.94–0.99 for grounded, but support_full is 0.0–0.1 for all hallucinated nouns.
- Caveat: no thresholds were tuned, but this is still the study set. The full-image features need no traces, so they can be checked on all 1004 AMBER-g captions.

## Next steps (plan for the next session, written 2026-10-03)

The oracle+zoom upper bound (−40% hallucinated nouns on the full set) shows that the refill works and the detector is the bottleneck. The goal is a better detection rule.

**What is known about confidence (do not re-test):**
- **Commit confidence is weak.** p(token | image) at the noun's commit gives AUROC 0.577 (hallucinated median 0.524 vs grounded 0.595), with no gap at all for late nouns (step ≥ 16).
  Within the CLIP-flagged set it is 0.52. Only ≥ 0.9 is informative ("probably grounded"). Hallucinations here are confident misreadings of real objects, not uncertain guesses.
- Other signals so far: CLIP full-image rank 0.86 (best), decoding order 0.78, attention concentration (late_top5) 0.62–0.68, and image ablation (dlogp) 0.45 (useless).

1. **Test re-prediction confidence in the finished caption (most promising, untested at scale).**
   On caption 163, masking one noun in the finished baseline caption and re-predicting it with the image (one forward pass, context on both sides) gave
   grounded nouns 0.94–1.00 and hallucinated ones 0.29–0.93 (cup 0.93, apples 0.80, eggs 0.69, drink 0.56, bowl 0.29).
   For every noun of the 144 study captions (`with_clean/nouns.csv`, 1050 nouns, 155 hallucinated), record p(original token) for its first token in the finished baseline caption:
   - with the image (plain re-prediction);
   - with the image + 5 random COCO captions as context;
   - with no image;
   - with the tight zoom crop in place of the image.
   Also record **stability** = min (or spread) of p over the perturbations. On 163, grounded nouns stayed at 0.92–1.00 under every context, while hallucinated ones moved
   (eggs 0.69 → 0.08, apples 0.80 → 0.16), so "flag if min p < 0.9" would catch all of them except cup.
   The code to reuse is in `rag_mmada_context.py` (slot re-prediction) and `slot_refill_study.py` (zoom crop). The cost is about 1–4 forwards per noun, ~10 min on the GPU.
   Report the AUROC of each signal alone, combined with CLIP rank_full (grouped 5-fold CV by caption), and **within CLIP's flags** (does it remove the background-word false flags?).
2. **Degrade the oracle to see whether recall or precision matters more.** Run oracle+zoom on the 144 study captions with:
   - (a) a random half of its flags dropped (recall ≈ 0.5);
   - (b) extra random flags on grounded words (precision ≈ 0.4).
   This tells us whether the detector work should go into finding more hallucinations or into fewer false flags.
3. **If re-prediction confidence is strong, use it as the gate or as a veto on CLIP flags,** and run it on the full set with the zoom refill. Two candidates:
   - the background-word exclusion (precision 0.77 on the study set), re-checked on the full set because it was written after seeing the flags;
   - online, re-run the detector after each refill so it can also catch the new wrong words (what makes the oracle self-correcting).
4. **Controls and validation, still open:**
   - a matched-rate random+zoom control on all 1004 for clip+cropveto+zoom;
   - thresholds were chosen on the 144 study captions, so validate any new rule on held-out ids (730–1004);
   - the old full-set random control for the remask rule is also still missing.

## Session 2026-10-03 (24 h GPU, Jupyter job 32117370 on gpgpu091)

### 1. Re-prediction confidence in the finished caption: negative (`repredict_confidence.py`, `analyze_repredict.py`)

All 1050 study nouns (155 hallucinated, 144 captions). The finished baseline caption is kept, the noun's tokens are masked, and one forward pass
re-predicts them. Conditions: plain, plain_nb (pos−1 and the next token masked too), random5 / random5b (two independent sets of 5 random COCO
captions before the query), noimage, zoom (tight crop from the baseline late_map in place of the image). Batched per caption, ~25 s / caption, 61 min.
Smoke check on 163 matches the earlier probe (cup 0.93, bowl 0.29, eggs 0.68, apples 0.77).
Outputs: `hallu_study/repredict/repredict.csv`, `analysis.json`, `repredict.log`, `breakdown.py`.

| signal (higher ⇒ hallucinated) | AUROC | 95% CI |
|---|---|---|
| 1 − p plain | 0.667 | 0.62–0.71 |
| 1 − p plain, neighbours masked | **0.682** | 0.63–0.73 |
| 1 − p random5 / random5b | 0.680 / 0.679 | |
| 1 − p no image | 0.658 | |
| 1 − p zoom | 0.670 | |
| 1 − min p (plain, random ×2, zoom) = "stability" | 0.676 | 0.63–0.72 |
| spread = p plain − min p | 0.641 | |
| entropy plain | 0.680 | |
| log p plain − log p no image | 0.493 | |
| commit confidence (baseline trace), for reference | 0.577 | |
| CLIP rank_full, for reference | **0.860** | 0.81–0.90 |

Grouped 5-fold CV: re-prediction (all conditions) 0.674, CLIP 0.867, **re-prediction + CLIP 0.868**, + decoding order 0.869.
**Within CLIP's flags** (148 nouns, 83 hallucinated) every re-prediction signal is 0.51–0.59 (CIs include 0.5).

- **Re-prediction is better than commit confidence (0.68 vs 0.58) but adds nothing to CLIP** (Spearman with rank_full only 0.22, yet CV 0.867 → 0.868).
  It does not separate CLIP's false flags from real hallucinations either. As a veto on CLIP flags it loses hallucinations about as fast as false flags
  (min p < 0.95: precision 0.56 → 0.62 but recall 0.54 → 0.32). As a union with CLIP it only trades precision for recall (CLIP or p < 0.3: P 0.43, R 0.61).
- **Caption 163 was not representative.** Over all nouns, "flag if min p < 0.9" has precision 0.24 (base rate 0.15) at recall 0.54, against CLIP's 0.56 at the same recall.
  Many grounded nouns are re-predicted with low confidence (synonym ambiguity: laptop 0.30, desk 0.57, grass 0.27), and these are not mostly background words
  (29% of low-p grounded nouns are background vs 31% of all grounded).
- Item 3 of the plan (re-prediction as the gate / veto) is therefore dropped.
- Side finding (`breakdown.py`): **CLIP rank_full is 0.921 on object words but only 0.684 on background words** (317 nouns, 36 hallucinated). Re-prediction is 0.70 / 0.62.

### CLIP detector on all 1004 baseline captions (`full_clip_nouns.py`, `results/amber_g/full_clip/`)

Every AMBER-labelled noun (ignored words dropped) of the 1004 baseline captions, CLIP full-image rank against the online detector's vocabulary
(same as `slot_refill_study.py --trigger clip`), with four prompts and their mean. 6664 nouns, 523 hallucinated (matches the baseline count).
Analysis: `full_clip/analyze_full_clip.py` → `analysis.json`. Background = the STUFF list from `inspect_false_flags.py` (written after seeing study flags).
These are baseline-caption nouns, offline. The online run's precision differs (it sees rerouted text, and the crop veto).

| split | nouns / hallucinated | AUROC "a photo of a {w}." | objects | background | other prompts (bare / there is / word / ensemble) |
|---|---|---|---|---|---|
| all 1004 | 6664 / 523 | 0.830 (0.80–0.85) | 0.878 | 0.721 | 0.821 / 0.786 / 0.766 / 0.818 |
| non-study (860 captions) | 5614 / 368 | 0.820 (0.79–0.85) | 0.865 | 0.756 | 0.805 / 0.776 / 0.748 / 0.804 |
| held-out 730–1004, non-study | 1665 / 125 | 0.837 (0.79–0.88) | 0.880 | 0.789 | 0.822 / 0.784 / 0.760 / 0.816 |

Operating points (precision / recall):

| detector | all 1004 | non-study | held-out 730–1004 |
|---|---|---|---|
| CLIP < 0.9 (current) | 0.28 / 0.46 | 0.22 / 0.42 | 0.29 / 0.47 |
| CLIP < 0.8 | 0.34 / 0.29 | 0.28 / 0.29 | 0.38 / 0.32 |
| never background, CLIP < 0.9 | 0.44 / 0.37 | 0.35 / 0.33 | 0.45 / 0.34 |
| background only if < 0.5, else < 0.9 | 0.41 / 0.39 | 0.32 / 0.35 | 0.42 / 0.36 |
| **never background, CLIP < 0.95** | **0.38 / 0.54** | 0.32 / 0.53 | 0.34 / 0.47 |
| never background, CLIP < 0.8 | 0.60 / 0.25 | 0.52 / 0.24 | 0.63 / 0.26 |

- **CLIP generalises:** 0.82–0.84 off the study set vs 0.86 on it. The original prompt is the best; prompt ensembling does not fix background words.
- **Excluding background words is better than tightening the threshold** (at recall ≈ 0.3: 0.44 vs 0.34 precision). "Never background, < 0.95" is at least as good as the current rule on both axes in every split.

## Session 2026-10-05 (no GPU at first, then L40S job 32263440 on gpgpu183, `gpu-l40s-preempt`, preempted 16:07)

### Why CLIP misses hallucinations and why the crop veto drops some (offline, study set)

Images of crops: `/tmp/ayazdanparas_veto/` (login node only; yellow = cluster the crop is built from, blue = other top-30 patches).
- **Vetoed hallucinations** (CLIP rank_full < 0.9 but tight crop ≥ 0.95, 8 nouns): 5 look like **real objects AMBER does not list**
  (252 backpack = a real bag, truth has "bag"; 32 dog collar; 670 rope; 304 sun glare; 180 canola field "flowers"), 1 shared misreading
  (354 glass table → "chair"), 1 crop of empty sky (2 sun). So the veto is mostly right and the label is wrong.
- **Missed hallucinations** (rank_full ≥ 0.9, 72 nouns = 51 distinct (caption, lemma); 21 are repeats, because CLIP gives one score per word per image):
  13 not on AMBER's absent list (maybe present), 22 where CLIP also thinks the crop shows the word (misreading of a related real object:
  517 horse → cow, 428 wild sheep → cows, 67 shutter → bench, 616 glass door → window), 16 where the crop does not show the word (scene guess,
  attention on empty sky / court: 526 kites → people, 206 tennis → ball, 312 laptop → desk).

### CLIP over the whole vocabulary (`rag/clip_relation/score_one_image.py`, `one_image/`)

The CLIP rank is `share of the 418 vocabulary words with a lower cosine(image, "a photo of a {w}.")`; only the caption's nouns were saved before.
Vocabulary = AMBER `relation.json`: 340 object words + 78 association words (study lemmas add nothing). Not in it: lime, goat, kayak, shutter.
517: **horse 1st (0.279)**, car 0.233, trough 0.224, **cow 4th (0.221)**. Over 11 missed words, a related rival beats the word in 5 (517 horse > cow,
428 sheep > cow, 470 bench > table, 616 door > window, 634 ground > road); not in 163 (lemon < apple, lime missing), 67 (shutter missing), or the scene guesses.
The association words add junk near the top (163: "ginger" 1st for a red-haired woman; napery, riff, cann, rpoe, cathole).

### Related-rival rule with a WordNet vocabulary: negative on held-out data (`rag/clip_relation/wordnet_rival_check.py`, `wordnet_rival/`)

Vocabulary: WordNet physical-object nouns with ≥ 20 COCO train-caption mentions + AMBER's 340 = 2651 words (7240 with ≥ 1 mention).
Rule: flag the noun if a related word (Wu-Palmer ≥ t on WordNet senses) scores higher on CLIP; variants exclude subtypes (beagle for dog),
or subtypes + synonyms + broader words ("siblings"). Views: full image or the tight crop; optional margin.
- Tuning set (10 hand-picked CLIP misses + 20 clean grounded; "people" has no physical sense, so 9): best was siblings, crop, wup ≥ 0.80,
  margin > 0.05, sinks removed: **5/9 vs 0/20**.
- **Held-out (50 random hallucinated + 100 random grounded, seed 1, settings fixed in advance): 12/50 = 24% vs 5/100 = 5%.**
  Margin AUROC 0.667 vs CLIP rank_full 0.859 on the same nouns. Among nouns CLIP does not flag it catches 2/17 hallucinated for 4/88 grounded,
  and CLIP or rival gives 70% / 16% vs CLIP 66% / 12%. Most winning rivals are junk WordNet words (setup, medium, drape, console, ornamental, ways, cave).
  **Dropped.** The genuine mix-ups (cow/sheep, apple/melon, table/bench) exist but are a small share.

### Attention sinks break the zoom crop (`hallu_study/zoom_miss_entropy.py`, `rag/clip_relation/crop_veto_nosink.py`)

The crop is built from `late_map[pos]`: attention of the masked position at its commit pass, image keys only, mean over 32 heads and layers 19–31,
then the connected top-30 cluster around the **single strongest patch**, ×1.5, ≥ 8 patches. When that patch is an attention sink the crop is empty background.
- Of 11 off-object crops seen by eye, 7 have a peak patch that is the **top patch for all 128 tokens** of the caption (function words included),
  holding 10–20% of image attention (100–600× a median patch) on a featureless spot (pixel std 1–3 in plain sky). Real-object peaks win for 20–40 tokens with ~1%.
- Off-object crops have **lower** per-head attention entropy (0.69 vs 0.74; proxy over 764 grounded nouns: lowest entropy quartile 24% miss vs 5–6%),
  and a cluster of 1–2 patches (all 11 misses vs 3/15 on-object). The sink concentrates attention, it does not spread it.
- Rule used: sink = patch that is the top patch for ≥ half of the caption's tokens; zero it before cropping. **A sink is present for 40% of study nouns
  (416/1050) and the crop moves for 34% (360).** Missed by the rule: the bottom-right corner patch (31,31) at 30–50 tokens, and second sinks (526).
- CLIP rank of the word on the tight crop: AUROC **0.765 → 0.834** with sinks removed (held-out 150: 0.736 → 0.824); for grounded nouns whose
  crop moved the median crop rank goes 0.96 → 0.99. The crop is now almost as good a detector as the full image (0.86).
- Effect on the CLIP + crop-veto rule is small: veto at 0.95 precision 0.66 → 0.67 at recall 0.48 (all 1050), 0.74 → 0.78 at 0.56 (held-out 150).

### Oracle + zoom with sink removal (`slot_refill_study.py --remove-sinks`, `rag/slot_refill/oracle_zoom_nosink/`, `oracle_zoom_l40s/`)

Online, sinks are taken from the commit pass itself: patches that are the top late-layer patch for ≥ half of the 128 answer positions.
**GPU drift:** on the L40S, 60/72 captions with no trigger differ from the A100 baseline (A100 run: 0/56), so L40S runs only compare with L40S runs.
The L40S control without sink removal was preempted at 88/144 (all 84 hallucinating, 4 clean); it resumes with the same command.

| 84 hallucinating captions | GPU | triggers | flagged word changed |
|---|---|---|---|
| oracle + mask only (plain refill) | A100 | 167 | 41 (25%) |
| oracle + zoom | A100 | 153 | 62 (41%) |
| oracle + zoom | L40S | 133 | 45 (34%) |
| oracle + zoom, sinks removed | L40S | 131 | 46 (35%) |

Paired first triggers (identical state in both L40S runs, 67): crop unchanged 45 → 21 vs 21 changed (consistency check); **crop moved 22 → 9 vs 11 changed,
mean p(original) 0.39 vs 0.39**, mixed cases both ways. **Better crops do not make the refill fix more.**
Official scores of the sink-removed run (L40S, vs the A100 baseline, so they include drift): hallucinating 90 hallucinated nouns, CHAIR 14.1, Cover 44.2,
Hal 60.7, Cog 10.7; clean 1 new hallucinated noun, Cover 48.3. Scores of all four oracle runs on 84 hallucinating + 4 clean captions:
CPU job 32273725 (`score_l40s_partial.slurm`) → `rag/slot_refill/eval_l40s_partial/eval_report.json`:

| 84 hallucinating, official scorer | GPU | CHAIR | Cover | Hal | Cog | hallu nouns | grounded |
|---|---|---|---|---|---|---|---|
| baseline | A100 | 22.3 | 43.5 | 98.8 | 21.3 | 155 | 521 |
| oracle + mask only | A100 | 18.7 | 43.5 | 85.7 | 16.7 | 123 | 516 |
| oracle + zoom | A100 | 15.3 | 45.0 | 78.6 | 13.8 | 102 | 548 |
| oracle + zoom | L40S | 14.8 | 44.2 | 63.1 | 10.9 | 93 | 519 |
| oracle + zoom, sinks removed | L40S | 14.1 | 44.2 | 60.7 | 10.7 | 90 | 528 |

Same GPU: sink removal is −3 hallucinated / +9 grounded nouns, a small gain within single-run noise. The GPU alone moves oracle + zoom by 9 nouns (102 vs 93).
The 4 clean captions scored so far give the same 1 new hallucination (caption 99, "forest") in both L40S runs, i.e. drift, not sinks.

The 22 moved crops, redrawn from the logged boxes (`/tmp/ayazdanparas_veto/moved/`, login node):
- better crop, better word: 137 hill (grass → "grassy area"), 248 people (train on the bridge → "two cars");
- **crop now on the misread object, and the model repeats the misreading more confidently**: 559 toilet tank → "sink" (p 0.33 → 0.69),
  371 white basin → "white bathtub", 632 surface → "table" (0.54 → 0.70). A sharp view of a stably confused object reinforces the error;
- **the sink rule removed a real object**: 660 the head of the person behind the umbrella, 137 the girl, 580 the toast (crop moved to the plate rim → "fabric").
  "Top patch for ≥ half the tokens" also catches the photo's main subject. True sinks are featureless (pixel std 1–3); next: also require a near-uniform patch.
The online oracle also fires on non-noun uses of object words ("can be seen", "drawing"); harmless, the refill keeps them.

### Word-first zoom refill (`slot_refill_study.py --refill-order word-first`, A100 job 32261020 on gpgpu096, `rag/slot_refill/oracle_zoom_wordfirst/`)

The current zoom refill commits the remasked span (pos−1, pos, pos+1) most-confident-first. Word-first commits the flagged position first, as the
crop's top token with the span masked, then fills the neighbours. It saves one forward pass. 144 study captions, oracle trigger, all on the A100
against the saved A100 runs. Log: `rag/slot_refill/oracle_zoom_wordfirst.log`; scores: `eval_wordfirst/eval_report.json`;
analysis: `hallu_study/wordfirst_analysis.py`. Each event logs `first_pass`: the probability mass at the flagged position in the first refill pass
(crop, span masked; the same pass in both orders) on correct object words (AMBER truth + synonyms), wrong object words (incl. the original),
and neutral tokens, plus the best correct and best wrong word. The object vocabulary has 1035 tokens that are whole AMBER object words.

| 84 hallucinating, official scorer (A100) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded | fewer / same / more |
|---|---|---|---|---|---|---|---|
| baseline | 22.3 | 43.5 | 98.8 | 21.3 | 155 | 521 | |
| oracle + mask only | 18.7 | 43.5 | 85.7 | 16.7 | 123 | 516 | 24 / 59 / 1 |
| oracle + zoom | 15.3 | 45.0 | 78.6 | 13.8 | 102 | 548 | 34 / 47 / 3 |
| **oracle + zoom, word-first** | **14.2** | **45.8** | **76.2** | 14.0 | **94** | **552** | 37 / 45 / 2 |

The 60 clean captions are unchanged in all runs (0 hallucinations, Cover 49.5).

Refill outcomes (147 refills each; the verb "can" is excluded). Wrong = the span contains an object not in the image:

| | wrong, same word | wrong, other word | neutral | correct |
|---|---|---|---|---|
| current order | 90 (61%) | 16 (11%) | 28 (19%) | 13 (9%) |
| word-first | 81 (55%) | 16 (11%) | 34 (23%) | 16 (11%) |

First refill pass at the flagged position (the answer to "what probability does the correct / neutral token get?"):

| final outcome | n | correct mass mean (median) | neutral | wrong | p(original word) |
|---|---|---|---|---|---|
| all | 147 | 0.07 (0.02) | 0.49 (0.47) | 0.43 (0.44) | 0.32 |
| wrong, same word | 81 | 0.04 (0.02) | 0.36 | 0.60 | 0.53 |
| wrong, other word | 16 | 0.05 (0.02) | 0.48 | 0.47 | 0.06 |
| neutral | 34 | 0.04 (0.01) | 0.82 | 0.14 | 0.08 |
| correct | 16 | 0.34 (0.32) | 0.48 | 0.18 | 0.06 |

- **The model rarely sees the right object in the crop.** Correct-object mass is ≥ 0.1 in 28/147 first passes, ≥ 0.3 in 12, and ≥ 0.5 in 3.
  The best correct word beats the original in 33/147. The crop's top word is the original hallucination in 76/147 (52%).
- Neutral words get about half the mass. The refill can usually avoid the error (neutral), but it can rarely correct it.
- Where the refill ends correct, the correct mass was already there (0.34). The order of commits does not create it.
- Paired first triggers (82 identical states; the text differs in 30): **7 better** (wrong → neutral 4, wrong → correct 2, neutral → correct 1),
  **4 worse** (251 "snowy slope." → "mountain.", 79 → "a clear sky", 32 "harness" → "collar", 60 "cat watches", where the verb "watches" is a labeller
  artefact), and 19 sideways.
- Watch: the crop's top token at the noun slot is sometimes "." (206, 251, 635). The neighbour then becomes `<|endoftext|>`, which ends the caption
  and in effect deletes the noun. Caption length was unaffected here (66.8 → 67.0 words), but check this on the full set.
- **Conclusion:** word-first is a small, consistent gain (−8 hallucinated / +4 grounded nouns at the same cost), within single-run noise but in the
  same direction on every metric except Cog. The remaining failures come from the model reading the crop as the wrong object and from scene guesses,
  where there is no correct object to find. The next check is the full 1004 on the A100 against `oracle_zoom_full` (523 → 314), about 3 h.

**What the wrong refills look like** (`hallu_study/wordfirst_wrong_examples.py` → `rag/slot_refill/wordfirst_wrong_examples.txt`, all 97;
crops drawn by `wordfirst_wrong_crops.py` → `figures/wordfirst_wrong_{misread,scene,absence,labels}.png`, 26 hand-picked).
Of the 26 drawn, by eye:
- **Off-object crop, the word comes from the text (9):** the crop is blank sky / sea / wall / plate rim / desk, yet the word gets 0.6–0.97
  (2 sun on empty sky, 720 sun on sea, 67 "phone" on the wooden wall, 371 "cup" on an empty wall corner, 541 "table" on the plate rim,
  587 "mouse" on the desk, 312 "table" on a keyboard corner). Here the crop gives no evidence, so the text context decides the word.
- **Real misreading of the object in the crop (4–5):** 517 a clean horse → cow 0.59 (horse 0.35); 428 horned wild sheep → cows 0.74 (sheep 0.00);
  407 a mini-keyboard remote → mouse 0.93; 130 white slide water → beach 0.81. A sharp view does not change these readings.
- **The model is right and the scorer counts it (12):** absence phrases (332 / 720 "no people" on empty sand or sea, 144 "other people or" next to a lone dog,
  430 "benches for people to sit", 670, 28); idioms (690 "bird's eye view", 526 "closer to the ground"); the crop shows the named object
  (289 the street lamp, 252 a bag → "backpack", 84 a shampoo bottle → "soap", 32 the dog's collar, 304 sun glare on the truck, 634 "dirt road" on dirt ground).
- The refills also leave broken grammar, which the scorer does not see: 67 "wearing against a wooden wall", "situated in a of a building";
  371 "close a bit,,".

### Detection: CLIP on the sink-free crop (offline, `rag/clip_relation/detect_nosink_crop.py` → `detect_nosink_crop.txt`)

All 1050 study nouns from `crop_veto_nosink.csv`. "Full-mix precision" re-weights the nouns to the full AMBER-g mix (285 hallucinating + 719 clean
captions); it gives a base rate of 0.077, against the true 0.078. The thresholds are swept on the study set.

| detector | AUROC | flags | caught / 155 | precision | full-mix precision |
|---|---|---|---|---|---|
| full image < 0.9 | 0.860 | 148 | 83 | 0.56 | 0.37 |
| tight crop < 0.9 (with sinks) | 0.765 | 293 | 83 | 0.28 | 0.17 |
| tight crop, no sink < 0.9 | 0.834 | 225 | 85 | 0.38 | 0.24 |
| mean(full, crop no sink) < 0.9 | 0.860 | 210 | 100 | 0.48 | 0.30 |
| full < 0.9 AND crop no sink < 0.95 (veto) | | 110 | 74 | 0.67 | 0.47 |
| never background, full < 0.95 AND crop no sink < 0.95 | | 102 | 77 | 0.75 | 0.57 |

- **Sink removal repairs the crop as a detector** (0.765 → 0.834; on the 360 nouns whose crop moved, 0.662 → 0.847), but it stays below the full image
  at every recall (full-mix precision at recall 0.5: 0.26 vs 0.38), and combining the two does not raise the AUROC (mean 0.860 = full alone).
- **It does not find more hallucinations.** Of the 72 the full image misses, the crop no sink < 0.8 catches 22 for 56 grounded flags.
  Its use is as a **veto**: it removes 29 of the full image's 65 false flags (65 → 36) for 9 hallucinations lost (83 → 74). That matches the earlier 0.66 → 0.67.
- The 45 nouns both views miss are the same kinds as the failed refills: misreadings (517 cow ×3, 428 cow, 470 table ×3, 67 bench),
  scene words (634 road ×5, 231 beach ×3 / sun, 699 sun), absence phrases (376 / 392 / 430 people), and 163 (drink, bowl, apple).
  Repeats make up a lot of them, because CLIP gives one score per word per image.
- Background exclusion was written after seeing the study flags. On held-out data, only the full-image rank has been checked (`full_clip/`, no crops).
- **Do false flags hurt under a one-word re-prediction?** (`flag_repredict.py` → `flag_repredict.txt`; top-1 from `repredict/repredict.csv`:
  the noun is masked in the finished baseline caption and re-predicted in one pass.) Crop no sink < 0.8 gives 76 false positives. With the full image,
  68 come back as the same word, 5 as another correct object (laptop → desk, basketball → ball), 3 as neutral, and 0 as a wrong object.
  With the crop (which still has sinks), the counts are 66 / 4 / 5 / 1. At < 0.9 (140 false positives) the pattern is the same, with 0–1 wrong.
  **So one-word re-prediction barely hurts, but it barely fixes either.** 56 of the 62 flagged hallucinations come back as the same word with the full image,
  and 46–50 with neighbours masked or with the crop. Online it was very different: clip+cropveto+zoom changed 221 of 435 grounded flags on the full set.
  That refill masks 3 tokens during decoding, and the rest of the caption is then decoded after it.
  Not yet tested: re-prediction with the sink-free crop as the input image. That needs the GPU.

**Oracle + zoom, word-first + sink removal** (A100, `rag/slot_refill/oracle_zoom_wordfirst_nosink/`, `eval_wordfirst_nosink/eval_report.json`):
84 hallucinating captions: CHAIR 13.1, Cover 44.6, Hal 71.4, Cog 13.1, **86 hallucinated nouns** (word-first alone: 94; zoom: 102; baseline: 155), 546 grounded (552).
Per caption vs baseline: 42 fewer, 40 same, 2 more. The 60 clean captions are unchanged.
Analysis: `hallu_study/wordfirst_sink_analysis.py` → `rag/slot_refill/wordfirst_sink_analysis.txt`.
- Refill outcomes (142 refills, verb "can" excluded): wrong, same word 79 (56%); wrong, other word 11 (8%); neutral 34 (24%); correct 18 (13%).
  Word-first alone: 55 / 11 / 23 / 11%. A sink was present in the commit pass for 77 / 142 refills.
- Paired first triggers (82): a sink was present in 46 and the crop moved in 33. Where the crop did not move, the text was identical in all 49 (A100 deterministic).
  Of the 33 moved crops the refill differs in 14: **6 better** (84 faucet → "running water", 248 people → "cars", 342 building → "fence",
  614 person → "street", 137 and 433 → neutral), **4 worse** (371 wall → "bathtub": the crop moved onto the misread basin again; 316 "visitors" → "people";
  559 "shower" → "sink"; 413 "computer" → "close-up"), and 4 sideways.
- When the crop moves off a blank patch onto content, the word often stays: 720 sun 0.96 → 0.97 (sea → beach), 67 bench (wall → man).
  These words come from the text, not the crop.
- **Grey-level std does not detect blank crops.** The crops that look empty by eye have std 6–65 (541 plate rim 35, 587 desk 56, 312 keyboard edge 65),
  and the crops that show an object have 21–70. "Blank" means *no object*, not *no texture*. Only 13 crops are below 12, and they end like the others.

**Complete A100 2×2, oracle + zoom: order × sink removal** (`eval_a100_2x2/eval_report.json`; new run `rag/slot_refill/oracle_zoom_nosink_a100/`, current order + `--remove-sinks`):

| 84 hallucinating captions (A100) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded | fewer / same / more |
|---|---|---|---|---|---|---|---|
| baseline | 22.3 | 43.5 | 98.8 | 21.3 | 155 | 521 | |
| oracle + mask only | 18.7 | 43.5 | 85.7 | 16.7 | 123 | 516 | 24 / 59 / 1 |
| zoom, current order | 15.3 | 45.0 | 78.6 | 13.8 | 102 | 548 | 34 / 47 / 3 |
| zoom, current order, sinks removed | 15.2 | 43.7 | 75.0 | 14.0 | 101 | 542 | 37 / 43 / 4 |
| zoom, word-first | 14.2 | 45.8 | 76.2 | 14.0 | 94 | 552 | 37 / 45 / 2 |
| zoom, word-first, sinks removed | **13.1** | 44.6 | **71.4** | **13.1** | **86** | 546 | 42 / 40 / 2 |

The 60 clean captions are unchanged in all runs (0 hallucinations, Cover 49.5). Sink removal alone: −1 hallucinated noun, −6 grounded (current order);
with word-first: −8 / −6. Word-first alone: −8 / +4. Only the combination improves CHAIR, Hal and Cog together. Each cell is a single run,
and differences of this size are within noise (the L40S showed a 9-noun drift).

**Wider remask span** (`--span-radius r`, new option, default 1, in `rag_slot_decoding.decode`; oracle + zoom + word-first + sinks removed, A100,
first 30 hallucinating ids, `rag/slot_refill/span_test/`, scored on those 30 against the r = 1 run):

| 30 hallucinating captions | CHAIR | Cover | Hal | Cog | hallu nouns | grounded | words | triggers / text changed / mean span |
|---|---|---|---|---|---|---|---|---|
| baseline | 24.1 | 45.0 | 96.7 | 20.9 | 57 | 173 | 61.6 | |
| r = 1 (3 tokens) | 13.1 | 47.3 | 70.0 | 13.5 | 29 | 183 | 65.7 | 50 / 28 / 2.7 |
| r = 2 (5 tokens) | 10.9 | **50.3** | 63.3 | 12.8 | 24 | **190** | 65.4 | 50 / 36 / 4.0 |
| r = 5 (11 tokens) | **9.8** | 49.7 | **50.0** | **9.5** | **21** | 185 | 64.5 | 62 / 53 / 7.6 |

- The wider span rewrites the phrase, not just the noun: "the sun shining" → "a sunny day, and the players";
  "a man sitting on a bench bench" → "on benches, leaning"; "the beach is relatively empty of people" → "seems to be empty, with no".
- Concerns:
  - Some rewrites are wording that avoids AMBER words, not grounding: "no other people or vehicles visible" → "no other skateboarders or pedestrians".
  - Occasional broken grammar: 84 "washing his face He is standing".
  - The oracle also fires on the verb "can" (47 "the ocean can be seen" → "there is a body of"), so an 11-token span rewrites correct text.
  - r = 5 rewrites the text in 53 of 62 triggers, against 28 of 50 at r = 1. That is more perturbation, which on selected captions helps by itself.
- 30 captions, single runs, oracle trigger. To check: all 84 + 60 clean; a matched random trigger at r = 2 / 5; and with CLIP, where false flags
  would rewrite 5–11 correct tokens.

### CLIP margin instead of rank (`rag/clip_relation/clip_margin.py` → `clip_margin.csv`, `clip_margin_analysis.py` → `.txt`)

margin = cosine(image, best vocabulary word) − cosine(image, noun), on the full image / the crop / the sink-free crop (boxes from `crop_veto_nosink.csv`;
ranks reproduce the saved ones). The hypothesis: a hallucinated noun is far below the top word, while a grounded noun is beaten only by a synonym
or a co-present object, so its gap is small.
- **Partly true.** Hallucinated nouns are almost never top-1 (1–3% vs 26–31% of grounded nouns), and their median margin is 0.11–0.13 vs 0.03–0.04.
- **But the margin is no better than the rank.** AUROC: full 0.838 vs rank 0.860; crop no sink 0.828 vs 0.834. Operating points match too:
  crop no sink margin > 0.1 catches 93 for 140 false positives, against rank < 0.9 at 85 / 140. Margin > 0.04 catches 145 (94%) for 400 false positives.
- **Why:** grounded nouns that are not top-1 lose by a median of 0.06–0.07, not by a small amount. The winner is usually another object in the image
  (546 / 625 on the full image), but it is the *salient* one: "beach" loses to "dog" in a photo of a dog on a beach. Hallucinated nouns also mostly lose
  to an object that is in the image (129 / 150): the real object they are a misreading of. So "winner is in the image" does not separate the groups either.
- Junk association words in the vocabulary win often: doghole beats dog 11 times on the sink-free crop, and cann, cathole and leave also appear.
  The only hallucinations that look grounded (margin ≤ 0.01) are 196 bush (to "cann"), 231 sun (top-1), and 470 table (to bench, 0.008).

### Vocabulary size and caption-aware rank: no real gain (`clip_vocab_embed.py` → `clip_vocab_embed.npz`; `clip_vocab_compare.py`, `clip_caption_exclude.py` → `.txt`)

CLIP embeddings of the 3 views × 1050 nouns and of 2700 words ("a photo of a {w}.") were saved once on the A100, so any vocabulary can be scored offline.
Vocabularies: current 418, AMBER main words 360, and WordNet physical objects with ≥ 1000 / ≥ 200 / ≥ 20 COCO mentions (587 / 1113 / 2654). Each includes the study lemmas.
`_syn` drops the noun's near-synonyms (spaCy > 0.8, WordNet synonyms and direct hypernyms / hyponyms); `_caption` also drops the caption's other object nouns
and their synonyms. Recall is reported at a fixed false-positive rate (10 / 20 / 30% of the 895 grounded nouns), which is fair across vocabulary sizes.

| | full image AUROC [recall at FP 10/20/30%] | crop no sink AUROC [recall] |
|---|---|---|
| rank, current 418 | 0.860 [0.59 0.73 0.88] | 0.834 [0.46 0.72 0.85] |
| rank, AMBER 360 | 0.856 [0.59 0.77 0.88] | 0.832 [0.47 0.71 0.85] |
| rank, WordNet 2654 | 0.860 [0.60 0.72 0.87] | 0.831 [0.43 0.67 0.84] |
| rank, current, synonyms removed | 0.859 [0.59 0.73 0.88] | 0.835 [0.46 0.71 0.85] |
| **rank, current, synonyms + caption nouns removed** | **0.868 [0.59 0.78 0.91]** | 0.837 [0.46 0.71 0.87] |
| margin, current, synonyms + caption nouns removed | 0.847 [0.53 0.75 0.86] | 0.836 [0.45 0.68 0.86] |

- **The vocabulary does not matter:** all five give AUROC within 0.005 on every view. The best-matching word is a real object, so it is already in AMBER's 340.
- Synonym removal does nothing. The winners over grounded nouns are the main object ("beach" → "dog"), or subtypes beyond one WordNet step
  (dog → retriever / labrador, umbrella → parasol, boat → rowboat), which only exist in the big vocabulary.
- Caption-aware rank is the only gain: full image 0.860 → 0.868, recall at 20% FP 0.73 → 0.78 (about +8 hallucinations for the same 179 false positives).
  This is within the CI (0.81–0.90) and does not hold with the WordNet vocabulary (0.861). The CLIP signal looks saturated at about 0.86 for these nouns.

**Caption-aware full image + sink-free crop as the second check** (`caption_aware_veto.py` → `caption_aware_veto.txt`; current vocabulary, synonyms removed;
the thresholds are swept on the study set, so the numbers are optimistic). Flag if full caption-aware rank < t1 AND crop no sink rank < t2.
Caught (of 155) / false positives (of 895), against the best earlier rule at about the same recall:

| recall | earlier best | caption-aware full + crop no sink |
|---|---|---|
| ~45–50% | full < 0.9 AND crop < 0.95: 74 / 36 | t1 0.9, t2 0.95: 72 / 33 |
| ~50%, never background | full < 0.95 AND crop < 0.95: 77 / 25 | t1 0.95, t2 0.95: 75 / 20; t1 0.9, t2 0.95: **68 / 11** |
| ~66% | full < 0.95: 103 / 131 | t1 0.97, t2 0.97: **104 / 109** |
| ~80% | full caption-aware alone: 121 / 179 | t1 0.99, t2 0.97: **125 / 160** |
| ~85% | full not in top 10: 131 / 243 | t1 0.99, t2 0.98: **131 / 185** (crop caption-aware: 130 / 171) |
| ~90% | full caption-aware alone: 141 / 268 | t1 0.99, t2 0.99: **138 / 213** (crop caption-aware: 137 / 204) |
| ~92–93% | crop no sink not in top 5: 144 / 349 | t1 0.99, t2 not top-1: **142 / 250** |

- **At high recall the second check helps:** 20–100 fewer false positives at 66–93% recall. At ~50% recall there is no gain over the earlier veto.
- The lowest false-positive point is never-background + caption-aware < 0.9 + crop < 0.95: 68 caught for 11 false positives (86% of flags real),
  but the background list was written after seeing the study flags.

### Full-set run: high-recall CLIP trigger, span ±1 and ±2 (2026-10-05 21:00 → 2026-10-06 03:18, COMPLETED)

`run_highrecall_full.sh`, in tmux session **`highrecall-full` on spartan-login3**, as step 32261020.11 of the A100 job 32261020 (gpgpu096).
Log: `results/amber_g/highrecall_full/run.log`. Outputs: `highrecall_full/span_r{1,2}/predictions.json` (+ `_events.json`, `amber_g_metrics.txt`).
Resumable: rerun the same command. Expected ~3–3.5 h per span, ~7 h in total.
- **Trigger** (new `slot_refill_study.py` options `--caption-aware --crop-not-top1`):
  flag if the full-image CLIP rank < 0.99, with the word's synonyms and the object words already committed in the caption (and their synonyms)
  removed from the vocabulary. Keep the flag unless the word is the top remaining word on its sink-free tight crop.
  Offline study-set operating point: 92% recall, 28% of grounded nouns flagged.
- **Refill:** zoom, word-first, sinks removed, `--span-radius` 1 then 2. `decode` gained an optional `state` argument (live sequence for the trigger).
- **Smoke test** (10 hallucinating study captions, `rag/slot_refill/smoke_highrecall/`): it fires on 52% of eligible words with precision 0.32,
  and the crop vetoes only 1 / 38. Many false flags are people words ("man" rank 0.98: "person" etc. beat it).
  Some refills damage the text ("The man of" → "The style of"). On the full set (72% clean captions) expect precision ~0.15.
  Online "caption objects" are those committed so far, not the finished caption, and the online vocabulary also contains all AMBER object words.
- To compare against: baseline 523 / 6141, clip+cropveto+zoom 503 / 6054, oracle+zoom 314 / 6229. There is no matched random control yet.

**Result** (step 32261020.11 COMPLETED 03:18 on 2026-10-06, 6h18m, exit 0; `span_r{1,2}/amber_g_metrics.txt`, paired counts in
`span_r{1,2}/final_eval/partial_report.json`, noun diffs in `span_r{1,2}/noun_diff.json`):

| full AMBER-g (1004) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded nouns |
|---|---|---|---|---|---|---|
| baseline | 7.5 | 48.4 | 28.4 | 2.3 | 523 | 6141 |
| clip+cropveto+zoom | 7.4 | 47.1 | 28.1 | 2.5 | 503 | 6054 |
| high-recall, span ±1 | 7.0 | 45.7 | 27.9 | 2.5 | 458 | 5783 |
| **high-recall, span ±2** | **6.0** | 45.0 | **22.7** | **2.0** | **376** | 5654 |
| oracle+zoom | 4.6 | 48.9 | 20.4 | 1.8 | 314 | 6229 |

| | span ±1 | span ±2 |
|---|---|---|
| hallucinating 285: hallucinated nouns 523 → | 333 | 271 |
| clean 719: new hallucinated nouns (captions) | 125 (91) | 105 (70) |
| per caption fewer / same / more | 143 / 741 / 120 | 169 / 740 / 95 |
| fired / eligible | 2726 / 7104 (38%) | 2771 / 7099 (39%) |
| precision (online oracle) | 0.21 | 0.19 |
| changed hallucinated / grounded words | 237 / 992 | 329 / 1276 |
| vetoed (hallucinated) | 396 (9) | 408 (10) |

- **Span ±2 is the first non-oracle setting that clearly beats baseline:** −147 hallucinated nouns (−28%) and better CHAIR, Hal and Cog together.
  The cost is −487 grounded nouns (−8%) and Cover −3.4. The clean captions get 105 new hallucinations.
- **No matched random control yet.** At a 39% fire rate with a 5-token span, much of the gain may be perturbation (as on the study set). This is the next run.
- False flags (span ±1, `span_r1/grounded_flag_outcomes.txt`, 2165 grounded flags): at the flagged position the refill keeps the same word 54%,
  gives another correct object 9%, neutral 32-34%, a wrong object 3-4%. Commonest swaps: man → person, trees → "a", table → ",".
  `grounded_flag_context.txt`: when the lemma is lost, the first refill pass already put little mass on it (median p(original) 0.05, neutral mass 0.74).

### Noun-constrained refill on top of span ±2 (`--noun-trigger --refill-vocab noun`): worse, cancelled

`--noun-trigger` fires only on object words used as nouns in context; `--refill-vocab noun` restricts the flagged position's refill to nouns.
Test on ids 1–100 (`highrecall_full/noun_test/`, `compare_noun_refill.txt`): hallucinated nouns 61 → 37 (span ±2) vs 38 (+ noun), grounded 569 → 529 vs 544.
Full run: `run_noun_full.sh` (seeded with the 100 test captions), step 32261020.20 from 13:08 on 2026-10-06, output `highrecall_full/span_r2_noun/`,
log `span_r2_noun_run.log`. **Cancelled at 746/1004 (15:09) by the user's decision, because it was worse.** Snapshot at ids 1–682
(`span_r2_noun_partial.py` → `span_r2_noun/partial_eval/report_1458.txt`; the 387-caption snapshot is `report.txt`):

| ids 1–682 | CHAIR | Cover | Hal | Cog | hallu nouns | grounded | hallucinating 202: removed / added | clean 480: new (captions) |
|---|---|---|---|---|---|---|---|---|
| baseline | 7.9 | 46.7 | 29.6 | 2.5 | 375 | 4217 | | |
| span ±2 | **6.1** | 43.1 | **23.8** | **2.2** | **263** | 3868 | 263 / 77 | 74 (53) |
| span ±2 + noun | 7.2 | 43.2 | 26.0 | 2.6 | 319 | 3910 | 247 / 97 | 94 (58) |

Refill at the flagged position (same word / another correct object / neutral noun / neutral non-noun / wrong object):
- span ±2: grounded flags (1624) 43 / 9 / 10 / 34 / 3%; hallucinated flags (372) 36 / 8 / 13 / 38 / 5%.
- span ±2 + noun: grounded flags (1408) 58 / 15 / 20 / 0 / 6%; hallucinated flags (329) 50 / 11 / 29 / 0 / 10%.

- **The noun constraint halves the gain** (−56 vs −112 hallucinated nouns) and saves almost no Cover (43.2 vs 43.1, +42 grounded nouns).
  At 387 captions the gap was 16 nouns; at 682 it was 56.
- Why: in plain span ±2, 38% of flagged hallucinations become a non-noun (e.g. "a", ","). That is what removes them. When forced to a noun,
  the slot gets the same hallucination back (50% vs 36%) or a wrong object (10% vs 5%), and clean captions gain 20 more hallucinations.

### Lower-false-positive detector on ids 1–100: helps (2026-10-06, `highrecall_full/precise_test/`)

Detector A = caption-aware full-image CLIP rank < 0.97 **AND** caption-aware rank on the sink-free tight crop < 0.97
(`--clip-threshold 0.97 --caption-aware --crop-veto 0.97 --remove-sinks`; no code change). Offline study-set point from `caption_aware_veto.txt`:
103/155 caught for 103/895 false positives, vs the high-recall detector's 142 / 264. Refill as span_r2: zoom, word-first, sinks removed, no noun options.
Ids 1–100 (`noun_test/ids_1_100.json`), same A100 job 32261020 as span_r2, so the captions are directly comparable. Script `run_precise_test.sh`
(steps .22 and .24), log `run.log`, outputs `A/`, `A3/`. Scores: `compare_precise.py [A A3]` → `eval/report_A_100.txt`, `eval/report_A_A3_100.txt`.
A stricter detector B (0.90 AND crop 0.95; offline 70 / 32) was stopped at 2/100 to make room for A with span ±3; `B/` is a 2-caption stub.

| ids 1–100 (35 hallucinating, 65 clean) | CHAIR | Cover | Hal | Cog | hallu nouns | grounded | hallucinating: removed / added | clean: new (captions) | words |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 9.5 | 45.6 | 35.0 | 2.7 | 61 | 569 | | | 6235 |
| span ±2, high-recall detector (span_r2) | 6.4 | 42.4 | 24.0 | 1.8 | 37 | 529 | 42 / 9 | 9 (5) | 6109 |
| **span ±2, detector A** | 6.6 | **45.3** | 27.0 | 2.0 | 41 | **568** | 31 / 5 | 6 (4) | 6313 |
| span ±3, detector A | 6.4 | 44.4 | 26.0 | 1.4 | 40 | 574 | 34 / 8 | 5 (3) | 6372 |

| detector | fired / eligible | precision | flags on hallucinated / grounded |
|---|---|---|---|
| high-recall, span ±2 | 288 / 661 (44%) | 0.18 | 53 / 235 |
| A, span ±2 | 113 / 659 (17%) | 0.35 | 40 / 73 |
| A, span ±3 | 108 / 661 (16%) | 0.37 | 40 / 68 |

Refill at the flagged position (same word / another correct object / neutral noun / neutral non-noun / wrong object):
- A ±2: grounded flags 41 / 11 / 11 / 32 / 5%; hallucinated flags 18 / 8 / 18 / 57 / 0%.
- A ±3: grounded flags 29 / 15 / 10 / 44 / 1%; hallucinated flags 15 / 2 / 20 / 62 / 0%.

- **Detector A keeps 83% of the hallucination reduction (−20 vs −24) for almost no grounded loss (−1 vs −40); Cover 45.3 vs 42.4 (baseline 45.6).**
  False flags drop by 69% (235 → 73) and precision doubles (0.18 → 0.35). At 78 captions it was the same picture (−12 vs −15, grounded −12 vs −34).
- **Span ±3 vs ±2 with detector A is a wash:** −1 hallucinated noun, CHAIR / Hal / Cog slightly better (Cog 1.4 vs 2.0), Cover −0.9, and more rewriting
  (a flagged correct word is kept 29% vs 41%). Within noise; ±2 stays the default.
- Caveats: 100 captions and 61 hallucinated nouns, single runs; the A thresholds were picked on the study set, which overlaps ids 1–100.
- Next: detector A + span ±2 on all 1004 (~3 h: the 100 captions took 18–19 min), with a matched random+zoom control at ~17%.

## Practical notes

- On the login node `module load GCCcore/11.3.0 Python/3.11.3` did not work from a non-login shell.
  `virtualenv/bin/python` then fails on `libpython3.11.so`. CPU-only analysis of the CSVs works with system
  `python3` + numpy.
- `hallu_study/trigger_estimate.py` overwrites `attention_threshold.json` when run.
- **User preference (2026-10-02):** run almost everything (analysis, scoring, `eval_remask.py`, quick Python checks) inside the user's running Jupyter
  job via `srun --jobid=<jupyter job> --overlap --ntasks=1 bash -c 'module load GCCcore/11.3.0 Python/3.11.3; source virtualenv/bin/activate; …'`,
  not on the login node. Long experiments go in their own `sbatch` job, not the Jupyter GPU. Find the Jupyter job with `squeue -u $USER` (job name `mdlm.slurm`).
- **New account uom00092 (2026-10-07):** same QOS as punim2198 (normal, publicgpu, gpgpudeeplearn, feit, publiccpu) and the same account-level
  raw share, but no usage yet, so fair-share 1.0 against 0.064 for punim2198. Submit with `sbatch -A uom00092 …`; the command-line flag overrides the
  `#SBATCH -A punim2198` line in `mdlm.slurm` / `mdlm_cpu.slurm`. Usage will lower it over time, so check `sshare -U -u $USER`.
- **GPU access (2026-10-05):** the project's fair-share is low (factor 0.065, project at ~3× its share), and gpu-a100 / deeplearn queues are full.
  `-p gpu-l40s-preempt --qos=publicgpu` (L40S 48 GB) is open to us and often has free GPUs, but jobs are **cancelled without warning** when the owners need
  the node, so only run resumable work there. `extremecfd` and `fos-gpu-l40s` are not available to punim2198.
- **Do not mix GPU types within a comparison.** Greedy MMaDA decoding drifts between A100 and L40S (bf16 kernels), so most captions change.
- The AMBER scorer (`eval_remask.py`) needs no GPU, only spaCy from the virtualenv: CPU partition `sapphire` (cascade is being retired).
- **User preference (2026-10-05): nothing on the login node, not even light CPU analysis.** CPU work goes in a CPU Jupyter job,
  `sbatch mdlm_cpu.slurm` (sapphire, 2 CPUs, 16 GB, 8 h, port 8890), via `srun --jobid=<job> --overlap --ntasks=1 bash -c '...'`.
  Figures go under `results/amber_g/hallu_study/figures/`, not `/tmp` (compute nodes do not see the login node's `/tmp`).
