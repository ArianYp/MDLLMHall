# Hallucination in a masked-diffusion multimodal LLM: detect and correct during decoding

MMaDA-8B on AMBER generative captioning. Progress update, 2026-10-06.

---

## 1. Question and setup

**Can we stop a diffusion VLM from hallucinating objects, by catching a wrong object word when it is committed and refilling it?**

- **Model:** `Gen-Verse/MMaDA-8B-MixCoT` (masked diffusion LLM + MAGVIT-v2 image tokens, 512 px = 1024 image tokens).
- **Decoding:** greedy, 128 new tokens, 64 steps, 4 semi-autoregressive blocks of 32. Each step commits the 2 most confident masked tokens.
  So every word has a commit step and can be *remasked* and refilled, which a left-to-right model cannot do.
- **Benchmark:** AMBER generative (1004 images), with the official scorer.
  - CHAIR: % of object mentions that are hallucinated (↓).
  - Cover: % of true objects mentioned (↑).
  - Hal: % of captions with ≥ 1 hallucination (↓).
  - Cog: overlap with human-like hallucinations (↓).
- **Baseline, full set:** CHAIR 7.5, Cover 48.4, Hal 28.4, Cog 2.3. 523 hallucinated and 6141 grounded nouns. 285 / 1004 captions hallucinate.
- **Study set:** 144 captions, 84 that hallucinate at baseline (155 hallucinated nouns) and 60 clean ones (1050 object nouns in total).
  Used for traces, ablations and the detector study.

---

## 1b. The problem: MMaDA describes objects that are not there (AMBER 163)

![hallucination example](presentation/hallucination_163.png)

- 5 of the 11 object mentions are not in the image: **cup, drink, bowl, eggs, apples** (AMBER labels).
- They are not random inventions. Each is a **misreading of a real object** the model is looking at:
  - "a white **cup** … sipping a **drink**" → she holds a **lime** to her mouth;
  - "a **bowl** of **eggs**" → the **halved melon** with its seeds;
  - "green **apples**" → the **limes**.
- Attention confirms it: eggs and bowl attend to the melon, apples to the limes, cup and drink to the hand and the lime.
  Removing those patches kills "eggs" just as it kills the grounded "broccoli".
- Backup example: `presentation/hallucination_537.png` (two kites, no people → *"a **man** in a **woman** flying a kite"*).

---

## 2. What the hallucinations are (analysis of baseline traces)

Per-noun features at the commit step, AUROC for "hallucinated vs grounded" (1050 nouns, 155 hallucinated):

| signal | AUROC |
|---|---|
| image ablation: log p(word \| image) − log p(word \| no image) | 0.45 (useless) |
| decoder confidence at commit | 0.58 |
| attention concentration (top-5 patch share, layers 19–31) | 0.62 |
| re-prediction confidence in the finished caption | 0.68 |
| decoding order (position, first mention, filled in after neighbours), CV | 0.78 |
| **CLIP: rank of the word on the full image** | **0.86** |

- **Not the language prior.** 64% of hallucinated nouns are *raised* by the image (dlogp > 0). Removing the image barely separates the groups.
- **Confident misreadings of real objects.** The attention of a hallucinated noun is on a real object: caption 163's "eggs" and "bowl" attend to a half melon,
  "apples" to limes. Removing the attended patches kills "eggs" as it kills the grounded "broccoli".
- **Late and fill-in.** They are committed later and after their neighbours, and they are first mentions (a repeated word is rarely hallucinated).
- So the internal confidence of MMaDA is a weak detector. An external check against the image (CLIP) is much stronger.

---

## 3. First attempts, and the control that changed the evaluation

**Remask rule** (remask a token and its neighbours when p(no image) > p(image) and image attention is high):

| 84 hallucinating captions | hallucinated nouns (baseline 155) |
|---|---|
| rule (fires on 13.7% of tokens) | 121 |
| **random remask, same rate** | **119** |
| neighbour rule (5.5%) | 137 |
| **random, same rate** | **140** |

On the full set the rule is a wash: CHAIR 7.3 vs 7.5, Hal 28.7 vs 28.4. A confidence-gated zoom pass is slightly worse (CHAIR 7.6, Cover 47.3).

**Lesson: any perturbation "helps" captions that were selected for hallucinating** (regression to the mean). So from here on:
1. every method is compared with a **matched-rate random trigger**;
2. clean captions are always scored (does it add hallucinations?);
3. **detection and correction are separated**: an **oracle trigger** (AMBER's own labels, applied online) isolates the correction step.

---

## 4. The method: trigger → remask → refill

During decoding, when a committed token is a whole AMBER object word:
1. **Trigger:** oracle, CLIP or random decides whether it is suspicious.
2. **Remask** the word and its committed neighbours (pos−1, pos, pos+1).
3. **Refill** the span in a second pass:
   - *mask only*: same image, same prompt;
   - *zoom*: the attended region is cropped from the original photo (connected top-30 late-layer patches, 1.5×), upscaled and put in place of the image;
   - *word-first*: the crop decides only the flagged word, then the neighbours are refilled with the full image;
   - *sinks removed*: attention-sink patches are zeroed before cropping (see slide 9).
4. Decoding then continues normally, with at most one trigger per position.

In the refill the model sees the crop (or the image), the question and **all committed caption tokens on both sides**. Only the span and the
not-yet-decoded positions are masked. Cost: about 1.2 triggers per caption, about 3 extra forward passes, +11% time for zoom (CLIP check: +0.2%).

---

## 4b. Example: zooming in recovers a misread object

![dog → lion](presentation/zoom_967.png)

- When MMaDA writes "dog", its attention (red box) is on a **lioness** between safari trucks. At full resolution it writes *"a brown **dog** standing on a dirt road"*.
- Zoom refill: the word is masked and refilled with the crop in place of the image → *"a majestic **lion** standing on a dirt road"*. The rest of the sentence is unchanged.
- Same pattern in other captions (one hallucination, one zoom each):

| | baseline → after zooming in |
|---|---|
| `presentation/zoom_1001.png` | a game of **tennis** → a game of **volleyball** |
| `presentation/zoom_926.png` | people on a small **boat** → on a **paddle board** |
| `presentation/zoom_342.png` | leaning against a **building** → against a **fence** |
| `presentation/zoom_84.png` | washing his hands under a **faucet** → under running **water** |
| `presentation/zoom_503.png` | placed on a cardboard **box** → a cardboard **plate** |

- These hallucinations are low-resolution misreadings of a real attended object, and a closer look fixes them. (Trigger: oracle.)

---

## 5. Ablation A: the correction step (oracle trigger, same GPU)

Oracle trigger, so every trigger is on a real hallucination. All runs on the same A100 (GPU type changes greedy outputs).

| 84 hallucinating captions | CHAIR ↓ | Cover ↑ | Hal ↓ | Cog ↓ | hallu nouns | grounded nouns | captions better / same / worse |
|---|---|---|---|---|---|---|---|
| baseline | 22.3 | 43.5 | 98.8 | 21.3 | 155 | 521 | |
| mask only (same image) | 18.7 | 43.5 | 85.7 | 16.7 | 123 | 516 | 24 / 59 / 1 |
| zoom | 15.3 | 45.0 | 78.6 | 13.8 | 102 | 548 | 34 / 47 / 3 |
| zoom + sinks removed | 15.2 | 43.7 | 75.0 | 14.0 | 101 | 542 | 37 / 43 / 4 |
| zoom + word-first | 14.2 | **45.8** | 76.2 | 14.0 | 94 | **552** | 37 / 45 / 2 |
| **zoom + word-first + sinks removed** | **13.1** | 44.6 | **71.4** | **13.1** | **86** | 546 | **42 / 40 / 2** |

The 60 clean captions are unchanged in every oracle run (0 hallucinations, Cover 49.5).

| hallucinated nouns | sinks kept | sinks removed |
|---|---|---|
| current order | 102 | 101 |
| word-first | 94 | 86 |

- **The zoom crop is the main gain** (123 → 102, and Cover goes *up*).
- **Word-first adds −8, and sink removal only helps together with word-first** (−1 alone, −8 with word-first).
  In word-first the crop decides only the noun, so a better crop goes straight into the word.
- Single runs; differences under ~8 nouns are within noise (a GPU change alone moved zoom by 9).

---

## 6. Ablation B: detection (refill fixed), with matched random controls

Study set; the random trigger is matched to CLIP's fire rate (~16% of object words; precision 0.18 = the base rate).

| hallucinated nouns on 84 hallucinating captions (new ones on 60 clean) | random trigger | CLIP trigger (rank < 0.9) | oracle trigger |
|---|---|---|---|
| mask-only refill | 149 (+1) | 138 (+2) | 123 (0) |
| zoom refill | 130 (+4) | 132 (+1) | **102 (0)** |
| zoom + crop veto (CLIP must also reject the crop) | 135 (+3), at a matched 12% | 136 (**0**) | |

- **With the zoom refill, CLIP is no better than random on hallucinating captions** (132 vs 130). Most of the zoom gain there is perturbation:
  changing any word re-routes the rest of the caption.
- CLIP's advantage is **safety**: fewer new hallucinations on clean captions (0–1 vs 3–4).
- **Oracle ≫ CLIP:** the same refill gives 102 with perfect detection. **Detection is the bottleneck.**

---

## 7. Full AMBER-g (1004 captions)

| full set | CHAIR ↓ | Cover ↑ | Hal ↓ | Cog ↓ | hallu nouns | grounded nouns |
|---|---|---|---|---|---|---|
| baseline | 7.5 | 48.4 | 28.4 | 2.3 | 523 | 6141 |
| remask rule | 7.3 | 48.4 | 28.7 | 2.1 | | |
| zoom, confidence gate | 7.6 | 47.3 | 29.1 | 2.4 | | |
| CLIP + crop veto + zoom | 7.4 | 47.1 | 28.1 | 2.5 | 503 | 6054 |
| **oracle + zoom (upper bound)** | **4.6** | **48.9** | **20.4** | **1.8** | **314** | **6229** |

- **With perfect detection the refill removes 40% of hallucinated nouns and *adds* grounded ones.** With CLIP it removes 4%.
- Why CLIP fails on the full set:
  - **Recall:** it reaches under half the hallucinations and never touches 131 of the 285 hallucinating captions.
  - **False flags:** online precision falls from 0.69 (study) to 0.41 (full), because 72% of full-set captions are clean. 221 grounded words are changed,
    and 51 new hallucinations appear in clean captions.

---

## 8. Ablation C: detectors (offline, 1050 study nouns)

CLIP rank = share of a 418-word object vocabulary that scores below the noun ("a photo of a {w}."). Low rank ⇒ flag.

| detector | AUROC |
|---|---|
| CLIP, attended crop | 0.765 |
| CLIP, attended crop, **sinks removed** | 0.834 |
| CLIP, full image | 0.860 |
| CLIP, full image, ignoring objects the caption already names | 0.868 |
| CLIP full image + re-prediction confidence (CV) | 0.868 |
| CLIP full image, all 1004 captions / held-out ids 730–1004 | 0.830 / 0.837 |

What did **not** help (all within ±0.005 AUROC):
- **Bigger vocabulary:** AMBER 360 words, WordNet+COCO 587 / 1113 / 2654 words.
- **Removing synonyms** of the noun from the vocabulary.
- **Similarity margin** (best word − noun) instead of rank: 0.838 vs 0.860.
- **WordNet "related rival" rule:** 24% vs 5% on held-out nouns; mostly junk rivals.

Why: when a grounded noun loses, it loses to the photo's **main object** ("beach" < "dog"), not to a synonym, and often by a large margin.
When a hallucinated noun loses, it loses to **the real object it misreads** (horse > "cow"). **The CLIP signal saturates at about 0.86.**

---

## 9. Attention sinks

The crop is built around the single strongest attended patch. In 40% of nouns that patch is an **attention sink**: the top patch for most tokens of the
caption (function words included), on a featureless spot, holding 10–20% of the attention. The crop then shows empty sky or wall.

![sink example](presentation/sink_84.png)

Example (AMBER 84, *"… washing his hands under a **faucet**"*; there is no faucet, the boy pours water from a cup):
- The attention of "faucet" is mostly on his arms and the water (red). But its single strongest patch is the top of the shampoo bottle (cyan),
  and that patch is the most-attended patch for **126 of the caption's 128 tokens**.
- **With the sink**, the crop is the shampoo bottle → top word "soap" (0.46) → the refill writes "**soap** soap." (still wrong).
- **With the sink removed**, the crop moves onto the hands and the pouring water → "water" (0.45) → the refill writes "running **water**." (correct; water is in AMBER's truth).

- Rule: a sink is a patch that is the top patch for ≥ half of the caption's tokens. It is zeroed before cropping, and the crop moves for 34% of nouns.
- As a **detector**, the crop improves 0.765 → 0.834.
- As a **refill**, it helps only with word-first (slide 5). Some moved crops land on the misread object and reinforce the error
  (371 basin → "bathtub"). The rule also sometimes removes the photo's main subject.

---

## 9b. The CLIP check, illustrated

**Why it works (full image):** for each object noun, count how many vocabulary words CLIP finds a better description of the photo than the noun,
ignoring the noun's synonyms and objects the caption already names. A real object is near the top; a hallucination is beaten by many words.

![CLIP idea](presentation/clip_idea_372.png)

AMBER 372: a rowing boat on a lake in the rain. The six real objects are each beaten by 0–2 words; the caption's **"sun"** is beaten by **263**.
Flag if more than ~4 words beat the noun (rank < 0.99).

**Why the veto (sink-free crop):** a real but small object loses to the scene's main subject on the full image and gets flagged.
On the crop of the region the model looked at, it is the main subject. If it is CLIP's top word there, the flag is dropped.

![CLIP veto](presentation/clip_veto_218_84.png)

AMBER 218: a small car half-hidden behind a stop sign. On the full image 63 words beat "car" (sign, signal, barrier…) → flagged.
On the crop "car" is CLIP's top word → vetoed. (Backup: `presentation/clip_veto_449_29.png`, a coffee cup next to a box of donuts.)

---

## 10. High-recall detection: full image (caption-aware) + sink-free crop as a second check

Flag a noun only if both checks say it fits badly: the full image (caption's other objects ignored) and the sink-free crop. Best trade-offs from a grid:

| t1 full | t2 crop | hallucinations caught (of 155) | false positives (of 895) | share of flags that are real |
|---|---|---|---|---|
| 0.97 | 0.95 | 94–97 (61–63%) | 79–84 (9%) | 54% |
| 0.97 | 0.97 | 103–104 (67%) | 103–109 (12%) | 49–50% |
| 0.99 | 0.95 | 108–113 (70–73%) | 116–125 (13–14%) | 47–48% |
| 0.98 | 0.98 | 116–117 (75%) | 143–153 (16–17%) | 43–45% |
| 0.99 | 0.97 | 123–125 (79–81%) | 149–160 (17–18%) | 44–45% |
| **0.99** | **0.98** | **130–131 (84–85%)** | **171–185 (19–21%)** | 41–43% |
| 0.99 | 0.99 | 137–138 (88–89%) | 204–213 (23–24%) | 39–40% |
| 0.99 | not top-1 | 142 (92%) | 250–264 (28–30%) | 35–36% |

(Ranges: crop rank plain / caption-aware. 0.99 ≈ "more than ~4 words fit better"; 0.97 ≈ "more than ~12".)

- The crop as a second check removes 20–100 false positives at the same recall (e.g. 85% recall: 243 → 171–185).
- Low false-positive point: never flag background words, t1 0.9, t2 0.95 → **68 caught (44%) for 11 false positives (1.2%)**.
- Caveat: the thresholds and the background list are tuned on these same nouns, and on the full set the share of real flags is about half.

---

## 11. Can we afford false positives?

**One-word re-prediction** (mask the flagged noun in the finished caption, one pass): of 76 grounded nouns flagged by the sink-free crop,
68 come back unchanged, 5 as another correct object, 3 as a neutral word, and **0 as a wrong object**. So a false flag is nearly harmless.

**But the online 3-token refill is not harmless:** on the full set it changed 221 of 435 grounded flags (about half), because the neighbours and the rest of the caption
are re-decoded after it.

→ A high-recall detector (85–90%) is only usable with a refill that leaves correct words alone. That is the target for the refill design.

---

## 12. Why the refill still fails, even with the oracle

Refill outcomes (oracle + zoom, 147 refills, word-first run):

| | same wrong word again | another wrong word | neutral | correct object |
|---|---|---|---|---|
| zoom, current order | 61% | 11% | 19% | 9% |
| zoom, word-first | 55% | 11% | 23% | 11% |
| zoom, word-first + sinks removed | 56% | 8% | 24% | 13% |

**Probability in the first refill pass at the flagged position** (crop, span masked): correct objects 0.07 on average (median 0.02), neutral words 0.49,
wrong objects 0.43. The crop's top word is the original hallucination in 52% of refills. Correct objects get ≥ 0.3 in only 12 / 147.

**What the wrong refills are** (26 crops inspected by eye; `results/amber_g/hallu_study/figures/wordfirst_wrong_*.png`):

| type | n | example |
|---|---|---|
| off-object crop, the word comes from the text | 9 | blank sky → "sun" 0.35; bare wall → "phone" 0.94; plate rim → "table" |
| real misreading of the object in the crop | 4–5 | a clear horse → "cow" 0.59; horned wild sheep → "cows" 0.74 (sheep 0.00) |
| model right, scorer counts it | 12 | "no people" on empty sand; "bird's eye view"; "street lamps"; a bag → "backpack" |

- The model rarely sees the right object in the crop, so **avoiding the error (neutral) is realistic, correcting it is not**.
- Part of the residual "error" is AMBER scoring: absence phrases, idioms and compounds.

![misreadings](results/amber_g/hallu_study/figures/wordfirst_wrong_misread.png)

---

## 13. Summary

1. MMaDA's object hallucinations are **confident, image-driven misreadings**, not language-prior guesses. Internal confidence is a weak signal (0.58).
2. On selected hallucinating captions, **random perturbation is a strong baseline**. Every claim here is checked against a matched random trigger and clean captions.
3. **Correction works when detection is right:** with the oracle, the zoom refill removes 40% of hallucinated nouns on the full set and raises Cover.
   On the study set, word-first + sink removal takes it from 102 to 86 (baseline 155).
4. **Detection is the bottleneck.** CLIP on the full image is the best detector (AUROC 0.86, held-out 0.84), but online it only gives −4% on the full set.
   Vocabulary, margins and synonyms don't move it. The sink-free crop is a good second check that cuts false positives at high recall.
5. The refill rarely finds the correct object. It mostly avoids the error. False flags are cheap for one-word re-prediction, but not for the 3-token online refill.

---

## 14. Next steps

1. **Full-set oracle run with word-first + sink removal**, to confirm slide 5 on 1004 captions (vs 523 → 314). ~40 min.
2. **High-recall CLIP detector + word-first + sink removal online** on the study set, then held-out ids 730–1004, against a matched random control.
3. **A refill that protects correct words**, so that high recall becomes affordable:
   - one-token refill, or keep the original word unless the crop clearly prefers another;
   - delete or neutralise instead of replacing (most achievable outcomes are neutral).
4. **Wider remask span: first result is promising.** Oracle + zoom + word-first + sinks removed, first 30 hallucinating captions, A100:

   | 30 captions | CHAIR ↓ | Cover ↑ | Hal ↓ | hallu nouns (baseline 57) | grounded (173) |
   |---|---|---|---|---|---|
   | span ±1 (3 tokens) | 13.1 | 47.3 | 70.0 | 29 | 183 |
   | span ±2 (5 tokens) | 10.9 | **50.3** | 63.3 | 24 | **190** |
   | span ±5 (11 tokens) | **9.8** | 49.7 | **50.0** | **21** | 185 |

   It rewrites the phrase, not just the noun ("the sun shining" → "a sunny day, and the players"). But it also rewrites more text overall
   (53 / 62 triggers vs 28 / 50), sometimes just avoids AMBER words ("no other skateboarders or pedestrians"), and occasionally breaks grammar.
   Needs the full study set, a matched random control, and a test with CLIP flags, where false flags would rewrite 5–11 correct tokens.
5. Open checks:
   - a degraded oracle (half recall / added false flags) to split recall vs precision (cut off at 59 / 144 captions);
   - a full-set random control for the CLIP run;
   - noting AMBER artefacts (negations, idioms, compounds) when reporting.

**Caveats:** single runs; the study set is selected (84 of 144 hallucinate); detector thresholds were tuned on the study set; GPU type changes greedy decoding,
so only same-GPU runs are compared.
