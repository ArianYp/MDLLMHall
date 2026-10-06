# Presentation material

The slide deck is `../presentation.md`. This folder holds the figures made for it.

## The problem (`hallucination_<id>.png`, made by `make_hallucination_example.py`)

The photo next to MMaDA's full baseline caption: hallucinated object nouns in red, grounded ones in green (AMBER labeler), plus AMBER's truth list.
- `hallucination_163.png` (main): woman with a lime, broccoli and a halved melon → cup, drink, bowl, eggs, apples (5 of 11 mentions).
  Each is a misreading of a real object (lime, melon, limes).
- `hallucination_537.png` (backup): two kites in the sky → "a man in a woman flying a kite" (4 people mentions, no people).

## The CLIP check (`clip_idea_<id>.png`, `clip_veto_<id>_<pos>.png`, made by `make_clip_examples.py`)

Offline, from `rag/clip_relation/clip_vocab_embed.npz`, with exactly the online criterion: caption-aware full-image rank < 0.99
(≈ more than 4 of the 418 words beat the noun), veto if the noun is CLIP's top word on the sink-free crop.
- `clip_idea_372.png` (main): rowing boat on a rainy lake. Real objects are beaten by 0–2 words, "sun" by 263.
  `clip_idea_660.png` (backup): a beach umbrella, no people. "person" is beaten by 27 words, "people" by 97.
- `clip_veto_218_84.png` (main): a small car behind a stop sign. 63 words beat it on the full image; it is the top word on the crop.
  `clip_veto_449_29.png` (backup): a coffee cup next to donuts (34 → top word). Extras: `clip_veto_342_41.png` (bike basket), `clip_veto_722_4.png` (man under a plane).
- `python presentation/make_clip_examples.py --list` prints more candidates. Some winning words are AMBER vocabulary junk
  ("waterfont", "doghole", "stopcock").

## A failure the zoom cannot fix (`misread_517.png`, made by `make_misread_example.py`)

A white horse → "a cow". On the zoomed crop the model still says cow 0.59 vs horse 0.35, and CLIP ranks cow 4th, so it is not flagged either.

## One hallucination, one zoom (`zoom_<id>.png`, made by `make_examples.py`)

Each figure shows one thing:
- the photo, with a red box where MMaDA looked when it wrote the hallucinated word;
- the zoomed crop that replaces the image in the refill;
- the **baseline sentence**, with the single hallucinated noun in red;
- the **same sentence after zooming in**, with the refilled object in green;
- AMBER's ground-truth objects.

Only each caption's first trigger is used. Up to that point decoding is identical to the baseline, so the two sentences differ only through this one zoom.

| file | baseline → after zooming in | run |
|---|---|---|
| `zoom_967.png` | "a brown **dog** standing on a dirt road" → "a majestic **lion**" (the crop is a lioness) | full-set oracle + zoom |
| `zoom_1001.png` | "playing a game of **tennis**" → "a game of **volleyball**" (the crop is the ball at the net) | full-set oracle + zoom |
| `zoom_926.png` | "people on a small **boat**" → "on a **paddle board**" (two tiny people far out at sea) | full-set oracle + zoom |
| `zoom_342.png` | "leaning against a **building**" → "against a **fence**" (crop: fence 0.37) | zoom + word-first + sinks removed |
| `zoom_84.png` | "washing his hands under a **faucet**" → "under running **water**" (crop: water 0.45) | zoom + word-first + sinks removed |
| `zoom_503.png` | "placed on a cardboard **box**" → "a cardboard **plate**" (crop: plate 0.49) | zoom + word-first + sinks removed |

Runs:
- "Full-set oracle + zoom" is `results/amber_g/oracle_zoom_full/missing/` (current refill order, sinks kept). Its events do not log the crop's top words.
- The other three are from `results/amber_g/hallu_study/rag/slot_refill/oracle_zoom_wordfirst_nosink/`.
- The trigger is always the **oracle** (AMBER labels). The figures show what the correction step does once a hallucination is found.

How the examples were picked:
- `python presentation/make_examples.py --list` lists every clean one-zoom fix over the five oracle + zoom runs (`candidates.txt`; 47 entries, many repeated across runs).
- Each chosen one was then checked by eye, so that the crop really shows the right object.
- Rejected: 655 (motorcycle → bicycle; the crop is only road and grass), 146 (sandy road → sandy beach; it is a desert), 263 (television → computer; the crop is a blank wall).

Draw more with `python presentation/make_examples.py full:967 wf_nosink:342 …` (CPU, from MDLLM/).

## The attention-sink problem (`sink_84.png`, made by `make_sink_example.py`)

Caption 84, with the same first trigger in the two word-first runs (sinks kept / removed). The panels are:
1. the attention map of "faucet", with the sink circled;
2. where the two crops are;
3. the crop with the sink (the shampoo bottle) → refill "soap soap.";
4. the crop with the sink removed (hands and pouring water) → refill "running water.".

The sink is the top patch for 126 of the 128 caption tokens. The attention map comes from the baseline trace, which is the same state as the first trigger.
Draw others with `python presentation/make_sink_example.py <id>`.
- Rejected: 248 ("people" → "cars": the cars are train cars, not convincing) and 342 (the two crops nearly overlap, and the sink holds only 2% of the attention).
- Not reviewed: 614. See `rag/slot_refill/wordfirst_sink_analysis.txt`.

## Older figures

`example_102.png` and `example_537.png` (`make_example.py`) show whole captions, where one zoom fixed several hallucinations at once.
They have been replaced by the one-hallucination figures above.

## Slide text (one line per example)

**Zooming in recovers misread objects.** MMaDA's hallucinations are often misreadings of a real object it is attending to. Masking the word and refilling it
from a zoomed crop of the attended region fixes the reading:
- dog → **lion**
- tennis → **volleyball**
- boat → **paddle board**
- building → **fence**
- faucet → **running water**
- box → **plate**

The rest of the sentence is unchanged.
