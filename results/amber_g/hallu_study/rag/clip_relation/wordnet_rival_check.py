"""Small test of a "related rival" CLIP check with a WordNet vocabulary.

Vocabulary: WordNet physical-object nouns mentioned >= --min-count times in COCO train captions,
plus AMBER's main object words (relation.json keys). Each word gets one sense: its most frequent
noun synset under physical_entity.n.01.

For a noun w on its image, rivals = vocabulary words whose sense has Wu-Palmer similarity >= t to w's
sense. The noun is flagged at threshold t if some rival has a higher CLIP score ("a photo of a {w}.",
full image) than w. Reported for several t, with the score margin of the best rival.

Nouns: --hallucinated id:lemma pairs, plus --n-grounded grounded nouns sampled from clean captions
(first mentions, one per caption, seed --seed).

Example (from MDLLM/):
    python results/amber_g/hallu_study/rag/clip_relation/wordnet_rival_check.py \
        --hallucinated 428:cow 67:bench 616:window 470:table 163:apple 163:bowl 634:road 526:people 206:ball 312:desk
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from rag_retrieve import CLIP_ID, COCO, CROPS, embed
from remask_decoding import STUDY, zoom_crop
from trace_amber_steps import AMBER_DATA, IMAGE_DIR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordnet_rival")
TRACE_DIRS = [os.path.join(STUDY, "traces"), os.path.join(STUDY, "traces_clean")]
PHYSICAL = wn.synset("physical_entity.n.01")
THRESHOLDS = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def physical_sense(word):
    """Most frequent noun sense of `word` that is a physical entity (None if there is none)."""
    senses = [s for s in wn.synsets(word.replace(" ", "_"), wn.NOUN)
              if any(PHYSICAL in path for path in s.hypernym_paths())]
    if not senses:
        return None
    count = lambda s: sum(l.count() for l in s.lemmas() if l.name().lower() == word.replace(" ", "_"))
    return max(senses, key=lambda s: (count(s), -senses.index(s)))


def build_vocabulary(min_count):
    counts = Counter()
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        for ann in json.load(handle)["annotations"]:
            counts.update(re.findall(r"[a-z]+", ann["caption"].lower()))
    lemmas = Counter()
    for token, c in counts.items():
        lemma = wn.morphy(token, wn.NOUN)
        if lemma and len(lemma) >= 3:
            lemmas[lemma] += c
    amber = set(json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8")))
    words = {w for w, c in lemmas.items() if c >= min_count} | amber
    sense = {w: physical_sense(w) for w in sorted(words)}
    return {w: s for w, s in sense.items() if s is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hallucinated", nargs="*", default=[], help="id:lemma pairs")
    parser.add_argument("--random-hallucinated", type=int, default=0, help="random held-out hallucinated nouns")
    parser.add_argument("--random-grounded", type=int, default=0, help="random held-out grounded nouns")
    parser.add_argument("--exclude", nargs="*", default=[], help="rows CSVs whose (id, lemma) nouns are left out")
    parser.add_argument("--tag", default="", help="suffix for the output CSV")
    parser.add_argument("--n-grounded", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", default="full", choices=["full"] + list(CROPS),
                        help="score the full photo or the noun's attended crop (tight = the zoom-refill crop)")
    parser.add_argument("--remove-sinks", action="store_true",
                        help="crops: ignore patches that are the top patch for >= half of the caption's tokens")
    args = parser.parse_args()
    tag = args.view + ("_nosink" if args.remove_sinks else "") + (f"_{args.tag}" if args.tag else "")
    device = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)

    sense = build_vocabulary(args.min_count)
    vocabulary = sorted(sense)
    print(f"vocabulary {len(vocabulary)} physical-object words (COCO count >= {args.min_count}, + AMBER main words)")

    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        nouns = list(csv.DictReader(handle))
    samples = []
    if args.random_hallucinated or args.random_grounded:
        # Held-out random sample: first mentions with a physical sense, minus nouns used for tuning.
        exclude = set()
        for path in args.exclude:
            with open(path, encoding="utf-8") as handle:
                exclude |= {(int(r["id"]), r["lemma"]) for r in csv.DictReader(handle)}
        pool = [r for r in nouns if r["label"] in ("grounded", "hallucinated") and float(r["repeat_mention"]) == 0
                and r["lemma"] in sense and (int(r["id"]), r["lemma"]) not in exclude]
        rng = random.Random(args.seed)
        for label, n in (("hallucinated", args.random_hallucinated), ("grounded", args.random_grounded)):
            group = [r for r in pool if r["label"] == label]
            for r in rng.sample(group, min(n, len(group))):
                samples.append(dict(id=int(r["id"]), pos=int(r["pos"]), lemma=r["lemma"], label=label))
        print(f"random sample (seed {args.seed}): {sum(s['label'] == 'hallucinated' for s in samples)} hallucinated "
              f"of {sum(r['label'] == 'hallucinated' for r in pool)}, {sum(s['label'] == 'grounded' for s in samples)} "
              f"grounded of {sum(r['label'] == 'grounded' for r in pool)} (excluded {len(exclude)} tuning nouns)")
        args.n_grounded = 0
    for p in args.hallucinated:
        item_id, lemma = int(p.split(":")[0]), p.split(":")[1]
        pos = min(int(r["pos"]) for r in nouns if int(r["id"]) == item_id and r["lemma"] == lemma
                  and r["label"] == "hallucinated")
        samples.append(dict(id=item_id, pos=pos, lemma=lemma, label="hallucinated"))
    clean = [r for r in nouns if r["label"] == "grounded" and r["caption_hallucinates"] == "False"
             and float(r["repeat_mention"]) == 0 and r["lemma"] in sense]
    by_caption = {}
    for r in clean:
        by_caption.setdefault(int(r["id"]), []).append(r)
    rng = random.Random(args.seed)
    for item_id in rng.sample(sorted(by_caption), args.n_grounded) if args.n_grounded else []:
        r = rng.choice(by_caption[item_id])
        samples.append(dict(id=item_id, pos=int(r["pos"]), lemma=r["lemma"], label="grounded"))

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        text = []
        for i in range(0, len(vocabulary), 256):
            inputs = processor(text=[f"a photo of a {w}." for w in vocabulary[i:i + 256]], return_tensors="pt",
                               padding=True).to(device)
            text.append(torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1))
    text = torch.cat(text)
    index = {w: i for i, w in enumerate(vocabulary)}

    # Full photo, or the crop around the noun's attended region (late-layer map at its first token, as in the zoom refill).
    image_sims = {}
    for s in samples:
        photo = Image.open(os.path.join(IMAGE_DIR, f"AMBER_{s['id']}.jpg")).convert("RGB")
        if args.view != "full":
            trace_dir = next(d for d in TRACE_DIRS if os.path.exists(os.path.join(d, f"{s['id']}.npz")))
            late_map = torch.from_numpy(np.load(os.path.join(trace_dir, f"{s['id']}.npz"))["late_map"].astype(np.float32))
            zoom, min_patches = CROPS[args.view]
            image_map = late_map[s["pos"]].clone()
            if args.remove_sinks:
                # Sink = a patch that is the top patch for >= half of the caption's tokens; drop it before cropping.
                counts = torch.bincount(late_map.argmax(1), minlength=late_map.shape[1])
                s["sinks"] = torch.nonzero(counts >= 0.5 * late_map.shape[0]).flatten().tolist()
                image_map[s["sinks"]] = 0
            photo, s["box"] = zoom_crop(photo, image_map, 512, zoom, 30, min_patches)
        image_sims[(s["id"], s["pos"])] = (text @ embed([photo], model, processor, device)[0]).cpu()

    def ancestors(synset):
        return set(synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))

    # Which rivals are allowed, given the noun's sense a and the rival's sense b:
    #   all       any related word (the first run)
    #   subtypes  not a subtype of the noun (beagle for dog)
    #   siblings  also not the same sense (lavatory for toilet) and not a broader word (device for computer)
    modes = {
        "all": lambda a, b: True,
        "subtypes": lambda a, b: a not in ancestors(b),
        "siblings": lambda a, b: a != b and a not in ancestors(b) and b not in ancestors(a),
    }

    rows = []
    for s in samples:
        w = s["lemma"]
        if w not in sense:
            print(f"skip {s['id']}:{w} (no physical WordNet sense)")
            continue
        sims = image_sims[(s["id"], s["pos"])]
        own = float(sims[index[w]])
        place = int((sims > own).sum()) + 1
        rivals = []
        for v in vocabulary:
            if v == w or float(sims[index[v]]) <= own:
                continue
            wup = sense[w].wup_similarity(sense[v]) or 0.0
            rivals.append((v, float(sims[index[v]]), wup))
        row = dict(id=s["id"], pos=s["pos"], lemma=w, label=s["label"], view=tag, box=s.get("box", ""),
                   sinks=s.get("sinks", ""),
                   sense=sense[w].name(), place=place, vocabulary=len(vocabulary), sim=own)
        for mode, allowed in modes.items():
            for t in THRESHOLDS:
                close = [r for r in rivals if r[2] >= t and allowed(sense[w], sense[r[0]])]
                best = max(close, key=lambda r: r[1]) if close else None
                row[f"{mode}_rival_{t}"] = f"{best[0]} ({sense[best[0]].name()}, wup {best[2]:.2f})" if best else ""
                row[f"{mode}_margin_{t}"] = best[1] - own if best else 0.0
        rows.append(row)

    nh = sum(r["label"] == "hallucinated" for r in rows)
    ng = sum(r["label"] == "grounded" for r in rows)
    for mode in modes:
        print(f"\n=== rivals: {mode}")
        print(f"{'id':>4} {'word':<10} {'label':<12}  best rival above it (wup >= 0.8)               margin  | flagged at wup >= " +
              " ".join(f"{t:.2f}" for t in THRESHOLDS))
        for r in rows:
            flags = " ".join("  X " if r[f"{mode}_margin_{t}"] > 0 else "  . " for t in THRESHOLDS)
            print(f"{r['id']:>4} {r['lemma']:<10} {r['label']:<12}  {r[f'{mode}_rival_0.8'] or '-':<45} "
                  f"{r[f'{mode}_margin_0.8']:+.3f}  | {flags}")
        for t in THRESHOLDS:
            h = sum(r[f"{mode}_margin_{t}"] > 0 for r in rows if r["label"] == "hallucinated")
            g = sum(r[f"{mode}_margin_{t}"] > 0 for r in rows if r["label"] == "grounded")
            print(f"  wup >= {t:.2f}: flags {h}/{nh} hallucinated, {g}/{ng} grounded")

    with open(os.path.join(OUT_DIR, f"rows_{tag}.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json.dump({w: s.name() for w, s in sense.items()}, open(os.path.join(OUT_DIR, "vocabulary.json"), "w"), indent=0)
    print("\nsaved", OUT_DIR)


if __name__ == "__main__":
    main()
