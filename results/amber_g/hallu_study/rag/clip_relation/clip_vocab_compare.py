"""Does a larger or cleaner CLIP vocabulary detect hallucinated nouns better? Offline, from clip_vocab_embed.npz.

Vocabularies (each also contains the 1050 study nouns' lemmas, so every noun has a rank):
  current      AMBER relation.json main + association words + study lemmas (418, the one used so far)
  amber340     AMBER relation.json main words only + study lemmas (drops the junk association words)
  wordnet_c1000 / wordnet_c200   WordNet physical-object nouns with >= 1000 / 200 COCO train-caption mentions + AMBER main words
  wordnet_all  the same with >= 20 mentions (2651, wordnet_rival_check.py)
Signals per view (full image, crop, sink-free crop), higher => hallucinated:
  rank  = -(share of vocabulary words below the noun);  margin = sim(best word) - sim(noun)
  *_syn = the same with the noun's near-synonyms removed from the vocabulary first (spaCy similarity > 0.8, AMBER's rule, plus WordNet
          synonyms and direct hypernyms / hyponyms), so "person" beating "man" does not count against "man".
Reported: AUROC and recall at a fixed false-positive rate (10 / 20 / 30% of the 895 grounded nouns), which is comparable across vocabulary sizes.

CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/clip_vocab_compare.py
"""

import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np
from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from amber_noun_labels import AmberLabeler
from detect_nosink_crop import auroc
from rag_retrieve import COCO
from trace_amber_steps import AMBER_DATA

VIEWS = (("full", "full image"), ("crop", "crop with sinks"), ("nosink", "crop no sink"))
FPR = (0.1, 0.2, 0.3)


def coco_counts(words):
    counts = Counter()
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        for ann in json.load(handle)["annotations"]:
            counts.update(re.findall(r"[a-z]+", ann["caption"].lower()))
    lemmas = Counter()
    for token, c in counts.items():
        lemma = wn.morphy(token, wn.NOUN)
        if lemma in words:
            lemmas[lemma] += c
    return lemmas


def recall_at_fpr(score, y, fpr):
    thr = np.quantile(score[~y], 1 - fpr)
    return (score[y] > thr).mean()


def main():
    data = np.load(os.path.join(HERE, "clip_vocab_embed.npz"))
    words = [str(w) for w in data["words"]]
    widx = {w: i for i, w in enumerate(words)}
    rows = list(csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv"), encoding="utf-8")))
    y = np.array([r["label"] == "hallucinated" for r in rows])
    lemmas = [r["lemma"] for r in rows]
    img_row = {int(i): k for k, i in enumerate(data["image_ids"])}
    sims = {"full": data["full"][[img_row[int(r["id"])] for r in rows]] @ data["text"].T,
            "crop": data["crop"] @ data["text"].T, "nosink": data["nosink"] @ data["text"].T}

    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    study = set(lemmas)
    wordnet = set(json.load(open(os.path.join(HERE, "wordnet_rival", "vocabulary.json"), encoding="utf-8")))
    counts = coco_counts(wordnet)
    vocabularies = {
        "current": {str(w) for w in data["current"]},
        "amber340": set(relation) | study,
        "wordnet_c1000": {w for w in wordnet if counts[w] >= 1000} | set(relation) | study,
        "wordnet_c200": {w for w in wordnet if counts[w] >= 200} | set(relation) | study,
        "wordnet_all": wordnet | study,
    }

    # Near-synonyms of each study lemma over the union vocabulary.
    labeler = AmberLabeler()
    vec = np.stack([labeler.nlp.vocab[w].vector for w in words])
    norm = np.linalg.norm(vec, axis=1)
    has_vec = norm > 0
    vec = vec / np.where(has_vec, norm, 1)[:, None]
    syn = {}
    for lemma in study:
        close = set()
        i = widx[lemma]
        if has_vec[i]:
            close |= {words[j] for j in np.flatnonzero((vec @ vec[i] > 0.8) & has_vec)}
        for s in wn.synsets(lemma, wn.NOUN):
            for t in [s] + s.hypernyms() + s.hyponyms():
                close |= {l.name().lower() for l in t.lemmas()}
        close.discard(lemma)
        syn[lemma] = np.array(sorted(widx[w] for w in close if w in widx), dtype=int)

    print(f"nouns {len(y)} (hallucinated {y.sum()}); vocabulary sizes: " + ", ".join(f"{k} {len(v)}" for k, v in vocabularies.items()))
    print(f"median near-synonyms per study lemma in the union vocabulary: {np.median([len(v) for v in syn.values()]):.0f}")
    print("\nAUROC (higher = better) | recall at false-positive rate " + " / ".join(f"{f:.0%}" for f in FPR))
    for v, view_name in VIEWS:
        S = sims[v]
        print(f"\n  {view_name}")
        for vname, vocab in vocabularies.items():
            cols = np.array(sorted(widx[w] for w in vocab))
            mask = np.zeros(len(words), bool)
            mask[cols] = True
            sig = {"rank": [], "margin": [], "rank_syn": [], "margin_syn": []}
            winners = Counter()
            for k, lemma in enumerate(lemmas):
                s, own = S[k], S[k, widx[lemma]]
                m = mask.copy()
                sig["rank"].append(-(s[m] < own).mean())
                sig["margin"].append(s[m].max() - own)
                m[syn[lemma]] = False
                sig["rank_syn"].append(-(s[m] < own).mean())
                sig["margin_syn"].append(s[m].max() - own)
                if not y[k]:
                    best = np.flatnonzero(m)[np.argmax(s[m])]
                    if words[best] != lemma:
                        winners[(lemma, words[best])] += 1
            cells = []
            for name, values in sig.items():
                a = np.array(values)
                cells.append(f"{name} {auroc(a, y):.3f} [" + " ".join(f"{recall_at_fpr(a, y, f):.2f}" for f in FPR) + "]")
            print(f"    {vname:<14} " + " | ".join(cells))
            if vname in ("current", "wordnet_all") and v == "nosink":
                print(f"    {'':<14} grounded nouns, top winners after synonym removal: "
                      + ", ".join(f"{a}->{b} {n}" for (a, b), n in winners.most_common(12)))


if __name__ == "__main__":
    main()
