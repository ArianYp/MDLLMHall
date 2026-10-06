"""Two-stage CLIP detector: caption-aware rank on the full image, then the sink-free crop as a second check. Offline.

Stage 1: flag a noun if its full-image rank is below t1, where the rank is taken after removing the caption's other object nouns
         and all near-synonyms from the vocabulary (current 418 words; see clip_caption_exclude.py). t1 = "top1" means flag unless
         the noun is the best remaining word.
Stage 2: keep the flag only if the noun's rank on the sink-free tight crop is below t2 (plain rank, or caption-aware).
Reported: hallucinations caught (of 155) and false positives (of 895 grounded) on a grid, the best trade-offs, and reference rules.

CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/caption_aware_veto.py
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from amber_noun_labels import AmberLabeler
from detect_nosink_crop import BACKGROUND


def main():
    data = np.load(os.path.join(HERE, "clip_vocab_embed.npz"))
    words = [str(w) for w in data["words"]]
    widx = {w: i for i, w in enumerate(words)}
    rows = list(csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv"), encoding="utf-8")))
    y = np.array([r["label"] == "hallucinated" for r in rows])
    lemmas = [r["lemma"] for r in rows]
    bg = np.array([l in BACKGROUND for l in lemmas])
    img_row = {int(i): k for k, i in enumerate(data["image_ids"])}
    S_full = data["full"][[img_row[int(r["id"])] for r in rows]] @ data["text"].T
    S_crop = data["nosink"] @ data["text"].T
    caption_lemmas = defaultdict(set)
    for r in rows:
        caption_lemmas[r["id"]].add(r["lemma"])

    labeler = AmberLabeler()
    vec = np.stack([labeler.nlp.vocab[w].vector for w in words])
    norm = np.linalg.norm(vec, axis=1)
    has_vec = norm > 0
    vec = vec / np.where(has_vec, norm, 1)[:, None]
    syn = {}
    for lemma in set(lemmas):
        close = {lemma}
        if has_vec[widx[lemma]]:
            close |= {words[j] for j in np.flatnonzero((vec @ vec[widx[lemma]] > 0.8) & has_vec)}
        for s in wn.synsets(lemma, wn.NOUN):
            for t in [s] + s.hypernyms() + s.hyponyms():
                close |= {l.name().lower() for l in t.lemmas()}
        syn[lemma] = {widx[w] for w in close if w in widx}

    base = np.zeros(len(words), bool)
    base[[widx[str(w)] for w in data["current"]]] = True

    def rank(S, k, caption_aware):
        lemma, s = lemmas[k], S[k]
        m = base.copy()
        drop = syn[lemma] - {widx[lemma]}
        if caption_aware:
            drop |= set().union(*(syn[o] for o in caption_lemmas[rows[k]["id"]] if o != lemma))
        m[list(drop)] = False
        m[widx[lemma]] = True
        own = s[widx[lemma]]
        return (s[m] < own).mean(), bool(s[m].max() > own)  # rank, not top-1

    full_cap = [rank(S_full, k, True) for k in range(len(y))]
    full_plain = np.array([float(r["rank_full"]) for r in rows])
    crop_plain = [rank(S_crop, k, False) for k in range(len(y))]
    crop_cap = [rank(S_crop, k, True) for k in range(len(y))]
    H, G = y.sum(), (~y).sum()

    def flag(sig, t):
        return np.array([not1 if t == "top1" else r < t for r, not1 in sig])

    def show(name, f):
        tp, fp = (f & y).sum(), (f & ~y).sum()
        print(f"  {name:<62} caught {tp:>3}/{H} ({tp / H:.0%}) | false positives {fp:>3}/{G} ({fp / G:.1%}) | real {tp / max(tp + fp, 1):.0%}")
        return tp, fp

    print("reference rules")
    show("full < 0.9 (plain)", full_plain < 0.9)
    show("full < 0.9 AND crop no sink < 0.95 (plain)", (full_plain < 0.9) & flag(crop_plain, 0.95))
    T1 = (0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, "top1")
    T2 = (0.8, 0.9, 0.95, 0.97, 0.98, 0.99, "top1", None)
    print("\nstage 1 alone: full image, caption-aware")
    for t1 in T1:
        show(f"full caption-aware < {t1}", flag(full_cap, t1))

    for crop_name, crop_sig in (("crop no sink (plain)", crop_plain), ("crop no sink (caption-aware)", crop_cap)):
        print(f"\nstage 1 + stage 2 = {crop_name}: grid, then the best trade-offs")
        points = []
        for t1 in T1:
            for t2 in T2:
                f = flag(full_cap, t1) & (flag(crop_sig, t2) if t2 is not None else True)
                points.append(((f & y).sum(), (f & ~y).sum(), t1, t2))
        best = [p for p in points if not any(q[0] >= p[0] and q[1] < p[1] or q[0] > p[0] and q[1] <= p[1] for q in points)]
        for tp, fp, t1, t2 in sorted(set(best)):
            print(f"  full caption-aware < {t1!s:<5} AND crop < {t2!s:<5}  caught {tp:>3}/{H} ({tp / H:.0%}) | "
                  f"false positives {fp:>3}/{G} ({fp / G:.1%}) | real {tp / max(tp + fp, 1):.0%}")

    print("\nsame, never flagging background words (list written after seeing the study flags)")
    for t1, t2 in ((0.9, 0.95), (0.95, 0.95), (0.97, 0.97), (0.98, 0.98), ("top1", 0.97), ("top1", "top1")):
        show(f"never background, full caption-aware < {t1} AND crop no sink < {t2}", ~bg & flag(full_cap, t1) & flag(crop_plain, t2))


if __name__ == "__main__":
    main()
