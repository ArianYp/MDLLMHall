"""CLIP rank / margin against the best word the caption does NOT already mention. Offline, from clip_vocab_embed.npz.

Idea: a grounded noun often loses to the image's main object ("beach" < "dog"), but that object is usually in the caption too.
A hallucinated noun loses to the real object it misreads ("cow" < "horse"), which the caption does not mention.
So before ranking a noun, drop from the vocabulary the caption's other object nouns (all AMBER-labelled nouns of the same caption)
and their near-synonyms, as well as the noun's own near-synonyms.

Reported per view and vocabulary: AUROC and recall at a 10 / 20 / 30% false-positive rate, for the plain signal, synonyms removed,
and synonyms + caption nouns removed. CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/clip_caption_exclude.py
"""

import csv
import json
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
from clip_vocab_compare import FPR, VIEWS, recall_at_fpr
from detect_nosink_crop import auroc


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
        i = widx[lemma]
        if has_vec[i]:
            close |= {words[j] for j in np.flatnonzero((vec @ vec[i] > 0.8) & has_vec)}
        for s in wn.synsets(lemma, wn.NOUN):
            for t in [s] + s.hypernyms() + s.hyponyms():
                close |= {l.name().lower() for l in t.lemmas()}
        syn[lemma] = {widx[w] for w in close if w in widx}

    wordnet = set(json.load(open(os.path.join(HERE, "wordnet_rival", "vocabulary.json"), encoding="utf-8")))
    vocabularies = {"current": {str(w) for w in data["current"]}, "wordnet_all": wordnet | set(lemmas)}
    print(f"nouns {len(y)} (hallucinated {y.sum()}); caption nouns per caption: median "
          f"{np.median([len(v) for v in caption_lemmas.values()]):.0f}")
    print("AUROC [recall at false-positive rate " + " / ".join(f"{f:.0%}" for f in FPR) + "]")
    for v, view_name in VIEWS:
        S = sims[v]
        print(f"\n  {view_name}")
        for vname, vocab in vocabularies.items():
            base = np.zeros(len(words), bool)
            base[[widx[w] for w in vocab]] = True
            out = {k: [] for k in ("rank", "rank_syn", "rank_caption", "margin", "margin_syn", "margin_caption")}
            for k, lemma in enumerate(lemmas):
                s, own = S[k], S[k, widx[lemma]]
                for name, drop in (("", set()), ("_syn", syn[lemma] - {widx[lemma]}),
                                   ("_caption", (syn[lemma] - {widx[lemma]}) | set().union(
                                       *(syn[o] for o in caption_lemmas[rows[k]["id"]] if o != lemma)))):
                    m = base.copy()
                    m[list(drop)] = False
                    m[widx[lemma]] = True
                    out["rank" + name].append(-(s[m] < own).mean())
                    out["margin" + name].append(s[m].max() - own)
            cells = []
            for name, values in out.items():
                a = np.array(values)
                cells.append(f"{name} {auroc(a, y):.3f} [" + " ".join(f"{recall_at_fpr(a, y, f):.2f}" for f in FPR) + "]")
            print(f"    {vname:<12} " + " | ".join(cells[:3]))
            print(f"    {'':<12} " + " | ".join(cells[3:]))


if __name__ == "__main__":
    main()
