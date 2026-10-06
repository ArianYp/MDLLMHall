"""Do the nouns flagged by CLIP on the sink-free crop survive a one-word re-prediction? Offline.

Flags: rank_tight_nosink < t (crop_veto_nosink.csv). Re-prediction (repredict/repredict.csv, 2026-10-03): the noun is masked in the
finished baseline caption and re-predicted in one forward pass; top-1 (first token) per condition:
  plain = full image; plain_nb = full image, pos-1 and the next token masked too; zoom = tight crop with sinks (not sink-free).
Top-1 is classed as: same word, another correct object (in the image), a wrong object (not in the image), or neutral (not an object word).
For false positives (grounded nouns flagged) "same" or "other correct" = not hurt; for flagged hallucinations "wrong" = not fixed.

CPU only (spaCy). Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/flag_repredict.py
"""

import csv
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler

STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))
CONDITIONS = ("plain", "plain_nb", "zoom")
KINDS = ("same word", "other correct", "neutral", "wrong object")


def main():
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe
    lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())

    def kind(item_id, original, top1):
        if top1.strip().lower() == original.strip().lower() or lem(top1) == lem(original):
            return "same word"
        w = top1.strip().lower()
        if not w.isalpha() or lem(w) not in objects:
            return "neutral"
        _, safe, _ = labeler.word_lists(item_id)
        l = lem(w)
        return "other correct" if l in safe or any(labeler.similar(l, s) for s in safe) else "wrong object"

    crop = {(r["id"], r["pos"]): r for r in csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv")))}
    rep = {(r["id"], r["pos"]): r for r in csv.DictReader(open(os.path.join(STUDY, "repredict", "repredict.csv")))}
    keys = [k for k in crop if k in rep]
    print(f"nouns in both files: {len(keys)} / {len(crop)} (hallucinated {sum(crop[k]['label'] == 'hallucinated' for k in keys)})")

    for t in (0.8, 0.9):
        flagged = [k for k in keys if float(crop[k]["rank_tight_nosink"]) < t]
        for label, title in (("grounded", "FALSE POSITIVES (grounded nouns flagged)"), ("hallucinated", "FLAGGED HALLUCINATIONS")):
            group = [k for k in flagged if crop[k]["label"] == label]
            print(f"\ncrop no sink < {t}: {title}: {len(group)}")
            for cond in CONDITIONS:
                c = Counter(kind(int(k[0]), rep[k]["token"], rep[k][f"{cond}_top1"]) for k in group)
                n = max(len(group), 1)
                print(f"  {cond:<9} " + " | ".join(f"{x} {c[x]:>3} ({c[x] / n:.0%})" for x in KINDS))
            if label == "grounded" and t == 0.8:
                print("  examples (word: plain / plain_nb / zoom top-1):")
                for k in group[:25]:
                    r = rep[k]
                    print(f"    {k[0]:>4} {r['word']:<12} " + " / ".join(f"{r[f'{c}_top1'].strip()!r}" for c in CONDITIONS))


if __name__ == "__main__":
    main()
