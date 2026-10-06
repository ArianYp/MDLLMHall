"""Is the CLIP margin (top-1 word similarity - noun similarity) larger for hallucinated nouns? Offline, from clip_margin.csv.

1) Margin distribution, hallucinated vs grounded, per view; AUROC of the margin vs the rank (95% CI over captions).
2) Grounded nouns that are not top-1: is the winner another object in the image (AMBER truth / synonym), and is the margin small?
3) Operating points "flag if margin > m": hallucinations caught, false positives, precision.
4) Hallucinated nouns with a small margin: what beats them.

CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/clip_margin_analysis.py
"""

import csv
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from amber_noun_labels import AmberLabeler
from detect_nosink_crop import auroc, boot_ci

VIEWS = (("full", "full image"), ("crop", "crop with sinks"), ("nosink", "crop no sink"))


def main():
    labeler = AmberLabeler()
    rows = list(csv.DictReader(open(os.path.join(HERE, "clip_margin.csv"))))
    y = np.array([r["label"] == "hallucinated" for r in rows])
    groups = np.array([int(r["id"]) for r in rows])
    H, G = y.sum(), (~y).sum()

    def in_image(item_id, word):
        _, safe, _ = labeler.word_lists(item_id)
        return word in safe or any(labeler.similar(word, s) for s in safe)

    print(f"nouns {len(rows)}: hallucinated {H}, grounded {G}")
    print("\n1) margin = sim(top-1 word) - sim(noun); 0 when the noun is CLIP's top word")
    for v, name in VIEWS:
        m = np.array([float(r[f"{v}_margin"]) for r in rows])
        rank = np.array([float(r[f"{v}_rank"]) for r in rows])
        q = lambda a: " ".join(f"{x:.3f}" for x in np.percentile(a, [25, 50, 75, 90]))
        lo, hi = boot_ci(m, y, groups)
        print(f"  {name:<16} AUROC margin {auroc(m, y):.3f} ({lo:.3f}-{hi:.3f}) vs rank {auroc(-rank, y):.3f} | top-1: hallucinated "
              f"{(m[y] == 0).mean():.0%}, grounded {(m[~y] == 0).mean():.0%}")
        print(f"  {'':<16} margin p25/p50/p75/p90: hallucinated {q(m[y])} | grounded {q(m[~y])} | grounded not top-1 {q(m[~y & (m > 0)])}")

    print("\n2) grounded nouns that are not top-1: who beats them")
    for v, name in VIEWS:
        g = [r for r, h in zip(rows, y) if not h and float(r[f"{v}_margin"]) > 0]
        present = [in_image(int(r["id"]), r[f"{v}_top1"]) for r in g]
        mp = [float(r[f"{v}_margin"]) for r, p in zip(g, present) if p]
        ma = [float(r[f"{v}_margin"]) for r, p in zip(g, present) if not p]
        print(f"  {name:<16} {len(g)} not top-1: winner in the image {sum(present)} (median margin {np.median(mp):.3f}), "
              f"winner not in the image {len(g) - sum(present)} (median margin {np.median(ma):.3f})")
        print(f"  {'':<16} most common (noun -> winner): " + ", ".join(f"{a}->{b} {n}" for (a, b), n in
                                                                   Counter((r['lemma'], r[f'{v}_top1']) for r in g).most_common(10)))
    h_rows = [r for r, h in zip(rows, y) if h]
    for v, name in VIEWS:
        win = [in_image(int(r["id"]), r[f"{v}_top1"]) for r in h_rows if float(r[f"{v}_margin"]) > 0]
        print(f"  for comparison, hallucinated not top-1 ({name}): winner in the image {sum(win)} / {len(win)}")

    print("\n3) flag if margin > m")
    for v, name in VIEWS:
        m = np.array([float(r[f"{v}_margin"]) for r in rows])
        print(f"  {name}")
        for t in (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1):
            f = m > t
            tp, fp = (f & y).sum(), (f & ~y).sum()
            print(f"    margin > {t:.3f}: caught {tp:>3}/{H} ({tp / H:.0%}) | false positives {fp:>3}/{G} ({fp / G:.0%}) | precision {tp / max(tp + fp, 1):.2f}")
    print("  combinations (margin on two views)")
    mf = np.array([float(r["full_margin"]) for r in rows])
    mn = np.array([float(r["nosink_margin"]) for r in rows])
    for name, f in (("full > 0.02 OR no sink > 0.05", (mf > 0.02) | (mn > 0.05)),
                    ("full > 0.01 AND no sink > 0.01", (mf > 0.01) & (mn > 0.01)),
                    ("full > 0.02 AND no sink > 0.02", (mf > 0.02) & (mn > 0.02)),
                    ("mean margin > 0.02", (mf + mn) / 2 > 0.02),
                    ("mean margin > 0.03", (mf + mn) / 2 > 0.03)):
        tp, fp = (f & y).sum(), (f & ~y).sum()
        print(f"    {name:<32} caught {tp:>3}/{H} ({tp / H:.0%}) | false positives {fp:>3}/{G} ({fp / G:.0%}) | precision {tp / max(tp + fp, 1):.2f}")

    print("\n4) hallucinated nouns with margin <= 0.01 on the sink-free crop (look like grounded)")
    for r in h_rows:
        if float(r["nosink_margin"]) <= 0.01:
            print(f"    {r['id']:>4} {r['lemma']:<10} margin {float(r['nosink_margin']):.3f} top5 {r['nosink_top5']}")


if __name__ == "__main__":
    main()
