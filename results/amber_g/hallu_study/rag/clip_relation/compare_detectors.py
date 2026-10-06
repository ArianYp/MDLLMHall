"""Full-image CLIP vs crop-only CLIP vs full + crop check, as detectors of hallucinated nouns (no generation)."""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "nouns_clip.csv"))))
y = np.array([r["label"] == "hallucinated" for r in rows])
col = lambda f: np.array([float(r[f]) for r in rows])
full = col("rank_full")
crops = {v: col(f"rank_{v}") for v in ("super_tight", "tight", "loose")}
print(f"{len(rows)} nouns: {y.sum()} hallucinated, {(~y).sum()} grounded\n")


def line(name, flag):
    print(f"{name:46s} flagged {flag.sum():4d} | hallucinated {(flag & y).sum():3d} / {y.sum()} | grounded {(flag & ~y).sum():3d} / {(~y).sum()} "
          f"| precision {y[flag].mean() if flag.any() else 0:.3f} recall {(flag & y).sum() / y.sum():.3f}")


def auroc(score):  # lower rank => hallucinated
    s = -score
    p, n = s[y], s[~y]
    return ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n))


print("AUROC over all nouns (ranking quality, threshold-free):")
print(f"  full image {auroc(full):.3f} | " + " | ".join(f"{v} crop {auroc(c):.3f}" for v, c in crops.items()))
two_stage = np.where(crops["tight"] >= 0.95, 1.0, full)  # vetoed nouns are never flagged at any full-image cut below 1
print(f"  full + tight crop check (veto >= 0.95) {auroc(two_stage):.3f}")
print(f"  mean of full and tight ranks {auroc((full + crops['tight']) / 2):.3f} | min of the two {auroc(np.minimum(full, crops['tight'])):.3f}\n")

print("At the thresholds used so far:")
line("1. full image: rank_full < 0.9", full < 0.9)
for v in crops:
    line(f"2. crop only: rank_{v} < 0.9", crops[v] < 0.9)
line("3. full < 0.9, then drop if rank_tight >= 0.95", (full < 0.9) & (crops["tight"] < 0.95))

for budget in (114, 148):
    print(f"\nEqual budget: flag the {budget} lowest-ranked nouns by each score")
    for name, score in [("1. full image", full)] + [(f"2. crop only ({v})", c) for v, c in crops.items()] + \
                        [("3. full + tight crop check", two_stage), ("   mean(full, tight)", (full + crops["tight"]) / 2)]:
        order = np.argsort(score, kind="stable")[:budget]
        flag = np.zeros(len(rows), bool)
        flag[order] = True
        line(name, flag)

print("\nOverlap at the operating points (full < 0.9 vs tight crop < 0.9):")
a, b = full < 0.9, crops["tight"] < 0.9
for name, m in [("both", a & b), ("full only", a & ~b), ("crop only", ~a & b)]:
    print(f"  {name:10s} flagged {m.sum():3d} | hallucinated {(m & y).sum():3d} | grounded {(m & ~y).sum():3d} | precision {y[m].mean():.3f}")
