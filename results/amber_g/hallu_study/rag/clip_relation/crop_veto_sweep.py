"""Second-stage crop check for the CLIP detector: keep a flag (rank_full < 0.9) only if CLIP does not find the word in the crop."""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "nouns_clip.csv"))))
y_all = np.array([r["label"] == "hallucinated" for r in rows])
flag = np.array([float(r["rank_full"]) < 0.9 for r in rows])
print(f"CLIP alone: flagged {flag.sum()} | precision {y_all[flag].mean():.3f} recall {y_all[flag].sum() / y_all.sum():.3f}")
for view in ("super_tight", "tight", "loose"):
    rank = np.array([float(r[f"rank_{view}"]) for r in rows])
    for veto in (0.9, 0.95, 0.97, 0.98, 0.99):
        keep = flag & (rank < veto)
        print(f"  veto if rank_{view:11s} >= {veto:.2f}: flagged {keep.sum():3d} | precision {y_all[keep].mean():.3f} "
              f"recall {y_all[keep].sum() / y_all.sum():.3f} | grounded removed {(flag & ~keep & ~y_all).sum():2d}/{(flag & ~y_all).sum()} "
              f"hallucinated lost {(flag & ~keep & y_all).sum():2d}/{(flag & y_all).sum()} | fire rate {keep.mean():.3f}")
