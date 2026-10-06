"""Does decoding order add anything beyond position in the caption?"""
import csv
import numpy as np
from analyze_hallu_study import auroc, logistic_cv

import sys
rows = list(csv.DictReader(open(sys.argv[1])))
y = np.array([r["label"] == "hallucinated" for r in rows], dtype=float)
g = np.array([int(r["id"]) for r in rows])
f = lambda k: np.array([float(r[k]) for r in rows])
pos, step, ctx, rep = f("relative_position"), f("step"), f("context_before"), f("repeat_mention")
ctx = np.where(np.isnan(ctx), np.nanmean(ctx), ctx)
in_block = (step - 1) % 16
top5 = f("late_top5"); conf = f("p_image")
print(f"AUROC step-within-block alone: {auroc(in_block, y.astype(bool)):.3f}")
for name, cols in [("position", [pos]), ("position + repeat", [pos, rep]),
                   ("position + repeat + context_before", [pos, rep, ctx]),
                   ("position + repeat + step-in-block", [pos, rep, in_block]),
                   ("position + repeat + context + top5", [pos, rep, ctx, top5]),
                   ("position + repeat + context + top5 + confidence", [pos, rep, ctx, top5, conf])]:
    print(f"  {name:<50} CV AUROC {auroc(logistic_cv(np.c_[tuple(cols)], y, g), y.astype(bool)):.3f}")
print("context_before AUROC within position thirds:")
for lo, hi in ((0, 1/3), (1/3, 2/3), (2/3, 1.01)):
    s = (pos >= lo) & (pos < hi)
    print(f"  position {lo:.2f}-{min(hi,1):.2f}: n={s.sum():3d} hallucinated={int(y[s].sum()):2d}  "
          f"context_before {auroc(ctx[s], y[s].astype(bool)):.3f}  top5 {auroc(top5[s], y[s].astype(bool)):.3f}")
