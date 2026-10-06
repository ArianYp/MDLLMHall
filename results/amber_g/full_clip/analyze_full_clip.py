"""Detector variants on the full-set baseline nouns (nouns_full.csv), split into study / non-study / held-out ids 730-1004."""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from analyze_hallu_study import auroc, bootstrap_auroc, logistic_cv  # noqa: E402
STUFF = {"sky", "cloud", "wall", "floor", "ground", "grass", "tree", "bush", "plant", "road", "street", "path", "sidewalk",
         "water", "sea", "ocean", "river", "lake", "beach", "sand", "snow", "mountain", "hill", "field", "forest", "dirt",
         "court", "building", "fence", "rock", "leaf", "sun"}
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "nouns_full.csv"))) if r["in_vocabulary"] == "True"]
y = np.array([r["label"] == "hallucinated" for r in rows]); g = np.array([int(r["id"]) for r in rows])
study = np.array([r["study"] == "True" for r in rows]); stuff = np.array([r["lemma"] in STUFF for r in rows])
col = lambda k: np.array([float(r[k]) for r in rows])
out = {}
splits = {"all 1004": np.ones(len(rows), bool), "non-study (860)": ~study, "held-out ids 730-1004, non-study": (g >= 730) & ~study}
print(f"{len(rows)} nouns in vocabulary, {y.sum()} hallucinated")
for sname, m in splits.items():
    print(f"\n== {sname}: {m.sum()} nouns, {y[m].sum()} hallucinated, {len(set(g[m]))} captions, base rate {y[m].mean():.3f}")
    out[sname] = dict(auroc={}, points={})
    for p in ["detector", "bare", "there", "word", "ensemble"]:
        s = 1 - col(f"rank_{p}")
        a = auroc(s[m], y[m]); ci = bootstrap_auroc(s[m], y[m], g[m], n=500)
        ao, ab = auroc(s[m & ~stuff], y[m & ~stuff]), auroc(s[m & stuff], y[m & stuff])
        out[sname]["auroc"][p] = dict(all=a, ci=ci, objects=ao, background=ab)
        print(f"  rank_{p:9s} AUROC {a:.3f} ({ci[0]:.2f}-{ci[1]:.2f})  objects {ao:.3f}  background {ab:.3f}")
    rk = col("rank_detector")
    pts = {"CLIP < 0.9 (current)": rk < 0.9, "CLIP < 0.85": rk < 0.85, "CLIP < 0.8": rk < 0.8,
           "never background, CLIP < 0.9": (rk < 0.9) & ~stuff,
           "background only if < 0.5, else < 0.9": np.where(stuff, rk < 0.5, rk < 0.9),
           "never background, CLIP < 0.95": (rk < 0.95) & ~stuff,
           "never background, CLIP < 0.8": (rk < 0.8) & ~stuff}
    for pname, f in pts.items():
        f = f & m; tp = int((f & y).sum())
        o = dict(flagged=int(f.sum()), tp=tp, precision=tp / max(f.sum(), 1), recall=tp / max((y & m).sum(), 1), rate=f.sum() / m.sum())
        out[sname]["points"][pname] = o
        print(f"  {pname:40s} flagged {o['flagged']:4d}  P {o['precision']:.2f}  R {o['recall']:.2f}  rate {o['rate']:.3f}")
json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)
