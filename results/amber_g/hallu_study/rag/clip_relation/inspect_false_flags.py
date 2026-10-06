"""Which grounded nouns does the CLIP detector (rank_full < 0.9) flag, and how do simple second-stage filters trade off?"""

import collections
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(os.path.dirname(HERE))
AMBER = os.path.join(STUDY, "..", "..", "..", "third_party", "AMBER", "data", "annotations.json")

base = {(r["id"], r["pos"]): r for r in csv.DictReader(open(os.path.join(STUDY, "with_clean", "nouns.csv")))}
rows = [dict(base[(r["id"], r["pos"])], **r) for r in csv.DictReader(open(os.path.join(HERE, "nouns_clip.csv")))]
flagged = [r for r in rows if float(r["rank_full"]) < 0.9]
for label in ("grounded", "hallucinated"):
    counts = collections.Counter(r["lemma"] for r in flagged if r["label"] == label)
    print(f"flagged {label} ({sum(counts.values())}): {counts.most_common(25)}")

annotations = json.load(open(AMBER))
print("\nfalse flags (grounded nouns flagged by CLIP):")
for r in [r for r in flagged if r["label"] == "grounded"][:16]:
    print(f"  id {r['id']:>4s} {r['word']:12s} rank_full={float(r['rank_full']):.2f} support={float(r['support_full']):.1f} "
          f"top5={float(r['late_top5']):.3f} | truth={annotations[int(r['id']) - 1]['truth']}")

y = np.array([r["label"] == "hallucinated" for r in flagged])
n_hallucinated = sum(r["label"] == "hallucinated" for r in rows)
col = lambda f: np.array([float(r[f]) for r in flagged])
support, top5, rank_tight = col("support_full"), col("late_top5"), col("rank_tight")


def show(name, keep):
    print(f"{name:52s} kept {keep.sum():3d} precision {y[keep].mean():.3f} recall {y[keep].sum() / n_hallucinated:.3f} | "
          f"grounded removed {(~y & ~keep).sum()}/{(~y).sum()}, hallucinated lost {(y & ~keep).sum()}/{y.sum()}")


print()
show("CLIP rank_full < 0.9 alone", np.ones(len(flagged), bool))
show("+ drop if support_full >= 0.3", support < 0.3)
show("+ drop if late_top5 < 0.06", top5 >= 0.06)
show("+ drop if support_full >= 0.3 or late_top5 < 0.06", (support < 0.3) & (top5 >= 0.06))
show("+ drop if rank_tight >= 0.95", rank_tight < 0.95)
for t in (0.8, 0.85):
    show(f"stricter CLIP alone: rank_full < {t}", col("rank_full") < t)

# Background ("stuff") words, COCO-Stuff-style scene categories. Written after seeing the flagged words above,
# so the gain of this filter on these nouns is optimistic; it needs checking on held-out captions.
STUFF = {"sky", "cloud", "wall", "floor", "ground", "grass", "tree", "bush", "plant", "road", "street", "path", "sidewalk",
         "water", "sea", "ocean", "river", "lake", "beach", "sand", "snow", "mountain", "hill", "field", "forest", "dirt",
         "court", "building", "fence", "rock", "leaf", "sun"}
is_stuff = np.array([r["lemma"] in STUFF for r in flagged])
print()
show("+ never flag background words", ~is_stuff)
show("+ background words only if rank_full < 0.5", ~is_stuff | (col("rank_full") < 0.5))
show("+ never flag background, drop if rank_tight >= 0.95", ~is_stuff & (rank_tight < 0.95))
print("flagged background words: grounded", int((is_stuff & ~y).sum()), "hallucinated", int((is_stuff & y).sum()),
      collections.Counter(r["lemma"] for r in flagged if r["lemma"] in STUFF and r["label"] == "hallucinated"))
all_stuff = np.array([r["lemma"] in STUFF for r in rows]); all_y = np.array([r["label"] == "hallucinated" for r in rows])
print(f"all nouns: background words are {all_stuff.mean():.0%} of nouns, hallucination rate {all_y[all_stuff].mean():.3f} "
      f"vs {all_y[~all_stuff].mean():.3f} for objects")
