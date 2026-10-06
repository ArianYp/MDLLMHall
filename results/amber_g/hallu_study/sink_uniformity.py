"""How uniform are the image patches that the token-count sink rule picks?

Candidate sink = patch that is the top late-layer patch for >= half of the caption's 128 tokens (the rule used so far).
For each candidate: rel_std = pixel std (grey) of the 16x16 patch in the model's 512 view / std of the whole view.
True sinks sit on featureless background (low rel_std); the main subject of the photo does not.

Output: figures/sink_candidates.csv and a printed summary. CPU only.

Example (from MDLLM/):
    python results/amber_g/hallu_study/sink_uniformity.py
"""

import csv
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IMAGES = os.path.join(ROOT, "data", "amber", "images")
OUT = os.path.join(HERE, "figures")


def model_view(item_id, resolution=512):
    """Grey image as the model sees it: short side resized to `resolution`, centre-cropped."""
    im = Image.open(os.path.join(IMAGES, f"AMBER_{item_id}.jpg")).convert("L")
    w, h = im.size
    s = resolution / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    left, top = (im.width - resolution) // 2, (im.height - resolution) // 2
    return np.asarray(im, dtype=np.float64)[top:top + resolution, left:left + resolution]


def late_maps(item_id):
    for d in ("traces", "traces_clean"):
        path = os.path.join(HERE, d, f"{item_id}.npz")
        if os.path.exists(path):
            return np.load(path)["late_map"].astype(np.float64)
    raise FileNotFoundError(item_id)


def candidates(item_id, frac=0.5):
    maps = late_maps(item_id)
    counts = np.bincount(maps.argmax(1), minlength=maps.shape[1])
    view = model_view(item_id)
    out = []
    for p in np.nonzero(counts >= frac * maps.shape[0])[0]:
        r, c = divmod(int(p), 32)
        patch = view[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        out.append(dict(id=item_id, patch=int(p), row=r, col=c, tokens=int(counts[p]),
                        share=float(maps[:, p].mean()), rel_std=float(patch.std() / view.std())))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = sorted({int(f.split(".")[0]) for d in ("traces", "traces_clean") for f in os.listdir(os.path.join(HERE, d))
                  if f.endswith(".npz")})
    rows = [c for i in ids for c in candidates(i)]
    with open(os.path.join(OUT, "sink_candidates.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(ids)} captions, {len(rows)} candidate sink patches in {len({r['id'] for r in rows})} captions")

    # Cases judged by eye (crop images of 2026-10-05): true sinks vs real objects the rule removed.
    known = {523: "sink (sky)", 738: "sink (grass)", 449: "sink (box corner)", 206: "sink (court)", 526: "sink (sky)",
             130: "sink (surf)", 312: "sink (laptop edge)", 371: "sink (corner)",
             660: "OBJECT (head behind umbrella)", 137: "OBJECT (the girl)", 580: "OBJECT (toast)"}
    print("\nknown cases:")
    for r in rows:
        if r["id"] in known:
            print(f"  {r['id']:>4} ({r['row']:>2},{r['col']:>2}) tokens {r['tokens']:>3} share {r['share']:.3f} "
                  f"rel_std {r['rel_std']:.2f}  {known[r['id']]}")
    v = np.array([r["rel_std"] for r in rows])
    print("\nrel_std over all candidates: " + ", ".join(f"p{q} {np.percentile(v, q):.2f}" for q in (10, 25, 50, 75, 90)))
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        print(f"  rel_std < {t}: {int((v < t).sum())} / {len(v)} candidates kept as sinks")


if __name__ == "__main__":
    main()
