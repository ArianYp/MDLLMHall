"""Do nouns whose zoom crop misses the object have more spread-out (higher-entropy) attention?

Features of the noun's late-layer attention map (first token, from the baseline trace):
  late_image_entropy, late_top5, late_image_mass   (from with_clean/nouns.csv)
  cluster    size of the connected top-30 cluster around the peak patch (what the crop is built from)
  peak_share max patch / total attention over the image
  map_entropy entropy of the normalised late_map (recomputed, nats)

1) Hand labels for crops inspected by eye (on = crop shows the word's object or the real thing it misread).
2) Proxy on all grounded nouns (object is present): crop misses = rank_tight < 0.5, crop hits = rank_tight >= 0.9.
AUROC is for "higher feature => crop misses". CPU only, numpy.
"""
import csv, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
N = list(csv.DictReader(open(os.path.join(H, "with_clean", "nouns.csv"))))
C = {(r["id"], r["pos"]): r for r in csv.DictReader(open(os.path.join(H, "rag", "clip_relation", "nouns_clip.csv")))}
maps = {}

def late_map(i):
    if i not in maps:
        d = "traces" if os.path.exists(os.path.join(H, "traces", f"{i}.npz")) else "traces_clean"
        maps[i] = np.load(os.path.join(H, d, f"{i}.npz"))["late_map"].astype(np.float64)
    return maps[i]

def map_features(i, pos):
    m = late_map(i)[int(pos)]
    p = m / m.sum()
    top = set(np.argsort(-m)[:30].tolist()); seed = int(np.argmax(m)); seen = {seed}; queue = [seed]
    while queue:
        r, c = divmod(queue.pop(), 32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr) * 32 + c + dc
                if 0 <= r + dr < 32 and 0 <= c + dc < 32 and n in top and n not in seen:
                    seen.add(n); queue.append(n)
    return dict(cluster=len(seen), peak_share=float(p.max()), map_entropy=float(-(p * np.log(p + 1e-12)).sum()))

def auroc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    pos, neg = x[y], x[~y]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))

FEATURES = ["late_image_entropy", "map_entropy", "late_top5", "peak_share", "cluster", "late_image_mass"]

def row(r):
    out = {k: float(r[k]) for k in ("late_image_entropy", "late_top5", "late_image_mass")}
    out.update(map_features(r["id"], r["pos"]))
    return out

# 1) hand labels: (id, lemma) -> crop misses the object?
HAND = {("523", "motorcycle"): 1, ("738", "horse"): 1, ("863", "chair"): 1, ("449", "cup"): 1, ("224", "snow"): 1,
        ("2", "sun"): 1, ("206", "ball"): 1, ("526", "people"): 1, ("130", "beach"): 1, ("4", "boat"): 1, ("312", "desk"): 1,
        ("60", "dog"): 0, ("159", "boat"): 0, ("494", "computer"): 0, ("428", "cow"): 0, ("517", "cow"): 0,
        ("616", "window"): 0, ("584", "ball"): 0, ("634", "road"): 0, ("660", "person"): 0, ("180", "flower"): 0,
        ("304", "sun"): 0, ("252", "backpack"): 0, ("32", "collar"): 0, ("354", "chair"): 0, ("670", "rope"): 0}
hand = []
for (i, lemma), miss in HAND.items():
    r = min((r for r in N if r["id"] == i and r["lemma"] == lemma), key=lambda r: int(r["pos"]))
    hand.append(dict(row(r), miss=miss, name=f"{i}:{lemma}"))
print(f"1) hand-labelled crops: {sum(h['miss'] for h in hand)} miss, {sum(1 - h['miss'] for h in hand)} on object")
print(f"   {'feature':<20}{'miss mean':>10}{'on mean':>10}{'AUROC':>8}")
for f in FEATURES:
    a = [h[f] for h in hand if h["miss"]]; b = [h[f] for h in hand if not h["miss"]]
    print(f"   {f:<20}{np.mean(a):>10.3f}{np.mean(b):>10.3f}{auroc([h[f] for h in hand], [h['miss'] for h in hand]):>8.2f}")
print("   per crop (sorted by late_image_entropy):")
for h in sorted(hand, key=lambda h: -h["late_image_entropy"]):
    print(f"     {'MISS' if h['miss'] else ' on '}  {h['name']:<16} entropy={h['late_image_entropy']:.2f} top5={h['late_top5']:.3f} peak={h['peak_share']:.3f} cluster={h['cluster']}")

# 2) proxy on grounded nouns
G = []
for r in N:
    if r["label"] != "grounded" or (r["id"], r["pos"]) not in C:
        continue
    t = float(C[(r["id"], r["pos"])]["rank_tight"])
    if t < 0.5 or t >= 0.9:
        G.append(dict(row(r), miss=int(t < 0.5), rank_tight=t))
print(f"\n2) grounded nouns, proxy: crop misses (rank_tight < 0.5) {sum(g['miss'] for g in G)} vs hits (>= 0.9) {sum(1 - g['miss'] for g in G)}")
print(f"   {'feature':<20}{'miss mean':>10}{'hit mean':>10}{'AUROC':>8}")
for f in FEATURES:
    a = [g[f] for g in G if g["miss"]]; b = [g[f] for g in G if not g["miss"]]
    print(f"   {f:<20}{np.mean(a):>10.3f}{np.mean(b):>10.3f}{auroc([g[f] for g in G], [g['miss'] for g in G]):>8.2f}")
miss_rate = lambda S: sum(g["miss"] for g in S) / max(1, len(S))
for f, cuts in (("cluster", [(1, 1), (2, 3), (4, 10), (11, 30)]),):
    print(f"   miss rate by {f}: " + ", ".join(f"{lo}-{hi}: {miss_rate([g for g in G if lo <= g[f] <= hi]):.2f} (n={len([g for g in G if lo <= g[f] <= hi])})" for lo, hi in cuts))
q = np.quantile([g["late_image_entropy"] for g in G], [0, .25, .5, .75, 1])
print("   miss rate by late_image_entropy quartile: " + ", ".join(
    f"Q{k + 1} [{q[k]:.2f},{q[k + 1]:.2f}]: {miss_rate([g for g in G if q[k] <= g['late_image_entropy'] <= q[k + 1]]):.2f}" for k in range(4)))
