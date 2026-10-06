import json
import numpy as np
def spearmanr(a, b):
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1]), float("nan")

d = np.load("results/amber_g/traces/image_attention_amber_163.npz")
meta = json.load(open("results/amber_g/traces/image_attention_amber_163.json"))
abl = json.load(open("results/amber_g/traces/image_ablation_amber_163.json"))
dl = {x["pos"]: x["image"]["logprob"] - x["no_image"]["logprob"] for s in abl["steps"] for x in s["commits"]}
L = slice(19, 32)
G = 32
mass = d["image_mass"][:, L].astype(float)
ent = d["image_entropy"][:, L].astype(float)
mp = d["image_map"][:, L].astype(float)
rows = []
for i, t in enumerate(meta["tokens"]):
    m = mp[i].mean(0)
    m /= m.sum()
    top = int(m.argmax())
    r, c = divmod(top, G)
    x = 333 + (c + 0.5) * 1333 / G
    y = (r + 0.5) * 1333 / G
    top5 = np.sort(m)[-5:].sum()
    pos = t["pos"]
    rows.append((mass[i].mean(), ent[i].mean(), top5, dl[pos]))
    print(f"{pos:3d} {t['token']!r:13} mass {mass[i].mean():.3f}  ent {ent[i].mean():.3f}  "
          f"minhead {ent[i].min(1).mean():.3f}  top {m[top]:.3f}  top5 {top5:.3f}  "
          f"patch {(r, c)}  px {(int(x), int(y))}  dlogp {dl[pos]:+.2f}")
a = np.array(rows)
for j, n in enumerate(["mass", "entropy", "top5 share"]):
    rho, p = spearmanr(a[:, j], a[:, 3])
    print(f"spearman({n}, dlogp) = {rho:+.2f} (p={p:.3f}, n={len(a)})")
