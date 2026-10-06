import json
import numpy as np

d = np.load("results/amber_g/traces/image_attention_amber_163.npz")
meta = json.load(open("results/amber_g/traces/image_attention_amber_163.json"))
tok = [t["token"] for t in meta["tokens"]]
mass, ent, mp = (d[k].astype(float) for k in ("image_mass", "image_entropy", "image_map"))
content = [i for i, t in enumerate(tok) if t.strip() in
           {"woman", "red", "hair", "mouth", "cup", "drink", "plate", "bowl", "eggs", "green", "apples", "broccoli"}]
def top5(i, layers):
    m = mp[i, layers].mean(0); m /= m.sum(); return np.sort(m)[-5:].sum(), divmod(int(m.argmax()), 32)
print("layer | img mass: The  image  objects | entropy: The  image  objects | top patch The / image")
for l in range(32):
    pt = [divmod(int(mp[i, l].argmax()), 32) for i in (0, 1)]
    print(f"{l:5d} | {mass[0,l].mean():14.3f} {mass[1,l].mean():6.3f} {mass[content,l].mean():8.3f} | "
          f"{ent[0,l].mean():12.3f} {ent[1,l].mean():6.3f} {ent[content,l].mean():8.3f} | {pt[0]} / {pt[1]}")
late = slice(19, 32)
for i in (0, 1):
    s, p = top5(i, late)
    print(f"{tok[i]!r}: late-layer top-5 share {s:.3f}, top patch {p}, effective patches {1024 ** ent[i, late].mean():.0f}")
s = np.mean([top5(i, late)[0] for i in content])
print(f"object words: late-layer top-5 share {s:.3f}, effective patches {np.mean([1024 ** ent[i, late].mean() for i in content]):.0f}")
rank = sorted(range(len(tok)), key=lambda i: top5(i, late)[0])
print("least focused (late layers):", [(tok[i], round(top5(i, late)[0], 3)) for i in rank[:5]])
