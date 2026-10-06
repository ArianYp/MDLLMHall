"""How often would the remask rule fire on the baseline trajectories? (approximate: trajectories change once it fires)"""
import glob, json
import numpy as np
L = slice(19, 32)
rows = []
for path in sorted(glob.glob("results/amber_g/hallu_study/traces/*.json")):
    t = json.load(open(path))
    for r in t["records"]:
        tok = t["tokens"][r["pos"]]
        if tok == "<|endoftext|>":
            continue
        rows.append((t["id"], r["step"], np.mean(r["image_mass_by_layer"][L]), r["logp_no_image"] - r["logp_image"], tok))
mass = np.array([r[2] for r in rows]); up = np.array([r[3] for r in rows]); step = np.array([r[1] for r in rows])
thr = float(np.median(mass))
print(f"content tokens {len(rows)}; median late-layer image mass = {thr:.4f}")
fire = (step >= 15) & (up > 0) & (mass >= thr)
print(f"tokens fixed at step>=15: {(step>=15).sum()}; rule fires on {fire.sum()} ({fire.sum()/max((step>=15).sum(),1):.1%}); per caption mean {fire.sum()/len(set(r[0] for r in rows)):.1f}")
from collections import Counter
print("most common triggering tokens:", Counter(rows[i][4] for i in np.flatnonzero(fire)).most_common(15))
json.dump(dict(late_image_mass_median=thr), open("results/amber_g/hallu_study/attention_threshold.json", "w"))
