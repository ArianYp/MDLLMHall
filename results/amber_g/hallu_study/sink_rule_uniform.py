"""Sink rule with a uniformity check: a sink must be the top patch for >= half the tokens AND near-uniform (rel_std < T).

1) Offline crop ranks (rag/clip_relation/crop_veto_nosink.csv): each caption has at most one candidate, so a noun's
   crop is either the original or the sink-removed one; no new CLIP needed.
2) The 22 first triggers whose crop moved in the L40S oracle+zoom runs: which still move under the new rule,
   redrawn to figures/moved_sinks/ (grey = crop with sink, red = sink removed, rel_std of each removed patch).

CPU only. Example (from MDLLM/): python results/amber_g/hallu_study/sink_rule_uniform.py --max-rel-std 0.35
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw

from sink_uniformity import HERE, IMAGES, model_view


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def rel_std(view, p):
    r, c = divmod(p, 32)
    return float(view[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].std() / view.std())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rel-std", type=float, default=0.35)
    args = parser.parse_args()
    t = args.max_rel_std

    # 1) offline crop ranks
    cand = {int(r["id"]): float(r["rel_std"]) for r in csv.DictReader(open(os.path.join(HERE, "figures", "sink_candidates.csv")))}
    rows = list(csv.DictReader(open(os.path.join(HERE, "rag", "clip_relation", "crop_veto_nosink.csv"))))
    held = {(r["id"], r["pos"]) for r in csv.DictReader(open(os.path.join(
        HERE, "rag", "clip_relation", "wordnet_rival", "rows_tight_nosink_heldout.csv")))}
    for r in rows:
        keep = int(r["id"]) in cand and cand[int(r["id"])] < t
        r["rank_tight_uniform"] = float(r["rank_tight_nosink"] if keep else r["rank_tight_orig"])
    n_sink = sum(1 for v in cand.values() if v < t)
    print(f"offline: {n_sink} / {len(cand)} candidate sinks pass rel_std < {t}; "
          f"nouns whose crop moves: {sum(r['rank_tight_uniform'] != float(r['rank_tight_orig']) or (int(r['id']) in cand and cand[int(r['id'])] < t and r['crop_moved'] == '1') for r in rows)}")
    for name, S in (("all 1050", rows), ("held-out 150", [r for r in rows if (r["id"], r["pos"]) in held])):
        H = [r for r in S if r["label"] == "hallucinated"]
        G = [r for r in S if r["label"] == "grounded"]
        print(f"\n== {name}: AUROC of tight-crop rank (lower => hallucinated)")
        for k in ("orig", "nosink", "uniform"):
            key = f"rank_tight_{k}"
            print(f"   {k:<8} {auroc([-float(r[key]) for r in H], [-float(r[key]) for r in G]):.3f}", end="")
            F = [r for r in S if float(r["rank_full"]) < 0.9]
            fh = sum(float(r[key]) < 0.95 for r in F if r["label"] == "hallucinated")
            fg = sum(float(r[key]) < 0.95 for r in F if r["label"] == "grounded")
            print(f"   | CLIP < 0.9 + veto >= 0.95: {fh} hallucinated, {fg} grounded, precision {fh / max(1, fh + fg):.2f}, recall {fh / len(H):.2f}")

    # 2) the 22 moved first triggers of the L40S runs
    hall = set(json.load(open(os.path.join(HERE, "ids.json")))["ids"])
    sr = os.path.join(HERE, "rag", "slot_refill")
    def first_events(name):
        return {c["id"]: c["events"][0] for c in json.load(open(os.path.join(sr, name, "predictions_events.json")))["captions"]
                if c["id"] in hall and c["events"]}
    A, B = first_events("oracle_zoom_l40s"), first_events("oracle_zoom_nosink")
    changed = lambda e: e["old_text"][e["remasked"].index(e["pos"])] != e["new_text"][e["remasked"].index(e["pos"])]
    out = os.path.join(HERE, "figures", "moved_sinks")
    os.makedirs(out, exist_ok=True)
    still = []
    print(f"\n22 moved first triggers under rel_std < {t}:")
    for i in sorted(set(A) & set(B)):
        a, b = A[i], B[i]
        if not (a["pos"] == b["pos"] and a["old_text"] == b["old_text"] and a["step"] == b["step"]) or a["box"] == b["box"]:
            continue
        view = model_view(i)
        stds = [rel_std(view, p) for p in b["sinks"]]
        moves = any(s < t for s in stds)
        still.append((i, moves))
        print(f"  {i:>4} {a['lemma']:<9} sinks rel_std {[round(s, 2) for s in stds]} -> {'still removed' if moves else 'KEPT (crop as with sink)'}"
              f" | with sink {''.join(a['new_text'])!r} {'changed' if changed(a) else 'kept'}"
              f" | removed {''.join(b['new_text'])!r} {'changed' if changed(b) else 'kept'}")
        photo = Image.open(os.path.join(IMAGES, f"AMBER_{i}.jpg")).convert("RGB")
        im = photo.copy()
        d = ImageDraw.Draw(im)
        lw = max(2, min(photo.size) // 120)
        d.rectangle(a["box"], outline=(160, 160, 160), width=lw)
        d.rectangle(b["box"], outline=(255, 0, 0), width=lw)
        h = 300
        full = im.resize((int(im.width * h / im.height), h))
        sheet = Image.new("RGB", (full.width + 2 * h + 20, h + 58), "white")
        sheet.paste(full, (0, 44))
        sheet.paste(photo.crop(tuple(a["box"])).resize((h, h)), (full.width + 10, 44))
        sheet.paste(photo.crop(tuple(b["box"])).resize((h, h)), (full.width + h + 20, 44))
        g = ImageDraw.Draw(sheet)
        g.text((5, 4), f"{i} flagged '{a['lemma']}' span {''.join(a['old_text'])!r}  sink rel_std {[round(s, 2) for s in stds]} "
                       f"-> {'removed under uniform rule' if moves else 'NOT a sink under uniform rule'}", fill="black")
        g.text((full.width + 12, 18), f"WITH sink (grey) -> {''.join(a['new_text'])!r}", fill=(90, 90, 90))
        g.text((full.width + 12, 30), f"p(orig)={a['p_original_refill']:.2f} {'CHANGED' if changed(a) else 'kept'}", fill=(90, 90, 90))
        g.text((full.width + h + 22, 18), f"sink REMOVED (red) -> {''.join(b['new_text'])!r}", fill=(200, 0, 0))
        g.text((full.width + h + 22, 30), f"p(orig)={b['p_original_refill']:.2f} {'CHANGED' if changed(b) else 'kept'}", fill=(200, 0, 0))
        sheet.save(os.path.join(out, f"{i}_{a['lemma']}.jpg"), quality=85)
    print(f"  still moved under the uniform rule: {sum(m for _, m in still)} / {len(still)}; drawings in {out}")


if __name__ == "__main__":
    main()
