"""AUROCs of re-prediction confidence (repredict_confidence.py) as a hallucination detector, alone, with CLIP and within CLIP's flags.

Score direction: higher => hallucinated (so confidences enter as 1 - p). 95% CIs are bootstrapped over captions.
Writes <repredict dir>/analysis.json and prints the tables.

Example:
    python analyze_repredict.py
"""

import argparse
import csv
import json
import os

import numpy as np

from analyze_hallu_study import auroc, bootstrap_auroc, logistic_cv
from remask_decoding import STUDY

PERTURB = ["plain", "random5", "random5b", "zoom"]


def load(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repredict", default=os.path.join(STUDY, "repredict", "repredict.csv"))
    parser.add_argument("--clip", default=os.path.join(STUDY, "rag", "clip_relation", "nouns_clip.csv"))
    args = parser.parse_args()
    rows = load(args.repredict)
    clip = {(r["id"], r["pos"]): r for r in load(args.clip)}
    rows = [dict(r, **{k: v for k, v in clip[(r["id"], r["pos"])].items() if k not in r}) for r in rows]
    y = np.array([r["label"] == "hallucinated" for r in rows])
    groups = np.array([int(r["id"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    p = {c: col(f"{c}_p") for c in ["plain", "plain_nb", "random5", "random5b", "noimage", "zoom"]}
    lp = {c: col(f"{c}_logp_mean") for c in p}
    p_min = np.min([p[c] for c in PERTURB], axis=0)
    p_min_ctx = np.min([p[c] for c in ["plain", "random5", "random5b"]], axis=0)
    signals = {
        "1 - p plain": 1 - p["plain"],
        "1 - p plain, neighbours masked": 1 - p["plain_nb"],
        "1 - p random5": 1 - p["random5"],
        "1 - p random5b": 1 - p["random5b"],
        "1 - p no image": 1 - p["noimage"],
        "1 - p zoom": 1 - p["zoom"],
        "1 - min p (plain, random x2, zoom)": 1 - p_min,
        "1 - min p (plain, random x2)": 1 - p_min_ctx,
        "spread = p plain - min p": p["plain"] - p_min,
        "- logp_mean plain (all word tokens)": -lp["plain"],
        "entropy plain": col("plain_entropy"),
        "-(log p plain - log p no image)": -(np.log(p["plain"] + 1e-9) - np.log(p["noimage"] + 1e-9)),
        "top-1 differs from original (plain)": np.array([r["plain_top1"] != r["token"] for r in rows], float),
        "1 - commit confidence (baseline trace)": 1 - col("p_image"),
        "1 - CLIP rank_full": 1 - col("rank_full"),
    }
    out = dict(n=len(rows), n_hallucinated=int(y.sum()), n_captions=len(set(groups)), auroc={}, cv={}, within_clip={},
               operating_points={}, two_stage={})
    print(f"{len(rows)} nouns, {int(y.sum())} hallucinated, {len(set(groups))} captions\n")
    print(f"{'signal':42s} AUROC  95% CI       halluc mean  grounded mean")
    for name, s in signals.items():
        a, ci = auroc(s, y), bootstrap_auroc(s, y, groups)
        out["auroc"][name] = dict(auroc=a, ci=ci, mean_h=float(s[y].mean()), mean_g=float(s[~y].mean()))
        print(f"{name:42s} {a:.3f}  {ci[0]:.2f}-{ci[1]:.2f}   {s[y].mean():8.3f}  {s[~y].mean():8.3f}")

    # Grouped CV logistic regression.
    logit = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
    rep = np.c_[[logit(p[c]) for c in p] + [lp[c] for c in p] + [col("plain_entropy")]].T
    clipf = np.c_[col("rank_full"), col("support_full")]
    order = np.c_[col("relative_position"), col("repeat_mention"), col("context_before")]
    sets = {"re-prediction (all conditions)": rep, "re-prediction plain only": np.c_[logit(p["plain"]), lp["plain"]],
            "CLIP rank_full + support_full": clipf, "re-prediction + CLIP": np.c_[rep, clipf],
            "re-prediction + CLIP + decoding order": np.c_[rep, clipf, order]}
    print("\ngrouped 5-fold CV logistic regression")
    for name, x in sets.items():
        a = auroc(logistic_cv(x, y.astype(float), groups), y)
        out["cv"][name] = a
        print(f"  {name:42s} {a:.3f}")

    # Fixed operating points of re-prediction alone, vs CLIP.
    print("\noperating points over all nouns (precision / recall / fire rate)")
    def op(flag):
        tp = int((flag & y).sum())
        return dict(flagged=int(flag.sum()), tp=tp, precision=tp / max(flag.sum(), 1), recall=tp / y.sum(),
                    fire_rate=float(flag.mean()))
    points = {"CLIP rank_full < 0.9": col("rank_full") < 0.9}
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        points[f"p plain < {t}"] = p["plain"] < t
        points[f"min p < {t}"] = p_min < t
    for name, flag in points.items():
        out["operating_points"][name] = op(flag)
        o = out["operating_points"][name]
        print(f"  {name:28s} {o['flagged']:4d} flagged  P {o['precision']:.2f}  R {o['recall']:.2f}  rate {o['fire_rate']:.3f}")

    # Within CLIP's flags: does re-prediction separate CLIP's false flags from real hallucinations?
    flagged = col("rank_full") < 0.9
    print(f"\nwithin CLIP flags (rank_full < 0.9): {int(flagged.sum())} nouns, {int((flagged & y).sum())} hallucinated")
    for name, s in signals.items():
        if "CLIP" in name:
            continue
        a, ci = auroc(s[flagged], y[flagged]), bootstrap_auroc(s[flagged], y[flagged], groups[flagged])
        out["within_clip"][name] = dict(auroc=a, ci=ci)
        print(f"  {name:42s} {a:.3f}  {ci[0]:.2f}-{ci[1]:.2f}")
    print("\ntwo-stage: CLIP flag kept only if the re-prediction signal is below t (precision / recall over all hallucinated)")
    for sig, values in (("p plain", p["plain"]), ("min p", p_min), ("p zoom", p["zoom"])):
        for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            keep = flagged & (values < t)
            o = op(keep)
            o.update(grounded_removed=int((flagged & ~y & ~keep).sum()), hallucinated_lost=int((flagged & y & ~keep).sum()))
            out["two_stage"][f"{sig} < {t}"] = o
            print(f"  {sig:8s} < {t:<5} P {o['precision']:.2f}  R {o['recall']:.2f}  "
                  f"grounded removed {o['grounded_removed']:3d}/{int((flagged & ~y).sum())}  "
                  f"hallucinated lost {o['hallucinated_lost']:3d}/{int((flagged & y).sum())}")
    print("\nunion: CLIP flag OR re-prediction flag")
    for t in (0.3, 0.5, 0.7):
        for sig, values in (("p plain", p["plain"]), ("min p", p_min)):
            o = op(flagged | (values < t))
            out["two_stage"][f"union CLIP or {sig} < {t}"] = o
            print(f"  CLIP or {sig} < {t}: P {o['precision']:.2f}  R {o['recall']:.2f}  rate {o['fire_rate']:.3f}")

    path = os.path.join(os.path.dirname(args.repredict), "analysis.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
