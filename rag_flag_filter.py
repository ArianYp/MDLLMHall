"""Among nouns the CLIP detector flags (rank_full < threshold), what separates the real objects from the hallucinations?

Joins rag/clip_relation/nouns_clip.csv (CLIP features) with with_clean/nouns.csv (MMaDA features at the
noun's commit step in the baseline trace: confidence, image ablation, attention, decoding order).
Label-derived columns (annotated_absent, caption_hallucinates) are left out.

Within the flagged set it reports, per feature, the AUROC for "higher => hallucinated" with a 95% bootstrap
CI over captions, grouped-CV logistic regression on feature groups, and, for single-feature second-stage
filters (keep a flag only on one side of a threshold), the precision and recall of the two-stage detector
over all hallucinated nouns. Output: rag/clip_relation/flag_filter.json.

Example:
    python rag_flag_filter.py --threshold 0.9
"""

import argparse
import csv
import os

import numpy as np

from rag_clip_relation import auroc, bootstrap
from remask_decoding import STUDY, write_json

MMADA = ["p_image", "entropy_image", "dlogp", "p_no_image", "no_image_top1", "kl", "late_image_mass",
         "late_image_entropy", "late_min_head_entropy", "late_top5", "step", "relative_position",
         "context_before", "right_before", "repeat_mention"]
CLIP = ["sim_full", "support_full", "ret_full", "top1_full"] + [f"{a}_{v}" for v in ("super_tight", "tight", "loose")
                                                                for a in ("sim", "rank", "support", "ret")]
GROUPS = dict(language_prior=["p_no_image", "dlogp", "kl", "no_image_top1"],
              attention=["late_image_mass", "late_image_entropy", "late_min_head_entropy", "late_top5"],
              confidence=["p_image", "entropy_image"],
              decoding_order=["step", "relative_position", "context_before", "right_before", "repeat_mention"],
              mmada_all=MMADA, clip_crops=CLIP, everything=MMADA + CLIP)


def cv_auroc(X, y, groups, folds=5, seed=0, l2=1e-2, iters=4000, lr=0.1):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    score = np.zeros(len(y))
    for k in range(folds):
        test = np.isin(groups, unique[k::folds])
        mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-9
        A = np.c_[np.ones((~test).sum()), (X[~test] - mu) / sd]
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            w -= lr * (A.T @ (p - y[~test]) / len(A) + l2 * np.r_[0, w[1:]])
        score[test] = np.c_[np.ones(test.sum()), (X[test] - mu) / sd] @ w
    return auroc(score, y.astype(bool))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--clip", default=os.path.join(STUDY, "rag", "clip_relation", "nouns_clip.csv"))
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"))
    args = parser.parse_args()

    with open(args.nouns, encoding="utf-8") as handle:
        base = {(r["id"], r["pos"]): r for r in csv.DictReader(handle)}
    with open(args.clip, encoding="utf-8") as handle:
        rows = [dict(base[(r["id"], r["pos"])], **r) for r in csv.DictReader(handle)]
    flagged = [r for r in rows if float(r["rank_full"]) < args.threshold]
    y = np.array([r["label"] == "hallucinated" for r in flagged])
    groups = np.array([int(r["id"]) for r in flagged])
    n_hallucinated = sum(r["label"] == "hallucinated" for r in rows)
    print(f"flagged {len(flagged)} of {len(rows)} nouns: {y.sum()} hallucinated, {(~y).sum()} grounded, "
          f"{len(set(groups))} captions | precision {y.mean():.3f} recall {y.sum() / n_hallucinated:.3f}\n")

    value = lambda f: np.array([float(r[f]) for r in flagged])
    report = dict(threshold=args.threshold, n_flagged=len(flagged), n_flagged_hallucinated=int(y.sum()),
                  n_hallucinated_total=n_hallucinated, features={}, groups={}, filters={})
    print(f"{'feature':24s} {'AUROC':>6s}  {'95% CI':>11s}  {'hallucinated':>12s} {'grounded':>9s}   (AUROC: higher => hallucinated)")
    for f in MMADA + CLIP:
        v = value(f)
        a, ci = auroc(v, y), bootstrap(v, y, groups)
        report["features"][f] = dict(auroc=a, ci=ci, hallucinated_mean=float(v[y].mean()), grounded_mean=float(v[~y].mean()))
        mark = " *" if ci[0] > 0.5 or ci[1] < 0.5 else ""
        print(f"{f:24s} {a:6.3f}  {ci[0]:.2f}-{ci[1]:.2f}   {v[y].mean():12.3f} {v[~y].mean():9.3f}{mark}")

    print("\ngrouped 5-fold CV logistic regression within the flagged set:")
    for name, cols in GROUPS.items():
        X = np.stack([value(c) for c in cols], 1)
        report["groups"][name] = a = cv_auroc(X, y.astype(float), groups)
        print(f"  {name:16s} {a:.3f}")

    # Single-feature second stage: drop flags on the side of a threshold that looks grounded. The threshold is a quantile
    # of the feature within the flagged set, so this is optimistic (chosen on the same nouns); it shows the trade-off.
    print("\nsecond-stage filters (keep the flag only if the feature is on the hallucinated side):")
    print(f"  {'feature':22s} {'cut':>8s} {'kept':>5s} {'precision':>9s} {'recall':>7s}  {'grounded removed':>16s} {'hallucinated lost':>17s}")
    for f, info in sorted(report["features"].items(), key=lambda kv: -abs(kv[1]["auroc"] - 0.5))[:8]:
        v = value(f)
        side = 1 if info["auroc"] > 0.5 else -1
        best = []
        for q in (0.1, 0.2, 0.3):
            cut = np.quantile(v, q if side == 1 else 1 - q)
            keep = v * side > cut * side if f != "repeat_mention" else v < 0.5
            best.append(dict(quantile=q, cut=float(cut), kept=int(keep.sum()), precision=float(y[keep].mean()),
                             recall=float(y[keep].sum() / n_hallucinated), grounded_removed=int((~y & ~keep).sum()),
                             hallucinated_lost=int((y & ~keep).sum())))
            b = best[-1]
            print(f"  {f:22s} {b['cut']:8.3f} {b['kept']:5d} {b['precision']:9.3f} {b['recall']:7.3f}  "
                  f"{b['grounded_removed']:>8d} / {(~y).sum():<6d} {b['hallucinated_lost']:>9d} / {y.sum()}")
            if f == "repeat_mention":
                break
        report["filters"][f] = best
    write_json(os.path.join(STUDY, "rag", "clip_relation", "flag_filter.json"), report, indent=1)
    print(f"\nwrote {os.path.join(STUDY, 'rag', 'clip_relation', 'flag_filter.json')}")


if __name__ == "__main__":
    main()
