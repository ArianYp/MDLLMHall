"""Compare hallucinated vs grounded object nouns across the batch traces.

Inputs: results/amber_g/hallu_study/traces/<id>.json from trace_amber_batch.py.
Each object noun in a caption is labelled with amber_noun_labels.AmberLabeler
(the official AMBER-g per-noun rules) and joined to its first answer token.

For every feature this prints the mean / median per group and the AUROC for
separating hallucinated from grounded nouns (0.5 = no signal; > 0.5 means the
feature is higher for hallucinated nouns), with a 95% bootstrap interval that
resamples captions. It also fits a small logistic regression with
caption-grouped cross-validation to see whether features combine.
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np

from amber_noun_labels import AmberLabeler

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(ROOT, "results", "amber_g", "hallu_study")
LATE = slice(19, 32)
EOT = "<|endoftext|>"

FEATURES = [
    ("p_image", "p(token | image) at commit  [model confidence]"),
    ("entropy_image", "entropy of the with-image prediction"),
    ("dlogp", "log p(image) - log p(no image)"),
    ("p_no_image", "p(token | no image) on same context"),
    ("no_image_top1", "no-image argmax == committed token"),
    ("kl", "KL(with image || without image)"),
    ("late_image_mass", "attention mass on image, layers 19-31"),
    ("late_image_entropy", "image attention entropy, layers 19-31"),
    ("late_min_head_entropy", "lowest-entropy head, layers 19-31"),
    ("late_top5", "top-5 patch share, layers 19-31"),
    ("step", "commit step (1-64)"),
    ("relative_position", "position / caption length"),
    ("context_before", "share of +-3 neighbours committed earlier"),
    ("right_before", "right neighbour committed earlier"),
    ("repeat_mention", "lemma already mentioned earlier"),
]


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    combined = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auroc(values, labels, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    out = []
    for _ in range(n):
        pick = np.concatenate([index[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        out.append(auroc(values[pick], labels[pick]))
    out = np.array(out)
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def logistic_cv(x, y, groups, folds=5, seed=0, l2=1.0, iters=500):
    """Caption-grouped k-fold logistic regression (numpy, standardised features). Returns out-of-fold scores."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    fold_of = {g: k % folds for k, g in enumerate(unique)}
    fold = np.array([fold_of[g] for g in groups])
    scores = np.zeros(len(y))
    for k in range(folds):
        train, test = fold != k, fold == k
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-8
        xt = np.c_[np.ones(train.sum()), (x[train] - mu) / sd]
        w = np.zeros(xt.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-xt @ w))
            grad = xt.T @ (p - y[train]) / len(p) + l2 * np.r_[0, w[1:]] / len(p)
            w -= 0.5 * grad
        scores[test] = np.c_[np.ones(test.sum()), (x[test] - mu) / sd] @ w
    return scores


def noun_rows(labeler, trace):
    text = trace["answer_raw"]
    records = trace["records"]
    tokens = trace["tokens"]
    content = [i for i, t in enumerate(tokens) if t != EOT]
    last = max(content) if content else 0
    steps = {r["pos"]: r["step"] for r in records}
    seen = set()
    rows = []
    for noun in labeler.label(trace["id"], text):
        if noun["label"] == "ignored" or noun["span"] is None:
            continue
        start, end = noun["span"]
        hits = [i for i, (a, b) in enumerate(trace["offsets"]) if a < end and b > start and tokens[i] != EOT]
        if not hits:
            continue
        pos = hits[0]
        r = records[pos]
        step = r["step"]
        neighbours = [p for p in range(pos - 3, pos + 4) if p != pos and 0 <= p <= last and p not in hits]
        rows.append(dict(
            id=trace["id"],
            word=noun["word"],
            lemma=noun["lemma"],
            label=noun["label"],
            annotated_absent=noun["annotated_absent"],
            pos=pos,
            n_tokens=len(hits),
            token=tokens[pos],
            p_image=math.exp(r["logp_image"]),
            entropy_image=r["entropy_image"],
            dlogp=r["logp_image"] - r["logp_no_image"],
            p_no_image=math.exp(r["logp_no_image"]),
            no_image_top1=float(r["no_image_argmax"] == r["token_id"]),
            kl=r["kl_image_vs_no_image"],
            late_image_mass=float(np.mean(r["image_mass_by_layer"][LATE])),
            late_image_entropy=float(np.mean(r["image_entropy_by_layer"][LATE])),
            late_min_head_entropy=float(np.mean(r["image_entropy_min_head_by_layer"][LATE])),
            late_top5=r["late_top5_share"],
            step=step,
            relative_position=pos / max(last, 1),
            context_before=float(np.mean([steps[p] < step for p in neighbours])) if neighbours else float("nan"),
            right_before=float(pos + len(hits) <= last and steps[pos + len(hits)] < step),
            repeat_mention=float(noun["lemma"] in seen),
        ))
        seen.add(noun["lemma"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Hallucinated vs grounded nouns across batch traces.")
    parser.add_argument("--traces", nargs="+", default=[os.path.join(STUDY, "traces")],
                        help="One or more trace directories (e.g. hallucinating and clean captions).")
    parser.add_argument("--out", default=STUDY)
    args = parser.parse_args()

    traces = []
    paths = sorted(p for directory in args.traces for p in glob.glob(os.path.join(directory, "*.json")))
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                traces.append(json.load(handle))
        except json.JSONDecodeError:
            print(f"skipping {path}: still being written")
    labeler = AmberLabeler()
    rows = [row for trace in traces for row in noun_rows(labeler, trace)]
    hall = np.array([r["label"] == "hallucinated" for r in rows])
    groups = np.array([r["id"] for r in rows])
    n_match = sum(t["matches_saved"] for t in traces)
    hallucinating_ids = set(groups[hall].tolist())
    for r in rows:
        r["caption_hallucinates"] = r["id"] in hallucinating_ids
    n_clean = len({t["id"] for t in traces} - hallucinating_ids)
    print(f"captions with no hallucinated noun: {n_clean}")

    print(f"captions {len(traces)} (regenerated text identical to saved: {n_match})")
    print(f"object nouns: {len(rows)}  hallucinated {hall.sum()}  grounded {(~hall).sum()}  "
          f"(annotated-absent among hallucinated: {sum(r['annotated_absent'] for r in rows if r['label'] == 'hallucinated')})")
    print(f"captions with >=1 hallucinated noun: {len(set(groups[hall]))}")
    print()
    print(f"{'feature':<24} {'hallu mean':>10} {'ground mean':>11} {'hallu med':>9} {'ground med':>10} "
          f"{'AUROC':>6} {'95% CI (by caption)':>20}  description")
    summary = {}
    for key, description in FEATURES:
        v = np.array([r[key] for r in rows], dtype=float)
        ok = ~np.isnan(v)
        a = auroc(v[ok], hall[ok])
        lo, hi = bootstrap_auroc(v[ok], hall[ok], groups[ok])
        summary[key] = dict(auroc=a, ci=[lo, hi], hallucinated_mean=float(v[ok & hall].mean()),
                            grounded_mean=float(v[ok & ~hall].mean()))
        print(f"{key:<24} {v[ok & hall].mean():10.3f} {v[ok & ~hall].mean():11.3f} "
              f"{np.median(v[ok & hall]):9.3f} {np.median(v[ok & ~hall]):10.3f} {a:6.3f} "
              f"   [{lo:.3f}, {hi:.3f}]  {description}")

    print("\nWithin-caption comparison (captions with both labels): mean(hallucinated) - mean(grounded)")
    for key in ("p_image", "dlogp", "late_image_mass", "late_top5", "step", "context_before"):
        diffs = []
        for g in np.unique(groups):
            sel = groups == g
            if hall[sel].any() and (~hall[sel]).any():
                v = np.array([r[key] for r, s in zip(rows, sel) if s], dtype=float)
                h = hall[sel]
                diffs.append(np.nanmean(v[h]) - np.nanmean(v[~h]))
        diffs = np.array(diffs)
        print(f"  {key:<18} median diff {np.median(diffs):+.3f}   hallucinated higher in "
              f"{(diffs > 0).sum()}/{len(diffs)} captions")

    print("\nQuadrants (split at the median over all nouns): share hallucinated in each cell")
    top5 = np.array([r["late_top5"] for r in rows])
    dl = np.array([r["dlogp"] for r in rows])
    t_med, d_med = np.median(top5), np.median(dl)
    print(f"  thresholds: top5 share {t_med:.3f}, dlogp {d_med:.2f}")
    for focus_name, f in (("focused", top5 >= t_med), ("diffuse", top5 < t_med)):
        for eff_name, e in (("image effect high", dl >= d_med), ("image effect low", dl < d_med)):
            cell = f & e
            print(f"  {focus_name:<8} + {eff_name:<18} n={cell.sum():4d}  hallucinated {hall[cell].mean():.2f}")

    in_hallucinating = np.array([r["caption_hallucinates"] for r in rows])
    grounded_h = ~hall & in_hallucinating
    grounded_c = ~hall & ~in_hallucinating
    if grounded_c.any():
        print("\nThree groups: H = hallucinated nouns, Gh = grounded nouns in hallucinating captions, "
              "Gc = grounded nouns in clean captions")
        print(f"  n: H {hall.sum()}  Gh {grounded_h.sum()}  Gc {grounded_c.sum()}")
        print(f"  {'feature':<24} {'H mean':>8} {'Gh mean':>8} {'Gc mean':>8}   "
              f"{'AUROC H vs Gc':>20} {'AUROC Gh vs Gc':>20} {'AUROC H vs Gh':>20}")
        three = {}
        for key, _ in FEATURES:
            v = np.array([r[key] for r in rows], dtype=float)
            ok = ~np.isnan(v)
            cells = []
            for a_mask, b_mask in ((hall, grounded_c), (grounded_h, grounded_c), (hall, grounded_h)):
                sel = ok & (a_mask | b_mask)
                a = auroc(v[sel], a_mask[sel])
                lo, hi = bootstrap_auroc(v[sel], a_mask[sel], groups[sel])
                cells.append((a, lo, hi))
            three[key] = cells
            means = [v[ok & m].mean() for m in (hall, grounded_h, grounded_c)]
            print(f"  {key:<24} {means[0]:8.3f} {means[1]:8.3f} {means[2]:8.3f}   " +
                  " ".join(f"{a:.3f} [{lo:.2f},{hi:.2f}]" for a, lo, hi in cells))
        summary["three_groups"] = {k: [list(c) for c in v] for k, v in three.items()}

    print("\nCombining features (logistic regression, 5-fold CV grouped by caption): out-of-fold AUROC")
    sets = {
        "confidence only (p_image, entropy_image)": ["p_image", "entropy_image"],
        "image removal (dlogp, p_no_image, kl, no_image_top1)": ["dlogp", "p_no_image", "kl", "no_image_top1"],
        "attention (mass, entropy, min-head, top5)": ["late_image_mass", "late_image_entropy",
                                                     "late_min_head_entropy", "late_top5"],
        "decoding order (step, position, context, right, repeat)": ["step", "relative_position", "context_before",
                                                                   "right_before", "repeat_mention"],
        "confidence + image removal": ["p_image", "entropy_image", "dlogp", "p_no_image", "kl", "no_image_top1"],
        "all": [k for k, _ in FEATURES],
    }
    combined = {}
    for name, keys in sets.items():
        x = np.array([[r[k] for k in keys] for r in rows], dtype=float)
        x = np.where(np.isnan(x), np.nanmean(x, 0), x)
        a = auroc(logistic_cv(x, hall.astype(float), groups), hall)
        combined[name] = a
        print(f"  {name:<58} {a:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "nouns.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(dict(n_captions=len(traces), n_nouns=len(rows), n_hallucinated=int(hall.sum()),
                       features=summary, combined=combined), handle, indent=2)
    print(f"\nsaved {os.path.join(args.out, 'nouns.csv')} and summary.json")


if __name__ == "__main__":
    main()
