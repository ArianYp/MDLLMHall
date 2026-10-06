"""How many hallucinated nouns does CLIP find on the sink-removed zoom crop? Offline, from crop_veto_nosink.csv.

All 1050 study nouns (155 hallucinated, 144 captions). rank_* = share of the vocabulary words that CLIP scores below the noun
on that view ("a photo of a {w}."); low rank => flag as hallucinated. Views: full image, tight crop, tight crop with sinks removed.

1) AUROC (95% CI bootstrapped over captions) of each view and of simple combinations, and grouped 5-fold CV logistic regression.
2) Operating points: precision / recall / flags, on the study set and re-weighted to the full AMBER-g mix
   (285 hallucinating + 719 clean captions; study: 84 + 60), since precision depends on the base rate.
3) What the crop adds: hallucinations the full image misses (rank_full >= 0.9) that the crop finds, and the cost in grounded flags.
4) Object vs background words.

Thresholds are swept on the study set, so the operating points are optimistic. CPU only (numpy, sklearn if present).
Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/detect_nosink_crop.py
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Background (COCO-Stuff-style scene words), as in slot_refill_study.py.
BACKGROUND = {"sky", "cloud", "wall", "floor", "ground", "grass", "tree", "bush", "plant", "road", "street", "path", "sidewalk",
              "water", "sea", "ocean", "river", "lake", "beach", "sand", "snow", "mountain", "hill", "field", "forest", "dirt"}


def auroc(score, y):
    """P(score of a hallucinated noun > score of a grounded one), ties count half."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, y, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    vals = []
    for _ in range(n):
        take = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        if y[take].any() and (~y[take]).any():
            vals.append(auroc(score[take], y[take]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv"))))
    y = np.array([r["label"] == "hallucinated" for r in rows])
    groups = np.array([int(r["id"]) for r in rows])
    hallu_cap = np.array([r["caption_hallucinates"] == "True" for r in rows])
    bg = np.array([r["lemma"] in BACKGROUND for r in rows])
    moved = np.array([r["crop_moved"] == "1" for r in rows])
    full = np.array([float(r["rank_full"]) for r in rows])
    crop = np.array([float(r["rank_tight_orig"]) for r in rows])
    nosink = np.array([float(r["rank_tight_nosink"]) for r in rows])
    # Re-weight to the full AMBER-g mix: nouns of hallucinating captions x 285/84, of clean captions x 719/60.
    w = np.where(hallu_cap, 285 / 84, 719 / 60)
    print(f"nouns {len(y)}, hallucinated {y.sum()}, captions {len(np.unique(groups))}, crop moved by sink removal {moved.sum()}")
    print(f"re-weighted base rate {w[y].sum() / w.sum():.3f} (study {y.mean():.3f}; full-set baseline 523/6664 = 0.078)")

    signals = {
        "full image": -full,
        "tight crop": -crop,
        "tight crop, no sink": -nosink,
        "min(full, crop no sink)": -np.minimum(full, nosink),
        "mean(full, crop no sink)": -(full + nosink) / 2,
    }
    print("\n1) AUROC, lower rank => hallucinated (95% CI over captions)")
    for name, s in signals.items():
        lo, hi = boot_ci(s, y, groups)
        print(f"  {name:<26} all {auroc(s, y):.3f} ({lo:.3f}-{hi:.3f}) | objects {auroc(s[~bg], y[~bg]):.3f} | background {auroc(s[bg], y[bg]):.3f}"
              f" | crop moved {auroc(s[moved], y[moved]):.3f} (n {moved.sum()}, hallu {y[moved].sum()})")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        for name, cols in (("full", [full]), ("crop no sink", [nosink]), ("full + crop no sink", [full, nosink]),
                           ("full + crop + crop no sink", [full, crop, nosink])):
            X = np.column_stack(cols)
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, groups):
                pred[te] = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            print(f"  CV logistic {name:<24} {auroc(pred, y):.3f}")
    except ImportError:
        print("  (sklearn not available, CV skipped)")

    def point(flag):
        tp, n = (flag & y).sum(), flag.sum()
        wp = (w * (flag & y)).sum() / max((w * flag).sum(), 1e-9)
        return f"flags {n:>4} | caught {tp:>3}/{y.sum()} (recall {tp / y.sum():.2f}) | precision {tp / max(n, 1):.2f} | full-mix precision {wp:.2f}"

    print("\n2) operating points (flag if rank < t)")
    for name, r in (("full image", full), ("tight crop", crop), ("tight crop, no sink", nosink)):
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            print(f"  {name:<20} < {t:.2f}: {point(r < t)}")
    print("  combinations:")
    for name, flag in (
        ("full < 0.9 OR crop no sink < 0.5", (full < 0.9) | (nosink < 0.5)),
        ("full < 0.9 OR crop no sink < 0.7", (full < 0.9) | (nosink < 0.7)),
        ("full < 0.9 AND crop no sink < 0.95 (crop veto)", (full < 0.9) & (nosink < 0.95)),
        ("full < 0.95 AND crop no sink < 0.95", (full < 0.95) & (nosink < 0.95)),
        ("full < 0.95 AND crop no sink < 0.9", (full < 0.95) & (nosink < 0.9)),
        ("mean(full, crop no sink) < 0.85", (full + nosink) / 2 < 0.85),
        ("mean(full, crop no sink) < 0.9", (full + nosink) / 2 < 0.9),
        ("never background, full < 0.95 AND crop no sink < 0.95", ~bg & (full < 0.95) & (nosink < 0.95)),
    ):
        print(f"  {name:<55} {point(flag)}")

    print("\n  precision at fixed recall (study set / full mix):")
    for name, s in signals.items():
        order = np.argsort(-s, kind="mergesort")
        cum = np.cumsum(y[order])
        cells = []
        for rec in (0.3, 0.5, 0.7, 0.9):
            k = int(np.searchsorted(cum, rec * y.sum())) + 1
            top = order[:k]
            cells.append(f"R{rec:.1f}: {y[top].mean():.2f} / {(w[top] * y[top]).sum() / w[top].sum():.2f}")
        print(f"  {name:<26} " + " | ".join(cells))

    print("\n3) what the sink-free crop adds to the full image")
    miss = full >= 0.9
    print(f"  full image misses (rank_full >= 0.9): {(miss & y).sum()} hallucinated, {(miss & ~y).sum()} grounded")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9):
        extra = miss & (nosink < t)
        print(f"    crop no sink < {t:.1f} among them: catches {(extra & y).sum():>3} hallucinated for {(extra & ~y).sum():>3} grounded"
              f" (old crop: {(miss & (crop < t) & y).sum():>3} / {(miss & (crop < t) & ~y).sum():>3})")
    print("  hallucinated nouns missed by both (full >= 0.9 and crop no sink >= 0.9):")
    both = np.flatnonzero(y & miss & (nosink >= 0.9))
    print("   ", ", ".join(f"{rows[i]['id']}:{rows[i]['lemma']}" for i in both))


if __name__ == "__main__":
    main()
