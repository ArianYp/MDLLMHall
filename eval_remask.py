"""Score remask decoding against the baseline on the study captions.

For each condition the predictions are split into the study set (ids.json,
captions that hallucinated at baseline) and the clean control set
(ids_clean.json). Each subset is scored with the untouched official
third_party/AMBER/inference.py --evaluation_type g, and per caption with
amber_noun_labels.AmberLabeler (hallucinated / grounded object nouns), so the
same image can be compared across conditions.
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

from amber_noun_labels import AmberLabeler, read_json

ROOT = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(ROOT, "results", "amber_g", "hallu_study")
OUT = os.path.join(ROOT, "results", "amber_g", "remask")
AMBER_REPO = os.path.join(ROOT, "third_party", "AMBER")
METRICS = ("CHAIR", "Cover", "Hal", "Cog")


def official_scores(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    out = subprocess.run([sys.executable, "inference.py", "--inference_data", os.path.abspath(path),
                          "--evaluation_type", "g"], cwd=AMBER_REPO, capture_output=True, text=True, check=True).stdout
    return {m: float(re.search(rf"^{m}:\s*([0-9.]+)", out, re.MULTILINE).group(1)) for m in METRICS}


def main():
    parser = argparse.ArgumentParser(description="Official AMBER-g scores and paired counts for remask runs.")
    parser.add_argument("--conditions", nargs="+", required=True, help="name=predictions.json pairs")
    parser.add_argument("--output-dir", default=OUT)
    args = parser.parse_args()

    saved = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
    subsets = {"hallucinating (84)": read_json(os.path.join(STUDY, "ids.json"))["ids"],
               "clean": read_json(os.path.join(STUDY, "ids_clean.json"))["ids"]}
    conditions = {"baseline": saved}
    for pair in args.conditions:
        name, path = pair.split("=", 1)
        conditions[name] = {int(r["id"]): r["response"] for r in read_json(path)}

    # Partial runs: keep only captions every condition has finished.
    subsets = {name.split()[0]: [i for i in ids if all(i in c for c in conditions.values())]
               for name, ids in subsets.items()}
    subsets = {f"{name} ({len(ids)})": ids for name, ids in subsets.items() if ids}

    labeler = AmberLabeler()
    per_caption = {}
    for name, responses in conditions.items():
        for subset_ids in subsets.values():
            for i in subset_ids:
                labels = labeler.label(i, responses[i])
                per_caption[(name, i)] = dict(
                    hallucinated=sum(l["label"] == "hallucinated" for l in labels),
                    grounded=sum(l["label"] == "grounded" for l in labels),
                    words=len(responses[i].split()),
                )

    os.makedirs(args.output_dir, exist_ok=True)
    report = {}
    for subset_name, subset_ids in subsets.items():
        print(f"\n=== {subset_name} captions")
        print(f"{'condition':<10} {'CHAIR':>6} {'Cover':>6} {'Hal':>6} {'Cog':>5}   {'hallu nouns':>11} "
              f"{'grounded':>8} {'words':>6}   vs baseline: fewer / same / more hallucinated nouns")
        for name, responses in conditions.items():
            records = [dict(id=i, response=responses[i]) for i in sorted(subset_ids)]
            tag = subset_name.split()[0]
            scores = official_scores(records, os.path.join(args.output_dir, f"subset_{tag}_{name}.json"))
            rows = [per_caption[(name, i)] for i in subset_ids]
            base = [per_caption[("baseline", i)] for i in subset_ids]
            diff = np.array([r["hallucinated"] - b["hallucinated"] for r, b in zip(rows, base)])
            report[f"{tag}/{name}"] = dict(official=scores,
                                           hallucinated_nouns=int(sum(r["hallucinated"] for r in rows)),
                                           grounded_nouns=int(sum(r["grounded"] for r in rows)),
                                           mean_words=float(np.mean([r["words"] for r in rows])),
                                           fewer=int((diff < 0).sum()), same=int((diff == 0).sum()),
                                           more=int((diff > 0).sum()))
            r = report[f"{tag}/{name}"]
            paired = "" if name == "baseline" else f"{r['fewer']:>4} / {r['same']:>3} / {r['more']:>3}"
            print(f"{name:<10} {scores['CHAIR']:6.1f} {scores['Cover']:6.1f} {scores['Hal']:6.1f} {scores['Cog']:5.1f}   "
                  f"{r['hallucinated_nouns']:>11} {r['grounded_nouns']:>8} {r['mean_words']:6.1f}   {paired}")
    with open(os.path.join(args.output_dir, "eval_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nsaved {os.path.join(args.output_dir, 'eval_report.json')}")


if __name__ == "__main__":
    main()
