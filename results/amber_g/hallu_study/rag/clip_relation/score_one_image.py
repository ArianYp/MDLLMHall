"""CLIP similarity of AMBER images to every vocabulary word, same setup as rag_clip_relation.py.

Saves all scores (not only the caption's nouns) so we can see which words outrank a hallucinated one.
With --check id:word, lists every word CLIP places above `word` on that image, marking those
AMBER says are in the image (* = truth list or its relation.json associations).

Examples (from MDLLM/):
    python results/amber_g/hallu_study/rag/clip_relation/score_one_image.py --id 517 --words cow horse car road
    python results/amber_g/hallu_study/rag/clip_relation/score_one_image.py --check 517:cow 428:cow 67:bench
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from rag_retrieve import CLIP_ID, embed
from remask_decoding import STUDY
from trace_amber_steps import AMBER_DATA, IMAGE_DIR

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "one_image")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, nargs="*", default=[])
    parser.add_argument("--words", nargs="*", default=[], help="words to report explicitly (with --id)")
    parser.add_argument("--check", nargs="*", default=[], help="id:word pairs, e.g. 517:cow")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    device = torch.device("cuda")

    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        lemmas = {r["lemma"] for r in csv.DictReader(handle) if r["label"] in ("grounded", "hallucinated")}
    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    annotations = json.load(open(os.path.join(AMBER_DATA, "annotations.json"), encoding="utf-8"))
    vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | lemmas)

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        inputs = processor(text=[f"a photo of a {w}." for w in vocabulary], return_tensors="pt", padding=True).to(device)
        text = torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1)

    checks = [(int(c.split(":")[0]), c.split(":")[1]) for c in args.check]
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for item_id in dict.fromkeys(args.id + [i for i, _ in checks]):
        photo = Image.open(os.path.join(IMAGE_DIR, f"AMBER_{item_id}.jpg")).convert("RGB")
        sims = (text @ embed([photo], model, processor, device)[0]).cpu()
        order = sims.argsort(descending=True).tolist()
        ranked = [(vocabulary[i], float(sims[i])) for i in order]
        place = {w: n + 1 for n, (w, _) in enumerate(ranked)}
        truth = annotations[item_id - 1]["truth"]
        present = set(truth) | {w for t in truth for w in relation.get(t, [])}
        json.dump({"id": item_id, "view": "full", "truth": truth, "scores": dict(ranked)},
                  open(os.path.join(OUT_DIR, f"{item_id}_full.json"), "w"), indent=1)

        if item_id in args.id:
            print(f"\nimage {item_id}, top {args.top} of {len(vocabulary)}:")
            for n, (w, s) in enumerate(ranked[: args.top], 1):
                print(f"  {n:>3}. {w:<14} sim={s:.4f}{' *' if w in present else ''}")
            for w in args.words:
                print(f"  {w:<14} " + (f"place {place[w]}, sim={ranked[place[w] - 1][1]:.4f}" if w in place else "not in vocabulary"))

        for _, word in [c for c in checks if c[0] == item_id]:
            if word not in place:
                print(f"\n{item_id}:{word} not in vocabulary")
                continue
            p, s = place[word], ranked[place[word] - 1][1]
            above = ranked[: p - 1]
            above_present = [(w, x) for w, x in above if w in present]
            print(f"\n{item_id}:{word}  place {p}, sim={s:.4f}   truth: {', '.join(truth)}")
            print("  above: " + ", ".join(f"{w}{'*' if w in present else ''} {x:.3f}" for w, x in above))
            best = above_present[0] if above_present else None
            print("  best present word above: " + (f"{best[0]} (+{best[1] - s:.3f})" if best else "none"))
            summary.append({"id": item_id, "word": word, "place": p, "sim": s, "truth": truth,
                            "above": [w for w, _ in above], "above_present": [w for w, _ in above_present],
                            "margin_best_present": best[1] - s if best else None})
    if summary:
        json.dump(summary, open(os.path.join(OUT_DIR, "check_summary.json"), "w"), indent=1)
        print("\nsaved", os.path.join(OUT_DIR, "check_summary.json"))


if __name__ == "__main__":
    main()
