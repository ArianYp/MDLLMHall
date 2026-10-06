"""Does CLIP separate hallucinated from grounded nouns? Text of the noun vs the real image, its attended crop, and retrieved images.

For every grounded / hallucinated noun of the study captions (with_clean/nouns.csv), with
t = CLIP text embedding of "a photo of a {lemma}.":

  sim_full            cos(t, full image)
  sim_<crop>          cos(t, attended crop), crops as in rag_retrieve.py
  ret_<view>          mean cos(t, top-10 COCO train images retrieved for the view)
  support_<view>      fraction of those whose COCO captions mention the noun
  top1_<view>         image-image cosine of the view to its best match (retrieval confidence)
  rank_<view>         percentile of the noun among a vocabulary of object words for that view's
                      embedding (1 = the best-matching word), which removes the word's own bias
  prior_<view>        sim_<view> minus the word's mean similarity to all study full images

AUROC is for "lower value => hallucinated" (1 - the usual AUROC of the feature), with a 95%
bootstrap CI over captions. Output: hallu_study/rag/clip_relation/{nouns_clip.csv,summary.json}.

Example:
    python rag_clip_relation.py
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from rag_retrieve import CLIP_ID, COCO, CROPS, EMBEDDINGS, embed, mentions
from remask_decoding import STUDY, write_json, zoom_crop
from trace_amber_steps import AMBER_DATA, IMAGE_DIR

VIEWS = ["full"] + list(CROPS)


def auroc(score, positive):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = score[positive], score[~positive]
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([pos, neg])
    for v in np.unique(values):
        tie = values == v
        ranks[tie] = ranks[tie].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap(score, positive, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n):
        take = np.concatenate([index[g] for g in rng.choice(unique, len(unique))])
        if positive[take].any() and (~positive[take]).any():
            values.append(auroc(score[take], positive[take]))
    return np.percentile(values, [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"))
    parser.add_argument("--trace-dirs", nargs="+", default=[os.path.join(STUDY, "traces"), os.path.join(STUDY, "traces_clean")])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--zoom-top-k", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output-dir", default=os.path.join(STUDY, "rag", "clip_relation"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    with open(args.nouns, encoding="utf-8") as handle:
        nouns = [r for r in csv.DictReader(handle) if r["label"] in ("grounded", "hallucinated")]
    ids = sorted({int(r["id"]) for r in nouns})
    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | {r["lemma"] for r in nouns})
    print(f"{len(nouns)} nouns in {len(ids)} captions, vocabulary {len(vocabulary)} words", flush=True)

    bank = torch.load(EMBEDDINGS, map_location="cpu")
    coco_ids = torch.as_tensor(bank["indices"]).tolist()
    bank = torch.nn.functional.normalize(bank["clip_embeds"].float(), dim=-1).to(device)
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        captions = {}
        for ann in json.load(handle)["annotations"]:
            captions.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        text = []
        for i in range(0, len(vocabulary), 256):
            inputs = processor(text=[f"a photo of a {w}." for w in vocabulary[i:i + 256]], return_tensors="pt",
                               padding=True).to(device)
            text.append(torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1))
    text = torch.cat(text)
    word_index = {w: i for i, w in enumerate(vocabulary)}

    # One embedding per (caption, view): the full image, and every noun's crop at each size.
    # Photos are large (up to ~5700 px), so each caption's views are embedded before the next photo is opened.
    embeddings, keys = [], []
    for n, item_id in enumerate(ids, 1):
        trace_dir = next(d for d in args.trace_dirs if os.path.exists(os.path.join(d, f"{item_id}.npz")))
        trace = json.load(open(os.path.join(trace_dir, f"{item_id}.json"), encoding="utf-8"))
        late_map = torch.from_numpy(np.load(os.path.join(trace_dir, f"{item_id}.npz"))["late_map"].astype(np.float32))
        photo = Image.open(os.path.join(IMAGE_DIR, trace["image"])).convert("RGB")
        resolution = trace["settings"]["resolution"]
        views = [photo]
        keys.append((item_id, "full", None))
        for pos in sorted({int(r["pos"]) for r in nouns if int(r["id"]) == item_id}):
            for name, (zoom, min_patches) in CROPS.items():
                views.append(zoom_crop(photo, late_map[pos], resolution, zoom, args.zoom_top_k, min_patches)[0])
                keys.append((item_id, name, pos))
        embeddings += [embed(views[i:i + args.batch], model, processor, device) for i in range(0, len(views), args.batch)]
        if n % 24 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} captions, {len(keys)} views", flush=True)
    embeddings = torch.cat(embeddings)
    view_of = {k: i for i, k in enumerate(keys)}
    print(f"embedded {len(keys)} views", flush=True)

    similarity = embeddings @ bank.T
    top = torch.topk(similarity, args.top_k, dim=-1)
    retrieved = bank[top.indices]                                    # views, k, d
    word_sim = embeddings @ text.T                                   # views, vocabulary
    full_rows = [view_of[(i, "full", None)] for i in ids]
    word_prior = word_sim[full_rows].mean(0)                         # vocabulary

    rows = []
    for r in nouns:
        item_id, pos, lemma = int(r["id"]), int(r["pos"]), r["lemma"]
        w = word_index[lemma]
        row = dict(id=item_id, pos=pos, word=r["word"], lemma=lemma, label=r["label"], step=int(r["step"]),
                   p_image=float(r["p_image"]))
        for view in VIEWS:
            v = view_of[(item_id, "full", None) if view == "full" else (item_id, view, pos)]
            sims = word_sim[v]
            row[f"sim_{view}"] = float(sims[w])
            row[f"rank_{view}"] = float((sims < sims[w]).float().mean())
            row[f"prior_{view}"] = float(sims[w] - word_prior[w])
            row[f"ret_{view}"] = float((retrieved[v] @ text[w]).mean())
            row[f"top1_{view}"] = float(top.values[v, 0])
            hits = [captions.get(coco_ids[i], []) for i in top.indices[v].tolist()]
            row[f"support_{view}"] = sum(mentions(r["word"], lemma, caps) for caps in hits) / len(hits)
        rows.append(row)

    with open(os.path.join(args.output_dir, "nouns_clip.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positive = np.array([r["label"] == "hallucinated" for r in rows])
    groups = np.array([r["id"] for r in rows])
    features = [k for k in rows[0] if k.split("_")[0] in ("sim", "rank", "prior", "ret", "support", "top1")]
    report = dict(n_nouns=len(rows), n_hallucinated=int(positive.sum()), n_captions=len(ids), features={})
    print(f"\n{'feature':22s} AUROC (lower => hallucinated)  95% CI    hallucinated mean  grounded mean")
    for f in features:
        score = -np.array([r[f] for r in rows])
        a, ci = auroc(score, positive), bootstrap(score, positive, groups)
        report["features"][f] = dict(auroc=a, ci=ci, hallucinated_mean=float(-score[positive].mean()),
                                     grounded_mean=float(-score[~positive].mean()))
        print(f"{f:22s} {a:.3f}                       {ci[0]:.2f}-{ci[1]:.2f}   {-score[positive].mean():.3f}"
              f"              {-score[~positive].mean():.3f}", flush=True)
    write_json(os.path.join(args.output_dir, "summary.json"), report, indent=1)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
