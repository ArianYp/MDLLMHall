"""Attention-guided retrieval from COCO train for the nouns of one AMBER-g caption.

For each noun (first token, at its commit step) the late-layer attention map in
the saved baseline trace gives the region it looked at. That region is cut from
the original photo with remask_decoding.zoom_crop (super tight, tight and loose crops), and
the crop and the full image are embedded with the CLIP model behind
hallucination-attack/clip_embeddings.pt (OpenCLIP ViT-H-14, laion2b_s32b_b79k).
The top-k COCO train images by cosine similarity are copied next to the query,
with their COCO captions, and a contact sheet is drawn per query.

"support" is the fraction of retrieved images whose captions mention the noun,
a first check of the "is the prediction right" use of retrieval.

Example:
    python rag_retrieve.py --id 163
"""

import argparse
import csv
import json
import os
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from remask_decoding import STUDY, write_json, zoom_crop
from trace_amber_steps import IMAGE_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS = os.path.join(ROOT, "..", "hallucination-attack", "clip_embeddings.pt")
COCO = "/data/gpfs/datasets/COCO"
CLIP_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# name: (side as a multiple of the attended cluster, minimum side in patches)
CROPS = dict(super_tight=(1.0, 4), tight=(1.5, 8), loose=(3.0, 8))


def coco_path(coco_id):
    return os.path.join(COCO, "train2017", f"{coco_id:012d}.jpg")


def mentions(word, lemma, captions):
    pattern = re.compile(rf"\b({re.escape(word)}|{re.escape(lemma)}e?s?)\b", re.IGNORECASE)
    return any(pattern.search(c) for c in captions)


@torch.no_grad()
def embed(images, model, processor, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    return F.normalize(model.get_image_features(**inputs).float(), dim=-1)


def contact_sheet(query, hits, path, thumb=224, cols=6):
    """Query in the first cell, then the hits, each labelled with rank, similarity and whether captions mention the noun."""
    cells = [(query, "query")] + [(Image.open(coco_path(h["coco_id"])).convert("RGB"),
                                   f"#{h['rank']} {h['similarity']:.3f}{' +' if h['mentions_noun'] else ''}") for h in hits]
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(cells):
        image = image.copy()
        image.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 18)
        sheet.paste(image, (x + (thumb - image.width) // 2, y + (thumb - image.height) // 2))
        draw.text((x + 4, y + thumb + 3), label, fill="black")
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", type=int, default=163)
    parser.add_argument("--traces", default=os.path.join(STUDY, "traces"))
    parser.add_argument("--nouns", default=os.path.join(STUDY, "with_clean", "nouns.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--zoom-top-k", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_root = args.output_dir or os.path.join(STUDY, "rag", str(args.id))
    os.makedirs(out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trace = json.load(open(os.path.join(args.traces, f"{args.id}.json"), encoding="utf-8"))
    late_map = torch.from_numpy(np.load(os.path.join(args.traces, f"{args.id}.npz"))["late_map"].astype(np.float32))
    resolution = trace["settings"]["resolution"]
    photo = Image.open(os.path.join(IMAGE_DIR, trace["image"])).convert("RGB")
    with open(args.nouns, encoding="utf-8") as handle:
        nouns = [r for r in csv.DictReader(handle) if int(r["id"]) == args.id and r["label"] != "ignored"]
    print(f"caption {args.id}: {trace['answer']}\nphoto {photo.size}, {len(nouns)} nouns", flush=True)

    bank = torch.load(EMBEDDINGS, map_location="cpu")
    coco_ids = torch.as_tensor(bank["indices"]).tolist()
    bank = F.normalize(bank["clip_embeds"].float(), dim=-1).to(device)
    print(f"bank {tuple(bank.shape)}, coco ids {min(coco_ids)}..{max(coco_ids)}", flush=True)
    with open(os.path.join(COCO, "annotations", "captions_train2017.json"), encoding="utf-8") as handle:
        captions = {}
        for ann in json.load(handle)["annotations"]:
            captions.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)

    # The bank should contain these images' own embeddings; same pipeline means cosine near 1.
    probe = [0, len(coco_ids) // 2, len(coco_ids) - 1]
    for resize in (None, 336):
        images = [Image.open(coco_path(coco_ids[i])).convert("RGB") for i in probe]
        if resize:
            images = [im.resize((resize, resize)) for im in images]
        cos = (embed(images, model, processor, device) * bank[probe]).sum(-1)
        print(f"self-check resize={resize}: cosine to stored {[round(c, 4) for c in cos.tolist()]}", flush=True)

    summary = dict(id=args.id, caption=trace["answer"], clip=CLIP_ID, top_k=args.top_k, queries=[])
    queries = [("full_image", None, photo, None)]
    for row in nouns:
        pos = int(row["pos"])
        for name, (zoom, min_patches) in CROPS.items():
            crop, box = zoom_crop(photo, late_map[pos], resolution, zoom, args.zoom_top_k, min_patches)
            queries.append((f"{row['label']}_{pos:03d}_{row['word']}/{name}", row, crop, box))

    for name, row, image, box in queries:
        folder = os.path.join(out_root, name)
        os.makedirs(folder, exist_ok=True)
        image.save(os.path.join(folder, "query.jpg"), quality=95)
        similarity = bank @ embed([image], model, processor, device)[0]
        top = torch.topk(similarity, args.top_k)
        hits = []
        for rank, (s, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
            coco_id = coco_ids[i]
            caps = captions.get(coco_id, [])
            hit = dict(rank=rank, coco_id=coco_id, similarity=s, captions=caps,
                       mentions_noun=bool(row) and mentions(row["word"], row["lemma"], caps))
            shutil.copy(coco_path(coco_id), os.path.join(folder, f"rank{rank:02d}_{s:.3f}_coco{coco_id:012d}.jpg"))
            hits.append(hit)
        contact_sheet(image, hits, os.path.join(folder, "sheet.jpg"))
        entry = dict(query=name, box=box, crop_size=image.size, hits=hits)
        if row:
            entry.update(word=row["word"], lemma=row["lemma"], label=row["label"], pos=int(row["pos"]),
                         p_image=float(row["p_image"]), support=sum(h["mentions_noun"] for h in hits) / len(hits))
        write_json(os.path.join(folder, "retrieved.json"), entry, indent=1)
        summary["queries"].append({k: v for k, v in entry.items() if k != "hits"})
        print(f"{name:32s} box={box} support={entry.get('support', '-')}  "
              f"top3={[(h['coco_id'], round(h['similarity'], 3)) for h in hits[:3]]}", flush=True)
        print(f"    #1 caption: {hits[0]['captions'][:1]}", flush=True)

    write_json(os.path.join(out_root, "summary.json"), summary, indent=1)
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
