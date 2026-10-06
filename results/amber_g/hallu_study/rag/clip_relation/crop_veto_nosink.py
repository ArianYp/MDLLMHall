"""CLIP rank of each study noun on its tight crop, with and without attention-sink removal.

Same setup as rag_clip_relation.py (AMBER relation.json vocabulary + study lemmas, "a photo of a {w}.",
tight crop = zoom_crop(late_map[pos], zoom 1.5, top 30, >= 8 patches)). `rank_tight_orig` should
reproduce nouns_clip.csv's rank_tight; `rank_tight_nosink` zeroes the caption's sink patches
(top patch for >= half of its 128 tokens) before cropping.

Output: crop_veto_nosink.csv (one row per noun), used to re-score the crop veto offline.

Example (from MDLLM/):
    python results/amber_g/hallu_study/rag/clip_relation/crop_veto_nosink.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from rag_retrieve import CLIP_ID, CROPS, embed
from remask_decoding import STUDY, zoom_crop
from trace_amber_steps import AMBER_DATA, IMAGE_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIRS = [os.path.join(STUDY, "traces"), os.path.join(STUDY, "traces_clean")]


def main():
    device = torch.device("cuda")
    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        nouns = [r for r in csv.DictReader(handle) if r["label"] in ("grounded", "hallucinated")]
    with open(os.path.join(HERE, "nouns_clip.csv"), encoding="utf-8") as handle:
        saved = {(r["id"], r["pos"]): r for r in csv.DictReader(handle)}
    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | {r["lemma"] for r in nouns})
    index = {w: i for i, w in enumerate(vocabulary)}
    print(f"{len(nouns)} nouns, vocabulary {len(vocabulary)}", flush=True)

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        text = []
        for i in range(0, len(vocabulary), 256):
            inputs = processor(text=[f"a photo of a {w}." for w in vocabulary[i:i + 256]], return_tensors="pt",
                               padding=True).to(device)
            text.append(torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1))
    text = torch.cat(text)
    zoom, min_patches = CROPS["tight"]

    rows = []
    ids = sorted({int(r["id"]) for r in nouns})
    for n, item_id in enumerate(ids, 1):
        trace_dir = next(d for d in TRACE_DIRS if os.path.exists(os.path.join(d, f"{item_id}.npz")))
        late_map = torch.from_numpy(np.load(os.path.join(trace_dir, f"{item_id}.npz"))["late_map"].astype(np.float32))
        counts = torch.bincount(late_map.argmax(1), minlength=late_map.shape[1])
        sinks = torch.nonzero(counts >= 0.5 * late_map.shape[0]).flatten().tolist()
        photo = Image.open(os.path.join(IMAGE_DIR, f"AMBER_{item_id}.jpg")).convert("RGB")
        caption_nouns = [r for r in nouns if int(r["id"]) == item_id]
        views, boxes = [], []
        for r in caption_nouns:
            m = late_map[int(r["pos"])]
            crop, box = zoom_crop(photo, m, 512, zoom, 30, min_patches)
            m2 = m.clone()
            m2[sinks] = 0
            crop2, box2 = zoom_crop(photo, m2, 512, zoom, 30, min_patches)
            views += [crop, crop2]
            boxes.append((box, box2))
        sims = torch.cat([embed(views[i:i + 32], model, processor, device) for i in range(0, len(views), 32)]) @ text.T
        for k, r in enumerate(caption_nouns):
            w = index[r["lemma"]]
            orig, nosink = sims[2 * k], sims[2 * k + 1]
            s = saved[(r["id"], r["pos"])]
            rows.append(dict(id=r["id"], pos=r["pos"], lemma=r["lemma"], label=r["label"],
                             caption_hallucinates=r["caption_hallucinates"], repeat_mention=r["repeat_mention"],
                             rank_full=float(s["rank_full"]), rank_tight_saved=float(s["rank_tight"]),
                             rank_tight_orig=float((orig < orig[w]).float().mean()),
                             rank_tight_nosink=float((nosink < nosink[w]).float().mean()),
                             sinks=json.dumps(sinks), crop_moved=int(boxes[k][0] != boxes[k][1]),
                             box_orig=json.dumps(boxes[k][0]), box_nosink=json.dumps(boxes[k][1])))
        if n % 24 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} captions", flush=True)

    out = os.path.join(HERE, "crop_veto_nosink.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diff = np.abs(np.array([r["rank_tight_orig"] - r["rank_tight_saved"] for r in rows]))
    print(f"reproduction check: max |rank_tight_orig - saved| = {diff.max():.4f}, mean {diff.mean():.5f}")
    print(f"sink present in {sum(r['sinks'] != '[]' for r in rows)} / {len(rows)} nouns, crop moved for {sum(r['crop_moved'] for r in rows)}")
    print("saved", out)


if __name__ == "__main__":
    main()
