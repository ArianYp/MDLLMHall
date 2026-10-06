"""CLIP similarity margin between the best vocabulary word and the noun, per view (full image, tight crop, tight crop with sinks removed).

Hypothesis: a hallucinated noun sits well below CLIP's top word; a grounded noun that is not top-1 is beaten only by a synonym or
another object in the crop, so the gap is small. Rank (crop_veto_nosink.csv) ignores the size of the gap.

Same setup as crop_veto_nosink.py: vocabulary = AMBER relation.json words + study lemmas, "a photo of a {w}.", ViT-H-14.
The crops are rebuilt from the saved boxes (box_orig, box_nosink), so no traces are needed. The ranks are checked against the saved ones.

Output: clip_margin.csv, one row per noun: per view the noun's similarity, rank, top-1 word and similarity, margin = top1 - noun (0 if the
noun is top-1), and the top-5 words. Analysis: clip_margin_analysis.py.

Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/clip_margin.py
"""

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

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = ("full", "crop", "nosink")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv"), encoding="utf-8")))
    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        lemmas = {r["lemma"] for r in csv.DictReader(handle) if r["label"] in ("grounded", "hallucinated")}
    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | lemmas)
    index = {w: i for i, w in enumerate(vocabulary)}
    print(f"{len(rows)} nouns, vocabulary {len(vocabulary)}, device {device}", flush=True)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        text = []
        for i in range(0, len(vocabulary), 256):
            inputs = processor(text=[f"a photo of a {w}." for w in vocabulary[i:i + 256]], return_tensors="pt",
                               padding=True).to(device)
            text.append(torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1))
        text = torch.cat(text)

        out, ids = [], sorted({int(r["id"]) for r in rows}, key=int)
        for n, item_id in enumerate(ids, 1):
            photo = Image.open(os.path.join(IMAGE_DIR, f"AMBER_{item_id}.jpg")).convert("RGB")
            caption_rows = [r for r in rows if int(r["id"]) == item_id]
            images = [photo]
            for r in caption_rows:
                images += [photo.crop(tuple(json.loads(r["box_orig"]))), photo.crop(tuple(json.loads(r["box_nosink"])))]
            sims = torch.cat([embed(images[i:i + 32], model, processor, device) for i in range(0, len(images), 32)]) @ text.T
            for k, r in enumerate(caption_rows):
                w = index[r["lemma"]]
                row = {key: r[key] for key in ("id", "pos", "lemma", "label", "caption_hallucinates", "repeat_mention", "crop_moved",
                                              "rank_full", "rank_tight_orig", "rank_tight_nosink")}
                for view, s in zip(VIEWS, (sims[0], sims[1 + 2 * k], sims[2 + 2 * k])):
                    top = torch.topk(s, 5)
                    row[f"{view}_sim"] = float(s[w])
                    row[f"{view}_rank"] = float((s < s[w]).float().mean())
                    row[f"{view}_top1"] = vocabulary[int(top.indices[0])]
                    row[f"{view}_top1_sim"] = float(top.values[0])
                    row[f"{view}_margin"] = float(top.values[0] - s[w])
                    row[f"{view}_top5"] = json.dumps([[vocabulary[int(i)], round(float(v), 4)] for v, i in zip(top.values, top.indices)])
                out.append(row)
            if n % 24 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)} captions", flush=True)

    path = os.path.join(HERE, "clip_margin.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for view, saved in (("full", "rank_full"), ("crop", "rank_tight_orig"), ("nosink", "rank_tight_nosink")):
        diff = [abs(float(r[f"{view}_rank"]) - float(r[saved])) for r in out]
        print(f"reproduction check {view}: max |rank - saved| = {max(diff):.4f}, mean {sum(diff) / len(diff):.5f}")
    print("saved", path)


if __name__ == "__main__":
    main()
