"""Save CLIP embeddings so that vocabularies can be compared offline (clip_vocab_compare.py).

Images: for each study noun (crop_veto_nosink.csv) the full photo, the tight crop and the sink-free tight crop (saved boxes).
Text: "a photo of a {w}." for the union of the current vocabulary (AMBER relation.json words + study lemmas, 418) and the
WordNet physical-object vocabulary of wordnet_rival_check.py (COCO count >= 20 + AMBER main words, 2651).

Output: clip_vocab_embed.npz (full [n_images], crop / nosink [n_nouns], text [n_words], image_index, words).
Example (from MDLLM/): python results/amber_g/hallu_study/rag/clip_relation/clip_vocab_embed.py
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

from rag_retrieve import CLIP_ID, embed
from remask_decoding import STUDY
from trace_amber_steps import AMBER_DATA, IMAGE_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    device = torch.device("cuda")
    rows = list(csv.DictReader(open(os.path.join(HERE, "crop_veto_nosink.csv"), encoding="utf-8")))
    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        lemmas = {r["lemma"] for r in csv.DictReader(handle) if r["label"] in ("grounded", "hallucinated")}
    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    current = set(relation) | {w for v in relation.values() for w in v} | lemmas
    wordnet = set(json.load(open(os.path.join(HERE, "wordnet_rival", "vocabulary.json"), encoding="utf-8")))
    words = sorted(current | wordnet)
    print(f"{len(rows)} nouns, current vocabulary {len(current)}, WordNet {len(wordnet)}, union {len(words)}", flush=True)

    model = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    with torch.no_grad():
        text = []
        for i in range(0, len(words), 256):
            inputs = processor(text=[f"a photo of a {w}." for w in words[i:i + 256]], return_tensors="pt", padding=True).to(device)
            text.append(torch.nn.functional.normalize(model.get_text_features(**inputs).float(), dim=-1).cpu())
        text = torch.cat(text).numpy()

        ids = sorted({int(r["id"]) for r in rows})
        full, crop, nosink = [], [None] * len(rows), [None] * len(rows)
        for n, item_id in enumerate(ids, 1):
            photo = Image.open(os.path.join(IMAGE_DIR, f"AMBER_{item_id}.jpg")).convert("RGB")
            ks = [k for k, r in enumerate(rows) if int(r["id"]) == item_id]
            images = [photo]
            for k in ks:
                images += [photo.crop(tuple(json.loads(rows[k]["box_orig"]))), photo.crop(tuple(json.loads(rows[k]["box_nosink"])))]
            e = torch.cat([embed(images[i:i + 32], model, processor, device) for i in range(0, len(images), 32)]).cpu().numpy()
            full.append(e[0])
            for j, k in enumerate(ks):
                crop[k], nosink[k] = e[1 + 2 * j], e[2 + 2 * j]
            if n % 24 == 0 or n == len(ids):
                print(f"  {n}/{len(ids)} captions", flush=True)

    path = os.path.join(HERE, "clip_vocab_embed.npz")
    np.savez_compressed(path, full=np.stack(full), crop=np.stack(crop), nosink=np.stack(nosink), text=text,
                        image_ids=np.array(ids), words=np.array(words), current=np.array(sorted(current)))
    print("saved", path)


if __name__ == "__main__":
    main()
