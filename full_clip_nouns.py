"""CLIP full-image rank of every AMBER-labelled noun in the 1004 baseline AMBER-g captions (no traces needed).

The vocabulary is the one the online detector uses (slot_refill_study.py --trigger clip): AMBER relation.json words,
study lemmas and the eligible AMBER object words. rank = share of the vocabulary that matches the image worse.
Besides the detector's prompt "a photo of a {w}." it scores three other prompts and their mean embedding,
to see whether the weak background-word AUROC is a prompt problem.
Text-only features (relative position among the caption's nouns, repeat mention) are added for comparison.
Output: results/amber_g/full_clip/nouns_full.csv

Example:
    python full_clip_nouns.py
"""

import csv
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from amber_noun_labels import AmberLabeler
from rag_retrieve import CLIP_ID, embed
from remask_decoding import ROOT, STUDY
from trace_amber_steps import AMBER_DATA, IMAGE_DIR, PREDICTIONS, QUERY_FILE, read_json

PROMPTS = dict(detector="a photo of a {}.", bare="a photo of {}.", there="there is {} in the image.", word="{}")


def main():
    out_dir = os.path.join(ROOT, "results", "amber_g", "full_clip")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    labeler = AmberLabeler()
    queries = {int(r["id"]): r for r in read_json(QUERY_FILE)}
    predictions = {int(r["id"]): r["response"] for r in read_json(PREDICTIONS)}
    study_ids = set(read_json(os.path.join(STUDY, "ids.json"))["ids"]) | set(read_json(os.path.join(STUDY, "ids_clean.json"))["ids"])

    relation = json.load(open(os.path.join(AMBER_DATA, "relation.json"), encoding="utf-8"))
    with open(os.path.join(STUDY, "with_clean", "nouns.csv"), encoding="utf-8") as handle:
        lemmas = {r["lemma"] for r in csv.DictReader(handle)}
    vocabulary = sorted(set(relation) | {w for v in relation.values() for w in v} | lemmas
                        | (labeler.object_words - labeler.global_safe))
    word_index = {w: i for i, w in enumerate(vocabulary)}

    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    text = {}
    with torch.no_grad():
        for name, template in PROMPTS.items():
            chunks = []
            for i in range(0, len(vocabulary), 256):
                inputs = processor(text=[template.format(w) for w in vocabulary[i:i + 256]], return_tensors="pt",
                                   padding=True).to(device)
                chunks.append(F.normalize(clip.get_text_features(**inputs).float(), dim=-1))
            text[name] = torch.cat(chunks)
        text["ensemble"] = F.normalize(sum(text[n] for n in PROMPTS), dim=-1)

    rows = []
    for n, item_id in enumerate(sorted(predictions), 1):
        nouns = [r for r in labeler.label(item_id, predictions[item_id]) if r["label"] != "ignored"]
        if not nouns:
            continue
        photo = Image.open(os.path.join(IMAGE_DIR, queries[item_id]["image"])).convert("RGB")
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            image = embed([photo], clip, processor, device)
            sims = {name: (image @ t.T)[0] for name, t in text.items()}
        seen = set()
        for k, noun in enumerate(nouns):
            lemma = noun["lemma"]
            row = dict(id=item_id, study=item_id in study_ids, word=noun["word"], lemma=lemma, label=noun["label"],
                       relative_position=k / max(len(nouns) - 1, 1), repeat_mention=int(lemma in seen),
                       in_vocabulary=lemma in word_index)
            seen.add(lemma)
            if lemma in word_index:
                for name, s in sims.items():
                    row[f"rank_{name}"] = float((s < s[word_index[lemma]]).float().mean())
                    row[f"sim_{name}"] = float(s[word_index[lemma]])
            rows.append(row)
        if n % 100 == 0:
            print(f"{n}/{len(predictions)} captions, {len(rows)} nouns", flush=True)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(out_dir, "nouns_full.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}: {len(rows)} nouns, {sum(r['label'] == 'hallucinated' for r in rows)} hallucinated")


if __name__ == "__main__":
    main()
