"""Pick AMBER-g captions with no hallucinated object noun (official per-noun rules) as controls."""

import json
import random
import sys

from amber_noun_labels import AmberLabeler, read_json

PREDICTIONS = "results/amber_g/mmada_predictions.json"
STUDY_IDS = "results/amber_g/hallu_study/ids.json"
OUT = "results/amber_g/hallu_study/ids_clean.json"
N, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20, 0

labeler = AmberLabeler()
exclude = set(read_json(STUDY_IDS)["ids"])
clean = []
for row in read_json(PREDICTIONS):
    item_id = int(row["id"])
    if item_id in exclude:
        continue
    labels = labeler.label(item_id, row["response"])
    if labels and all(l["label"] != "hallucinated" for l in labels) and any(l["label"] == "grounded" for l in labels):
        clean.append(item_id)
ids = sorted(random.Random(SEED).sample(clean, N))
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(dict(source=PREDICTIONS, rule="no noun labelled hallucinated, >=1 grounded noun, not in ids.json",
                   n_clean_pool=len(clean), seed=SEED, ids=ids), handle)
print(f"clean captions available: {len(clean)} / {1004 - len(exclude)}; sampled {N}: {ids}")
