"""Baseline-hallucinating AMBER-g ids (AmberLabeler, as in score_partial.py) not already run in the study-set oracle+zoom."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
baseline = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results/amber_g/mmada_predictions.json"))}
study = {int(r["id"]) for r in read_json(os.path.join(ROOT, "results/amber_g/hallu_study/rag/slot_refill/oracle_zoom/predictions.json"))}
labeler = AmberLabeler()
hallucinating = [i for i in sorted(baseline) if any(n["label"] == "hallucinated" for n in labeler.label(i, baseline[i]))]
missing = [i for i in hallucinating if i not in study]
for name, ids in (("ids_baseline_hallucinating.json", hallucinating), ("ids_missing.json", missing)):
    with open(os.path.join(here, name), "w", encoding="utf-8") as handle:
        json.dump(dict(source="results/amber_g/mmada_predictions.json, AmberLabeler", ids=ids), handle)
print(f"baseline captions {len(baseline)}, hallucinating {len(hallucinating)}, in study {len(set(hallucinating) & study)}, missing {len(missing)}")
