"""Full AMBER-g oracle+zoom predictions: study-set run where available, the run on the missing hallucinating ids
otherwise, and the baseline text for baseline-clean captions (the oracle has nothing to fix there; on the 60 study
clean captions it changed 3 texts and added 0 hallucinations). Official scores and paired noun counts vs baseline.

    python results/amber_g/oracle_zoom_full/merge_and_score.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
load = lambda path: {int(r["id"]): r["response"] for r in read_json(path)}
baseline = load(os.path.join(ROOT, "results/amber_g/mmada_predictions.json"))
study = load(os.path.join(ROOT, "results/amber_g/hallu_study/rag/slot_refill/oracle_zoom/predictions.json"))
missing = load(os.path.join(here, "missing/predictions.json"))
needed = read_json(os.path.join(here, "ids_missing.json"))["ids"]
assert set(needed) <= set(missing), f"{len(set(needed) - set(missing))} missing ids not run yet"

ids = sorted(baseline)
source = {i: "study" if i in study else "missing" if i in missing else "baseline" for i in ids}
method = {i: study.get(i, missing.get(i, baseline[i])) for i in ids}
merged = [dict(id=i, response=method[i]) for i in ids]
with open(os.path.join(here, "predictions.json"), "w", encoding="utf-8") as handle:
    json.dump(merged, handle)

scores = {name: official_scores([dict(id=i, response=r[i]) for i in ids], os.path.join(here, f"{name}_official.json"))
          for name, r in (("baseline", baseline), ("oracle_zoom", method))}

labeler = AmberLabeler()
count = lambda text, i, lab: sum(n["label"] == lab for n in labeler.label(i, text))
h_base = np.array([count(baseline[i], i, "hallucinated") for i in ids])
h_meth = np.array([count(method[i], i, "hallucinated") for i in ids])
g_base = np.array([count(baseline[i], i, "grounded") for i in ids])
g_meth = np.array([count(method[i], i, "grounded") for i in ids])
hallucinating = h_base > 0

events = [e for c in read_json(os.path.join(here, "missing/predictions_events.json"))["captions"] for e in c["events"]]
events += [e for c in read_json(os.path.join(ROOT, "results/amber_g/hallu_study/rag/slot_refill/oracle_zoom/predictions_events.json"))["captions"] for e in c["events"]]
changed = lambda e: e["old_text"][e["remasked"].index(e["pos"])] != e["new_text"][e["remasked"].index(e["pos"])]
report = dict(
    n_captions=len(ids), sources={s: sum(v == s for v in source.values()) for s in ("study", "missing", "baseline")}, official=scores,
    hallucinated_nouns=dict(baseline=int(h_base.sum()), method=int(h_meth.sum())),
    grounded_nouns=dict(baseline=int(g_base.sum()), method=int(g_meth.sum())),
    text_changed=int(sum(baseline[i] != method[i] for i in ids)),
    per_caption=dict(fewer=int((h_meth < h_base).sum()), same=int((h_meth == h_base).sum()), more=int((h_meth > h_base).sum())),
    baseline_hallucinating=dict(n=int(hallucinating.sum()), baseline=int(h_base[hallucinating].sum()), method=int(h_meth[hallucinating].sum()),
                                grounded_baseline=int(g_base[hallucinating].sum()), grounded_method=int(g_meth[hallucinating].sum())),
    baseline_clean=dict(n=int((~hallucinating).sum()), method=int(h_meth[~hallucinating].sum())),
    triggers=dict(fired=len(events), changed=sum(map(changed, events))),
)
with open(os.path.join(here, "report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("full AMBER-g (1004), official scorer:")
for name, s in scores.items():
    print(f"  {name:12s} CHAIR {s['CHAIR']:5.1f}  Cover {s['Cover']:5.1f}  Hal {s['Hal']:5.1f}  Cog {s['Cog']:4.1f}")
print(json.dumps({k: v for k, v in report.items() if k not in ("official", "n_captions")}, indent=1))
