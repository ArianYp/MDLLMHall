"""Official AMBER-g scores of a partial full-set run vs the baseline on the same ids, plus paired noun counts and trigger stats.

Example (from MDLLM/):
    python results/amber_g/slot_refill_full/score_partial.py results/amber_g/slot_refill_full/clip_cropveto_zoom
"""

import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

run_dir = os.path.abspath(sys.argv[1])
out = os.path.join(run_dir, sys.argv[2] if len(sys.argv) > 2 else "partial_eval")
os.makedirs(out, exist_ok=True)
for name in ("predictions.json", "predictions_events.json"):
    shutil.copy(os.path.join(run_dir, name), os.path.join(out, "snapshot_" + name))
method = {int(r["id"]): r["response"] for r in read_json(os.path.join(out, "snapshot_predictions.json"))}
events = read_json(os.path.join(out, "snapshot_predictions_events.json"))
baseline = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
ids = sorted(method)

scores = {}
for name, responses in (("baseline", baseline), ("clip_cropveto_zoom", method)):
    records = [dict(id=i, response=responses[i]) for i in ids]
    scores[name] = official_scores(records, os.path.join(out, f"{name}_ids_{ids[0]}-{ids[-1]}.json"))

labeler = AmberLabeler()
count = lambda text, i, lab: sum(n["label"] == lab for n in labeler.label(i, text))
h_base = np.array([count(baseline[i], i, "hallucinated") for i in ids])
h_meth = np.array([count(method[i], i, "hallucinated") for i in ids])
g_base = np.array([count(baseline[i], i, "grounded") for i in ids])
g_meth = np.array([count(method[i], i, "grounded") for i in ids])
hallucinating = h_base > 0

caps = events["captions"]
ev = [e for c in caps for e in c["events"]]
changed = lambda e: e["old_text"][e["remasked"].index(e["pos"])] != e["new_text"][e["remasked"].index(e["pos"])]
report = dict(
    n_captions=len(ids), id_range=[ids[0], ids[-1]], official=scores,
    hallucinated_nouns=dict(baseline=int(h_base.sum()), method=int(h_meth.sum())),
    grounded_nouns=dict(baseline=int(g_base.sum()), method=int(g_meth.sum())),
    text_changed=int(sum(baseline[i] != method[i] for i in ids)),
    per_caption=dict(fewer=int((h_meth < h_base).sum()), same=int((h_meth == h_base).sum()), more=int((h_meth > h_base).sum())),
    baseline_hallucinating=dict(n=int(hallucinating.sum()), baseline=int(h_base[hallucinating].sum()), method=int(h_meth[hallucinating].sum())),
    baseline_clean=dict(n=int((~hallucinating).sum()), method=int(h_meth[~hallucinating].sum()),
                        now_hallucinating=int((h_meth[~hallucinating] > 0).sum())),
    triggers=dict(eligible=sum(c["eligible"] for c in caps), fired=len(ev), on_hallucinated=sum(e["oracle_hallucinated"] for e in ev),
                  changed_hallucinated=sum(changed(e) for e in ev if e["oracle_hallucinated"]),
                  changed_grounded=sum(changed(e) for e in ev if not e["oracle_hallucinated"]),
                  vetoed=sum(c.get("vetoed", 0) for c in caps), vetoed_hallucinated=sum(c.get("vetoed_oracle_hallucinated", 0) for c in caps)),
)
with open(os.path.join(out, "partial_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print(f"ids {ids[0]}-{ids[-1]} ({len(ids)} captions), official AMBER-g:")
for name, s in scores.items():
    print(f"  {name:20s} CHAIR {s['CHAIR']:5.1f}  Cover {s['Cover']:5.1f}  Hal {s['Hal']:5.1f}  Cog {s['Cog']:4.1f}")
print(json.dumps({k: v for k, v in report.items() if k not in ("official", "n_captions", "id_range")}, indent=1))
