"""Base vs MixCoT baseline on the ids the Base baseline has finished so far: official scores and noun counts.

CPU only. Example (from MDLLM/):
    python results/amber_g_base/score_baselines_partial.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

base = {int(r["id"]): r["response"] for r in read_json(os.path.join(HERE, "mmada_base_predictions_1-100.json"))}
mix = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
ids = sorted(base)
out = os.path.join(HERE, "partial_baselines")
os.makedirs(out, exist_ok=True)
labeler = AmberLabeler()

print(f"ids {ids[0]}-{ids[-1]} ({len(ids)} captions finished by the Base baseline)")
print(f"  {'model':<8} {'CHAIR':>6} {'Cover':>6} {'Hal':>6} {'Cog':>5} | {'halluc.':>7} {'grounded':>8} "
      f"{'hallucinating captions':>22} {'words / caption':>16}")
for name, texts in (("MixCoT", mix), ("Base", base)):
    s = official_scores([dict(id=i, response=texts[i]) for i in ids], os.path.join(out, f"{name}_ids_{ids[0]}-{ids[-1]}.json"))
    labels = {i: labeler.label(i, texts[i]) for i in ids}
    h = sum(n["label"] == "hallucinated" for i in ids for n in labels[i])
    g = sum(n["label"] == "grounded" for i in ids for n in labels[i])
    caps = sum(any(n["label"] == "hallucinated" for n in labels[i]) for i in ids)
    words = sum(len(texts[i].split()) for i in ids) / len(ids)
    print(f"  {name:<8} {s['CHAIR']:>6.1f} {s['Cover']:>6.1f} {s['Hal']:>6.1f} {s['Cog']:>5.1f} | {h:>7} {g:>8} "
          f"{caps:>22} {words:>16.1f}")
print(f"captions identical in both models: {sum(base[i] == mix[i] for i in ids)} / {len(ids)}")
for i in ids[:2]:
    print(f"\n[{i}] MixCoT: {mix[i][:400]}\n[{i}] Base:   {base[i][:400]}")
