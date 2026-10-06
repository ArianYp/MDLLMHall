"""Snapshot of the running span_r2_noun full-set run vs the baseline and span_r2 (no noun constraint) on the same ids:
official scores, noun counts with gross changes, and what the refill puts at the flagged position.

CPU only. Safe while the run is going (reads copies). Example (from MDLLM/):
    python results/amber_g/highrecall_full/span_r2_noun_partial.py
"""

import os
import shutil
import sys
from collections import Counter

from nltk.tag import PerceptronTagger

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

RUN = os.path.join(HERE, "span_r2_noun")
out = os.path.join(RUN, "partial_eval")
os.makedirs(out, exist_ok=True)
for name in ("predictions.json", "predictions_events.json"):
    shutil.copy(os.path.join(RUN, name), os.path.join(out, "snapshot_" + name))

read = lambda path: {int(r["id"]): r["response"] for r in read_json(path)}
texts = {"baseline": read(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")),
         "span ±2": read(os.path.join(HERE, "span_r2", "predictions.json")),
         "span ±2 + noun": read(os.path.join(out, "snapshot_predictions.json"))}
logs = {"span ±2": {int(c["id"]): c for c in read_json(os.path.join(HERE, "span_r2", "predictions_events.json"))["captions"]},
        "span ±2 + noun": {int(c["id"]): c for c in read_json(os.path.join(out, "snapshot_predictions_events.json"))["captions"]}}
ids = sorted(texts["span ±2 + noun"])

print(f"ids {ids[0]}-{ids[-1]} ({len(ids)} captions finished), official scorer")
for name, t in texts.items():
    s = official_scores([dict(id=i, response=t[i]) for i in ids], os.path.join(out, name.replace(" ", "_").replace("±", "pm") + ".json"))
    print(f"  {name:<16} CHAIR {s['CHAIR']:5.1f}  Cover {s['Cover']:5.1f}  Hal {s['Hal']:5.1f}  Cog {s['Cog']:4.1f}")

labeler = AmberLabeler()
tagger = PerceptronTagger()
objects = labeler.object_words - labeler.global_safe
labels = {name: {i: labeler.label(i, t[i]) for i in ids} for name, t in texts.items()}
lem = lambda name, i, lab: Counter(n["lemma"] for n in labels[name][i] if n["label"] == lab)
group = {i: "hallucinating" if lem("baseline", i, "hallucinated") else "clean" for i in ids}
n = Counter(group.values())
print(f"\nnoun counts ({n['hallucinating']} captions hallucinate at baseline, {n['clean']} clean)")
print(f"  {'run':<16} {'halluc.':>7} {'grounded':>8} | hallucinating captions: removed / added | clean: new halluc. (captions) | grounded lost / gained | words")
for name in texts:
    h = sum(sum(lem(name, i, "hallucinated").values()) for i in ids)
    g = sum(sum(lem(name, i, "grounded").values()) for i in ids)
    words = sum(len(texts[name][i].split()) for i in ids)
    if name == "baseline":
        print(f"  {name:<16} {h:>7} {g:>8} | {'':>39} | {'':>29} | {'':>22} | {words}")
        continue
    rem = add = cn = cc = lost = gained = 0
    for i in ids:
        hb, hm = lem("baseline", i, "hallucinated"), lem(name, i, "hallucinated")
        gb, gm = lem("baseline", i, "grounded"), lem(name, i, "grounded")
        lost += sum((gb - gm).values())
        gained += sum((gm - gb).values())
        if group[i] == "hallucinating":
            rem += sum((hb - hm).values())
            add += sum((hm - hb).values())
        else:
            cn += sum(hm.values())
            cc += bool(hm)
    print(f"  {name:<16} {h:>7} {g:>8} | {rem:>23} / {add:<13} | {cn:>18} ({cc:>3}) {'':>5} | {lost:>12} / {gained:<7} | {words}")


def kind(i, old, token):
    word = token.strip().lower()
    if word == old.strip().lower():
        return "same word"
    if word.isalpha():
        lemma = labeler.lemmatizer.lemmatize(word)
        if lemma in objects:
            _, safe, _ = labeler.word_lists(i)
            return "another correct object" if lemma in safe or any(labeler.similar(lemma, s) for s in safe) else "wrong object"
        if tagger.tag([word])[0][1].startswith("NN"):
            return "neutral noun"
    return "neutral, not a noun"


ORDER = ("same word", "another correct object", "neutral noun", "neutral, not a noun", "wrong object")
for name, log in logs.items():
    ev = [(i, e) for i in ids for e in log[i]["events"]]
    hits = sum(e["oracle_hallucinated"] for _, e in ev)
    elig = sum(log[i]["eligible"] for i in ids)
    print(f"\n{name}: fired {len(ev)} / {elig} eligible ({len(ev) / max(elig, 1):.0%}), precision {hits / max(len(ev), 1):.2f}")
    for flagged in ("grounded", "hallucinated"):
        cnt = Counter(kind(i, e["old_text"][e["remasked"].index(e["pos"])], e["new_text"][e["remasked"].index(e["pos"])])
                      for i, e in ev if bool(e["oracle_hallucinated"]) == (flagged == "hallucinated"))
        tot = sum(cnt.values())
        print(f"  {flagged} flags n={tot}: " + " | ".join(f"{k} {cnt[k]} ({cnt[k] / max(tot, 1):.0%})" for k in ORDER))
