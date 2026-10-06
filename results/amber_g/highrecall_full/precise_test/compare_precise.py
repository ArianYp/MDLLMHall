"""Lower-false-positive CLIP detectors (precise_test/A, B) vs the baseline and span_r2 (high-recall detector) on the same ids:
official scores, noun counts with gross changes, detector precision, and what the refill puts at the flagged position.

CPU only. Safe while the run is going (reads copies). Example (from MDLLM/):
    python results/amber_g/highrecall_full/precise_test/compare_precise.py [A A3]
"""

import os
import shutil
import sys
from collections import Counter

from nltk.tag import PerceptronTagger

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

out = os.path.join(HERE, "eval")
os.makedirs(out, exist_ok=True)
RUNS = {"span ±2 (0.99, crop not top-1)": os.path.join(FULL, "span_r2"),
        "A (0.97 AND crop 0.97)": os.path.join(HERE, "A"),
        "A, span ±3": os.path.join(HERE, "A3")}
# optional argv: run directory names to include (default all), e.g. `compare_precise.py A` to score A alone on all its ids
if len(sys.argv) > 1:
    RUNS = {k: v for k, v in RUNS.items() if k.startswith("span") or os.path.basename(v) in sys.argv[1:]}

read = lambda path: {int(r["id"]): r["response"] for r in read_json(path)}
texts = {"baseline": read(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
logs = {}
for k, (name, d) in enumerate(RUNS.items()):
    if not os.path.exists(os.path.join(d, "predictions.json")):
        continue
    for f in ("predictions.json", "predictions_events.json"):
        shutil.copy(os.path.join(d, f), os.path.join(out, f"snapshot_{k}_{f}"))
    texts[name] = read(os.path.join(out, f"snapshot_{k}_predictions.json"))
    logs[name] = {int(c["id"]): c for c in read_json(os.path.join(out, f"snapshot_{k}_predictions_events.json"))["captions"]}
ids = sorted(set.intersection(*(set(t) for t in texts.values())) & set(range(1, 101)))
W = 31

print(f"ids {ids[0]}-{ids[-1]} ({len(ids)} captions finished in every run), official scorer")
for k, (name, t) in enumerate(texts.items()):
    s = official_scores([dict(id=i, response=t[i]) for i in ids], os.path.join(out, f"scores_{k}.json"))
    print(f"  {name:<{W}} CHAIR {s['CHAIR']:5.1f}  Cover {s['Cover']:5.1f}  Hal {s['Hal']:5.1f}  Cog {s['Cog']:4.1f}")

labeler = AmberLabeler()
tagger = PerceptronTagger()
objects = labeler.object_words - labeler.global_safe
labels = {name: {i: labeler.label(i, t[i]) for i in ids} for name, t in texts.items()}
lem = lambda name, i, lab: Counter(n["lemma"] for n in labels[name][i] if n["label"] == lab)
group = {i: "hallucinating" if lem("baseline", i, "hallucinated") else "clean" for i in ids}
n = Counter(group.values())
print(f"\nnoun counts ({n['hallucinating']} captions hallucinate at baseline, {n['clean']} clean)")
print(f"  {'run':<{W}} {'halluc.':>7} {'grounded':>8} | hallucinating: removed / added | clean: new halluc. (captions) | grounded lost / gained | words")
for name in texts:
    h = sum(sum(lem(name, i, "hallucinated").values()) for i in ids)
    g = sum(sum(lem(name, i, "grounded").values()) for i in ids)
    words = sum(len(texts[name][i].split()) for i in ids)
    if name == "baseline":
        print(f"  {name:<{W}} {h:>7} {g:>8} | {'':>30} | {'':>29} | {'':>22} | {words}")
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
    print(f"  {name:<{W}} {h:>7} {g:>8} | {rem:>14} / {add:<13} | {cn:>18} ({cc:>3}) {'':>5} | {lost:>12} / {gained:<7} | {words}")


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
    print(f"\n{name}: fired {len(ev)} / {elig} eligible ({len(ev) / max(elig, 1):.0%}), precision {hits / max(len(ev), 1):.2f}, "
          f"on hallucinated {hits}, on grounded {len(ev) - hits}")
    for flagged in ("grounded", "hallucinated"):
        cnt = Counter(kind(i, e["old_text"][e["remasked"].index(e["pos"])], e["new_text"][e["remasked"].index(e["pos"])])
                      for i, e in ev if bool(e["oracle_hallucinated"]) == (flagged == "hallucinated"))
        tot = sum(cnt.values())
        print(f"  {flagged} flags n={tot}: " + " | ".join(f"{k} {cnt[k]} ({cnt[k] / max(tot, 1):.0%})" for k in ORDER))
