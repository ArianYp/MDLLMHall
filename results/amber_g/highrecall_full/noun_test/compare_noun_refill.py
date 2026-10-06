"""Noun-constrained run vs the unconstrained span-2 run on the same ids: noun counts vs baseline, and what the refill
puts at the flagged position (same word / another correct object / neutral noun / neutral non-noun / wrong object),
for grounded and hallucinated flags.

CPU only (spaCy). Example (from MDLLM/):
    python results/amber_g/highrecall_full/noun_test/compare_noun_refill.py
"""

import os
import sys
from collections import Counter

from nltk.tag import PerceptronTagger

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402

SPAN = os.path.join(HERE, "..", "span_r2")
NEW = os.path.join(HERE, "noun_r2")


def preds(run_dir):
    return {int(r["id"]): r["response"] for r in read_json(os.path.join(run_dir, "predictions.json"))}


def events(run_dir):
    return {int(c["id"]): c for c in read_json(os.path.join(run_dir, "predictions_events.json"))["captions"]}


new = preds(NEW)
ids = sorted(new)
baseline_all = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
runs = {"baseline": baseline_all, "span ±2": preds(SPAN), "span ±2 + noun": new}
logs = {"span ±2": events(SPAN), "span ±2 + noun": events(NEW)}
labeler = AmberLabeler()
tagger = PerceptronTagger()
objects = labeler.object_words - labeler.global_safe

labels = {name: {i: labeler.label(i, r[i]) for i in ids} for name, r in runs.items()}


def lemmas(name, i, label):
    return Counter(n["lemma"] for n in labels[name][i] if n["label"] == label)


group = {i: "hallucinating" if lemmas("baseline", i, "hallucinated") else "clean" for i in ids}
n_group = Counter(group.values())
print(f"ids {ids[0]}-{ids[-1]}: {len(ids)} captions ({n_group['hallucinating']} hallucinating, {n_group['clean']} clean at baseline)\n")

print("noun counts (AmberLabeler, same as the scorer's counts)")
print(f"  {'run':<16} {'halluc.':>8} {'grounded':>9} | hallucinating captions: removed / added | "
      f"clean: new halluc. (captions) | grounded lost / gained | words")
for name in runs:
    h = sum(sum(lemmas(name, i, "hallucinated").values()) for i in ids)
    g = sum(sum(lemmas(name, i, "grounded").values()) for i in ids)
    words = sum(len(runs[name][i].split()) for i in ids)
    if name == "baseline":
        print(f"  {name:<16} {h:>8} {g:>9} | {'':>38} | {'':>28} | {'':>22} | {words}")
        continue
    rem = add = clean_new = clean_caps = lost = gained = 0
    for i in ids:
        hb, hm = lemmas("baseline", i, "hallucinated"), lemmas(name, i, "hallucinated")
        gb, gm = lemmas("baseline", i, "grounded"), lemmas(name, i, "grounded")
        lost += sum((gb - gm).values())
        gained += sum((gm - gb).values())
        if group[i] == "hallucinating":
            rem += sum((hb - hm).values())
            add += sum((hm - hb).values())
        else:
            clean_new += sum(hm.values())
            clean_caps += bool(hm)
    print(f"  {name:<16} {h:>8} {g:>9} | {rem:>17} / {add:<18} | {clean_new:>15} ({clean_caps:>3}) {'':>7} | "
          f"{lost:>10} / {gained:<9} | {words}")

# same decoding until the first trigger: captions where neither run fired must be identical (A100 determinism)
quiet = [i for i in ids if not logs["span ±2"][i]["events"] and not logs["span ±2 + noun"][i]["events"]]
print(f"\nno trigger in either run: {len(quiet)} captions, text identical in {sum(runs['span ±2'][i] == new[i] for i in quiet)}")


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
    totals = Counter()
    for c in log.values():
        if int(c["id"]) in new:
            totals.update({k: c.get(k, 0) for k in ("eligible", "fired", "vetoed", "not_noun")})
    print(f"\n{name}: eligible {totals['eligible']}, not a noun in context {totals['not_noun']}, "
          f"vetoed by crop {totals['vetoed']}, fired {totals['fired']}")
    for flagged in ("grounded", "hallucinated"):
        cnt, swaps = Counter(), Counter()
        for i in ids:
            for e in log[i]["events"]:
                if bool(e["oracle_hallucinated"]) != (flagged == "hallucinated"):
                    continue
                k = e["remasked"].index(e["pos"])
                old, tok = e["old_text"][k], e["new_text"][k]
                kd = kind(i, old, tok)
                cnt[kd] += 1
                if kd != "same word":
                    swaps[(old.strip(), tok.strip())] += 1
        tot = sum(cnt.values())
        print(f"  {flagged} flags n={tot}: " + " | ".join(f"{k} {cnt[k]} ({cnt[k] / max(tot, 1):.0%})" for k in ORDER))
        print(f"    commonest swaps: {swaps.most_common(12)}")
