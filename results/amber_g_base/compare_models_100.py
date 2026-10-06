"""MixCoT vs Base on AMBER-g ids 1-100: official scores of each model's baseline and of the method (high-recall CLIP
trigger, zoom word-first refill, span +-2, noun constraints), noun counts against each model's own baseline, and what
the refill puts at the flagged position.

CPU only (spaCy + the official scorer). Example (from MDLLM/):
    python results/amber_g_base/compare_models_100.py
"""

import os
import sys
from collections import Counter

from nltk.tag import PerceptronTagger

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402
from eval_remask import official_scores  # noqa: E402

MIX = os.path.join(ROOT, "results", "amber_g")
# name: (model, predictions file, events file or None, the model's baseline run)
RUNS = {
    "MixCoT baseline": ("MixCoT", os.path.join(MIX, "mmada_predictions.json"), None, None),
    "MixCoT span ±2": ("MixCoT", os.path.join(MIX, "highrecall_full", "span_r2", "predictions.json"),
                       os.path.join(MIX, "highrecall_full", "span_r2", "predictions_events.json"), "MixCoT baseline"),
    "MixCoT span ±2 + noun": ("MixCoT", os.path.join(MIX, "highrecall_full", "noun_test", "noun_r2", "predictions.json"),
                              os.path.join(MIX, "highrecall_full", "noun_test", "noun_r2", "predictions_events.json"),
                              "MixCoT baseline"),
    "Base baseline": ("Base", os.path.join(HERE, "mmada_base_predictions_1-100.json"), None, None),
    "Base span ±2 + noun": ("Base", os.path.join(HERE, "noun_r2", "predictions.json"),
                            os.path.join(HERE, "noun_r2", "predictions_events.json"), "Base baseline"),
}
IDS = read_json(os.path.join(MIX, "highrecall_full", "noun_test", "ids_1_100.json"))["ids"]

texts = {name: {int(r["id"]): r["response"] for r in read_json(path)} for name, (_, path, _, _) in RUNS.items()}
logs = {name: {int(c["id"]): c for c in read_json(ev)["captions"]} for name, (_, _, ev, _) in RUNS.items() if ev}
for name, t in texts.items():
    missing = [i for i in IDS if i not in t]
    assert not missing, f"{name}: {len(missing)} of the ids missing"

out = os.path.join(HERE, "compare_100")
os.makedirs(out, exist_ok=True)
print(f"AMBER-g ids {IDS[0]}-{IDS[-1]} ({len(IDS)} captions), official scorer")
print(f"  {'run':<24} {'CHAIR':>6} {'Cover':>6} {'Hal':>6} {'Cog':>5}")
for name in RUNS:
    s = official_scores([dict(id=i, response=texts[name][i]) for i in IDS],
                        os.path.join(out, name.replace(" ", "_").replace("±", "pm") + ".json"))
    print(f"  {name:<24} {s['CHAIR']:>6.1f} {s['Cover']:>6.1f} {s['Hal']:>6.1f} {s['Cog']:>5.1f}")

labeler = AmberLabeler()
tagger = PerceptronTagger()
objects = labeler.object_words - labeler.global_safe
labels = {name: {i: labeler.label(i, texts[name][i]) for i in IDS} for name in RUNS}


def lemmas(name, i, label):
    return Counter(n["lemma"] for n in labels[name][i] if n["label"] == label)


print("\nnoun counts (AmberLabeler); method runs against their own model's baseline")
print(f"  {'run':<24} {'halluc.':>7} {'grounded':>8} {'words':>6} | captions hallucinating at baseline | "
      f"halluc. removed / added | clean: new halluc. (captions) | grounded lost / gained")
for name, (model, _, _, base) in RUNS.items():
    h = sum(sum(lemmas(name, i, "hallucinated").values()) for i in IDS)
    g = sum(sum(lemmas(name, i, "grounded").values()) for i in IDS)
    words = sum(len(texts[name][i].split()) for i in IDS)
    line = f"  {name:<24} {h:>7} {g:>8} {words:>6} | "
    if base is None:
        n_hal = sum(bool(lemmas(name, i, "hallucinated")) for i in IDS)
        print(line + f"{n_hal:>34} |")
        continue
    rem = add = clean_new = clean_caps = lost = gained = 0
    for i in IDS:
        hb, hm = lemmas(base, i, "hallucinated"), lemmas(name, i, "hallucinated")
        gb, gm = lemmas(base, i, "grounded"), lemmas(name, i, "grounded")
        lost += sum((gb - gm).values())
        gained += sum((gm - gb).values())
        if hb:
            rem += sum((hb - hm).values())
            add += sum((hm - hb).values())
        else:
            clean_new += sum(hm.values())
            clean_caps += bool(hm)
    print(line + f"{'':>34} | {rem:>15} / {add:<5} | {clean_new:>18} ({clean_caps:>2}) {'':>6} | {lost:>12} / {gained}")

# the method's decoding equals the baseline until the first trigger, so untriggered captions must match it
for name, (_, _, _, base) in RUNS.items():
    if name in logs:
        quiet = [i for i in IDS if not logs[name][i]["events"]]
        same = sum(texts[name][i] == texts[base][i] for i in quiet)
        print(f"{name}: {len(quiet)} captions without a trigger, identical to {base} in {same}")


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
    for i in IDS:
        totals.update({k: log[i].get(k, 0) for k in ("eligible", "fired", "vetoed", "not_noun")})
    hits = sum(e["oracle_hallucinated"] for i in IDS for e in log[i]["events"])
    print(f"\n{name}: eligible {totals['eligible']}, not a noun {totals['not_noun']}, vetoed by crop {totals['vetoed']}, "
          f"fired {totals['fired']} ({totals['fired'] / max(totals['eligible'], 1):.0%}), precision {hits / max(totals['fired'], 1):.2f}")
    for flagged in ("grounded", "hallucinated"):
        cnt, swaps = Counter(), Counter()
        for i in IDS:
            for e in log[i]["events"]:
                if bool(e["oracle_hallucinated"]) != (flagged == "hallucinated"):
                    continue
                k = e["remasked"].index(e["pos"])
                kd = kind(i, e["old_text"][k], e["new_text"][k])
                cnt[kd] += 1
                if kd != "same word":
                    swaps[(e["old_text"][k].strip(), e["new_text"][k].strip())] += 1
        tot = sum(cnt.values())
        print(f"  {flagged} flags n={tot}: " + " | ".join(f"{k} {cnt[k]} ({cnt[k] / max(tot, 1):.0%})" for k in ORDER))
        print(f"    commonest swaps: {swaps.most_common(10)}")

print("\nexample captions (first 3 ids where the two baselines differ):")
shown = 0
for i in IDS:
    if texts["MixCoT baseline"][i] != texts["Base baseline"][i] and shown < 3:
        shown += 1
        for name in ("MixCoT baseline", "Base baseline", "Base span ±2 + noun"):
            print(f"  [{i}] {name}: {texts[name][i][:300]}")
