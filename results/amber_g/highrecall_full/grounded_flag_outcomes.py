"""What the word-first zoom refill does to falsely flagged (grounded) words, in the categories of flag_repredict.py:
same word / another correct object / neutral / wrong object, at the flagged position and over the remasked span.
Also the refill's first pass at the flagged position (p of the original word, crop's top token) and the commonest swaps.

CPU only (spaCy). Example (from MDLLM/):
    python results/amber_g/highrecall_full/grounded_flag_outcomes.py results/amber_g/highrecall_full/span_r1
"""

import json
import os
import statistics as st
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402

run_dir = os.path.abspath(sys.argv[1])
caps = read_json(os.path.join(run_dir, "predictions_events.json"))["captions"]
baseline = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
labeler = AmberLabeler()
objects = labeler.object_words - labeler.global_safe


def label(item_id, token):
    word = token.strip().lower()
    if not word.isalpha():
        return None
    lemma = labeler.lemmatizer.lemmatize(word)
    if lemma not in objects:
        return None
    _, safe_words, _ = labeler.word_lists(item_id)
    return lemma, ("correct" if lemma in safe_words or any(labeler.similar(lemma, w) for w in safe_words) else "wrong")


order = ("same word", "another correct object", "neutral", "wrong object")
at_pos = {g: Counter() for g in ("hallucinating", "clean")}
span_wrong = {g: Counter() for g in at_pos}
swaps = Counter()
p_orig, crop_top_is_orig, n = [], 0, 0
for c in caps:
    i = int(c["id"])
    g = "hallucinating" if any(r["label"] == "hallucinated" for r in labeler.label(i, baseline[i])) else "clean"
    for e in c["events"]:
        if e["oracle_hallucinated"]:
            continue
        k = e["remasked"].index(e["pos"])
        old, new = e["old_text"][k], e["new_text"][k]
        if old.strip().lower() == new.strip().lower():
            kind = "same word"
        else:
            lab = label(i, new)
            kind = "neutral" if lab is None else ("another correct object" if lab[1] == "correct" else "wrong object")
            swaps[(old.strip(), new.strip())] += 1
        at_pos[g][kind] += 1
        span_wrong[g][any(l and l[1] == "wrong" for l in (label(i, t) for t in e["new_text"]))] += 1
        if e.get("p_original_refill") is not None:
            p_orig.append(e["p_original_refill"])
        wf = e.get("word_first") or {}
        if wf.get("token") is not None:
            n += 1
            crop_top_is_orig += wf["token"].strip().lower() == old.strip().lower()

print("grounded (false) flags, outcome at the flagged position:")
for g, cnt in at_pos.items():
    tot = sum(cnt.values())
    print(f"  {g:<13} n={tot:>4} | " + " | ".join(f"{k} {cnt[k]} ({cnt[k] / tot:.0%})" for k in order)
          + f" | span has a wrong object: {span_wrong[g][True]} ({span_wrong[g][True] / tot:.0%})")
if p_orig:
    print(f"first refill pass (crop, span masked), p(original word): mean {st.mean(p_orig):.2f} median {st.median(p_orig):.2f}, "
          f"< 0.5 in {sum(p < 0.5 for p in p_orig)} / {len(p_orig)}")
if n:
    print(f"crop's top token is the original word: {crop_top_is_orig} / {n}")
print("commonest swaps:", swaps.most_common(30))
