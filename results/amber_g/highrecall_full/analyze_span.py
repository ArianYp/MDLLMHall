"""Where the high-recall run gains and loses nouns vs the baseline: per-caption noun multiset diffs and trigger breakdown.

Example (from MDLLM/):
    python results/amber_g/highrecall_full/analyze_span.py results/amber_g/highrecall_full/span_r1
"""

import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402

run_dir = os.path.abspath(sys.argv[1])
method = {int(r["id"]): r["response"] for r in read_json(os.path.join(run_dir, "predictions.json"))}
events = read_json(os.path.join(run_dir, "predictions_events.json"))
baseline = {int(r["id"]): r["response"] for r in read_json(os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json"))}
ids = sorted(method)
labeler = AmberLabeler()


def lemmas(i, text, label):
    return Counter(n["lemma"] for n in labeler.label(i, text) if n["label"] == label)


groups = {g: {k: Counter() for k in ("h_removed", "h_added", "g_lost", "g_gained")} for g in ("hallucinating", "clean")}
totals = {g: Counter() for g in groups}
for i in ids:
    hb, hm = lemmas(i, baseline[i], "hallucinated"), lemmas(i, method[i], "hallucinated")
    gb, gm = lemmas(i, baseline[i], "grounded"), lemmas(i, method[i], "grounded")
    g = "hallucinating" if sum(hb.values()) else "clean"
    d = groups[g]
    d["h_removed"].update(hb - hm)
    d["h_added"].update(hm - hb)
    d["g_lost"].update(gb - gm)
    d["g_gained"].update(gm - gb)
    t = totals[g]
    t["captions"] += 1
    t["h_base"] += sum(hb.values())
    t["h_meth"] += sum(hm.values())
    t["g_base"] += sum(gb.values())
    t["g_meth"] += sum(gm.values())
    t["words_base"] += len(baseline[i].split())
    t["words_meth"] += len(method[i].split())
    t["text_changed"] += baseline[i] != method[i]

ev = [e for c in events["captions"] for e in c["events"]]
word = lambda e, key: e[key][e["remasked"].index(e["pos"])]
flag = {True: Counter(), False: Counter()}
outcome = Counter()
for e in ev:
    h = bool(e["oracle_hallucinated"])
    flag[h][word(e, "old_text").strip().lower()] += 1
    outcome[(h, word(e, "old_text") != word(e, "new_text"))] += 1

report = dict(
    totals={g: dict(t) for g, t in totals.items()},
    sums={g: {k: sum(c.values()) for k, c in d.items()} for g, d in groups.items()},
    top={g: {k: c.most_common(15) for k, c in d.items()} for g, d in groups.items()},
    triggers=dict(fired=len(ev),
                  on_hallucinated=sum(flag[True].values()), on_grounded=sum(flag[False].values()),
                  hallucinated_word_changed=outcome[(True, True)], hallucinated_word_kept=outcome[(True, False)],
                  grounded_word_changed=outcome[(False, True)], grounded_word_kept=outcome[(False, False)],
                  top_flagged_grounded=flag[False].most_common(20), top_flagged_hallucinated=flag[True].most_common(20)),
)
out = os.path.join(run_dir, "noun_diff.json")
with open(out, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=1)
print(json.dumps(report, indent=1))
print("saved", out)
