"""Why falsely flagged grounded words change online when the offline one-word re-prediction keeps them (flag_repredict.txt).
For each grounded flag: was the right-hand text still masked at the trigger, does the original lemma survive anywhere
in the refilled span, and the word-first pass (crop, span masked): p(original), correct / neutral mass, the argmax.

CPU only (spaCy). Example (from MDLLM/):
    python results/amber_g/highrecall_full/grounded_flag_context.py results/amber_g/highrecall_full/span_r1
"""

import os
import statistics as st
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from amber_noun_labels import AmberLabeler, read_json  # noqa: E402

caps = read_json(os.path.join(os.path.abspath(sys.argv[1]), "predictions_events.json"))["captions"]
labeler = AmberLabeler()
lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())
is_mask = lambda t: "mask" in t

rows = []
for c in caps:
    for e in c["events"]:
        if e["oracle_hallucinated"]:
            continue
        k = e["remasked"].index(e["pos"])
        old, new = e["old_text"], e["new_text"]
        right_masked = k + 1 < len(old) and is_mask(old[k + 1])
        same_at_pos = old[k].strip().lower() == new[k].strip().lower()
        in_span = lem(old[k]) in {lem(t) for t in new if t.strip().isalpha()}
        fp, wf = e.get("first_pass") or {}, e.get("word_first") or {}
        rows.append(dict(right_masked=right_masked, n_masked_before=sum(map(is_mask, old)), same=same_at_pos, in_span=in_span,
                         p_orig=e.get("p_original_refill"), p_correct=fp.get("p_correct"), p_neutral=fp.get("p_neutral"),
                         top=wf.get("token"), top_p=wf.get("p"), top5=wf.get("top5")))

n = len(rows)
print(f"grounded flags: {n}")
print(f"  word kept at its position: {sum(r['same'] for r in rows)} ({sum(r['same'] for r in rows) / n:.0%}); "
      f"original lemma anywhere in the new span: {sum(r['in_span'] for r in rows)} ({sum(r['in_span'] for r in rows) / n:.0%})")
print(f"  masked tokens in the span before the remask: {dict(sorted(Counter(r['n_masked_before'] for r in rows).items()))}")
for flag in (False, True):
    S = [r for r in rows if r["right_masked"] == flag]
    if S:
        print(f"  right neighbour still masked = {flag}: n={len(S)}, kept {sum(r['same'] for r in S) / len(S):.0%}, "
              f"lemma in span {sum(r['in_span'] for r in S) / len(S):.0%}")

lost = [r for r in rows if not r["in_span"]]
kept = [r for r in rows if r["in_span"]]
for name, S in (("lemma kept", kept), ("lemma lost", lost)):
    S = [r for r in S if r["p_orig"] is not None]
    if S:
        print(f"  {name}: n={len(S)} | first pass p(original) median {st.median(r['p_orig'] for r in S):.2f} | "
              f"correct-object mass median {st.median(r['p_correct'] for r in S):.2f} | neutral mass median {st.median(r['p_neutral'] for r in S):.2f} | "
              f"argmax p median {st.median(r['top_p'] for r in S):.2f}")
print(f"  lost although correct-object mass > neutral mass (argmax went to a neutral token): "
      f"{sum(r['p_correct'] > r['p_neutral'] and r['top'] and lem(r['top']) not in labeler.object_words for r in lost)} / {len(lost)}")
print("  commonest word-first argmax when the lemma was lost:", Counter((r["top"] or "").strip() for r in lost).most_common(20))
print("  examples (lost): top-5 of the word-first pass")
for r in lost[:12]:
    print("   ", r["top5"])
