"""Word-first refill vs the current (most-confident-first) refill, oracle + zoom, both on the A100.

1) Refill outcomes per run: the refilled span is wrong (an object not in the image), correct (an object in the image,
   none wrong) or neutral (no object word); "same wrong word again" split out. Verb "can" excluded.
2) First refill pass of the word-first run (crop, span masked; identical to the first pass of the current order):
   probability mass at the flagged position on correct object words, wrong object words (incl. the original) and
   neutral tokens, overall and by the final outcome.
3) Paired first triggers: until its first refill a caption is in the same state in both runs, so the outcomes of the
   same trigger can be compared directly.

CPU only (spaCy). Example (from MDLLM/): python results/amber_g/hallu_study/wordfirst_analysis.py
"""

import json
import os
import statistics as st
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler

SR = os.path.join(HERE, "rag", "slot_refill")


def main():
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

    def outcome(item_id, e):
        k = e["remasked"].index(e["pos"])
        flagged = labeler.lemmatizer.lemmatize(e["old_text"][k].strip().lower())
        labels = [l for l in (label(item_id, t) for t in e["new_text"]) if l]
        wrong = [l for l, kind in labels if kind == "wrong"]
        if wrong:
            return ("wrong, same word" if flagged in wrong else "wrong, other word"), flagged
        return ("correct" if labels else "neutral"), flagged

    runs = {}
    for name in ("oracle_zoom", "oracle_zoom_wordfirst"):
        caps = json.load(open(os.path.join(SR, name, "predictions_events.json")))["captions"]
        runs[name] = {c["id"]: [(e, *outcome(c["id"], e)) for e in c["events"]] for c in caps}

    order = ("wrong, same word", "wrong, other word", "neutral", "correct")
    print("1) refill outcomes (verb 'can' excluded)")
    for name, caps in runs.items():
        rows = [o for evs in caps.values() for _, o, f in evs if f != "can"]
        c = Counter(rows)
        print(f"  {name:<22} refills {len(rows):>3} | " + " | ".join(f"{k} {c[k]:>3} ({c[k] / len(rows):.0%})" for k in order))

    print("\n2) first refill pass at the flagged position (word-first run), probability mass")
    evs = [(e, o) for caps in [runs["oracle_zoom_wordfirst"]] for v in caps.values() for e, o, f in v if f != "can" and e.get("first_pass")]
    def summary(S, name):
        if not S:
            return
        m = lambda k: st.median(e["first_pass"][k] for e, _ in S)
        a = lambda k: st.mean(e["first_pass"][k] for e, _ in S)
        print(f"  {name:<22} n={len(S):>3} | correct mean {a('p_correct'):.2f} median {m('p_correct'):.2f} | "
              f"neutral mean {a('p_neutral'):.2f} median {m('p_neutral'):.2f} | wrong mean {a('p_wrong'):.2f} median {m('p_wrong'):.2f} "
              f"(original word mean {st.mean(e['p_original_refill'] for e, _ in S):.2f})")
    summary(evs, "all")
    for k in order:
        summary([x for x in evs if x[1] == k], f"final: {k}")
    print("  how much of the first pass is on a correct object word:")
    for t in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"    p_correct >= {t}: {sum(e['first_pass']['p_correct'] >= t for e, _ in evs)} / {len(evs)}")
    print(f"  best correct word beats the original word: {sum((e['first_pass']['best_correct'] or ['', 0])[1] > e['p_original_refill'] for e, _ in evs)} / {len(evs)}")
    print(f"  crop's top word (step 1) is the original word: {sum(e['word_first']['token'] == e['old_text'][e['remasked'].index(e['pos'])] for e, _ in evs)} / {len(evs)}")
    print("  examples with the most mass on correct words:")
    for e, o in sorted(evs, key=lambda x: -x[0]["first_pass"]["p_correct"])[:8]:
        fp = e["first_pass"]
        print(f"    {''.join(e['old_text'])!r:<22} correct {fp['p_correct']:.2f} (best {fp['best_correct'][0]!r} {fp['best_correct'][1]:.2f}) "
              f"wrong {fp['p_wrong']:.2f} neutral {fp['p_neutral']:.2f} | step1 {e['word_first']['token']!r} -> {''.join(e['new_text'])!r} [{o}]")

    print("\n3) paired first triggers (same state in both runs)")
    A, B = runs["oracle_zoom"], runs["oracle_zoom_wordfirst"]
    pairs = [(i, A[i][0], B[i][0]) for i in A if A[i] and B.get(i) and A[i][0][0]["pos"] == B[i][0][0]["pos"]
             and A[i][0][0]["old_text"] == B[i][0][0]["old_text"] and A[i][0][0]["step"] == B[i][0][0]["step"]]
    pairs = [p for p in pairs if p[1][2] != "can"]
    print(f"  pairs: {len(pairs)}; refilled text differs: {sum(a[0]['new_text'] != b[0]['new_text'] for _, a, b in pairs)}")
    trans = Counter((a[1], b[1]) for _, a, b in pairs if a[0]["new_text"] != b[0]["new_text"])
    for (x, y), n in sorted(trans.items(), key=lambda t: -t[1]):
        print(f"    current: {x:<18} -> word-first: {y:<18} {n}")
    print("  changed pairs:")
    for i, a, b in pairs:
        if a[0]["new_text"] != b[0]["new_text"]:
            print(f"    {i:>4} {''.join(a[0]['old_text'])!r:<22} current {''.join(a[0]['new_text'])!r:<22} [{a[1]}] | "
                  f"word-first {''.join(b[0]['new_text'])!r:<22} [{b[1]}]  (crop's word {b[0]['word_first']['token']!r} {b[0]['word_first']['p']:.2f})")


if __name__ == "__main__":
    main()
