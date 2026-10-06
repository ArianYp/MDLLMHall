"""Wrong refills of the word-first run (oracle + zoom, A100): what the crop proposed and what is really in the image.

For each refill whose span still holds an object not in the image: AMBER truth objects, the flagged text and the refill,
the crop's top-5 at the flagged position (step 1 of word-first), the best correct and best wrong word of the first pass,
and whether each wrong word is on AMBER's absent list ("absent") or just not annotated ("unlisted", maybe present).

CPU only (spaCy). Example (from MDLLM/): python results/amber_g/hallu_study/wordfirst_wrong_examples.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler

EVENTS = os.path.join(HERE, "rag", "slot_refill", "oracle_zoom_wordfirst", "predictions_events.json")


def main():
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe
    lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())

    def wrong_words(item_id, tokens):
        _, safe, hallu = labeler.word_lists(item_id)
        out = []
        for t in tokens:
            w = t.strip().lower()
            if not w.isalpha() or lem(w) not in objects:
                continue
            l = lem(w)
            if l in safe or any(labeler.similar(l, s) for s in safe):
                continue
            listed = l in hallu or any(labeler.similar(l, h) for h in hallu)
            out.append((l, "absent" if listed else "unlisted"))
        return out

    caps = json.load(open(EVENTS))["captions"]
    rows = []
    for c in caps:
        for e in c["events"]:
            flagged = lem(e["old_text"][e["remasked"].index(e["pos"])])
            if flagged == "can":
                continue
            wrong = wrong_words(c["id"], e["new_text"])
            if wrong:
                rows.append((c["id"], e, flagged, wrong))

    same = [r for r in rows if r[2] in [w for w, _ in r[3]]]
    other = [r for r in rows if r not in same]
    print(f"wrong refills: {len(rows)} (same word {len(same)}, other word {len(other)})")
    print("wrong words by AMBER annotation:", dict(Counter(k for r in rows for _, k in r[3])))
    print("most repeated wrong words:", Counter(w for r in rows for w, _ in r[3]).most_common(15))

    for title, group in (("SAME WRONG WORD AGAIN", same), ("A DIFFERENT WRONG WORD", other)):
        print(f"\n=== {title} ({len(group)}) ===")
        for item_id, e, flagged, wrong in group:
            truth = labeler.word_lists(item_id)[0]["truth"]
            fp, wf = e["first_pass"], e["word_first"]
            bc = fp["best_correct"] or ["-", 0.0]
            print(f"[{item_id}] {''.join(e['old_text'])!r} -> {''.join(e['new_text'])!r}  wrong: {wrong}")
            print(f"      truth: {', '.join(truth)}")
            print(f"      crop top5: {', '.join(f'{t.strip()} {p:.2f}' for t, p in wf['top5'])}")
            print(f"      mass correct {fp['p_correct']:.2f} (best {bc[0].strip()} {bc[1]:.2f}) | wrong {fp['p_wrong']:.2f} "
                  f"(best {fp['best_wrong'][0].strip()} {fp['best_wrong'][1]:.2f}) | neutral {fp['p_neutral']:.2f} | box {e['box']}")


if __name__ == "__main__":
    main()
