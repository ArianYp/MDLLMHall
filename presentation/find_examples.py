"""List refills of the best oracle run (zoom + word-first + sinks removed, A100) that replace a hallucinated word with a correct object.

For each: the baseline vs final hallucinated nouns of the caption, the refill, the crop's top-5 and the crop box.
CPU only. Example (from MDLLM/): python presentation/find_examples.py
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from amber_noun_labels import AmberLabeler

SR = os.path.join(ROOT, "results", "amber_g", "hallu_study", "rag", "slot_refill")
RUN = os.path.join(SR, "oracle_zoom_wordfirst_nosink")
BASELINE = os.path.join(ROOT, "results", "amber_g", "mmada_predictions.json")


def main():
    labeler = AmberLabeler()
    objects = labeler.object_words - labeler.global_safe
    lem = lambda t: labeler.lemmatizer.lemmatize(t.strip().lower())
    base = {int(r["id"]): r["response"] for r in json.load(open(BASELINE))}
    final = {int(r["id"]): r["response"] for r in json.load(open(os.path.join(RUN, "predictions.json")))}
    caps = json.load(open(os.path.join(RUN, "predictions_events.json")))["captions"]

    def hallu(i, text):
        return [r["word"] for r in labeler.label(i, text) if r["label"] == "hallucinated"]

    for c in caps:
        i = c["id"]
        for e in c["events"]:
            _, safe, _ = labeler.word_lists(i)
            new = [lem(t) for t in e["new_text"] if t.strip().isalpha() and lem(t) in objects]
            kinds = ["correct" if w in safe or any(labeler.similar(w, s) for s in safe) else "wrong" for w in new]
            if new and all(k == "correct" for k in kinds) and e["old_text"] != e["new_text"]:
                hb, hf = hallu(i, base[i]), hallu(i, final[i])
                print(f"[{i}] {''.join(e['old_text'])!r} -> {''.join(e['new_text'])!r} | step {e['step']} | box {e['box']} | "
                      f"crop top5 {e['word_first']['top5'] if e.get('word_first') else None}")
                print(f"      caption hallucinated: baseline {hb} -> final {hf}")


if __name__ == "__main__":
    main()
